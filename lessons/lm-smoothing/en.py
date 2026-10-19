# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-smoothing. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-smoothing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How good is a language model?](/course/open/lm-nll?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="On the same text, model A gets an average NLL of 2.3 and model B 2.6. Which predicted the text better?",
        options={
            "right": "A: it was less surprised by the text",
            "b": "B: its number is larger",
            "neither": "Neither: the NLL says nothing about how good a model is",
        },
        correct="right",
        why={
            "b": "The NLL is the average surprise: the larger it is, the worse the model predicted the text. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "neither": "The average NLL measures exactly that: how well the model predicted the real text. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "right": "Lower is better: A gave the characters that really came higher probabilities, on average.",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def quick_check_zero():
    quick_zero = L.choice(
        "quick-zero",
        question="A text has 1,000 pairs. A model gives 999 of them the probability 0.1, and one of them the probability 0. What is the average NLL?",
        options={
            "near": "About 2.3: one pair out of 1,000 hardly counts",
            "more": "A little more than 2.3: the zero pulls it up a bit",
            "right": "Infinite: one pair with probability 0 is enough",
        },
        correct="right",
        why={
            "near": "The surprise of probability 0 is minus log 0: infinite, and an average with an infinite term is infinite. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "more": "Not a little more: the surprise of that one pair is infinite. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "right": "Minus log 0 is infinite, and one infinite surprise makes the total, and so the average, infinite.",
        },
    )
    quick_zero
    return (quick_zero,)


@app.cell(hide_code=True)
def text_why():
    mo.md(f"""
    ## Why this matters

    In the last lesson, "Zebra" got an infinite NLL from your model of the plays, only because a "Z" was never followed by an "e" there. The model calls such a pair impossible, while it is just rare. Of the {C.grouped(65 * 65, L.lang)} cells of your table, {C.grouped(C.zero_cells(), L.lang)} are 0. Any new text with one such pair gets an infinite NLL, so the NLL cannot tell you how good the model is on text it has not seen, and that is the text a language model is for.

    **An everyday example.** A shop that sold no umbrellas in a dry month, and so orders none for the next one, has nothing to sell on the first rainy day: not yet is not the same as never.
    """)
    return


@app.cell(hide_code=True)
def predict_last():
    guess_last = L.guess(
        "last-part",
        question=f"Count the pairs of the first 90% of the plays, and look at the last 10%, text that the counts did not include. Of its {C.grouped(C.unseen_in_last()[1], L.lang)} pairs, how many do you think never came in the first 90%?",
        options={
            "none": "None: it is more of the same plays",
            "right": "A few: fewer than 1 in 100 of them",
            "many": "Many: more than 1 in 10, as most cells of the table are 0",
        },
        reveal=f"**A few: {C.grouped(C.unseen_in_last()[0], L.lang)} of the {C.grouped(C.unseen_in_last()[1], L.lang)}**, fewer than 1 in 500. Most zeros of the table are pairs that hardly ever come. But a few are enough: the model gives each of them probability 0, so the average NLL on the last 10% is infinite, while it is {C.number(C.nll_split(0)[0], 2, L.lang)} on the first 90%. The text is not stranger than the rest of the plays; the model is just sure that what it never saw cannot happen.",
    )
    guess_last
    return (guess_last,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Smoothing** gives every pair a little probability, even the pairs that never came. The simplest way is **add-k smoothing**: add a small number k to every count, before you divide by the row sums. A pair that never came then gets a probability of k divided by the new row sum: small, but not 0. With k = 1, it is called Laplace smoothing. For the pair of a and b, with $N$ the count matrix and 65 characters:
        """),
        L.formula(r"[[pred:P(b \mid a)]] = \frac{[[data:N]][a, b] + [[hyper:k]]}{(\text{sum of row } a \text{ of } [[data:N]]) + 65 [[hyper:k]]}"),
        mo.md(f"""
        The price: the pairs that did come give up a little of their probability, so the NLL on the counted text goes up a little. The gain is on text the model did not count, where the zeros made the NLL infinite. k is a number you choose. The table: the first 90% of the plays counted, with a few values of k, and the average NLL on that counted part and on the last 10%.

        | k | NLL on the counted 90% | NLL on the last 10% |
        |---|---|---|
        | 0 | {C.number(C.nll_split(0)[0], 4, L.lang)} | {C.number(C.nll_split(0)[1], 4, L.lang)} |
        | 0.01 | {C.number(C.nll_split(0.01)[0], 4, L.lang)} | {C.number(C.nll_split(0.01)[1], 4, L.lang)} |
        | 0.1 | {C.number(C.nll_split(0.1)[0], 4, L.lang)} | {C.number(C.nll_split(0.1)[1], 4, L.lang)} |
        | 1 | {C.number(C.nll_split(1)[0], 4, L.lang)} | {C.number(C.nll_split(1)[1], 4, L.lang)} |
        | 10 | {C.number(C.nll_split(10)[0], 4, L.lang)} | {C.number(C.nll_split(10)[1], 4, L.lang)} |
        | 100 | {C.number(C.nll_split(100)[0], 4, L.lang)} | {C.number(C.nll_split(100)[1], 4, L.lang)} |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **smoothing** | | giving every possible pair at least a little probability, even the pairs that never came | |
        | **add-k smoothing** | add k | adding the same small number k to every count before dividing by the row sums; with k = 1, Laplace smoothing | `N + k` |
        """),
        L.watch_out("Smoothing makes the NLL on the counted text worse, not better: there, k = 0 always looks best. You only see its gain on text the model did not count, so choose k on such text."),
        mo.md(r"""
        The picture: the 65 probabilities after "Z", in the order of the table, without smoothing and with k = 1.
        """),
        mo.Html(C.row_bars("Z", {"title": "After \"Z\": without and with smoothing", "before": "without smoothing: {n} of 65 are 0",
                                 "after": "with k = 1: none are 0", "axis": "the next character, in the order of the table",
                                 "alt": "Two rows of 65 bars, the probabilities of each character after Z. Without smoothing, only 4 bars are visible, 2 of them tiny, and 61 are 0. With k = 1, all 65 bars are visible: the 2 large ones are about a quarter lower, and the other 63 are small but not 0."})),
        mo.md(f"""
        A "Z" came {C.row_count("Z")} times in the plays, followed by only four different characters. With k = 1, the row sum becomes {C.row_count("Z")} + 65 = {C.row_count("Z") + 65}, and every character that never came after "Z" gets 1/{C.row_count("Z") + 65}. Together, those 61 get 61/{C.row_count("Z") + 65}, almost a quarter of the row, and that is what the four that did come lose: in a row with few counts, k = 1 is a lot.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To smooth a count matrix `N` with add-k smoothing, and choose k:

    1. Add k to every count: `M = N + k`.
    2. Divide every row by its new sum: `P = M / M.sum(axis=1, keepdims=True)`.
    3. To choose k, measure the NLL on text that was not counted, for a few values of k, and take the k with the lowest NLL.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The counts of "abab ba", and a new text, "ba ab", that was not counted: the pairs "a " and " a" never came in "abab ba". Without smoothing, its NLL would be infinite. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_smooth():
    import numpy as np

    _N = np.array([[0, 0, 1], [0, 0, 2], [1, 2, 0]])   # the counts of "abab ba"
    _row_of = {" ": 0, "a": 1, "b": 2}
    _new = "ba ab"   # a text that was not counted
    for _k in [0.01, 0.1, 1, 10]:
        # 1. k added to every count
        _M = _N + _k
        # 2. every row divided by its new sum
        _P = _M / _M.sum(axis=1, keepdims=True)
        # 3. the NLL on the text that was not counted
        _total = 0
        for _i in range(len(_new) - 1):
            _total = _total - np.log(_P[_row_of[_new[_i]], _row_of[_new[_i + 1]]])
        print(_k, _total / (len(_new) - 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With k = 0.01, the two pairs that never came get so little that the NLL is still high, about 2.59. With k = 1 it is lowest, about 1.05. With k = 10, every row is almost even, so the pairs that did come lose too much, and the NLL goes up again, to about 1.07. On this text, k = 1 is the best of the four.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A pair that never came
    """)
    return


@app.cell(hide_code=True)
def choice_unseen_cell():
    choice_unseen = L.choice(
        "unseen-prob",
        question="The counts in one row of a table with 65 characters add up to 935. With k = 1, what probability does add-k smoothing give a pair of that row that never came?",
        options={
            "zero": "0: it never came",
            "right": "0.001: 1 out of 935 + 65",
            "row": "1 out of 935: one count more",
            "even": "1 out of 65: as if every character were equal",
        },
        correct="right",
        why={
            "zero": "That is without smoothing. With k = 1, its count becomes 0 + 1.",
            "row": "Its count becomes 1, but the row sum grows too: each of the 65 counts gets 1 more.",
            "even": "That would be a row without any counts. This row has 935, and those still count.",
            "right": "Its count becomes 1, and the row sum 935 + 65 = 1,000, so 1/1,000 = 0.001.",
        },
    )
    choice_unseen
    return (choice_unseen,)


@app.cell(hide_code=True)
def text_exercise_smooth():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Smooth a count matrix

        `smooth(N, k)` gives the probabilities of the count matrix `N` with add-k smoothing. Replace the `...` with steps 1 and 2 of the recipe. The check calls `smooth(np.array([[3, 1, 0], [0, 0, 2]]), 1)`, which should give about `[[0.571, 0.286, 0.143], [0.2, 0.2, 0.6]]`, and tries other counts and other values of `k`.
        """),
        L.task("Make `smooth` add `k` to every count and divide every row by its new sum."),
    ])
    return


@app.cell(hide_code=True)
def exercise_smooth():
    box_smooth = L.code_box("smooth", lines=5)
    box_smooth
    return (box_smooth,)


@app.cell(hide_code=True)
def check_smooth(box_smooth):
    result_smooth = L.check("smooth", box_smooth.value)
    result_smooth
    return (result_smooth,)


@app.cell(hide_code=True)
def hints_smooth():
    L.hints(
        "smooth",
        [
            "Steps 1 and 2 of the recipe, as in the worked example.",
            "First a new matrix with `k` added to every count. Then the sum of each of its rows, kept as a column with `keepdims=True`, so that broadcasting divides each row by its own sum.",
            "`M = N + ___` and `return M / M.sum(axis=___, keepdims=True)`.",
        ],
        why="`[[3, 1, 0], [0, 0, 2]]` plus 1 is `[[4, 2, 1], [1, 1, 3]]`, with the row sums 7 and 5. Divided: `[[4/7, 2/7, 1/7], [1/5, 1/5, 3/5]]`, about `[[0.571, 0.286, 0.143], [0.2, 0.2, 0.6]]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ### Exercise 3. Choose k
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick-k",
        question="Look at the table in *The idea*. Which k would you choose for your model of the plays, to use on text it has not seen?",
        options={
            "zero": "0: its NLL on the counted 90% is the lowest",
            "small": "0.01: as little change to the counts as possible",
            "right": "1: its NLL on the last 10% is the lowest",
            "big": "100: no pair comes close to 0 any more",
        },
        correct="right",
        why={
            "zero": "On the counted text, k = 0 always looks best; on new text it gives an infinite NLL. See the watch-out in *The idea*.",
            "small": "Better than 0, but the NLL on the last 10% is higher than with k = 1: the pairs that never came still get too little.",
            "big": "With k = 100, the rows with few counts get almost even, so the pairs that did come lose too much there: the NLL on the last 10% is the highest of the table, after 0.",
            "right": "Of the values in the table, k = 1 gives the lowest NLL on text the counts did not include, and that is what the model is for.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The NLL of [How good is a language model?](/course/open/lm-nll?lang=en) became infinite for any text with a pair the model never saw; add-k smoothing gives every pair a little probability, and the NLL on new text shows which k works best. In the next lesson, *Project: the same bigram model, learned with gradient descent*, your model learns this table instead of counting it, and you see what takes the place of k there.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pick, choice_unseen, result_smooth):
    L.wrap_up(choice_unseen, result_smooth, choice_pick)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
