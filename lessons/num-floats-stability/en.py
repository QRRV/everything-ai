# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson num-floats-stability. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("num-floats-stability", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) and [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loss():
    quick_loss = L.choice(
        "quick-loss",
        question="Softmax gives the right class a probability of 0.9. Training makes the logit of that class a little larger. What happens to the cross-entropy loss?",
        options={
            "right": "It goes down: towards 0, but never below it",
            "up": "It goes up: a larger logit gives a larger loss",
            "same": "It stays the same: 0.9 is already almost right",
        },
        correct="right",
        why={
            "up": "A larger logit for the right class gives it a larger probability, and minus the log of a larger probability is smaller. See [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "same": "The loss is minus the log of the probability, about 0.105 at 0.9; it only becomes 0 at a probability of 1. See [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "right": "The probability of the right class goes up, so minus its log goes down, towards 0, which it reaches only at a probability of 1.",
        },
    )
    quick_loss
    return (quick_loss,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="What is $\\log(e^2 \\cdot e^5)$?",
        options={
            "times": "10: the exponents multiplied",
            "right": "7: the exponents added",
            "power": "About 1,097: e to the power 7",
        },
        correct="right",
        why={
            "times": "Multiplying powers of $e$ adds their exponents: $e^2 \\cdot e^5 = e^7$. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "power": "$e^2 \\cdot e^5$ is indeed $e^7$, about 1,097, but the log undoes the $e$-power. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "right": "$e^2 \\cdot e^5 = e^7$, and the log undoes the $e$-power: 7. You need exactly this step for the trick in this lesson.",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A computer keeps every number in a fixed number of bits. Usually you never notice, but a model computes with very large and very small numbers: $e$ to the power of a logit, probabilities that are almost 0, sums of millions of terms. Then a step in between can be too large to store, although the answer you want is a normal number, and a the loss of a training run becomes `nan`. GPUs often use even fewer bits, so it happens there sooner.

    **An everyday example.** A pocket calculator shows ten digits. Ask it for $10^{100} / 10^{99}$ by typing $10^{100}$ first, and it shows an error, although the answer is 10.
    """)
    return


@app.cell(hide_code=True)
def predict_point_three():
    guess_sum = L.guess(
        "point-three",
        question="In Python, what do you think `0.1 + 0.2 == 0.3` gives?",
        options={
            "true": "`True`: 0.1 plus 0.2 is 0.3",
            "error": "An error: `==` does not work on floats",
            "right": "`False`: the two sides differ a little",
        },
        reveal="**`False`.** `0.1 + 0.2` gives `0.30000000000000004`. A computer stores numbers in powers of 2, and 0.1 has no exact form in powers of 2, just as 1/3 has none in decimals: 0.3333… stops somewhere. So 0.1, 0.2 and 0.3 are each rounded, and the rounding of `0.1 + 0.2` lands on a different number than the rounding of 0.3.",
    )
    guess_sum
    return (guess_sum,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **floating-point** number, or float, is stored as a number of about 16 significant digits times a power of 2. NumPy's usual float, float64, uses 64 bits for that. The power lets the same 64 bits hold $10^{-300}$ as well as $10^{300}$; the 16 digits mean almost every number is rounded. The gap between a float and the next one grows with the number: next to 1 it is about $2.2 \cdot 10^{-16}$, next to $10^{16}$ it is 2, so `1e16 + 1 == 1e16` is `True`.

        The largest float64 is about $1.8 \cdot 10^{308}$. A result above it becomes `inf`, infinitely large: **overflow**. $e^{709}$ still fits, $e^{710}$ does not. A result closer to 0 than about $10^{-308}$ keeps fewer digits, and from about $5 \cdot 10^{-324}$ down it becomes 0: **underflow**. $e^{-746}$ is 0, and NumPy does not warn you, as it does for overflow. GPUs often compute with fewer bits: in float32, $e^{89}$ already overflows; in float16, $e^{12}$.

        The loss of a classifier or a language model takes $e$ to the power of each logit, adds them up and takes the log: the **log-sum-exp**. With logits of a thousand, the $e$-powers overflow, although the answer, about a thousand, fits easily. The trick of [Many classes: softmax](/course/open/ml-softmax?lang=en) helps here too: take the largest value $m$ off first, and add it back after the log.
        """),
        L.formula(r"\mathrm{LSE}(\mathbf{z}) = \log \sum_k e^{z_k} = m + \log \sum_k e^{z_k - m}, \quad m = \max_k z_k"),
        mo.md(r"""
        That is allowed because $e^{z_k} = e^m \cdot e^{z_k - m}$: the sum is $e^m$ times the sum of the shifted powers, the log of a product is the sum of the logs, and $\log(e^m) = m$. Every exponent $z_k - m$ is 0 or below, so no power overflows; and the largest power is $e^0 = 1$, so the sum is at least 1 and you never take the log of 0.

        The cross-entropy loss of [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) is minus the log of the softmax probability of the right class $c$. Written out, that is the log-sum-exp minus the logit of the right class:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\log \frac{e^{z_c}}{\sum_k e^{z_k}} = \mathrm{LSE}(\mathbf{z}) - z_c"),
        mo.md(r"""
        Work out the softmax first and the log after, and a very small probability can underflow to 0, and the log of 0 is minus infinity. With this formula you never write the probability down, and the loss stays a normal number. That is why the cross-entropy loss in libraries usually takes the logits, not the probabilities.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **floating point**, float | | a number stored as a fixed number of significant digits, about 16 in float64, times a power of 2 | `0.1`, `1e16` |
        | **overflow** | | a result too large for a float; it becomes `inf` | `np.exp(710)` |
        | **underflow** | | a result too close to 0 for a float; it becomes 0 | `np.exp(-746)` |
        | **log-sum-exp**, $\mathrm{LSE}(\mathbf{z})$ | log-sum-exp of z | the log of the sum of $e$ to the power of each value, worked out as $m + \log \sum_k e^{z_k - m}$ | `m + np.log(np.sum(np.exp(z - m)))` |
        | $m$ | m | here: the largest value of $\mathbf{z}$ | `z.max()` |
        """),
        L.watch_out("A float is not wrong, but rounded. Never compare two results with `==`: `0.1 + 0.2 == 0.3` is `False`. Compare with a small margin instead: `abs(a - b) < 1e-9`, where `abs` gives a number without its sign. And a result that is `inf` or `nan` spreads: every sum, product or mean with it is `inf` or `nan` too, so one overflow early on spoils everything after it."),
        mo.md(r"""
        The picture: for numbers from 1 to $10^{16}$, the gap to the next float64. On these two log scales it is a straight line: the gap grows in step with the number (up close it climbs in tiny steps, because it only changes at each power of 2), so a float always keeps about 16 digits, wherever its point floats.
        """),
        mo.Html(C.gap_picture({"title": "The gap to the next float", "x": "the number", "y": "gap to the next float64",
                               "at_one": "at 1: 2.2 · 10⁻¹⁶", "at_big": "at 10¹⁶: 2",
                               "alt": "A line on two log scales, from the gap 2.2 times 10 to the power minus 16 at the number 1, straight up to the gap 2 at the number 10 to the power 16."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To work out the log-sum-exp without overflow:

    1. Find the largest value: $m = \max_k z_k$.
    2. Take it off every value: $z_k - m$, so every exponent is 0 or below.
    3. Take $e$ to the power of each, add them up, and take the log.
    4. Add $m$ back. For the cross-entropy loss, take off the logit of the right class: $\mathrm{LSE}(\mathbf{z}) - z_c$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three logits of about a thousand, first with the plain formula, then by the recipe. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_lse():
    import numpy as np

    _z = np.array([1000.0, 1001.0, 999.0])
    # the plain formula: e to the power 1000 overflows
    print(np.log(np.sum(np.exp(_z))))
    # 1. the largest value
    _m = _z.max()
    # 2. take it off: every exponent is 0 or below
    _shifted = _z - _m
    print(_shifted)
    # 3. the e-powers, their sum, and its log
    _log_sum = np.log(np.sum(np.exp(_shifted)))
    # 4. add m back. The loss of class 0 is that minus its logit, _z[0]: print it too
    print(_m + _log_sum)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The plain formula gives `inf`, and NumPy warns about the overflow, in a line with `RuntimeWarning` above the results. Shifted, the logits are −1, 0 and −2; their powers, about 0.37, 1 and 0.14, add up to about 1.50, whose log is about 0.408. With 1001 added back, the log-sum-exp is about 1001.408: just above the largest logit, as a log-sum-exp always is. With step 4 complete, the loss of class 0 is about 1001.408 − 1000 = 1.408: class 0 has the second-largest logit, and softmax gives it $e^{-1.408}$, about 0.24.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_underflow():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A very small number
    """)
    return


@app.cell(hide_code=True)
def choice_underflow_cell():
    choice_underflow = L.choice(
        "underflow",
        question="What does `np.exp(-800)` give?",
        options={
            "tiny": "A tiny positive number: floats go down to about 10⁻³⁰⁸",
            "right": "`0.0`: the answer is too close to 0 for a float",
            "minus": "`-inf`: the exponent is very negative",
        },
        correct="right",
        why={
            "tiny": "$e^{-800}$ is about $10^{-347}$, far below even the smallest float, about $5 \\cdot 10^{-324}$. Already $e^{-746}$ is 0.",
            "minus": "$e$ to any power is above 0, never negative: a very negative exponent gives a number close to 0, not a negative one.",
            "right": "$e^{-800}$ is about $10^{-347}$: underflow. Already $e^{-746}$ is 0.",
        },
    )
    choice_underflow
    return (choice_underflow,)


@app.cell(hide_code=True)
def text_exercise_lse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Log-sum-exp

        Write `logsumexp(z)`: the log of the sum of $e$ to the power of each value of the array `z`, by the recipe, so that it works for any values. The check calls `logsumexp(np.array([1, 2, 3]))`, which should give about `3.4076`, and tries values of a thousand and of minus a thousand.
        """),
        L.task("Make `logsumexp` give the log-sum-exp of `z`, without overflow."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lse():
    box_lse = L.code_box("logsumexp", lines=6)
    box_lse
    return (box_lse,)


@app.cell(hide_code=True)
def check_lse(box_lse):
    result_lse = L.check("logsumexp", box_lse.value)
    result_lse
    return (result_lse,)


@app.cell(hide_code=True)
def hints_lse():
    L.hints(
        "logsumexp",
        [
            "Steps 1 and 2 of the recipe: `m = z.max()`, and `z - m` for all values at once.",
            "Steps 3 and 4: `np.exp`, `np.sum` and `np.log` of the shifted values, and then `m` back on top.",
        ],
        why="For `[1, 2, 3]`: $m$ = 3, the shifted values are −2, −1 and 0, their powers add up to about 1.503, and $3 + \\log(1.503) \\approx 3.4076$, the same as the plain formula gives for these small values.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ce():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The loss straight from the logits

        Write `ce_from_logits(z, c)`: the cross-entropy loss of the logits `z` when the right class is `c`, as the log-sum-exp minus the logit of the right class, without overflow. The check calls `ce_from_logits(np.array([2, 1, 0]), 0)`, which should give about `0.4076`, and tries other classes and logits of a thousand and more.
        """),
        L.task("Make `ce_from_logits` give the cross-entropy loss of `z` for class `c`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ce():
    box_ce = L.code_box("ce-from-logits", lines=6)
    box_ce
    return (box_ce,)


@app.cell(hide_code=True)
def check_ce(box_ce):
    result_ce = L.check("ce-from-logits", box_ce.value)
    result_ce
    return (result_ce,)


@app.cell(hide_code=True)
def hints_ce():
    L.hints(
        "ce-from-logits",
        [
            "First the log-sum-exp of `z`, by the recipe, as in exercise 2: you may copy its lines into this box.",
            "Then take off the logit of the right class: `z[c]`.",
        ],
        why="For `[2, 1, 0]` the log-sum-exp is about 2.4076 and the logit of class 0 is 2: the loss is about 0.4076, minus the log of the softmax probability 0.665.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Taking the largest logit off, from [Many classes: softmax](/course/open/ml-softmax?lang=en), also keeps the log-sum-exp from overflowing, because $\log(e^m \cdot S) = m + \log S$, as in [Exponents and logarithms](/course/open/math-exp-log?lang=en). With it, the cross-entropy loss of [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) comes straight from the logits, without a probability that can become 0. You meet the trick again in *Attention weights from dot products*, where softmax gets large scores. In the next lesson, *How sure is a number? Confidence intervals and the bootstrap*, you find out how much a number you measured could have come out differently.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_underflow, result_lse, result_ce):
    L.wrap_up(choice_underflow, result_lse, result_ce)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
