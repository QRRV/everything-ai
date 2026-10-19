# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-exp-log. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-exp-log", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Calculating without loops](/course/open/np-vectorization?lang=en) and [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_power():
    quick_power = L.choice(
        "quick-power",
        question="What does `np.array([1, 2, 3]) ** 2` give?",
        options={
            "right": "`array([1, 4, 9])`",
            "times": "`array([2, 4, 6])`",
            "sum": "`14`",
        },
        correct="right",
        why={
            "times": "That is `* 2`. What does `**` do? See the table of new words in [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "sum": "That adds the squares up as well. Does `**` do that? See the table of new words in [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "right": "`** 2` squares every number of the array: 1, 4 and 9.",
        },
    )
    quick_power
    return (quick_power,)


@app.cell(hide_code=True)
def quick_check_line():
    quick_line = L.choice(
        "quick-line",
        question="What does `plt.plot([0, 1, 2], [0, 1, 4])` draw?",
        options={
            "bars": "Three bars, as high as 0, 1 and 4",
            "points": "Three loose points, without a line",
            "right": "A line through the points (0, 0), (1, 1) and (2, 4)",
        },
        correct="right",
        why={
            "bars": "Bars come from another function. Which one? See the table of new words in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "points": "Loose points come from another function. Which one? See the table of new words in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "right": "`plt.plot` draws a line chart: the first x goes with the first y, and a line joins the points.",
        },
    )
    quick_line
    return (quick_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Some things grow by adding: a step of the same size each time. Others grow by multiplying: a factor each time, and then the numbers get huge very fast. The reverse happens too: multiply many probabilities, and the result gets so small that a computer rounds it to 0. The exponent describes growth by multiplying, and the logarithm turns multiplying into adding. Later in the course, a model for classification uses both: to turn a score into a probability, and to count its mistakes.

    **An everyday example.** A sheet of paper is about 0.1 millimeter thick. Fold it in half, and it is 0.2; fold again, 0.4. Each fold doubles it.
    """)
    return


@app.cell(hide_code=True)
def predict_folds():
    guess_folds = L.guess(
        "fold",
        question="You fold that sheet of 0.1 millimeter in half, 30 times, if only you could. How thick do you think it gets?",
        options={
            "mm": "About 3 millimeters",
            "m": "About 10 meters",
            "km": "More than 100 kilometers",
        },
        reveal="**More than 100 kilometers.** 30 folds double the thickness 30 times: 0.1 millimeter times $2^{30}$, and $2^{30}$ is about 1.07 billion. That makes about 107 kilometers. 3 millimeters would be adding 0.1 millimeter 30 times; doubling grows much faster.",
    )
    guess_folds
    return (guess_folds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A power multiplies a number, the base, by itself: $2^3 = 2 \cdot 2 \cdot 2 = 8$. The raised number, the **exponent**, says how many times; in Python that is `2 ** 3`. One step up in the exponent multiplies by 2 once more, which is why powers grow faster and faster. Any number to the power 0 is 1, and a negative exponent divides: $2^{-3} = 1/8$.

        AI uses one base above all: the number $e$, about 2.718. A later lesson shows why it is exactly this number; for now it is a base like 2 or 10. $e^x$, also written $\exp(x)$, is always positive, is 1 at $x = 0$, and grows faster and faster.

        The **logarithm** goes the other way: it gives the exponent you need. $\log_2(8) = 3$, because $2^3 = 8$. Without a base, in this course, log always means base $e$: $\log(e^x) = x$, and $e^{\log(x)} = x$. And that makes log useful: it turns multiplying into adding. Multiply two powers of the same base and you add their exponents: $2^2 \cdot 2^3 = 2^{2+3}$, two factors of 2, then three more. The log gives that exponent, so:
        """),
        L.formula(r"\log(a \cdot b) = \log(a) + \log(b)"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **exponent** | | the raised number in a power: how many times the base is multiplied by itself | `2 ** 3` |
        | $e$ | e | the number 2.718…, the base AI uses most | `np.e` |
        | $e^x$, $\exp(x)$ | e to the x, exp of x | $e$ to the power $x$; always positive | `np.exp(x)` |
        | **logarithm**, $\log(x)$ | log of x | the exponent that gives $x$; in this course, without a base, base $e$: the inverse of exp | `np.log(x)` |
        | $\log_2(x)$, $\log_{10}(x)$ | log base 2 of x | the exponent of 2, or of 10, that gives $x$ | `np.log2(x)`, `np.log10(x)` |
        """),
        L.watch_out("The log only exists for numbers above 0: no power of $e$ is 0 or negative. `np.log(0)` gives `-inf`, minus infinity, and the log of a negative number gives `nan`, not a number, both with a warning."),
        mo.md(r"""
        Move the slider. The point $(x, e^x)$ lies on exp, and the point $(e^x, x)$ on log: the same two numbers, swapped. That is what an inverse does, and it makes the two curves mirror images in the dashed line, where $y = x$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_x_cell():
    def _frame(x):
        y = math.exp(x)
        return (C.picture(x, {"title": "exp and log, mirror images",
                              "alt": f"The curves of exp and log, mirror images in the dashed line y = x. exp rises faster and faster and is always above 0; log rises slower and slower and only exists right of 0. A point at ({L.number(x, 2)}, {L.number(y, 3)}) on exp, and its mirror image at ({L.number(y, 3)}, {L.number(x, 2)}) on log."}),
                f"""At <i>x</i> = <span class="ea-num">{L.number(x, 2)}</span>: <i>e</i><sup><i>x</i></sup> = <span class="ea-num">{L.number(y, 3)}</span>, and log(<span class="ea-num">{L.number(y, 3)}</span>) = <span class="ea-num">{L.number(x, 2)}</span>. log undoes exp.""")

    explore_x = L.explore("exp-x", "x", _frame, start=1, low=-2, high=2, step=0.25, role="", task="Move $x$ below 0: between which numbers does $e^x$ lie then, and is its log positive or negative?")
    explore_x
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    For $x$ below 0, $e^x$ lies between 0 and 1: the point on exp is just above the x-axis. Its mirror image on log is below the x-axis: the log of a number between 0 and 1 is negative.

    ### Recipe

    To work with powers and logarithms:

    1. A power: the base, `**`, and the exponent; $e^x$ is `np.exp(x)`.
    2. A logarithm: ask which exponent gives the number. `np.log` for base $e$, `np.log2` and `np.log10` for base 2 and 10.
    3. Check with the inverse: `np.exp(np.log(x))` gives `x` back.
    4. A product of many numbers: add their logs instead.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Powers of 2, a logarithm that undoes one, base $e$, and a product as a sum of logs. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_exp_log():
    import numpy as np

    # 1. powers of 2: one step up in the exponent multiplies by 2 once more
    _n = np.arange(11)
    print(2 ** _n)
    # 2. which exponent of 2 gives 1024?
    print(np.log2(1024))
    # 1. and 2. base e: e itself, and the log of e to the power 3
    print(np.exp(1), np.log(np.exp(3)))
    # 3. the inverse: exp of the log of 5
    print(np.exp(np.log(5)))
    # 4. the log of a product, and the sum of the logs
    print(np.log(0.5 * 0.2), np.log(0.5) + np.log(0.2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The powers of 2 go 1, 2, 4, 8 and on to 1024: ten doublings. `np.log2(1024)` gives that 10 back. `np.exp(1)` is $e$, 2.718…, and the log of $e^3$ is 3. `np.exp(np.log(5))` gives 4.999999999999999: 5, with a rounding error in the last digit, the kind computers make with decimals. And the log of 0.5 × 0.2 = 0.1 is the same as the log of 0.5 plus the log of 0.2: −2.30.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_negative():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which one is negative?
    """)
    return


@app.cell(hide_code=True)
def choice_negative_cell():
    choice_negative = L.choice(
        "negative",
        question="Which of these numbers is negative?",
        options={
            "exp-neg": "$e^{-3}$",
            "right": r"$\log(0.5)$",
            "log5": r"$\log(5)$",
            "exp0": "$e^{0}$",
        },
        correct="right",
        why={
            "exp-neg": "A negative exponent divides: $e^{-3} = 1/e^3$, small, but above 0. See *The idea*.",
            "log5": "Which exponent of $e$ gives 5? $e^1$ is only 2.718, so more than 1. See the picture.",
            "exp0": "Any number to the power 0 is 1. See *The idea*.",
            "right": "$e$ to the power 0 is 1, so to get 0.5, below 1, you need a negative exponent: $\\log(0.5) \\approx -0.69$.",
        },
    )
    choice_negative
    return (choice_negative,)


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Folding to a kilometer

        The sheet of 0.1 millimeter again. How many times must you fold it to make it a kilometer thick: 1,000,000 millimeters? The box has `thickness` and `height`, both in millimeters. Put the number of folds in `folds` exactly as a logarithm gives it, decimals and all; do not round it to whole folds. Replace the `...`. The last line is `folds`, so the check sees it. The check also tries another height.
        """),
        L.task("In `folds`, put the number of folds that takes the paper from `thickness` to `height`, worked out with a logarithm."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("folds", lines=7)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "folds",
        [
            "By what factor must the paper grow? And how many doublings give that factor?",
            "The factor is `height / thickness`. Which logarithm counts doublings? See step 2 of the recipe.",
            "`folds = np.___(height / thickness)`.",
        ],
        why="`height / thickness` is the factor, 10,000,000, and `np.log2` gives the exponent of 2 that makes it: about 23.3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_product():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A product as a sum of logs

        Finish `log_product(ps)`: the log of the product of a list of numbers above 0, worked out as the sum of their logs, so that you never multiply. Replace the `...`. The check calls `log_product([0.5, 0.2, 0.1])`: that should give about `-4.605`, the log of 0.01. It also tries other lists, including very small numbers such as `1e-200`, Python's way of writing $10^{-200}$: 1 divided by 10, two hundred times.
        """),
        L.task("Make `log_product` add up the log of each number."),
    ])
    return


@app.cell(hide_code=True)
def exercise_product():
    box_product = L.code_box("log-product", lines=8)
    box_product
    return (box_product,)


@app.cell(hide_code=True)
def check_product(box_product):
    result_product = L.check("log-product", box_product.value)
    result_product
    return (result_product,)


@app.cell(hide_code=True)
def hints_product():
    L.hints(
        "log-product",
        [
            "The log of a product is the sum of the logs: see the formula in *The idea*.",
            "A loop over the numbers. What do you add to `total` for each number?",
            "`for p in ps:` and inside it `total = total + ___`.",
        ],
        why="Each round adds `np.log(p)` to `total`. For `[0.5, 0.2, 0.1]`: −0.693 − 1.609 − 2.303 = −4.605, the log of 0.5 × 0.2 × 0.1 = 0.01.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Calculating without loops](/course/open/np-vectorization?lang=en), `**` raised every number of an array to a power; `np.exp` and `np.log` also work on a whole array at once. Later in the course, exp turns a model's score into a probability, and log turns the product of many probabilities into a sum, as in exercise 3. In the next lesson, *Log scales and power laws*, you draw charts whose axis goes in factors instead of steps.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_negative, result_folds, result_product):
    L.wrap_up(choice_negative, result_folds, result_product)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
