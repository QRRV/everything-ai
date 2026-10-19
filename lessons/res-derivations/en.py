# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-derivations. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-derivations", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From formula to code](/course/open/res-formula-to-code?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="A paper writes $q(k)$ for a distribution over $K$ classes, for every example. Which shape does it get in your code, for a batch of $B$ examples?",
        options={
            "k": "(K,)",
            "right": "(B, K)",
            "kb": "(K, B)",
        },
        correct="right",
        why={
            "k": "Every example has a distribution of its own: the batch axis comes first. See [From formula to code](/course/open/res-formula-to-code?lang=en).",
            "kb": "The batch axis comes first, as in the logits: a row per example. See [From formula to code](/course/open/res-formula-to-code?lang=en).",
            "right": "A row per example, a column per class. This lesson goes one step further: from a formula to the lines that led to it.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="$X$ is 0 or 2, each with probability ½. What is $\\mathrm{Var}(X)$?",
        options={
            "two": "2",
            "half": "0.5",
            "right": "1",
        },
        correct="right",
        why={
            "two": "2 is the mean of the squares of the values themselves, $\\tfrac12 \\cdot 0^2 + \\tfrac12 \\cdot 2^2$. The variance squares their distances to the expected value 1: $\\tfrac12 \\cdot 1^2 + \\tfrac12 \\cdot 1^2$. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "half": "Each value lies 1 from the expected value 1, and that distance is squared: $\\tfrac12 \\cdot 1 + \\tfrac12 \\cdot 1$. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "$\\mathbb{E}[X] = 1$, and $\\tfrac12 (0 - 1)^2 + \\tfrac12 (2 - 1)^2 = 1$. Note that $\\mathbb{E}[X^2] = 2$: not the variance. This lesson needs that difference.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Papers save space: "it follows that", "a simple calculation shows". The step they skip is often where an assumption hides, and an assumption that does not hold for your model makes the result wrong for it. Working through a derivation line by line, filling in what was skipped and checking every line with numbers, tells you what a result rests on.
    """)
    return


@app.cell(hide_code=True)
def predict_variance():
    guess_variance = L.guess(
        "relu-inputs",
        question="The formula of [Initialization](/course/open/nn-initialization?lang=en) is $\\mathrm{Var}(z) = n_\\text{in} \\cdot \\mathrm{Var}(w) \\cdot \\mathrm{Var}(x)$. A layer has 256 inputs and weights with $\\mathrm{Var}(w) = 1/256$. Its inputs come from a ReLU, $x = \\max(0, y)$ with $y$ standard normal, and $\\mathrm{Var}(x) \\approx 0.34$. What is $\\mathrm{Var}(z)$, measured on 20,000 inputs?",
        options={
            "formula": "About 0.34",
            "right": "About 0.5",
            "half": "About 0.17",
        },
        reveal=f"**About 0.5:** {L.number(C.guess_value(), 3)}. The formula assumed inputs with mean 0, and after a ReLU the mean is about 0.4. What counts is not $\\mathrm{{Var}}(x)$ but the mean of the squares, $\\mathbb{{E}}[x^2] = 0.5$. This lesson finds where that comes from.",
    )
    guess_variance
    return (guess_variance,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The derivation.** He and others (2015) derive their initialization in a few lines, for a layer $y_l = W_l x_l$ with bias terms 0, whose $n_l$ inputs come from a ReLU, $x_l = \max(0, y_{l-1})$. Every weight is drawn independently, with mean 0:
        """),
        L.formula(r"(1)\quad \mathrm{Var}[[[hidden:y_l]]] = n_l \, \mathrm{Var}[[[param:w_l]] [[hidden:x_l]]]"),
        L.formula(r"(2)\quad \mathrm{Var}[[[param:w_l]] [[hidden:x_l]]] = \mathrm{Var}[[[param:w_l]]] \, \mathbb{E}[[[hidden:x_l]]^2]"),
        L.formula(r"(3)\quad \mathbb{E}[[[hidden:x_l]]^2] = \tfrac{1}{2} \mathrm{Var}[[[hidden:y_{l-1}]]]"),
        L.formula(r"(4)\quad \mathrm{Var}[[[hidden:y_l]]] = \tfrac{1}{2} n_l \, \mathrm{Var}[[[param:w_l]]] \, \mathrm{Var}[[[hidden:y_{l-1}]]]"),
        mo.md(r"""
        So the variance stays the same from layer to layer when $\tfrac12 n_l \mathrm{Var}[w_l] = 1$: $\mathrm{Var}[w_l] = 2 / n_l$, He initialization. The numbers (1) to (4) are this lesson's: the paper numbers its equations in its own way, and you find these four lines in its section 2.2. The paper gives lines (2) and (3) one sentence each. Fill them in.

        **Line (1)** is the rule of [Initialization](/course/open/nn-initialization?lang=en): $y_l$ adds up $n_l$ independent products, and the variances of independent terms add up.

        **Line (2).** Write the definition of the variance out: $\mathrm{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$. The square is $X^2 - 2X\,\mathbb{E}[X] + (\mathbb{E}[X])^2$; the expected value of a sum is the sum of the expected values, and fixed numbers such as $\mathbb{E}[X]$ come out of an expected value unchanged. So the variance is $\mathbb{E}[X^2] - 2(\mathbb{E}[X])^2 + (\mathbb{E}[X])^2 = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$. For $X = w x$, with $w$ and $x$ independent, an expected value of a product is the product of the expected values: $\mathbb{E}[w^2 x^2] = \mathbb{E}[w^2]\,\mathbb{E}[x^2]$ and $\mathbb{E}[w x] = \mathbb{E}[w]\,\mathbb{E}[x]$. With $\mathbb{E}[w] = 0$, the second is 0, and $\mathbb{E}[w^2] = \mathrm{Var}[w]$. What is left is line (2): $\mathbb{E}[x^2]$, not $\mathrm{Var}[x]$, because $x$ does not have mean 0.

        **Line (3).** $y_{l-1}$ has mean 0 and is symmetric around 0, because its weights are. Then the positive half and the negative half of $y$ add the same to $\mathbb{E}[y^2]$. The ReLU keeps the positive half and makes the rest 0, so $\mathbb{E}[x^2] = \tfrac12 \mathbb{E}[y^2]$, and with mean 0, $\mathbb{E}[y^2] = \mathrm{Var}[y]$.

        **Check every line with numbers.** Draw many samples that meet the assumptions, work out both sides of a line, and compare: they should agree within the noise of the samples. Then break one assumption, and see which line no longer holds.

        The picture: the variance of $y_l$ after each of 10 layers of 256 neurons, as line (4) predicts it, the grey lines, and as it comes out, the dots, averaged over 8 runs. The y-axis has a log scale: equal steps down are equal factors.
        """),
        mo.Html(C.layers_picture({"title": "Line (4) and the measurement", "x": "layer", "y": "variance of y",
                                  "two": "Var[w] = 2/n", "one": "Var[w] = 1/n", "lang": L.lang,
                                  "alt": "Two grey lines with dots on a log scale over 10 layers. With Var[w] = 2/n the prediction stays at 1, and the dots lie between about 0.97 and 1. With Var[w] = 1/n the prediction halves at every layer, from 0.5 to about 0.001, and the dots lie on it."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **derivation** | | the lines that lead from what is assumed to a result, each following from the ones before by a rule | |
        | **numerical check** of a line | | both sides of the line worked out from many samples that meet the assumptions, and compared | |
        | $\mathbb{E}[x^2]$ | E of x squared | the expected value of the square of $x$: the mean of the squares, not the square of the mean | `np.mean(x ** 2)` |
        | $\mathrm{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | | the variance, written out: the mean of the squares minus the square of the mean | `np.var(x)` |
        """),
        L.watch_out("Two sides that agree on samples do not prove a line: your samples may meet an assumption that your model does not, such as weights with mean exactly 0 or a symmetric y. Check the line on the samples the derivation assumes, and then on samples like the ones in your model."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To follow a derivation:

    1. Write it out, a line per step, and number the lines.
    2. For every line, name the rule it uses and what it assumes, such as independence, mean 0 or symmetry.
    3. Where a step is skipped, write out a definition or a rule until the next line follows.
    4. Check every line with numbers: on samples that meet the assumptions, both sides agree within the noise; then break one assumption at a time, and see which line fails.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4 for line (1): the variance of a sum of $n = 16$ independent products, against $n$ times the variance of one product, on 100,000 samples. `np.var(a)` is the variance of an array: the square of `np.std(a)`.
    """)
    return


@app.cell
def example_line1():
    import numpy as np

    _rng = np.random.default_rng(0)
    _n = 16
    _w = _rng.normal(0, 0.25, size=(100_000, _n))  # weights with mean 0
    _x = np.maximum(0, _rng.normal(size=(100_000, _n)))  # inputs from a ReLU
    _terms = _w * _x  # the n products of every sample
    _left = np.var(_terms.sum(axis=1))  # Var[y]: y adds up the n products
    _right = _n * np.var(_terms)  # n times Var[w x]
    print(round(_left, 4), round(_right, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The two sides agree to within about 1%: the noise of 100,000 samples. Line (1) holds here. Its assumption is that the 16 products are independent: every weight and every input is drawn on its own. With $\mathrm{Var}[w] = 0.25^2$ and $\mathbb{E}[x^2] = 0.5$, both are about $16 \times 0.0625 \times 0.5 = 0.5$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_line2():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The two sides of line (2)

        Complete `line2_sides(ws, xs)`: for samples of the weights `ws` and of the inputs `xs`, give back the left side of line (2), $\mathrm{Var}[w x]$, and the right side, $\mathrm{Var}[w]\,\mathbb{E}[x^2]$. The check calls `line2_sides([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 3.0, 3.0])`, which should give about (3.3333, 3.3333), and tries lists in which the two sides differ.
        """),
        L.task("Make `line2_sides` give both sides of line (2) for the samples."),
    ])
    return


@app.cell(hide_code=True)
def exercise_line2():
    box_line2 = L.code_box("line2-sides", lines=10)
    box_line2
    return (box_line2,)


@app.cell(hide_code=True)
def check_line2(box_line2):
    result_line2 = L.check("line2-sides", box_line2.value)
    result_line2
    return (result_line2,)


@app.cell(hide_code=True)
def hints_line2():
    L.hints(
        "line2-sides",
        [
            "`ws * xs` gives the products. $\\mathbb{E}[x^2]$ is the mean of the squares: first square, then average.",
        ],
        why="`left = np.var(ws * xs)` and `right = np.var(ws) * np.mean(xs ** 2)`. In the example, the weights are ±1 with mean 0, so $\\mathrm{Var}[w] = 1$, and $\\mathbb{E}[x^2] = 20 / 6$; the products have the same variance.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_line3():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The two sides of line (3)

        Complete `line3_sides(ys)`: for samples `ys` of the values $y$ before the ReLU, give back the left side of line (3), $\mathbb{E}[x^2]$ with $x = \max(0, y)$, and the right side, $\tfrac12 \mathrm{Var}[y]$. The check calls `line3_sides([-2.0, -1.0, 0.0, 1.0, 2.0])`, which should give (1, 1), and tries lists that are not symmetric around 0.
        """),
        L.task("Make `line3_sides` give both sides of line (3) for the samples."),
    ])
    return


@app.cell(hide_code=True)
def exercise_line3():
    box_line3 = L.code_box("line3-sides", lines=9)
    box_line3
    return (box_line3,)


@app.cell(hide_code=True)
def check_line3(box_line3):
    result_line3 = L.check("line3-sides", box_line3.value)
    result_line3
    return (result_line3,)


@app.cell(hide_code=True)
def hints_line3():
    L.hints(
        "line3-sides",
        [
            "`np.maximum(0, ys)` is the ReLU of every sample. Square after the ReLU, not before.",
        ],
        why="`left = np.mean(np.maximum(0, ys) ** 2)` and `right = 0.5 * np.var(ys)`. For the example, $x = (0, 0, 0, 1, 2)$, so $\\mathbb{E}[x^2] = 5 / 5 = 1$, and $\\mathrm{Var}[y] = 10 / 5 = 2$.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your checks on many samples

    The cell below gives your two functions 200,000 samples: once as the derivation assumes, and once with one assumption broken: weights with mean 1 for line (2), and a $y$ with mean 0.5 for line (3). Until your functions are right, the course's stand in.
    """)
    return


@app.cell(hide_code=True)
def demo_lines(box_line2, box_line3, result_line2, result_line3):
    _d = C.demo(box_line2.value if result_line2.state == "correct" else None,
                box_line3.value if result_line3.state == "correct" else None)
    _r = _d["rows"]
    _who = ("your functions" if all(_d["yours"]) else "your `line2_sides` and the course's `line3_sides`" if _d["yours"][0]
            else "the course's `line2_sides` and your `line3_sides`" if _d["yours"][1] else "the course's functions")
    mo.md("\n".join([
        f"With {_who}:",
        "",
        "| line | samples | left | right |",
        "|---|---|---|---|",
        f"| (2) | as assumed | {L.number(_r['2-holds'][0], 4)} | {L.number(_r['2-holds'][1], 4)} |",
        f"| (2) | weights with mean 1 | {L.number(_r['2-broken'][0], 4)} | {L.number(_r['2-broken'][1], 4)} |",
        f"| (3) | as assumed | {L.number(_r['3-holds'][0], 4)} | {L.number(_r['3-holds'][1], 4)} |",
        f"| (3) | $y$ with mean 0.5 | {L.number(_r['3-broken'][0], 4)} | {L.number(_r['3-broken'][1], 4)} |",
        "",
        "As assumed, both sides agree within the noise. With weights whose mean is not 0, line (2) misses the term $\\mathbb{E}[w]^2 \\, \\mathrm{Var}[x]$; with a $y$ shifted up, the ReLU keeps more than half of it, and line (3) fails. A bias term that is not 0 shifts $y$ in the same way. And line (3) holds for ReLU only: GELU, the smoother ReLU of GPT-2, does not keep half of $\\mathbb{E}[y^2]$, even for a symmetric $y$.",
    ]))
    return


@app.cell(hide_code=True)
def text_exercise_assumption():
    mo.md(r"""
    ### Exercise 3. What line (3) assumes
    """)
    return


@app.cell(hide_code=True)
def choice_assumption_cell():
    choice_assumption = L.choice(
        "assumption",
        question="Which assumption about $y_{l-1}$ does line (3), $\\mathbb{E}[x_l^2] = \\tfrac12 \\mathrm{Var}[y_{l-1}]$, use?",
        options={
            "right": "That $y_{l-1}$ has mean 0 and is symmetric around it",
            "independent": "That $y_{l-1}$ is independent of the weights $w_l$",
            "unit": "That $y_{l-1}$ has variance 1",
        },
        correct="right",
        why={
            "independent": "Independence of the weights is what line (2) uses. Line (3) is about the ReLU alone: how much of $y$ it keeps.",
            "unit": "Line (3) holds for any variance: both sides scale with it. What it needs is that half of the square comes from each side of 0.",
            "right": "Then the ReLU keeps exactly half of $\\mathbb{E}[y^2]$, and $\\mathbb{E}[y^2] = \\mathrm{Var}[y]$. The demo above broke it with a mean of 0.5.",
        },
    )
    choice_assumption
    return (choice_assumption,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The He initialization of [Initialization](/course/open/nn-initialization?lang=en) is now derived line by line, with the variance of [Expected value and variance](/course/open/prob-expectation?lang=en) written out, and checked with numbers, as [From formula to code](/course/open/res-formula-to-code?lang=en) checked a formula against code. In a reproduction, a derivation you have checked tells you which assumptions your setting must meet before you can expect the paper's result. The next lesson, [Critical reading: claims versus evidence](/course/open/res-critical-reading?lang=en), asks whether a paper's experiments really support what it claims.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_assumption, result_line2, result_line3):
    L.wrap_up(result_line2, result_line3, choice_assumption)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
