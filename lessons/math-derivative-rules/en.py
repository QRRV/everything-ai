# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-derivative-rules. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-derivative-rules", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) and [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="$y = u^3$, with $u = 2x$. What is the slope of $y$ at $x = 1$?",
        options={
            "outer": "12: the slope of $u^3$ at $u = 2$",
            "right": "24: the slope of $u^3$ at $u = 2$, times 2",
            "atx": "6: the slope of $u^3$ at 1, times 2",
        },
        correct="right",
        why={
            "outer": "That is the outer slope only. The chain rule multiplies it by the slope of the inner step, $u = 2x$. See [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "atx": "The outer slope $3u^2$ is worked out at the inner value $u = 2$, not at $x = 1$. See [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "right": "The outer slope $3u^2$ at $u = 2$ is 12, and the inner step $u = 2x$ has the slope 2: 12 · 2 = 24.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="For which $x$ does $\\log x$ give a number?",
        options={
            "right": "Only for $x$ above 0: no power of $e$ is 0 or below",
            "all": "For every $x$: each number has an exponent of $e$ that gives it",
            "one": "Only for $x$ above 1: below that, the log would be negative",
        },
        correct="right",
        why={
            "all": "$e$ to any power is above 0, so 0 and negative numbers have no exponent that gives them. See the watch-out in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "one": "Between 0 and 1 the log is negative, but it is there: $e^{-1}$ is about 0.37, so $\\log(0.37)$ is about −1. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "right": "Every power of $e$ is above 0, so only a number above 0 has an exponent that gives it: its log.",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    With a tiny step you can measure a slope, but only roughly, and each slope costs two runs of the function. Most formulas of machine learning are built from a few pieces: sums, products, powers, $e^x$ and $\log$. For each piece there is a short rule that gives the exact derivative, and the chain rule joins the pieces. The libraries that train neural networks work this way: one rule per piece, and a tiny step only to check.

    **An everyday example.** A cookbook with a thousand recipes uses only a handful of techniques: boiling, frying, baking, mixing. Once you know the techniques, you can cook any recipe built from them, without trying it out first.
    """)
    return


@app.cell(hide_code=True)
def predict_product():
    guess_product = L.guess(
        "x-times-x",
        question="The slope of $x$ is 1, at every $x$. What do you think the slope of $x \\cdot x$ is at $x = 4$?",
        options={
            "one": "1: the slopes multiply, 1 · 1",
            "right": "8: each factor adds to the change",
            "nine": "16: the value of $x \\cdot x$ there",
        },
        reveal="**8.** $x \\cdot x$ is $x^2$, and the power rule gives $2x$: 8 at $x = 4$. Multiplying the slopes gives 1, which is far too little: when both factors grow, the product grows because of each of them. That is the product rule: the slope of the first factor times the second, plus the first times the slope of the second, $1 \\cdot 4 + 4 \\cdot 1 = 8$.",
    )
    guess_product
    return (guess_product,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The derivative $f'(x)$ is the exact slope of a function $f$ at every $x$, as in [The derivative as a slope](/course/open/math-slope?lang=en). **Derivative rules** give the derivative of a formula from the derivatives of its parts. You know three already, written with a prime after brackets for the derivative of what is inside them: the power rule, $(x^n)' = n \cdot x^{n-1}$; a number times $x$, $(a \cdot x)' = a$; and the chain rule. A few more go without saying: a number on its own has the derivative 0, the derivative of a sum is the sum of the derivatives, and a number in front stays in front, $(3f)' = 3f'$. Three are new:
        """),
        L.formula(r"[[grad:(f \cdot g)']] = [[grad:f']] \cdot g + f \cdot [[grad:g']]"),
        L.formula(r"[[grad:(e^x)']] = e^x \qquad [[grad:(\log x)']] = \frac{1}{x}"),
        mo.md(r"""
        The **product rule**: when both factors change, the product changes because of each of them, so the rule has two terms. The derivative of $e^x$ is $e^x$ itself: at every $x$, the slope of $e^x$ equals its height. The derivative of $\log x$ is $\frac{1}{x}$: steep close to 0, and flatter and flatter further on.

        With the chain rule, these rules also work on an inner part $g(x)$: the outer derivative at the inner value, times the derivative of the inner part. So $(e^{g(x)})' = e^{g(x)} \cdot g'(x)$, and $(\log g(x))' = \frac{g'(x)}{g(x)}$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **derivative rules** | | rules that give the derivative of a formula from the derivatives of its parts | |
        | **product rule** | | the derivative of a product: $(f \cdot g)' = f' \cdot g + f \cdot g'$ | `df * g + f * dg` |
        | $(e^x)' = e^x$ | the derivative of e to the x is e to the x | the slope of $e^x$ equals its height | `np.exp(x)` |
        | $(\log x)' = \frac{1}{x}$ | the derivative of log x is 1 over x | the slope of $\log x$ is 1 divided by $x$ | `1 / x` |
        """),
        L.watch_out("The derivative of a product is not the product of the derivatives: $(x \\cdot x)' = 2x$, not $1 \\cdot 1$. And $e^x$ is not a power of $x$: $x$ is in the exponent, so the power rule does not apply, and $(e^x)'$ is not $x \\cdot e^{x - 1}$."),
        mo.md(r"""
        The picture: $e^x$ and $\log x$, each with the <span class="c-grad">slope its rule gives</span> at one point.
        """),
        mo.Html(C.picture({"exp": "eˣ: the slope equals the height", "exp_y": "eˣ",
                           "exp_note": f"slope {L.number(2.718281828459045, 2)} at x = 1",
                           "log": "log x: the slope is 1/x", "log_y": "log x",
                           "log_note": f"slope {L.number(0.5, 1)} at x = 2",
                           "alt": "Two charts. Left: the curve of e to the x from −2 to 2, with a green line touching it at x = 1, where the height and the slope are both about 2.72. Right: the curve of log x from 0 to 4, with a green line touching it at x = 2, with the slope 0.5."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the derivative of a formula:

    1. Find the last step the formula takes: a sum, a product, a power, $e$ to the power, or a log, of some part.
    2. Use that step's rule. For a sum, the sum of the derivatives; for a product, the product rule; for a power, $e$ to the power or a log of an inner part, the chain rule: the outer derivative at the inner value, times the derivative of the inner part.
    3. Do the same for every part inside, until only $x$ and numbers are left.
    4. Check the formula at one $x$ with a tiny step: $(f(x + h) - f(x)) / h$ should come close.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The derivative of $x^2 \cdot e^x$ at $x = 1$, and of $\log(x^2 + 1)$ at $x = 2$. The comments with `#` say which step of the recipe each part is. The second derivative is not finished: run the cell as it is, then complete step 3 where the comment says, and run it again.
    """)
    return


@app.cell
def example_rules():
    import numpy as np

    def _f(x):
        return x ** 2 * np.exp(x)

    # 1. the last step of _f is a product: u = x² and v = eˣ
    # 2. the product rule: u' · v + u · v'
    # 3. the parts: u' = 2x by the power rule, v' = eˣ
    def _df(x):
        return 2 * x * np.exp(x) + x ** 2 * np.exp(x)

    # 4. check at x = 1 with a tiny step
    _h = 0.001
    print(_df(1), (_f(1 + _h) - _f(1)) / _h)

    def _k(x):
        return np.log(x ** 2 + 1)

    # 1. the last step of _k is a log, of the inner part x² + 1
    # 2. the chain rule: the log's derivative at the inner value, 1 / (x² + 1), times the inner derivative
    # 3. the inner part: (x² + 1)' = 2x. The log's part is here; multiply it by the inner derivative
    def _dk(x):
        return 1 / (x ** 2 + 1)

    # 4. check at x = 2
    print(_dk(2), (_k(2 + _h) - _k(2)) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    _e = 2.718281828459045
    _h = 0.001
    _step_f = ((1 + _h) ** 2 * _e ** (1 + _h) - _e) / _h
    mo.md(f"""
    By the rules, the derivative of $x^2 \\cdot e^x$ at 1 is $2e + e = 3e$, about {L.number(3 * _e, 3)}; the tiny step gives {L.number(_step_f, 3)}, close but a little off. For $\\log(x^2 + 1)$ at 2 the rules give $\\frac{{1}}{{5}} \\cdot 4 = 0.8$ exactly; as long as step 3 is not finished, `_dk(2)` gives only the log's part, 0.2, and the tiny step shows that it is too small. The rules are exact, and they cost one run each; the tiny step only checks them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_exp():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. e to the power 3x
    """)
    return


@app.cell(hide_code=True)
def choice_exp_cell():
    choice_exp = L.choice(
        "exp-3x",
        question="What is the derivative of $e^{3x}$?",
        options={
            "same": "$e^{3x}$: $e$ to the power stays itself",
            "power": "$3x \\cdot e^{3x - 1}$: the power rule",
            "right": "$3e^{3x}$: the chain rule",
        },
        correct="right",
        why={
            "same": "$(e^x)' = e^x$, but here the exponent is $3x$, an inner part: the chain rule multiplies by its derivative, 3.",
            "power": "The power rule is for $x^n$, $x$ to a fixed power. In $e^{3x}$, $x$ is in the exponent, so the rule for $e$ to the power applies, with the chain rule.",
            "right": "The outer step, $e$ to the power, gives $e^{3x}$ at the inner value, and the inner step $3x$ has the derivative 3: $3e^{3x}$.",
        },
    )
    choice_exp
    return (choice_exp,)


@app.cell(hide_code=True)
def text_exercise_product():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The product rule

        `f(x)` is $x^3 \cdot \log(x)$, and `df(x)` should give its derivative. Replace the `...` with the formula, by the recipe. The check calls `df(2)`, which should give about `12.318`, and tries other values of `x` above 0.
        """),
        L.task("Make `df` give the exact derivative of `f`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_product():
    box_product = L.code_box("product", lines=10)
    box_product
    return (box_product,)


@app.cell(hide_code=True)
def check_product(box_product):
    result_product = L.check("product", box_product.value)
    result_product
    return (result_product,)


@app.cell(hide_code=True)
def hints_product():
    L.hints(
        "product",
        [
            "Step 1 of the recipe: the last step of `f` is a product. Of which two parts?",
            "The parts are $x^3$ and $\\log(x)$. The power rule gives the derivative of the first, the rule for the log that of the second; the product rule puts them together, in two terms.",
        ],
        why="At $x = 2$: $3 \\cdot 4 \\cdot \\log 2 \\approx 8.318$, plus $8 \\cdot \\frac{1}{2} = 4$, makes about 12.318.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_softplus():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A log of an inner part

        `k(x)` is $\log(1 + e^x)$, and `dk(x)` should give its derivative. Replace the `...` with the formula, by the recipe. The check calls `dk(1)`, which should give about `0.731`, and tries other values of `x`, also below 0.
        """),
        L.task("Make `dk` give the exact derivative of `k`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_softplus():
    box_softplus = L.code_box("softplus", lines=10)
    box_softplus
    return (box_softplus,)


@app.cell(hide_code=True)
def check_softplus(box_softplus):
    result_softplus = L.check("softplus", box_softplus.value)
    result_softplus
    return (result_softplus,)


@app.cell(hide_code=True)
def hints_softplus():
    L.hints(
        "softplus",
        [
            "Step 1 of the recipe: the last step of `k` is a log. Of which inner part?",
            "The chain rule: the log's derivative at the inner value, times the derivative of the inner part. The inner part is $1 + e^x$; its derivative comes from the rule for a sum and the rule for $e^x$.",
        ],
        why="At $x = 1$, the inner value is $1 + e \\approx 3.718$, so the log's derivative there is $\\frac{1}{3.718}$; the inner derivative is $e \\approx 2.718$. Together about 0.731.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The derivative as a slope](/course/open/math-slope?lang=en) you measured a slope with a tiny step, and in [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en) the power rule gave it exactly. Now a few rules give the exact derivative of any formula built from sums, products, powers, $e^x$ and $\log$, with the chain rule of [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) for every inner part. In the next lesson, *Partial derivatives and the gradient, formally*, a formula gets more than one input.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_exp, result_product, result_softplus):
    L.wrap_up(choice_exp, result_product, result_softplus)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
