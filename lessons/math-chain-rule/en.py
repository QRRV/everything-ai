# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-chain-rule. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-chain-rule", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_square():
    quick_square = L.choice(
        "quick-square",
        question="What is the slope of $x^2$ at $x = 5$?",
        options={
            "right": "10",
            "value": "25",
            "x": "5",
        },
        correct="right",
        why={
            "value": "That is the value of $x^2$ at 5, not its slope. What does the power rule give? See *The idea* in [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en).",
            "x": "Which number comes to the front? See the power rule in [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en).",
            "right": "The power rule gives the slope formula $2x$, and 2 × 5 = 10.",
        },
    )
    quick_square
    return (quick_square,)


@app.cell(hide_code=True)
def quick_check_line():
    quick_line = L.choice(
        "quick-line",
        question="What is the slope of the line $-4x$?",
        options={
            "line": "$-4x$",
            "right": "$-4$, at every $x$",
            "positive": "4, at every $x$",
        },
        correct="right",
        why={
            "line": "That is the line itself. What is the slope of a line $a \\cdot x$? See the table in [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en).",
            "positive": "The line goes down as $x$ goes up. Which sign does its slope get? See the table in [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en).",
            "right": "A line $a \\cdot x$ has the slope $a$ everywhere: here −4.",
        },
    )
    quick_line
    return (quick_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The power rule gives the slope of $x^2$ and $x^3$. But a loss is not a plain power of $w$. The loss of one student, with the model $\hat{y} = w \cdot x$, is $(y - w \cdot x)^2$: a square of something that contains $w$. You can measure its slope, but that costs two calculations, is never quite exact, and a large model would have to do it for millions of parameters. The chain rule gives the exact slope of such a function, piece by piece.

    **An everyday example.** On a bike, one turn of the pedals turns the back wheel 2.5 times, and one turn of the wheel takes you 2 meters. So one turn of the pedals takes you 2.5 × 2 = 5 meters: the rates multiply.
    """)
    return


@app.cell(hide_code=True)
def predict_rates():
    guess_rates = L.guess(
        "rates",
        question="$u = 3x$, and $y = 2u$. When $x$ goes up by 1, how much do you think $y$ goes up?",
        options={
            "outer": "2",
            "added": "5",
            "multiplied": "6",
        },
        reveal="**6.** When $x$ goes up by 1, $u$ goes up by 3. Each time $u$ goes up by 1, $y$ goes up by 2, so $y$ goes up 3 × 2 = 6: the rates multiply. Check it: $y = 2 \\cdot 3x = 6x$.",
    )
    guess_rates
    return (guess_rates,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Many functions are built in steps: an inner step $u = g(x)$, and an outer step $y = f(u)$. If $u$ changes $g'$ times as fast as $x$, and $y$ changes $f'$ times as fast as $u$, then $y$ changes $f' \cdot g'$ times as fast as $x$. That is the chain rule. Write each slope as $\frac{dy}{dx}$, and you can see which change is measured against which:
        """),
        L.formula(r"[[grad:\frac{dy}{dx}]] = [[grad:\frac{dy}{du}]] \cdot [[grad:\frac{du}{dx}]]"),
        mo.md(r"""
        For the loss of one student, the inner step is the <span class="c-residual">residual</span> $r = y - w \cdot x$. Here $x$ and $y$ are the student's fixed numbers, and only $w$ changes: when $w$ goes up by 1, the prediction $w \cdot x$ goes up by $x$, so $r$ goes down by $x$. So $r$ is a line in $w$ with the slope $-x$. The outer step is $r^2$, with the slope $2r$ from the power rule. Multiplied:
        """),
        L.formula(r"[[grad:\frac{d\mathcal{L}}{dw}]] = [[grad:\frac{d\mathcal{L}}{dr}]] \cdot [[grad:\frac{dr}{dw}]] = 2[[residual:r]] \cdot (-x)"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **chain rule** | | the slope of a function built in steps is the product of the slopes of the steps | `dloss_dr * dr_dw` |
        | $\frac{dy}{dx}$ | d y by d x | the slope of $y$ with respect to $x$: how fast $y$ changes when $x$ changes. $\frac{d\mathcal{L}}{dw}$ is the $\mathcal{L}'(w)$ of [The derivative as a slope](/course/open/math-slope?lang=en) | `dy_dx` |
        """),
        L.watch_out("Work out the outer slope at the inner value, not at $x$. For $(2x)^2$ at $x = 3$, the inner value is $u = 6$, so the outer slope $2u$ is 12, not 6."),
        mo.md(r"""
        The student studied 4 hours and got a 7. Move the slider, or drag across the chart: the curve is the loss $(7 - 4w)^2$, the <span class="c-param">point</span> is at the $w$ you chose, and the <span class="c-grad">green line</span> has the slope that the chain rule gives there.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_w_cell():
    def _frame(w):
        r, outer, inner, slope = C.parts(w)
        return (C.picture(w, {"title": "The loss of one student", "loss": "loss",
                              "alt": f"The loss (7 − 4w)² for w from 0 to 3.5, a valley with its lowest point at w = 1.75. At w = {L.number(w, 2)} there is a point, with a short green line that has the slope there: {L.number(slope, 0)}."}),
                f"""At <i>w</i> = {L.number(w, 2)}: the <span class="c-residual">residual</span> is <i>r</i> = 7 − 4 × {L.number(w, 2)} = <span class="c-residual ea-num">{L.number(r, 0)}</span>. The outer slope 2<i>r</i> is <span class="c-grad ea-num">{L.number(outer, 0)}</span>, the inner slope is <span class="c-grad ea-num">{L.number(inner, 0)}</span>, and their product is the <span class="c-grad">slope</span> of the loss: <span class="c-grad ea-num">{L.number(slope, 0)}</span>.""")

    explore_w = L.explore("chain-w", "w", _frame, start=1, low=0, high=3.5, step=0.25, role="param", task="Find the $w$ where the slope of the loss is 0. What is the residual there?")
    explore_w
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Where the residual is 0, at $w$ = 1.75, the outer slope $2r$ is 0, so the slope of the loss, $2r \cdot (-x)$, is 0 too: the lowest point.

    ### Recipe

    To find a slope with the chain rule:

    1. Split the function into an inner step and an outer step, such as $r = y - w \cdot x$ and $r^2$.
    2. Find the slope of each step with the rules you know: $-x$ for the line, $2r$ for the square.
    3. Fill in the numbers: first the inner value, then the outer slope at that value.
    4. Multiply the two slopes. If you like, check by measuring with a tiny $h$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The slope of the loss of the student, 4 hours of study and a grade of 7, at $w = 1.5$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_chain():
    _x, _y = 4, 7
    _w = 1.5
    # 1. the inner step, the residual, and the outer step, its square
    _r = _y - _w * _x
    _loss = _r ** 2
    # 2. and 3. the slope of each step, at these numbers
    _dloss_dr = 2 * _r
    _dr_dw = -_x
    # 4. multiply the slopes
    _slope = _dloss_dr * _dr_dw
    print(_r, _dloss_dr, _dr_dw, _slope)
    # and check by measuring, with a tiny h
    _h = 0.001
    print(((_y - (_w + _h) * _x) ** 2 - _loss) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The residual is 7 − 1.5 × 4 = 1, so the outer slope is 2 and the inner slope −4: the chain rule gives −8. Measuring gives about −7.984: close, so the chain rule passes the check. The slope is negative, so a bigger $w$ lowers the loss, as the chart of the loss shows.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nested():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A square of a line
    """)
    return


@app.cell(hide_code=True)
def choice_nested_cell():
    choice_nested = L.choice(
        "nested",
        question="What is the slope of $(3x)^2$ at $x = 1$?",
        options={
            "outer": "6",
            "inner": "3",
            "right": "18",
        },
        correct="right",
        why={
            "outer": "That is the outer slope alone, or the outer slope worked out at $x$ instead of at $u$. See the formula and the watch-out in *The idea*.",
            "inner": "That is the inner slope only, of $3x$. What is the outer slope at $u = 3$? See the recipe.",
            "right": "The inner step $u = 3x$ has the slope 3, and $u = 3$. The outer step $u^2$ has the slope $2u = 6$ there. Multiplied: 18. Check it: $(3x)^2$ is 9 times $x^2$, so its slope is 9 × $2x$ = $18x$.",
        },
    )
    choice_nested
    return (choice_nested,)


@app.cell(hide_code=True)
def text_exercise_slope():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The slope of the loss as a function

        Finish `loss_slope(w, x, y)`. It gives back the slope of the loss $(y - w \cdot x)^2$ with respect to $w$, for one example with feature `x` and label `y`, worked out with the chain rule. Replace the `...`. The check calls `loss_slope(1.5, 4, 7)`: the student from the worked example, which should give `-8.0`. It also tries other examples.
        """),
        L.task("Make `loss_slope` give back the slope of the loss, with the chain rule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slope():
    box_slope = L.code_box("loss-slope", lines=5)
    box_slope
    return (box_slope,)


@app.cell(hide_code=True)
def check_slope(box_slope):
    result_slope = L.check("loss-slope", box_slope.value)
    result_slope
    return (result_slope,)


@app.cell(hide_code=True)
def hints_slope():
    L.hints(
        "loss-slope",
        [
            "Steps 1 and 2 of the recipe: the inner step is the residual, the outer step its square. What is the slope of each?",
            "Work out `r` first. The outer slope uses `r`; the inner slope is the slope of the line $y - w \\cdot x$ in $w$.",
            "`r = y - w * x`, and then `return 2 * r * ___`.",
        ],
        why="`r = y - w * x` is the residual. The outer step $r^2$ has the slope `2 * r`, the inner step the slope `-x`, so `return 2 * r * -x`. For the student: 2 × 1 × (−4) = −8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The slope with respect to b

        The model now has a bias term: $\hat{y} = w \cdot x + b$, so the residual is $r = y - (w \cdot x + b)$. The box has the student, $w$, $b$ and $r$. Put the slope of the loss $r^2$ with respect to $b$ in `slope_b`, with the chain rule. Replace the `...`. The last line is `slope_b`, so the check sees it. The check also tries another $b$.
        """),
        L.task("Put the slope of the loss with respect to `b` in `slope_b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_b():
    box_b = L.code_box("slope-b", lines=7)
    box_b
    return (box_b,)


@app.cell(hide_code=True)
def check_b(box_b):
    result_b = L.check("slope-b", box_b.value)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "slope-b",
        [
            "The outer step is the same as before: $r^2$, with the slope $2r$. Only the inner step changes.",
            "When $b$ goes up by 1 and the rest stays the same, how much does $r = y - (w \\cdot x + b)$ change?",
            "`slope_b = 2 * r * ___`.",
        ],
        why="When $b$ goes up by 1, the prediction goes up by 1 and $r$ goes down by 1: the inner slope is −1. So `slope_b = 2 * r * -1`, which is −1.5 for this student: $r$ = 0.75.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Slope formulas you can trust](/course/open/math-slope-formulas?lang=en) you found slopes of powers. The chain rule takes a function built in steps, finds the slope of each step, and multiplies them: so you get the exact slope of a loss such as $(y - w \cdot x)^2$ without measuring. Later in the course, a neural network is a long chain of such steps, and using this rule over and over gives the slope for every parameter. In the next lesson, *Linear regression on many features*, you use these slopes to fit a model with many features at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_nested, result_b, result_slope):
    L.wrap_up(choice_nested, result_slope, result_b)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
