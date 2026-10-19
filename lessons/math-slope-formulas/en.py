# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-slope-formulas. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-slope-formulas", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The derivative as a slope](/course/open/math-slope?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="The loss is 10.000 at $w = 2$, and 9.996 at $w = 2.001$. What slope do you measure, with $h$ = 0.001?",
        options={
            "change": "−0.004",
            "right": "−4",
            "flipped": "4",
        },
        correct="right",
        why={
            "change": "That is the change in the loss itself. What do you do with it in step 3 of the recipe? See [The derivative as a slope](/course/open/math-slope?lang=en).",
            "flipped": "The loss went down. Which sign does the slope get then? See *The idea* in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "right": "The change is 9.996 − 10.000 = −0.004, and divided by 0.001 that is −4.",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def quick_check_way():
    quick_way = L.choice(
        "quick-way",
        question="The slope of the loss at your $w$ is −4. What lowers the loss?",
        options={
            "bigger": "Making $w$ bigger",
            "smaller": "Making $w$ smaller",
            "stay": "Leaving $w$ as it is",
        },
        correct="bigger",
        why={
            "smaller": "A negative slope: what does the loss do when $w$ goes up? See the recipe in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "stay": "At the lowest point, the slope is 0. Is it 0 here? See *The idea* in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "bigger": "A negative slope means the loss goes down when $w$ goes up.",
        },
    )
    quick_way
    return (quick_way,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Measuring a slope takes two calculations, and the answer is only close: it depends a little on $h$. For simple functions there is a better way: a short formula that gives the exact slope at once. A model learns from slopes millions of times, so being exact and fast matters.

    **An everyday example.** A car speeds up from standing still: after $t$ seconds it has gone $t^2$ meters. Time it over the next 0.1 second at $t = 3$, and it covers 0.61 m: you measure 6.1 m/s. The speedometer shows exactly 6 m/s, because the speed follows a formula: $2t$.
    """)
    return


@app.cell(hide_code=True)
def predict_pattern():
    guess_pattern = L.guess(
        "pattern",
        question="You measure the slope of $x^2$ at $x = 1$, at $x = 2$ and at $x = 3$, with a tiny $h$. What do you think comes out?",
        options={
            "same": "Always 2",
            "double": "2, 4 and 6",
            "square": "1, 4 and 9",
        },
        reveal="**2, 4 and 6: twice $x$.** The slope of $x^2$ follows a short formula: $2x$. Check it below.",
    )
    guess_pattern
    return (guess_pattern,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        For many functions, the slope follows a short formula: the slope formula. For a power of $x$, the power rule gives it: the power, $n$, comes to the front, and the new power is $n - 1$.
        """),
        L.formula(r"f(x) = x^n \quad\Longrightarrow\quad [[grad:f'(x)]] = n \cdot x^{n-1}"),
        mo.md(r"""
        So $x^2$ has the slope $2x$, and $x^3$ the slope $3x^2$. A straight line $a \cdot x$ has the same slope everywhere: $a$, as the weight $w$ of a model is the slope of its line.

        | Function | Slope formula | Slope at $x = 3$ |
        |---|---|---|
        | $x^2$ | $2x$ | 6 |
        | $x^3$ | $3x^2$ | 27 |
        | $a \cdot x$, such as $5x$ | $a$, such as 5 | 5 |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **slope formula** | | a short formula that gives the exact slope of a function at every $x$, such as $2x$ for $x^2$ | `2 * x` |
        | **power rule** | | the slope formula of $x$ to the power $n$: $n$ times $x$ to the power $n - 1$ | for $x^3$: `3 * x ** 2` |
        | $f'(x)$ | f prime of x | the slope of the function $f$ at $x$ | |

        Move the slider, or drag across the chart. The curve is $x^2$; the <span class="c-grad">green line</span> has its slope at the $x$ you chose.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_x_cell():
    def _frame(x):
        return (C.picture(x, {"title": "The slope of x²",
                              "alt": f"The curve of x squared from −3 to 3, a valley with its lowest point at x = 0. At x = {L.number(x, 1)} there is a point, with a short green line that has the slope there: {L.number(2 * x, 1)}."}),
                f"""At <i>x</i> = {L.number(x, 1)}: measured with <i>h</i> = 0.001, the <span class="c-grad">slope</span> is <span class="c-grad ea-num">{L.number(C.measured(x), 3)}</span>; the formula 2<i>x</i> gives <span class="c-grad ea-num">{L.number(2 * x, 3)}</span>.""")

    explore_x = L.explore("slope-x", "x", _frame, start=1.5, low=-3, high=3, step=0.5, role="", task="Try a few $x$, and compare the two slopes under the chart: by how much do they differ?")
    explore_x
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    For $x^2$, the measured slope is always exactly $h$ too high, here 0.001: close, but not exact. The formula is exact.

    ### Recipe

    To find a slope with a formula:

    1. Write the function as a power of $x$, such as $x^2$ or $x^3$, or as a line $a \cdot x$.
    2. Use the rule: $x^n$ has the slope $n \cdot x^{n-1}$; $a \cdot x$ has the slope $a$.
    3. Fill in $x$.
    4. Check it by measuring, with a tiny $h$: the two should be close.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The slope of $x^2$ at $x = 3$, from the formula and measured. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_formula():
    def _f(x):
        return x ** 2

    _x = 3
    # 1. and 2. x² is x to the power 2: the 2 comes to the front, the power becomes 1
    # 3. fill in x
    print(2 * _x)
    # 4. check it by measuring, with a tiny h
    _h = 0.001
    print((_f(_x + _h) - _f(_x)) / _h)
    # a line, 5x, measured at the same x: its slope is 5
    print((5 * (_x + _h) - 5 * _x) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The formula gives exactly 6, the measurement 6.000999…: close, so the formula passes the check. The measurement is 0.001 too high, as on the slider: for $x^2$, a step of $h$ always measures $2x + h$. The line $5x$ measures 4.99999…: for a line, measuring is exact, apart from the last decimals, and the slope is 5 at every $x$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_x4():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The slope of x⁴
    """)
    return


@app.cell(hide_code=True)
def choice_x4_cell():
    choice_x4 = L.choice(
        "x4",
        question="What is the slope formula of $x^4$?",
        options={
            "4x": "$4x$",
            "x3": "$x^3$",
            "right": "$4x^3$",
        },
        correct="right",
        why={
            "4x": "Does $x$ keep a power? See the power rule in *The idea*: the power goes down by one, not to one.",
            "x3": "The power went down by one, but what comes to the front? See the power rule in *The idea*.",
            "right": "The power rule: the 4 comes to the front, and the power goes down by one: $4x^3$.",
        },
    )
    choice_x4
    return (choice_x4,)


@app.cell(hide_code=True)
def text_exercise_power():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The power rule as a function

        Finish `power_slope(n, x)`. It gives back the slope of $x$ to the power $n$, at $x$, from the power rule. Replace the `...`. The check calls `power_slope(3, 2)`: the slope of $x^3$ at $x = 2$, which should give `12`. It also tries other powers and other $x$.
        """),
        L.task("Make `power_slope` give back the slope from the power rule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_power():
    box_power = L.code_box("power", lines=4)
    box_power
    return (box_power,)


@app.cell(hide_code=True)
def check_power(box_power):
    result_power = L.check("power", box_power.value)
    result_power
    return (result_power,)


@app.cell(hide_code=True)
def hints_power():
    L.hints(
        "power",
        [
            "The power rule in *The idea*: what comes to the front, and what happens to the power?",
            "In Python, $x$ to a power is `x ** ...`. Mind which part belongs to the new power.",
            "`return n * x ** (___)`.",
        ],
        why="`n * x ** (n - 1)`: the old power $n$ to the front, and the power one lower. The brackets make Python work out `n - 1` first. `power_slope(3, 2)` is 3 · 2² = 12.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Check a claim

        Someone says: "The slope formula of $x^3$ is $3x$." Check it by measuring. The box has the function `f`, which gives $x^3$, the point $x = 2$ and a tiny step `h`. Put the slope of `f` at `x` in `measured`, and compare it with what $3x$ gives. Replace the `...`. The last line is `measured`, so the check sees it. The check also tries another $x$.
        """),
        L.task("Put the measured slope of `f` at `x` in `measured`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_claim():
    box_claim = L.code_box("claim", lines=9)
    box_claim
    return (box_claim,)


@app.cell(hide_code=True)
def check_claim(box_claim):
    result_claim = L.check("claim", box_claim.value)
    result_claim
    return (result_claim,)


@app.cell(hide_code=True)
def hints_claim():
    L.hints(
        "claim",
        [
            "Measure as in step 4 of the worked example: `f` where you are, `f` a tiny step further, and divide.",
            "Here the function is `f` and the point is `x`. By what do you divide the change?",
            "`measured = (f(x + h) - ___) / h`.",
        ],
        why="`(f(x + h) - f(x)) / h` measures about 12.006 at $x = 2$. The claim $3x$ gives 6, far off; the power rule gives $3x^2$, and 3 · 2² = 12. The measurement shows the claim is wrong.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The derivative as a slope](/course/open/math-slope?lang=en) you measured slopes with a tiny step. Now you know that for powers of $x$ a formula gives the exact slope at once, and that measuring is a good way to check a formula. A loss is built from pieces like these; later lessons show how to put their slopes together. In the next lesson, *Rolling downhill: gradient descent*, the slope tells a model which way to move its parameter, step after step.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_x4, result_claim, result_power):
    L.wrap_up(choice_x4, result_power, result_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
