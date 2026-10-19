# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-log-scales. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-log-scales", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log10():
    quick_log10 = L.choice(
        "quick-log10",
        question="What does `np.log10(1000)` give?",
        options={
            "divided": "`100.0`",
            "natural": "`6.907755278982137`",
            "right": "`3.0`",
        },
        correct="right",
        why={
            "divided": "That is 1000 / 10. Which exponent of 10 gives 1000? See *The idea* in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "natural": "That is `np.log(1000)`, with base $e$. Which base does `np.log10` have? See the table of new words in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "right": "$10^3 = 1000$, so the log with base 10 of 1000 is 3.",
        },
    )
    quick_log10
    return (quick_log10,)


@app.cell(hide_code=True)
def quick_check_product():
    quick_product = L.choice(
        "quick-product",
        question=r"What is $\log(a \cdot b)$?",
        options={
            "right": r"$\log(a) + \log(b)$",
            "times": r"$\log(a) \cdot \log(b)$",
            "sum": r"$\log(a + b)$",
        },
        correct="right",
        why={
            "times": "The log turns multiplying into something else. Into what? See the formula in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "sum": "Inside the log, the product stays a product. What happens to it outside? See the formula in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "right": "The log turns multiplying into adding: the log of a product is the sum of the logs.",
        },
    )
    quick_product
    return (quick_product,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Some numbers span many factors of 10. In a text, the most common word appears thousands of times, and most words only a few times. A small language model has a few million parameters, and a large one hundreds of billions. On a normal axis, everything but the biggest numbers squeezes into a line at the bottom. A log scale gives each factor of 10 the same room. And some relations that curve on a normal axis become a straight line on log axes, such as how much better a model gets as it grows: researchers draw that on exactly such charts.

    **An everyday example.** The scale for earthquakes: each step up means 10 times as much shaking. A 6 is not a bit more than a 5: it is 10 times as much.
    """)
    return


@app.cell(hide_code=True)
def predict_factor():
    guess_factor = L.guess(
        "factor",
        question="A chart has the gridlines 1, 10, 100 and 1000 up its side, equally far apart. A straight line goes from 1 at x = 0 to 1000 at x = 3. What do you think happens to y each time x goes up by 1?",
        options={
            "plus333": "y goes up by 333",
            "times10": "y is multiplied by 10",
            "plus10": "y goes up by 10",
        },
        reveal="**y is multiplied by 10.** Each step up the axis is a factor of 10: from 1 to 10, to 100, to 1000. A straight line climbs the same number of steps per x, so it multiplies y by the same factor each time. On a normal axis, the same numbers would make a curve that shoots up at the end.",
    )
    guess_factor
    return (guess_factor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        On a normal axis, equal steps are equal differences: 0, 10, 20, 30. On a **log scale**, equal steps are equal factors: 1, 10, 100, 1000. The axis puts each number at the place of its log with base 10: 100 at 2, 1000 at 3. That is also why 0 and negative numbers have no place on it.

        Two kinds of growth become a straight line:

        - Growth by the same factor per step, $y = a \cdot b^x$, is a straight line when only the y-axis has a log scale: $\log(y) = \log(a) + x \cdot \log(b)$.
        - A **power law**, $y = a \cdot x^k$, is a straight line on a **log-log plot**, with a log scale on both axes: $\log(y) = \log(a) + k \cdot \log(x)$. For a whole number $k$, $x^k$ is $k$ factors $x$, and the log turns each factor into a term, so $\log(x^k) = k \cdot \log(x)$; that rule holds for every $k$, also $-1$ or $0.5$.

        The slope of that straight line is the exponent $k$. From two points on it, with logs in any base, as long as you use the same base up and along:
        """),
        L.formula(r"k = \frac{\log(y_2) - \log(y_1)}{\log(x_2) - \log(x_1)}"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **log scale** | | an axis on which equal steps are equal factors, such as 1, 10, 100, 1000 | `plt.yscale("log")` |
        | **log-log plot** | | a chart with a log scale on both axes | `plt.loglog(x, y)` |
        | **power law** | | $y = a \cdot x^k$: $y$ is a power of $x$; a straight line on a log-log plot, with slope $k$ | `y = a * x ** k` |
        """),
        L.watch_out("On a log scale, the room between gridlines is a factor, not a difference: from 1 to 10 takes as much room as from 100 to 1000. Read the numbers off the axis, not the distances."),
        mo.md(r"""
        Move the slider: it sets the exponent $k$ of $y = x^k$, drawn on normal axes and on a log-log plot. The points are at $x$ = 1, 10 and 100: equally far apart on the log scale.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        factor = 10 ** (k if k >= 0 else -k)
        times = L.number(factor, 0 if factor == round(factor) else 2)
        y = f"<i>y</i> gets {times} times as big" if k > 0 else (f"<i>y</i> is divided by {times}" if k < 0 else "<i>y</i> stays the same")
        normal = "a straight line" if k in (0, 1) else "a curve"
        return (C.picture(k, {"normal": "Normal axes", "loglog": "Log-log axes",
                              "alt": f"Two charts of y = x to the power {L.number(k, 1)}, for x from 1 to 100. On normal axes it is {normal}; on log-log axes it is a straight line with slope {L.number(k, 1)}, with points at x = 1, 10 and 100."}),
                f"""With <i>k</i> = {L.number(k, 1)}: each time <i>x</i> gets 10 times as big, {y}. On the log-log plot that is the same step each time, so the line is straight.""")

    explore_k = L.explore("power-k", "k", _frame, start=2, low=-1, high=2, step=0.5, role="", task="Try every $k$: on the log-log plot, how many gridlines does the line go up or down from $x$ = 1 to $x$ = 100?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    On the log-log plot, the line goes $2k$ gridlines up from $x$ = 1 to $x$ = 100: four for $k$ = 2, from 1 to 10,000, and two down for $k$ = −1, to 0.01. Along, that is two factors of 10, so up divided by along is $k$: the line is straight, with slope $k$, for every $k$. On normal axes it bends, up or down, unless $k$ is 0 or 1.

    ### Recipe

    To draw and read a chart with a log scale:

    1. Numbers that span many factors of 10? Put that axis on a log scale: `plt.yscale("log")`, `plt.xscale("log")`, or both at once with `plt.loglog(x, y)`.
    2. Read the axis in factors: each gridline is 10 times the one below.
    3. A straight line with only the y-axis on a log scale: the same factor per step.
    4. A straight line on a log-log plot: a power law. Its exponent is the slope: the change in $\log(y)$ divided by the change in $\log(x)$, both with the same base. With base 10, `np.log10`, the logs are easy to read off the axis.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A number of cells that doubles every hour, from 100, drawn with a log scale on the y-axis. Then a power law, the area of a square against its side, on a log-log plot, and its exponent from two points. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_log_scale():
    import matplotlib.pyplot as plt
    import numpy as np

    _hours = np.arange(10)
    _cells = 100 * 2 ** _hours
    # 1. from 100 to 51,200: several factors of 10, so a log scale on the y-axis
    plt.plot(_hours, _cells)
    plt.yscale("log")
    plt.xlabel("hour")
    plt.ylabel("cells")
    plt.show()
    # 2. and 3. the logs with base 10 of the counts go up by the same step every hour
    print(np.log10(_cells))
    # 1. from 1 to 10,000: a log scale on both axes, a log-log plot
    _sides = np.array([1, 2, 5, 10, 20, 50, 100])
    plt.loglog(_sides, _sides ** 2)
    plt.xlabel("side")
    plt.ylabel("area")
    plt.show()
    # 4. a straight line: its slope between the sides 1 and 100 is the exponent
    _side_a = 1
    _area_a = _side_a ** 2
    _side_b = 100
    _area_b = _side_b ** 2
    print(_area_a, _area_b)
    _k = (np.log10(_area_b) - np.log10(_area_a)) / (np.log10(_side_b) - np.log10(_side_a))
    print(_k)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first line is straight: every hour the count is multiplied by 2, the same factor. Matplotlib writes the gridlines as powers of 10: $10^2$ is 100, $10^3$ is 1000. The logs with base 10 go 2, 2.30, 2.60 and so on: up by 0.30 every hour, $\log_{10}(2)$. The squares make a straight line on the log-log plot too, and its slope is `2.0`: the exponent 2 of the square. A side 100 times as long gives an area $100^2$ = 10,000 times as big.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_halfway():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Halfway on a log scale
    """)
    return


@app.cell(hide_code=True)
def choice_halfway_cell():
    choice_halfway = L.choice(
        "halfway",
        question="An axis with a log scale has the gridlines 1, 10, 100 and 1000. Which number lies exactly halfway between the gridlines 10 and 100?",
        options={
            "middle": "55",
            "right": "About 32",
            "log": "1.5",
        },
        correct="right",
        why={
            "middle": "55 is halfway on a normal axis, where equal steps are equal differences. What are equal steps on a log scale? See *The idea*.",
            "log": "1.5 is the log with base 10 of the number halfway, its place on the axis. Which number has that log? See *The idea*.",
            "right": "Halfway is halfway in the logs with base 10: 1.5, between 1 and 2. $10^{1.5}$ is about 31.6. From 10 to 31.6 is a factor 3.16, and from 31.6 to 100 too.",
        },
    )
    choice_halfway
    return (choice_halfway,)


@app.cell(hide_code=True)
def text_exercise_plot():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Words on a log-log plot

        The course counted the words of *Tiny Shakespeare*, a file in the course folder with about 40,000 lines from plays by William Shakespeare, and put them in order from most to least common. For twelve places on that list, the box has the rank in `ranks` and how often that word appears in `counts`: the most common, "the", 6,283 times; word 5,000 only 3 times. Draw `counts` against `ranks` as a line, with both axes on a log scale. Replace the `...`.
        """),
        L.task("Draw `counts` against `ranks` on a log-log plot."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plot():
    box_plot = L.code_box("zipf-plot", lines=7)
    box_plot
    return (box_plot,)


@app.cell(hide_code=True)
def check_plot(box_plot):
    result_plot = L.check("zipf-plot", box_plot.value)
    result_plot
    return (result_plot,)


@app.cell(hide_code=True)
def hints_plot():
    L.hints(
        "zipf-plot",
        [
            "The ranks go along the bottom, the counts up the side, and both span several factors of 10.",
            "One function draws a line with both axes on a log scale. Which one? See step 1 of the recipe.",
            "`plt.loglog(___, ___)`, before `plt.show()`.",
        ],
        why="`plt.loglog(ranks, counts)` draws a line chart with a log scale on both axes, the ranks along the bottom.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_exponent():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The exponent of the words

        The line of exercise 2 is almost straight: a power law, count = $a \cdot \text{rank}^k$. Work out $k$ from two of its points: word 10, "in", appears 2,375 times, and word 1,000, "secret", 21 times. The box has them as `rank_a` and `count_a`, and `rank_b` and `count_b`. Put the exponent in `k`. Replace the `...`. The last line is `k`, so the check sees it. The check also tries another count.
        """),
        L.task("Put the exponent of the power law through the two points in `k`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_exponent():
    box_exponent = L.code_box("zipf-exponent", lines=9)
    box_exponent
    return (box_exponent,)


@app.cell(hide_code=True)
def check_exponent(box_exponent):
    result_exponent = L.check("zipf-exponent", box_exponent.value)
    result_exponent
    return (result_exponent,)


@app.cell(hide_code=True)
def hints_exponent():
    L.hints(
        "zipf-exponent",
        [
            "The exponent is the slope on the log-log plot: see the formula in *The idea*.",
            "Up the side are the logs of the counts, along the bottom the logs of the ranks. What do you divide by what?",
            "`k = (np.log10(count_b) - np.log10(___)) / (np.log10(___) - np.log10(rank_a))`.",
        ],
        why="With base 10, the logs of the counts go from 3.376 to 1.322, a change of −2.053; the logs of the ranks from 1 to 3, a change of 2. Divided: about −1.03.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Exponents and logarithms](/course/open/math-exp-log?lang=en) the log turned multiplying into adding; a log scale does the same on an axis, so equal factors take equal room. In [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) you drew line charts; one extra line puts an axis on a log scale. Later in the course you meet scaling laws, power laws for how much better a model gets with more parameters and data, drawn on log-log plots. In the next lesson, *From score to probability: the sigmoid*, a function built with exp turns any score into a probability.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_halfway, result_exponent, result_plot):
    L.wrap_up(choice_halfway, result_plot, result_exponent)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
