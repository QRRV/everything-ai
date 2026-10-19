# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-plotting. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import plots
    from everything_ai.lesson import lesson

    L = lesson("py-plotting", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Lists](/course/open/py-lists?lang=en) and [Using code from modules](/course/open/py-imports?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="What does this program show?\n\n```python\ntemps = [12, 15, 9, 14]\nprint(len(temps))\n```",
        options={
            "last": "`14`",
            "count": "`4`",
            "index": "`3`",
        },
        correct="count",
        why={
            "last": "That is the last item. `len` counts the items. See the table of new words in [Lists](/course/open/py-lists?lang=en).",
            "count": "`len` counts the items in the list: four temperatures.",
            "index": "3 is the index of the last item, because indexes start at 0. `len` counts the items themselves. See [Lists](/course/open/py-lists?lang=en).",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def quick_check_import():
    quick_import = L.choice(
        "quick-import",
        question="The program starts with `import random`. Which line rolls a die?",
        options={
            "bare": "`randint(1, 6)`",
            "module": "`random(1, 6)`",
            "dot": "`random.randint(1, 6)`",
        },
        correct="dot",
        why={
            "bare": "After `import random`, every name from the module needs `random.` in front. See the watch-out in [Using code from modules](/course/open/py-imports?lang=en).",
            "module": "`random` is the module, a box of tools; you call one tool in it, with a dot. See the recipe in [Using code from modules](/course/open/py-imports?lang=en).",
            "dot": "The module's name, a dot, and the name of the function in it: `random.randint(1, 6)`.",
        },
    )
    quick_import
    return (quick_import,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A list of a hundred numbers is hard to read. Draw it, and you see at a glance whether it goes up, down, or jumps around. Later in the course, you watch a model learn in exactly this way: a line of its loss, one number for how wrong it still is, step after step, shows whether it is still getting better.

    **An everyday example.** A weather app does not give you a table of temperatures for the coming week: it draws a line. One look tells you that it gets warmer on Thursday.
    """)
    return


@app.cell(hide_code=True)
def predict_line():
    guess_line = L.guess(
        "line",
        question="What do you think this program draws?\n\n```python\nimport matplotlib.pyplot as plt\n\nplt.plot([1, 2, 3], [2, 4, 6])\nplt.show()\n```",
        options={
            "line": "A straight line that goes up",
            "bars": "Three bars, 2, 4 and 6 high",
            "numbers": "The numbers 2, 4 and 6, one per line",
        },
        reveal="**A straight line that goes up.** `plt.plot` puts the first list, `[1, 2, 3]`, along the bottom and the second, `[2, 4, 6]`, up the side, puts a point at each pair, and draws a line through the points. Each step to the right, the line goes up by 2.",
    )
    guess_line
    return (guess_line,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Matplotlib draws charts from lists of numbers. Its module `pyplot` has a function for each kind of chart, and you import it under the short name `plt`.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **Matplotlib** | a library: a set of modules that someone else wrote, installed with the course; it draws charts, and `pyplot` is the module you use | `import matplotlib.pyplot as plt` |
    | **`as`** | gives a module a short name of your own | `as plt` |
    | **x-axis, y-axis** | the line along the bottom, and the line up the side; each point has an x value and a y value | `plt.xlabel("day")` |
    | **line chart** | points joined by a line, for values in an order, such as over time | `plt.plot(x, y)` |
    | **scatter plot** | loose points, for pairs of measurements without an order | `plt.scatter(x, y)` |
    | **bar chart** | a bar per name, as high as its value, to compare separate things | `plt.bar(names, heights)` |

    Each function gets the x values first and the y values second, as two lists of the same length: the first x goes with the first y, and so on. `plt.title`, `plt.xlabel` and `plt.ylabel` say what the reader sees. `plt.show()` shows the chart under the cell.
    """)
    return


@app.cell(hide_code=True)
def picture_kinds():
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 3, figsize=(9.6, 2.6))
    _axes[0].plot([1, 2, 3, 4, 5], [12, 15, 9, 14, 16])
    _axes[0].set_title("Line chart: in order")
    _axes[1].scatter([150, 160, 165, 172, 180, 186], [48, 55, 60, 64, 75, 79])
    _axes[1].set_title("Scatter plot: pairs")
    _axes[2].bar(["read", "code", "walk"], [30, 45, 20])
    _axes[2].set_title("Bar chart: compare")
    _fig.tight_layout()
    _picture = mo.Html(plots.render(_fig))
    _plt.close(_fig)
    mo.vstack([
        _picture,
        mo.md(r"""
        The same kind of numbers can look very different. A line joins points in their order, such as temperatures day after day. A scatter plot leaves the points loose, such as the height and weight of six people. A bar chart compares separate things side by side, such as minutes per activity.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("The x values and the y values must be two lists of the same length: one x for each y. And `plt.show()` comes last: it shows the chart, and the next `plt.plot` starts a new one."),
        mo.md(r"""
        ### Recipe

        To draw a chart:

        1. At the top, import pyplot under the short name `plt`: `import matplotlib.pyplot as plt`.
        2. Draw with one function: `plt.plot(x, y)` for a line, `plt.scatter(x, y)` for loose points, `plt.bar(names, heights)` for bars.
        3. Say what the reader sees: `plt.title(...)`, `plt.xlabel(...)` and `plt.ylabel(...)`.
        4. Show the chart with `plt.show()`, as the last line.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    How many minutes the walk to school took on each day of a week, and three ways to get there. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_plot():
    # 1. import pyplot under the short name plt
    import matplotlib.pyplot as plt

    _days = [1, 2, 3, 4, 5, 6, 7]
    _walk_minutes = [14, 12, 15, 11, 13, 16, 12]
    # 2. draw: the x values first, then the y values
    plt.plot(_days, _walk_minutes)
    # 3. say what the reader sees
    plt.title("Minutes to school per day")
    plt.xlabel("day")
    plt.ylabel("minutes")
    # 4. show it; the next drawing starts a new chart
    plt.show()

    # 2. bars: the names first, then the heights
    plt.bar(["bike", "bus", "walk"], [9, 14, 25])
    # 3. what the reader sees
    plt.title("Minutes to school, three ways")
    plt.ylabel("minutes")
    # 4. show it
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear two charts. The first is a line through the seven days: day 1 at 14 minutes, day 2 at 12, and so on. `plt.show()` finishes it, so the bars start on a new chart: one bar per way to get to school. Change `plt.plot` into `plt.scatter` and run the cell again: now the days are loose points. The name `plt` is made in this cell, so other cells can use it right away; do not write `import matplotlib.pyplot as plt` in another cell, because that breaks *one name, one cell*. The exercise boxes run on their own, so each box imports it itself.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_temps():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A week of temperatures

        Draw `temps` as a line chart, with `days` along the bottom, a title, and a label on both axes. Replace the `...`. The check looks at what your chart shows: a line through the seven temperatures, day 1 to 7, with a title and two axis labels. Any title and labels that say what the chart shows are fine. In an exercise box, `plt.show()` does not show the chart; the check shows it under its message.
        """),
        L.task("Draw `temps` as a line with `days` along the bottom, with `plt.title`, `plt.xlabel` and `plt.ylabel`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_temps():
    box_temps = L.code_box("temps-line", lines=10)
    box_temps
    return (box_temps,)


@app.cell(hide_code=True)
def check_temps(box_temps):
    result_temps = L.check("temps-line", box_temps.value)
    result_temps
    return (result_temps,)


@app.cell(hide_code=True)
def hints_temps():
    L.hints(
        "temps-line",
        [
            "Steps 2 and 3 of the recipe. Temperatures day after day have an order: which kind of chart joins points in their order?",
            "Draw the line with the days as x values and the temperatures as y values; then add a title and the two axis labels.",
            "`plt.plot(___, ___)`, then `plt.title(\"...\")`, `plt.xlabel(\"...\")` and `plt.ylabel(\"...\")`.",
        ],
        why="`plt.plot(days, temps)` puts a point at each day and its temperature, and joins them in order. `plt.title(\"Temperature per day\")`, `plt.xlabel(\"day\")` and `plt.ylabel(\"degrees\")` say what the reader sees.",
    )
    return


@app.cell(hide_code=True)
def text_which_chart():
    mo.md(r"""
    ### Exercise 2. Which chart?
    """)
    return


@app.cell(hide_code=True)
def choice_which_chart_cell():
    choice_which_chart = L.choice(
        "which-chart",
        question="You measured the height and the weight of thirty people, and you want to see whether taller people weigh more. Which chart fits best?",
        options={
            "scatter": "A scatter plot, with height along the bottom and weight up the side",
            "line": "A line chart, with height along the bottom and weight up the side",
            "bar": "A bar chart, with a bar per person, as high as their weight",
        },
        correct="scatter",
        why={
            "scatter": "Each person is a pair of measurements, and the people have no order. Loose points show whether the points go up to the right.",
            "line": "A line joins points in an order, as day after day. Do thirty people have such an order? Look at the picture of the three kinds again.",
            "bar": "Thirty bars compare the people one by one, but where is the height then? Which chart shows two measurements of each person?",
        },
    )
    choice_which_chart
    return (choice_which_chart,)


@app.cell(hide_code=True)
def text_exercise_minutes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Minutes per activity

        Draw `minutes` as a bar chart, with a bar for each name in `names`, a title, and a label on the y-axis. Replace the `...`. The check looks at what your chart shows: four bars, `read`, `code`, `walk` and `sleep`, as high as their minutes, with a title and a y-axis label. Your chart appears under the check's message.
        """),
        L.task("Draw a bar per name, as high as its minutes, with a title and a y-axis label."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minutes():
    box_minutes = L.code_box("minutes-bar", lines=10)
    box_minutes
    return (box_minutes,)


@app.cell(hide_code=True)
def check_minutes(box_minutes):
    result_minutes = L.check("minutes-bar", box_minutes.value)
    result_minutes
    return (result_minutes,)


@app.cell(hide_code=True)
def hints_minutes():
    L.hints(
        "minutes-bar",
        [
            "Step 2 of the recipe: which function draws a bar for each name?",
            "The names go along the bottom and the minutes give the heights, in that order; then a title and a label for the y-axis.",
            "`plt.bar(___, ___)`, then `plt.title(\"...\")` and `plt.ylabel(\"...\")`.",
        ],
        why="`plt.bar(names, minutes)` draws a bar for each name, as high as its minutes. The bar for `sleep` towers over the rest: a chart shows that at once. `plt.title` and `plt.ylabel` say what the bars mean.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Lists](/course/open/py-lists?lang=en) you kept numbers in a list; now you draw them. In [Using code from modules](/course/open/py-imports?lang=en) you imported `math` and `random`; Matplotlib is imported the same way, with a short name of your own. Later in the course you draw the loss of a model, how wrong it still is, as a line chart, step after step. In the next lesson you count letters in a text and draw them as a bar chart.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which_chart, result_minutes, result_temps):
    L.wrap_up(result_temps, choice_which_chart, result_minutes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
