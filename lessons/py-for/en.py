# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-for. Route: course/route/lessen/stam-1-basis.yaml. Checks and the loop in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-for", "en")
    C = L.checks   # the loop in the picture, round by round


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Lists](/course/open/py-lists?lang=en) and [Making choices](/course/open/py-if?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_append():
    quick_append = L.choice(
        "quick-append",
        question="What does this program show?\n\n```python\nnums = [3, 5, 7]\nnums.append(9)\nprint(nums[-1])\n```",
        options={
            "seven": "`7`",
            "nine": "`9`",
            "three": "`3`",
        },
        correct="nine",
        why={
            "seven": "`7` was the last item before the second line. What does `append` do to the list? See the recipe in [Lists](/course/open/py-lists?lang=en).",
            "nine": "`append` adds `9` at the end, and `-1` takes the last item.",
            "three": "`-1` counts from the end, not from the start. See the idea in [Lists](/course/open/py-lists?lang=en).",
        },
    )
    quick_append
    return (quick_append,)


@app.cell(hide_code=True)
def quick_check_if():
    quick_if = L.choice(
        "quick-if",
        question="What does this program show?\n\n```python\nx = 4\nif x > 5:\n    print(\"big\")\nelse:\n    print(\"small\")\n```",
        options={
            "both": "`big` and `small`",
            "big": "`big`",
            "small": "`small`",
        },
        correct="small",
        why={
            "both": "Only one of the two paths runs. See the picture in [Making choices](/course/open/py-if?lang=en).",
            "big": "`4 > 5` is `False`, so the lines under `if` are skipped. See [Making choices](/course/open/py-if?lang=en).",
            "small": "`4 > 5` is `False`, so the line under `else` runs.",
        },
    )
    quick_if
    return (quick_if,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A list can hold a thousand values. Adding them up one line at a time would take a thousand lines. A loop runs the same few lines once for every item, however long the list is. Later in the course, a program learns by going through its examples again and again, in a loop.

    **An everyday example.** Handing out cards in a game: for each player, you give one card. You do not write a separate rule for every player; you do the same thing for each one, until everyone has a card.
    """)
    return


@app.cell(hide_code=True)
def predict_loop():
    guess_loop = L.guess(
        "loop",
        question="What do you think this program shows?\n\n```python\nfor color in [\"red\", \"green\", \"blue\"]:\n    print(color)\n```",
        options={
            "one": "One line: `red green blue`",
            "three": "Three lines: `red`, `green` and `blue`",
            "last": "Only `blue`",
        },
        reveal="**Three lines.** The indented line runs once for every item. The first time, `color` is `\"red\"`; then `\"green\"`; then `\"blue\"`. Each `print` puts its text on a new line.",
    )
    guess_loop
    return (guess_loop,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    `for` goes through a list item by item. Each round, the name after `for` gets the next item, and the indented lines run.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **loop** | lines of code that run again and again | |
    | **round** | one time through the indented lines of a loop | |
    | **for** | runs the indented lines under it once for every item; each round, the name after `for` gets the next item | `for temp in temps:` |
    | **range** | the whole numbers from a start up to, not including, a stop, to loop over | `range(1, 4)` gives 1, 2 and 3 |

    `range(4)` with one number starts at 0: 0, 1, 2 and 3. To add up in a loop, start with `total = 0` before the loop, and in each round give `total` a new value: `total = total + temp`. Python works out the right side first, with the value `total` has at that moment.
    """)
    return


@app.cell(hide_code=True)
def picture_rounds():
    mo.vstack([
        mo.md(f"```python\ntotal = 0\nfor temp in {C.PICTURE_ITEMS}:\n    total = total + temp\n```"),
        draw.program_trace(C.PICTURE_LINES["en"], C.PICTURE_SHOWN, code_label="Step", names=C.PICTURE_NAMES,
                           names_label="Names after it", output_label="Under the cell", title="The loop above, step by step"),
        mo.md(r"""
        `total` starts at 0. Each round, `temp` gets the next item, and Python works out the right side first, with the old `total`. After the last round, the line at the left edge runs once and shows 36.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Only the indented lines run in every round. A line after the loop, at the left edge, runs once, after the last round. And a name that keeps track of a total or a count gets its starting value before the loop, not inside it. In the box, Enter keeps the next line indented; to go back a level, press Backspace once."),
        mo.md(r"""
        ### Recipe

        To do something with every item of a list:

        1. Before the loop, give the names that keep track of a total or a count their starting value, such as `total = 0`.
        2. Write `for`, a name for one item, `in`, the list, and a colon: `for temp in temps:`.
        3. Under it, indented, write what should happen in each round; an `if` there lets its indented lines run only for some items.
        4. After the loop, at the left edge, use the result; that line runs once.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Rain in millimeters, from Monday to Friday: how much in total, and how many dry days? The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_rain():
    _rain = [3, 0, 12, 5, 0]
    # 1. before the loop: the names that keep track start at 0
    _total = 0
    _dry_days = 0
    # 2. the loop: one round for each item of the list
    for _mm in _rain:
        # 3. indented: this runs in every round
        _total = _total + _mm
        # and this only in the rounds where the condition is True
        if _mm == 0:
            _dry_days = _dry_days + 1
    # 4. after the loop, at the left edge: this runs once
    print(f"{_total} mm of rain, {_dry_days} dry days")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appears `20 mm of rain, 2 dry days`. In each of the five rounds, `_total` grew by that day's rain; in the two rounds with 0, `_dry_days` grew by 1. With `range`, a loop goes over numbers instead of a list. Run the cell below.
    """)
    return


@app.cell
def example_range():
    # range(1, 4) gives 1, 2 and 3: it stops just before 4
    for _n in range(1, 4):
        print(f"{_n} times 7 is {_n * 7}")
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The total price

        Replace the `...` inside the loop with the line that adds the price of each round to `total`. The check also tries other lists, and an empty one.
        """),
        L.task("Make the loop add up all prices in `total`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_total():
    box_total = L.code_box("total", lines=6)
    box_total
    return (box_total,)


@app.cell(hide_code=True)
def check_total(box_total):
    result_total = L.check("total", box_total.value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total",
        [
            "Step 3 of the recipe: what should happen to `total` in each round?",
            "Each round, `total` gets a new value: what it already was, plus the price of this round. Which name holds that price?",
            "`total = total + ___`, indented under the `for` line.",
        ],
        why="`total = total + price` adds the price of each round to what `total` already was. After four rounds it is 2.5 + 4 + 1.5 + 3, which is 11.0. `print(total)` stays at the left edge, so it runs once.",
    )
    return


@app.cell(hide_code=True)
def text_range():
    mo.md(r"""
    ### Exercise 2. Counting with range
    """)
    return


@app.cell(hide_code=True)
def choice_range_cell():
    choice_range = L.choice(
        "range",
        question="What does this program show?\n\n```python\nfor n in range(2, 5):\n    print(n)\n```",
        options={
            "to-four": "`2`, `3` and `4`, each on its own line",
            "to-five": "`2`, `3`, `4` and `5`, each on its own line",
            "from-zero": "`0`, `1`, `2`, `3` and `4`, each on its own line",
        },
        correct="to-four",
        why={
            "to-five": "`range` stops just before its stop, like a slice. Look at the word *range* in the table again.",
            "to-four": "`range(2, 5)` starts at 2 and stops just before 5: 2, 3 and 4.",
            "from-zero": "With two numbers, `range` starts at the first one. Look at the word *range* in the table again.",
        },
    )
    choice_range
    return (choice_range,)


@app.cell(hide_code=True)
def text_exercise_frost():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Frost days

        How many days in `temps` are below zero? Replace the `...` with a loop that adds 1 to `frost_days` for each day below zero. The check also tries other temperatures.
        """),
        L.task("Count the days below zero in `frost_days`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_frost():
    box_frost = L.code_box("frost", lines=7)
    box_frost
    return (box_frost,)


@app.cell(hide_code=True)
def check_frost(box_frost):
    result_frost = L.check("frost", box_frost.value)
    result_frost
    return (result_frost,)


@app.cell(hide_code=True)
def hints_frost():
    L.hints(
        "frost",
        [
            "Look at the worked example: the dry days were counted with an `if` inside the loop.",
            "Step 2 of the recipe: `for temp in temps:`. Inside it, an `if` whose condition is `True` for a temperature below zero.",
            "`for temp in temps:`, then indented `if temp < 0:`, and under it, indented once more, ___.",
        ],
        why="The loop goes through every temperature. `temp < 0` is `True` for -3 and -1, and in those rounds `frost_days = frost_days + 1` adds one. 0 is not below zero, so the result is 2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Lists](/course/open/py-lists?lang=en) you took items one by one, by hand; a loop does that for every item. With [Making choices](/course/open/py-if?lang=en), an `if` inside the loop picks the items that count. A total built up in a loop comes back later as the average of a list of numbers. In the next lesson, a loop runs as long as a condition is `True`, with `while`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_range, result_frost, result_total):
    L.wrap_up(result_total, choice_range, result_frost)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
