# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson np-vectorization. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-vectorization", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en) and [Repeating: for loops](/course/open/py-for?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_make():
    quick_make = L.choice(
        "quick-make",
        question="Which line makes an array of the numbers 1, 2 and 3?",
        options={
            "list": "`[1, 2, 3]`",
            "array": "`np.array([1, 2, 3])`",
            "loose": "`np.array(1, 2, 3)`",
        },
        correct="array",
        why={
            "list": "That is a list. What turns a list into an array? See the recipe in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "array": "`np.array` gets one list, and makes an array of it.",
            "loose": "`np.array` gets the numbers as one list, between square brackets. See the recipe in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
        },
    )
    quick_make
    return (quick_make,)


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="What does this program show?\n\n```python\ntotal = 0\nfor x in [1, 2, 3]:\n    total = total + x * 2\nprint(total)\n```",
        options={
            "list": "`[2, 4, 6]`",
            "six": "`6`",
            "twelve": "`12`",
        },
        correct="twelve",
        why={
            "list": "`total` is a number that grows with each round; nothing makes a list here. See the worked example in [Repeating: for loops](/course/open/py-for?lang=en).",
            "six": "That is 1 + 2 + 3. But each round adds `x * 2`, not `x`. See the recipe in [Repeating: for loops](/course/open/py-for?lang=en).",
            "twelve": "The loop adds 2, 4 and 6 to `total`, one round at a time: 12.",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model calculates with millions of numbers, over and over again. A Python loop does that one number at a time, and that is slow: for every number, Python works out again what `*` means for it. NumPy does the whole array in one go, in fast code of its own. That is why AI always calculates with arrays: with NumPy, the same work is often many times faster than with a loop. And the code gets shorter too: one line instead of a loop.

    **An everyday example.** A shop that raises all its prices by 10%. With one change in the register, every price is new, instead of a new sticker on every product, one at a time.
    """)
    return


@app.cell(hide_code=True)
def predict_times():
    guess_times = L.guess(
        "times",
        question="What do you think `np.array([1, 2, 3]) * 2` gives?",
        options={
            "repeat": "`[1, 2, 3, 1, 2, 3]`",
            "double": "`array([2, 4, 6])`",
            "error": "An error: you cannot multiply an array",
        },
        reveal="**`array([2, 4, 6])`.** On an array, `* 2` multiplies every number. On a list, `[1, 2, 3] * 2` gives `[1, 2, 3, 1, 2, 3]`: the list twice. That is why you make an array first.",
    )
    guess_times
    return (guess_times,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    You write a calculation once, as if it were for one number, and NumPy does it for every number of the array. That is called vectorization: calculating with whole arrays instead of a loop.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **vectorization** | calculating with whole arrays at once, without a loop of your own | `prices * 1.1` |
    | **array and number** | `+`, `-`, `*`, `/` and `**` with one number work on every number of the array | `np.array([1, 2, 3]) * 2` gives `array([2, 4, 6])` |
    | **two arrays** | with two arrays of the same shape, the numbers in the same place go together | `np.array([1, 2]) + np.array([10, 20])` gives `array([11, 22])` |
    | **`np.arange`** | the numbers 0, 1, 2 and so on, as an array, the way `range` counts | `np.arange(4)` gives `array([0, 1, 2, 3])` |

    The result is always a new array, of the same shape; the array you started with stays as it was.
    """)
    return


@app.cell(hide_code=True)
def picture_together():
    mo.vstack([
        draw.value_rows(C.picture_rows({"plus": "the numbers in the same place, added", "times": "every number times 2"}),
                        title="Calculating with whole arrays"),
        mo.md(r"""
        In `a + b`, the first number of `a` goes with the first number of `b`, the second with the second, and so on. In `a * 2`, the 2 goes with every number of `a`. No loop is needed for either.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`*` and `+` on a list do something else: `[1, 2] * 2` repeats the list, and `[1, 2] + [3]` sticks lists together. Make an array first. And for now, use two arrays of the same shape: a later lesson, *Broadcasting*, shows when NumPy allows other shapes."),
        mo.md(r"""
        ### Recipe

        To calculate without a loop:

        1. Make an array of your numbers: `np.array(...)`.
        2. Write the calculation once, as if it were for one number: `prices * 1.1`.
        3. NumPy does it for every number, and gives a new array of the same shape.
        4. With two arrays, check that they have the same shape: the numbers in the same place go together.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Temperatures from degrees Celsius to degrees Fahrenheit: times 9, divided by 5, plus 32. Once with NumPy, and once with a loop, to compare. Then two arrays: the price of each product times how many you buy. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_fahrenheit():
    import numpy as np

    # 1. the numbers as an array
    _celsius = np.array([12.5, 17.0, 14.0, 11.0])
    # 2. and 3. the calculation once, done for every number
    _fahrenheit = _celsius * 9 / 5 + 32
    print(_fahrenheit)
    # the same with a loop, one number at a time
    _by_loop = []
    for _c in [12.5, 17.0, 14.0, 11.0]:
        _by_loop.append(_c * 9 / 5 + 32)
    print(_by_loop)
    # 4. two arrays of the same shape: the numbers in the same place go together
    _prices = np.array([2.5, 1.0, 4.0])
    _counts = np.array([3, 10, 2])
    print(_prices.shape, _counts.shape)
    print(_prices * _counts)
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first two lines show the same temperatures, `54.5`, `62.6`, `57.2` and `51.8`: once as an array, from one line of code, and once as a list, from a loop of three lines. Then the two shapes, `(3,) (3,)`: the same, so the numbers can go together. The last line is `[ 7.5 10.   8. ]`: 2.5 times 3, 1 times 10, and 4 times 2.

    | With a loop | With NumPy |
    |---|---|
    | `for c in celsius:` and `out.append(c * 9 / 5 + 32)` | `celsius * 9 / 5 + 32` |
    | one number at a time, and you make the list | every number at once, and you get a new array |
    """)
    return


@app.cell(hide_code=True)
def demo_speed():
    _loop, _whole = C.measure()
    _ms_loop, _ms_whole = C.decimal(_loop * 1000, 0, L.lang), C.decimal(_whole * 1000, 1, L.lang)
    _times = C.decimal(_loop / max(_whole, 1e-9), 0, L.lang)
    mo.md(f"""
    **How much faster?** When this lesson opened, your computer squared the numbers 0 to 999,999 in two ways. With a loop, `for i in range(1000000):` and `squares.append(i * i)`, that took {_ms_loop} milliseconds. With NumPy, `np.arange(1000000) ** 2`, it took {_ms_whole} milliseconds: about {_times} times as fast.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_add():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two arrays together
    """)
    return


@app.cell(hide_code=True)
def choice_add_cell():
    choice_add = L.choice(
        "add-arrays",
        question="`a = np.array([4, 5, 6])` and `b = np.array([10, 20, 30])`. What does `a + b` give?",
        options={
            "pairs": "`array([14, 25, 36])`",
            "joined": "`array([4, 5, 6, 10, 20, 30])`",
            "total": "`75`",
        },
        correct="pairs",
        why={
            "pairs": "The numbers in the same place go together: 4 + 10, 5 + 20 and 6 + 30.",
            "joined": "That is what `+` does with two lists. Look at the *Watch out* box above the recipe again.",
            "total": "That would be the sum of all six numbers. What does `+` do with each number? Look at the picture again.",
        },
    )
    choice_add
    return (choice_add,)


@app.cell(hide_code=True)
def text_exercise_prices():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Everything 10% more expensive

        A shop raises all its prices by 10%. `prices` holds its four prices, shape `(4,)`. Make `new_prices`, with every price 10% higher, without a loop. Replace the `...`. The last line is `new_prices`, so the check sees it.
        """),
        L.task("Make `new_prices`: every price of `prices` 10% higher, in one calculation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prices():
    box_prices = L.code_box("new-prices", lines=6)
    box_prices
    return (box_prices,)


@app.cell(hide_code=True)
def check_prices(box_prices):
    result_prices = L.check("new-prices", box_prices.value)
    result_prices
    return (result_prices,)


@app.cell(hide_code=True)
def hints_prices():
    L.hints(
        "new-prices",
        [
            "Steps 2 and 3 of the recipe: write the calculation once, with `prices`.",
            "10% more is the price plus a tenth of the price. What do you multiply a price by, to get that?",
            "`new_prices = prices * ___`.",
        ],
        why="A price plus 10% of it is the price times 1.1. `prices * 1.1` does that for every price at once: `array([ 2.75,  1.1 ,  4.4 , 13.2 ])`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_speeds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How fast was each trip?

        Finish `speeds(km, minutes)`: it gets the distance of each trip, in km, and its time, in minutes, and gives back the speed of each trip, in km per hour, as an array. The function already makes arrays of both; they have the same shape, one number per trip. Replace the `...`, without a loop. The check calls `speeds([10, 3, 42], [30, 15, 120])`: that should give `array([20., 12., 21.])`.
        """),
        L.task("Make `speeds(km, minutes)` give back the speed of every trip in km per hour, without a loop."),
    ])
    return


@app.cell(hide_code=True)
def exercise_speeds():
    box_speeds = L.code_box("speeds", lines=9)
    box_speeds
    return (box_speeds,)


@app.cell(hide_code=True)
def check_speeds(box_speeds):
    result_speeds = L.check("speeds", box_speeds.value)
    result_speeds
    return (result_speeds,)


@app.cell(hide_code=True)
def hints_speeds():
    L.hints(
        "speeds",
        [
            "A speed is the distance divided by the time. For km per hour, the time has to be in hours.",
            "First turn the minutes into hours: an hour has 60 minutes. Then divide the distances by the hours: the numbers in the same place go together.",
            "`hours = minutes ___ 60`, and then `return ___`.",
        ],
        why="`minutes / 60` turns every time into hours: 30 minutes is 0.5 hours. `km / hours` divides each distance by its own time: 10 km in half an hour is 20 km per hour.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Repeating: for loops](/course/open/py-for?lang=en) a loop went through a list one item at a time; with an array, one line does the same for every number, and much faster. In [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en) the shape told you how many numbers an array has; two arrays of the same shape now go together number by number. In the next lesson, *Sums and means along an axis*, you go the other way: from a whole array back to one number, such as its sum.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add, result_prices, result_speeds):
    L.wrap_up(choice_add, result_prices, result_speeds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
