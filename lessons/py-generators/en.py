# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-generators. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-generators", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Functions](/course/open/py-functions?lang=en) and [Repeating: for loops](/course/open/py-for?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="What does this program show?\n\n```python\ndef total(numbers):\n    t = 0\n    for n in numbers:\n        t = t + n\n    return t\n\nprint(total([2, 3, 4]))\n```",
        options={
            "four": "`4`",
            "list": "`[2, 3, 4]`",
            "nine": "`9`",
        },
        correct="nine",
        why={
            "four": "That is only the last number. Each round adds a number to `t`. See the worked example in [Functions](/course/open/py-functions?lang=en).",
            "list": "`total` gives back `t`, not the list it got. Follow `t` through the rounds of the loop.",
            "nine": "The loop adds 2, 3 and 4 to `t`, and `return t` gives back 9.",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Which numbers does this program show, one per line?\n\n```python\nfor i in range(3):\n    print(i * 2)\n```",
        options={
            "zero": "`0`, `2` and `4`",
            "one": "`2`, `4` and `6`",
            "upto": "`0`, `2`, `4` and `6`",
        },
        correct="zero",
        why={
            "zero": "`range(3)` gives 0, 1 and 2; each is doubled.",
            "one": "`range(3)` starts at 0, not at 1. See the table of new words in [Repeating: for loops](/course/open/py-for?lang=en).",
            "upto": "`range` stops just before its stop number, so `i` is never 3. See the table of new words in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A function with `return` works out everything first, and then gives it all back at once. That is fine for three numbers. But a model learns from millions of examples, in small groups called batches, and they often do not all fit in memory at once. A generator hands out one value, or one batch, at a time, only when it is asked for the next one.

    **An everyday example.** The ticket dispenser at a deli counter. It does not print all the numbers of the day in the morning: each time someone pulls, it hands out the next one, and it remembers where it was.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="What do you think this program shows?\n\n```python\ndef three():\n    yield \"a\"\n    yield \"b\"\n    yield \"c\"\n\nfor letter in three():\n    print(letter)\n```",
        options={
            "first": "Only `a`",
            "all": "`a`, `b` and `c`, one per line",
            "error": "It stops: a function can give back only one value",
        },
        reveal="**`a`, `b` and `c`, one per line.** Each `yield` hands out one value, and the function pauses there. The `for` loop asks for the next value each round, and the function goes on from where it paused.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A function with `yield` in it is a generator function. Calling it does not run it yet: it gives you a generator, which hands out values one at a time, each time it is asked.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **`yield`** | hands out one value, and pauses the function on that line; the next time, it goes on from there | `yield i * i` |
    | **generator** | what you get when you call a function with `yield`; it hands out values one at a time | `g = squares(4)` |
    | **iterator** | something that hands out values one at a time, the way `for` asks for them, until it is used up; a generator is one, a list is not: you can loop over a list again and again, and `next` does not work on it | `for s in g:` |
    | **`next(g)`** | asks `g` for its next value | `next(g)` |
    | **`list(g)`** | asks `g` for all its values, and puts them in a list | `list(squares(4))` |

    A generator remembers where it was, and the values of its names, between two values. It hands out each value once: after the last one, it is used up.
    """)
    return


@app.cell(hide_code=True)
def picture_turns():
    mo.vstack([
        draw.marked_lines([("The generator function", [("def countdown(n):", ""), ("    for i in range(n):", ""), ("        yield n - i", "2")]),
                           ("The loop that asks", [("for s in countdown(3):", "1"), ("    print(s)", "3")])],
                          {"1": "asks for the next value, each round",
                           "2": "hands out one value, and pauses here",
                           "3": "uses the value, and then asks again"},
                          title="Values one at a time"),
        mo.md(r"""
        The loop and the generator take turns. The loop asks (1); the generator runs until its next `yield`, hands out a value and pauses (2); the loop uses it (3) and asks again. When the generator's own loop is done, the `for` ends too.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A generator is used up after one pass. A second `for` over the same generator gets nothing. Do you need the values again? Call the function again: that gives a new generator."),
        mo.md(r"""
        ### Recipe

        To hand out values one at a time:

        1. Write a function as usual, but hand out each value with `yield`, instead of collecting them all in a list.
        2. Call the function: that gives a generator. Nothing in the function has run yet.
        3. Ask it for values: with `for` one per round, with `next(g)` one at a time, or with `list(g)` all of them.
        4. Need the values again? Call the function again for a new generator.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Even numbers one at a time, and a running total. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_generators():
    # 1. a function with yield: it hands out one value at a time
    def _evens(n):
        print("start")
        for i in range(n):
            yield 2 * i

    # 2. calling it gives a generator; nothing has run yet
    _g = _evens(4)
    print("made _g")
    # 3. ask for values: one with next, the rest with a loop
    print(next(_g))
    for _x in _g:
        print(_x)
    # 4. used up; a new call gives a new generator
    print(list(_g))
    print(list(_evens(4)))

    # a generator remembers its names between two values
    def _running_total(numbers):
        total = 0
        for n in numbers:
            total = total + n
            yield total

    print(list(_running_total([3, 1, 2])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `made _g`, `start`, `0`, `2`, `4`, `6`, `[]`, `start`, `[0, 2, 4, 6]` and `[3, 4, 6]`. `start` comes after `made _g`: `_evens` only starts when `next(_g)` asks for the first value. The loop got the other three values. After that, `_g` was used up: `list(_g)` is `[]`, an empty list. A new call, `_evens(4)`, starts again. `_running_total` keeps `total` between two values, and hands out the total so far each time.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_squares():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Squares, one at a time

        Write `squares(n)`, a generator function that hands out the squares of 0 up to, but not including, `n`: `0`, `1`, `4`, `9`, and so on. The check asks `squares(4)` for all its values: they should be `[0, 1, 4, 9]`. It also tries other numbers.
        """),
        L.task("Make `squares(n)` hand out `i * i` for each `i` in `range(n)`, with `yield`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_squares():
    box_squares = L.code_box("squares", lines=5)
    box_squares
    return (box_squares,)


@app.cell(hide_code=True)
def check_squares(box_squares):
    result_squares = L.check("squares", box_squares.value)
    result_squares
    return (result_squares,)


@app.cell(hide_code=True)
def hints_squares():
    L.hints(
        "squares",
        [
            "Step 1 of the recipe, and `_evens` in the worked example: a loop, and `yield` in it.",
            "Loop over `range(n)`; in each round, hand out the square of `i`.",
            "`for i in range(n):`, and under it `yield ___`.",
        ],
        why="`for i in range(n):` goes over 0 up to, not including, `n`, and `yield i * i` hands out each square, one at a time.",
    )
    return


@app.cell(hide_code=True)
def text_when_runs():
    mo.md(r"""
    ### Exercise 2. When does it run?
    """)
    return


@app.cell(hide_code=True)
def choice_when_runs_cell():
    choice_when_runs = L.choice(
        "when-runs",
        question="Which lines does this program show, in which order?\n\n```python\ndef numbers():\n    print(\"start\")\n    yield 1\n    yield 2\n\ng = numbers()\nprint(\"made\")\nfor x in g:\n    print(x)\n```",
        options={
            "start-first": "`start`, `made`, `1`, `2`",
            "no-start": "`made`, `1`, `2`",
            "made-first": "`made`, `start`, `1`, `2`",
        },
        correct="made-first",
        why={
            "start-first": "That is what a function without `yield` would do. Look at step 2 of the recipe again.",
            "no-start": "The `print` in `numbers` does run, but when? Look at the picture again: what happens when the loop asks for the first value?",
            "made-first": "`numbers()` only makes the generator. The function starts when the loop asks for the first value: then it shows `start` and hands out 1.",
        },
    )
    choice_when_runs
    return (choice_when_runs,)


@app.cell(hide_code=True)
def text_exercise_batches():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Batches

        A model learns from its data in batches: small groups of examples, one group at a time. Finish `batches(items, size)`: a generator function that collects items in `batch`, and hands out `batch` as soon as it has `size` items. What is left at the end goes out as a last, smaller batch. Replace the `...`, and add the lines after the loop. The check asks `batches([1, 2, 3, 4, 5], 2)` for all its values: they should be `[[1, 2], [3, 4], [5]]`. It also tries other lists, also an empty one.
        """),
        L.task("Make `batches(items, size)` hand out full batches one at a time, and what is left at the end."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batches():
    box_batches = L.code_box("batches", lines=10)
    box_batches
    return (box_batches,)


@app.cell(hide_code=True)
def check_batches(box_batches):
    result_batches = L.check("batches", box_batches.value)
    result_batches
    return (result_batches,)


@app.cell(hide_code=True)
def hints_batches():
    L.hints(
        "batches",
        [
            "Like `_running_total` in the worked example, your generator keeps a name between two values: here the list `batch`.",
            "In the loop: add the item to `batch`; when `len(batch)` is `size`, hand it out and start a new, empty `batch`. After the loop: hand out what is left, if anything.",
            "`batch.append(item)`, then `if len(batch) == ___:` with the hand-out and the new batch under it. After the loop, as far in as `for`: `if ___:`.",
        ],
        why="Each item goes into `batch`; a full batch goes out with `yield`, and `batch = []` starts the next one. After the loop, `if len(batch) > 0:` hands out the last, smaller batch, but never an empty one.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Functions](/course/open/py-functions?lang=en) a function gave back one value with `return`, and stopped; a generator hands out values with `yield`, and goes on. In [Repeating: for loops](/course/open/py-for?lang=en) you looped over lists and `range`; now you know that `for` asks for one value at a time, from anything that hands them out. Later in the course, the data of a model comes in batches, just like from your `batches`: one batch per step of training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when_runs, result_batches, result_squares):
    L.wrap_up(result_squares, choice_when_runs, result_batches)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
