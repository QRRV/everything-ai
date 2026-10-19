# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-errors. Route: course/route/lessen/stam-1-basis.yaml. Checks and the traceback in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-errors", "en")
    C = L.checks   # the cell and the traceback in the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Functions](/course/open/py-functions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_half():
    quick_half = L.choice(
        "quick-half",
        question="What does this program show?\n\n```python\ndef half(x):\n    return x / 2\n\nprint(half(10) + half(4))\n```",
        options={
            "five": "`5.0`",
            "seven": "`7.0`",
            "fourteen": "`14.0`",
        },
        correct="seven",
        why={
            "five": "`half(10)` gives 5.0, but `+` adds `half(4)` to it. See the guess in [Functions](/course/open/py-functions?lang=en).",
            "seven": "`half(10)` gives back 5.0 and `half(4)` gives back 2.0, and `print` shows their sum.",
            "fourteen": "Each call gives back half of its own argument, not the whole number. See the picture in [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_half
    return (quick_half,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="What does this program show?\n\n```python\ndef first_big(nums):\n    for n in nums:\n        if n > 5:\n            return n\n    return 0\n\nprint(first_big([3, 8, 9]))\n```",
        options={
            "zero": "`0`",
            "nine": "`9`",
            "eight": "`8`",
        },
        correct="eight",
        why={
            "zero": "`return 0` only runs when the loop finds no number above 5. See the watch-out in [Functions](/course/open/py-functions?lang=en).",
            "nine": "`return` ends the function at once, so the loop never gets to 9. See the idea in [Functions](/course/open/py-functions?lang=en).",
            "eight": "At 8, `n > 5` is `True`, and `return n` ends the function right there.",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Everyone's code goes wrong, again and again; even experts spend much of their time finding bugs. An error message is not a punishment: Python tells you what went wrong, and where. Until now the checks put errors in plain words for you. From this lesson on, you read Python's own messages, and that saves you hours as your programs grow.

    **An everyday example.** A warning light in a car. It does not just say "something is wrong": the oil light says which part, and the manual says where to look. An error message works the same way: its last line says what went wrong, and the lines above it say where.
    """)
    return


@app.cell(hide_code=True)
def predict_stop():
    guess_stop = L.guess(
        "stop",
        question="Where do you think this program stops?\n\n```python\ndef area(width, height):\n    return width * hight\n\nprint(area(3, 4))\n```",
        options={
            "def": "On the `def` line",
            "return": "On the line with `return`, at `hight`",
            "none": "It does not stop: it shows `12`",
        },
        reveal="**On the line with `return`.** Python reads the `def` line without trouble: the lines of a function only run when it is called. When the last line calls `area(3, 4)`, Python gets to `hight`, does not know that name, and stops with a `NameError`. Python even adds: *Did you mean: 'height'?*",
    )
    guess_stop
    return (guess_stop,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    When Python stops with an error, it shows a traceback: what went wrong, and the calls that led there. Read it from the bottom up.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **bug** | a mistake in a program: it stops with an error, or it gives a wrong answer | |
    | **traceback** | what Python shows when it stops: the calls that led to the error, the first at the top, and at the bottom the kind of error with an explanation | `Traceback (most recent call last):` |
    | **error kind** | the name at the start of the last line; it says what kind of thing went wrong | `NameError` |

    The kinds you meet most:

    | Kind | What it usually means |
    |---|---|
    | `NameError` | Python does not know a name: a typo, or the line that makes the name comes later |
    | `TypeError` | two things that do not go together, such as a number and a list, or a call with the wrong number of values |
    | `IndexError` | a place in a list that is not there |
    | `ZeroDivisionError` | dividing by zero |
    | `UnboundLocalError` | a name inside a function is used before it has a value there |
    | `SyntaxError`, `IndentationError` | Python cannot read the code at all, so nothing in the cell runs; it points at the line |

    In a lesson, marimo shows an error under the cell: the kind in red, Python's explanation under it, and the lines involved colored in your cell. Under *Traceback* is the whole path. `in average` means inside the function `average`; `in <module>` means in the cell itself, outside any function. In other languages a traceback is called a stack trace. A cell that uses a name from a cell with an error does not run either; it says *An ancestor raised an exception*. Fix the first cell, the one with the red error. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def picture_traceback():
    mo.vstack([
        draw.marked_lines([("A cell", C.PICTURE_CELL), ("What Python shows under it", C.PICTURE_TRACEBACK)],
                          {"1": "what went wrong: read this first",
                           "2": "the line where it happened",
                           "3": "the call that led there"},
                          title="A traceback, read from the bottom up"),
        mo.md(r"""
        The list is empty, so `len(numbers)` is 0, and you cannot divide by 0. The last line says what (1); one step up is the line where it happened (2); above that is the call that led there (3). Here the bug is not a typo: the function does not handle an empty list. The course writes the traceback a little shorter than marimo: marimo puts a link to the cell after *Cell*, and under some lines it adds `~~~^^^`, pointing at the exact part that failed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("The line in the traceback is where Python noticed the problem, not always where the mistake is: a wrong value can come from an earlier line. And no error does not mean right: a function can give a wrong answer without any error. Test it with an input whose answer you know."),
        mo.md(r"""
        ### Recipe

        To find a bug:

        1. Read the last line: the kind of error, and Python's explanation.
        2. Go one step up: the line in your cell where it happened, with its line number.
        3. Further up: the call that led there.
        4. Look at the values on that line. Is a name misspelled? Is a list empty? Is something another kind of value than you think? A `print` just above that line shows them.
        5. Test small pieces: call the function with a small input whose answer you know.
        6. Fix it, run the cell again, and test once more.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Here are two bugs, found and fixed. The first is the one in the picture: steps 1 to 3 gave `ZeroDivisionError`, on the line `return total / len(numbers)`, after the call `average([])`. The second gives no error at all, only a wrong answer. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_bug_hunt():
    # 1. to 3. the traceback said: ZeroDivisionError, on the line
    # return total / len(numbers), after the call average([])
    def _average(numbers):
        # 4. a print(len(numbers)) just above that line showed 0 for []
        # 6. the fix: an empty list gets 0, before the division
        if len(numbers) == 0:
            return 0
        total = 0
        for n in numbers:
            total = total + n
        return total / len(numbers)

    # 5. test small pieces: inputs whose answer you know
    print(_average([]))
    print(_average([4]))
    print(_average([2, 4]))

    def _largest(nums):
        # an empty list has no largest number: give back None, Python's word for nothing
        if len(nums) == 0:
            return None
        # 6. the fix: start at the first number; it was best = 0
        best = nums[0]
        for n in nums:
            if n > best:
                best = n
        return best

    # 5. a small test whose answer is below 0: with best = 0, it gave 0
    print(_largest([5, 9, 2]))
    print(_largest([-3, -7]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `0`, `4.0`, `3.0`, `9` and `-3`. The comment at step 4 is a test that was done and then removed: a `print` just above the line that failed. The second function gave the right answer for `[5, 9, 2]` even with the bug. Only a test whose answer is below 0, `[-3, -7]`, showed it: with `best = 0`, no number is ever bigger than `best`. That is why you test with inputs that are a bit unusual.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The total price

        The function below should add up the prices in a list, but it has a bug. The check under it shows the traceback. From this lesson on, an exercise can be an ordinary cell: change the code in the cell itself, and run it. Want to try your function? Add a line under it, such as `print(total_price([1, 2]))`. If that line stops with an error, marimo shows it in red, and the check says *An ancestor raised an exception*: read the error, fix it, and run again.
        """),
        L.task("Find the bug in `total_price` with the traceback, and fix it."),
    ])
    return


@app.cell
def exercise_total():
    def total_price(prices):
        total = 0
        for price in prices:
            total = total + prices
        return total
    return (total_price,)


@app.cell(hide_code=True)
def check_total(total_price):
    try:
        _value = total_price
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_total = L.check("total", _value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total",
        [
            "Step 1 of the recipe: read the last line of the traceback in the check. Which kind of error, and what does Python say about it?",
            "Step 2: one step up is the line where it happened. Which two things does it add? Is each of them a number?",
            "In the loop, `price` is one price and `prices` is the whole list. Each round should add ___.",
        ],
        why="`total = total + prices` adds the whole list to a number, and Python cannot do that: a `TypeError`. In the loop, the price of this round is `price`, without an s: `total = total + price`.",
    )
    return


@app.cell(hide_code=True)
def text_traceback():
    mo.md(r"""
    ### Exercise 2. Read a traceback
    """)
    return


@app.cell(hide_code=True)
def choice_traceback_cell():
    choice_traceback = L.choice(
        "traceback",
        question="A cell has this code, and marimo shows this traceback under it. What went wrong?\n\n```python\ndef to_minutes(hours):\n    return hours * 60\n\ntimes = [1, 2, 3]\nprint(to_minutes(times[3]))\n```\n\n```\nTraceback (most recent call last):\n  Cell, line 5, in <module>\n    print(to_minutes(times[3]))\nIndexError: list index out of range\n```",
        options={
            "multiply": "A list cannot be multiplied by 60",
            "index": "`times` has no place 3: its places are 0, 1 and 2",
            "function": "`to_minutes` does not exist yet on line 5",
        },
        correct="index",
        why={
            "multiply": "The traceback never gets into `to_minutes`: it stops on line 5. Read the last line again: which kind of error is it?",
            "index": "The last line says `IndexError`, and the line above it is line 5: `times[3]` asks for a place that is not there.",
            "function": "Then the last line would say `NameError`. Read the last line again: which kind of error is it?",
        },
    )
    choice_traceback
    return (choice_traceback,)


@app.cell(hide_code=True)
def text_exercise_passed():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A bug without an error

        `count_passed` should count the scores of 5.5 or more. It gives no error, but its answer is wrong. Test it on small lists whose answer you know, find the bug, and fix it in the cell.
        """),
        L.task("Find the bug in `count_passed` by testing small pieces, and fix it."),
    ])
    return


@app.cell
def exercise_passed():
    def count_passed(scores):
        count = 0
        for score in scores:
            if score >= 5.5:
                count = count + 1
            return count
    return (count_passed,)


@app.cell(hide_code=True)
def check_passed(count_passed):
    try:
        _value = count_passed
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_passed = L.check("passed", _value)
    result_passed
    return (result_passed,)


@app.cell(hide_code=True)
def hints_passed():
    L.hints(
        "passed",
        [
            "Step 5 of the recipe: under the function, add `print(count_passed([7.0]))` and `print(count_passed([4.5, 8.0]))`, and run the cell. Which answer is wrong?",
            "With `[4.5, 8.0]` the answer is 0: the function stops after the first score. Which line ends a function at once?",
            "`return count` should run once, after the loop: in line with ___.",
        ],
        why="`return` ends the function at once. Indented inside the loop, it runs in the first round, after one score. In line with `for`, it runs after the loop, when every score has been counted.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Until now the checks put every error in plain words. From here on you read Python's own tracebacks, and some exercises are ordinary cells. You used what [Functions](/course/open/py-functions?lang=en) taught: `return` ends a function, and the check calls it with several inputs. Later in the course, a traceback tells you which line got numbers it could not use. The next lesson is an extra one: there you catch errors yourself, and raise your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_traceback, result_passed, result_total):
    L.wrap_up(result_total, choice_traceback, result_passed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
