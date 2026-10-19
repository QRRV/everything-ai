# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-exceptions. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-exceptions", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading error messages and finding bugs](/course/open/py-errors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_last_line():
    quick_last_line = L.choice(
        "quick-last-line",
        question="A traceback ends with `ZeroDivisionError: division by zero`. What does that last line tell you?",
        options={
            "what": "What went wrong: something was divided by zero",
            "where": "On which line of the cell it went wrong",
            "call": "Which call started it all",
        },
        correct="what",
        why={
            "what": "The last line has the error kind and Python's explanation: what went wrong.",
            "where": "The line where it happened is one step higher in the traceback. See the picture in [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
            "call": "The calls are higher up in the traceback. See the picture in [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
        },
    )
    quick_last_line
    return (quick_last_line,)


@app.cell(hide_code=True)
def quick_check_no_error():
    quick_no_error = L.choice(
        "quick-no-error",
        question="A function gives no error in your three tests. Is it right?",
        options={
            "yes": "Yes: no error means it is right",
            "not-sure": "Not for sure: it can still give wrong answers; test with inputs whose answer you know",
            "red": "Only if marimo shows nothing in red under the check",
        },
        correct="not-sure",
        why={
            "yes": "A function can give a wrong answer without any error. See the watch-out in [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
            "not-sure": "No error does not mean right: only a test whose answer you know shows whether the answer is right.",
            "red": "Red only shows errors, and a wrong answer gives none. See the worked example in [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
        },
    )
    quick_no_error
    return (quick_no_error,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Some problems are not bugs, but things you can expect: a person types `twelve` where you asked for a number, or a list turns out to be empty. You do not want your whole program to stop then. With `try` and `except`, your program handles such a problem and goes on. And with `raise`, your own function can refuse an input that cannot be right, with a clear message, instead of quietly giving nonsense.

    **An everyday example.** A vending machine. Put in a foreign coin, and it does not break down: it gives the coin back and shows a message. And when a product is sold out, it refuses to sell it, and says so.
    """)
    return


@app.cell(hide_code=True)
def predict_try():
    guess_try = L.guess(
        "try",
        question="What do you think this program shows?\n\n```python\ntry:\n    print(int(\"12\"))\n    print(int(\"twelve\"))\n    print(\"done\")\nexcept ValueError:\n    print(\"not a number\")\n```",
        options={
            "stops": "`12`, then an error, and nothing more",
            "caught": "`12` and `not a number`",
            "all": "`12`, `not a number` and `done`",
        },
        reveal="**`12` and `not a number`.** `int(\"12\")` turns the text into the number 12. `int(\"twelve\")` cannot, and stops with a `ValueError`. Python jumps straight to `except ValueError:`, so `print(\"done\")` in the `try` never runs. The program itself does not stop.",
    )
    guess_try
    return (guess_try,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## The idea

    `try` runs lines that might fail. If one of them stops with the error kind named after `except`, Python jumps to the lines under `except`, and the program goes on. With `raise`, your own code stops with an error of your choice.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **try** | runs the indented lines under it; when one stops with an error, the rest of them are skipped | `try:` |
    | **except** | with an error kind: its lines run only when that kind of error happens in the `try`; after them, the program goes on | `except ValueError:` |
    | **raise** | stops with an error of your choice, with your own message | `raise ValueError("age below 0")` |
    | **ValueError** | the error kind for a value of the right sort that still cannot be used, such as the text `"twelve"` where `int` needs a whole number | |
    | **int(...)** | turns a text into a whole number; a text that is not one gives a `ValueError` | `int("12")` gives `12` |

    A text that is not a whole number, such as `"twelve"`, `"3.5"` or the empty text `""`, gives a `ValueError`; `"-7"` is fine. An error of another kind than the one after `except` is not caught: the program stops with a traceback, as before. A `raise` in a function stops the function at once, just like `return`, but with an error; the code that called it can catch that error with `try` and `except`.
    """)])
    return


@app.cell(hide_code=True)
def picture_paths():
    mo.vstack([
        mo.md("```python\ntry:\n    n = int(text)\n    print(n * 2)\nexcept ValueError:\n    print(\"not a number\")\nprint(\"done\")\n```"),
        draw.if_else("n = int(text)", ["print(n * 2)"], ['print("not a number")'], ['print("done")'],
                     yes_label="no error", no_label="ValueError", title="try and except: two ways through"),
        mo.md(r"""
        If `int(text)` works, the rest of the `try` runs, and the `except` is skipped. If it stops with a `ValueError`, Python skips the rest of the `try` and runs the lines under `except`. Either way, the program goes on after it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Always name the error kind after `except`. A bare `except:` catches every error, also a `NameError` from a typo, so it hides your bugs. And keep the `try` small: the line that can fail for a reason you expect, and what needs its result."),
        mo.md(r"""
        ### Recipe

        To handle a problem you can expect, or to refuse an input:

        1. Find the line that can fail for a reason you expect, not because of a bug: such as `int(text)` on a text a person typed.
        2. Write `try:`, with that line indented under it.
        3. Write `except`, the error kind and a colon, and under it what should happen instead.
        4. In your own function, check the input first: if it cannot be right, `raise ValueError("what is wrong")`.
        5. Test both ways: an input that works, and one that fails.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Hours that people typed, of which one is not a number; and a function that refuses to divide by zero. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_try_raise():
    # 1. a typed answer may not be a whole number
    def _minutes(text):
        # 2. the line that can fail, under try
        try:
            return int(text) * 60
        # 3. except, the error kind, and what to do instead
        except ValueError:
            return 0

    for _text in ["2", "two", "5"]:
        print(f"{_text} hours is {_minutes(_text)} minutes")

    # 4. your own function refuses an input that cannot be right
    def _percent(part, whole):
        if whole == 0:
            raise ValueError("whole is 0: there is no percentage")
        return part / whole * 100

    # 5. test both ways; the except catches the error the function raises
    print(_percent(1, 4))
    try:
        print(_percent(1, 0))
    except ValueError:
        print("no percentage")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear five lines. For `two`, `int` stopped with a `ValueError`, so `_minutes` gave back 0, and the loop went on with `5`. `_percent(1, 4)` gives `25.0`; `_percent(1, 0)` raised its own `ValueError`, and the `except` around the call caught it: `no percentage`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_safe_int():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A safe number

        Fill in `safe_int(text, default)` in the cell below: it gives back `int(text)` when the text is a whole number, and `default` when `int` stops with a `ValueError`. The check calls it with `"42"`, with `"-7"`, and with texts that are not whole numbers.
        """),
        L.task("Make `safe_int` give back the number, or `default` when the text is not a whole number."),
    ])
    return


@app.cell
def exercise_safe_int():
    def safe_int(text, default):
        ...
    return (safe_int,)


@app.cell(hide_code=True)
def check_safe_int(safe_int):
    try:
        _value = safe_int
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_safe_int = L.check("safe-int", _value)
    result_safe_int
    return (result_safe_int,)


@app.cell(hide_code=True)
def hints_safe_int():
    L.hints(
        "safe-int",
        [
            "Look at the loop in the worked example: `int(...)` is under `try:`, and the `except` says what happens instead.",
            "Under `try:`, give back `int(text)`. Under `except ___:`, give back the default.",
            "`try:`, under it `return int(text)`; then `except ___:`, and under it `return default`.",
        ],
        why="`try: return int(text)` gives back the number when the text is a whole number. When `int` stops with a `ValueError`, Python jumps to `except ValueError:`, and `return default` gives back the default instead.",
    )
    return


@app.cell(hide_code=True)
def text_raise_caught():
    mo.md(r"""
    ### Exercise 2. A raised error, caught
    """)
    return


@app.cell(hide_code=True)
def choice_raise_caught_cell():
    choice_raise_caught = L.choice(
        "raise-caught",
        question="What does this program show?\n\n```python\ndef check_age(age):\n    if age < 0:\n        raise ValueError(\"age below 0\")\n    return age\n\ntry:\n    print(check_age(-5))\nexcept ValueError:\n    print(\"bad age\")\nprint(\"done\")\n```",
        options={
            "minus": "`-5` and `done`",
            "stops": "Only the error `ValueError: age below 0`",
            "caught": "`bad age` and `done`",
        },
        correct="caught",
        why={
            "minus": "With -5, `age < 0` is `True`, so `raise` runs before `return`. Look at the word *raise* in the table again.",
            "stops": "The call is under `try:`, and the error kind matches the `except`. Look at the picture again: which way does Python go?",
            "caught": "`check_age(-5)` raises a `ValueError`, so the `print` in the `try` never shows anything. The `except` shows `bad age`, and then the program goes on with `done`.",
        },
    )
    choice_raise_caught
    return (choice_raise_caught,)


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A clear error of your own

        For an empty list, this `average` stops with a `ZeroDivisionError`: Python's word for what it noticed, not for what is wrong. Make it raise a `ValueError` with a message of your own when the list is empty, before the division. Giving back 0 is not enough: 0 looks like a real average, and the calling code cannot tell. The check also calls it with lists that do have numbers. To test the empty list yourself, put the call under `try:` with `except ValueError:`, as in step 5 of the worked example; without it, your own error stops the cell, and the check cannot run.
        """),
        L.task("Make `average` raise a `ValueError` with a message for an empty list."),
    ])
    return


@app.cell
def exercise_average():
    def average(numbers):
        total = 0
        for n in numbers:
            total = total + n
        return total / len(numbers)
    return (average,)


@app.cell(hide_code=True)
def check_average(average):
    try:
        _value = average
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_average = L.check("average", _value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average",
        [
            "Step 4 of the recipe: check the input first, at the top of the function. For which list can `average` give no right answer?",
            "For the empty list: `len(numbers) == 0`. Then stop with an error of your own, before the division.",
            "`if len(numbers) == 0:` and under it `raise ___(\"the list is empty\")`.",
        ],
        why="`if len(numbers) == 0: raise ValueError(\"the list is empty\")` stops the function before the division, with a message that says what is wrong. Code that calls `average` can catch it with `except ValueError:`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Reading error messages and finding bugs](/course/open/py-errors?lang=en) you read the errors Python gave; now you catch the ones you expect, and raise your own. Later in the course, ready-made code raises a clear error when numbers do not fit together, and you will check inputs in your own code the same way. In the next lesson you read a text file, and a missing file gives an error you can expect.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_raise_caught, result_average, result_safe_int):
    L.wrap_up(result_safe_int, choice_raise_caught, result_average)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
