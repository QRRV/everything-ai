# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-decorators. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-decorators", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Functions](/course/open/py-functions?lang=en) and [Reading a text file](/course/open/py-files?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="What does this program show?\n\n```python\ndef triple(x):\n    return 3 * x\n\nprint(triple(triple(2)))\n```",
        options={
            "six": "`6`",
            "eighteen": "`18`",
            "nine": "`9`",
        },
        correct="eighteen",
        why={
            "six": "That is `triple(2)`. Its result goes into `triple` once more. See the guess in [Functions](/course/open/py-functions?lang=en).",
            "eighteen": "The inner call goes first: `triple(2)` gives 6, and `triple(6)` gives 18.",
            "nine": "Each call multiplies by 3; nothing is added. Work out the inner call first, as in [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def quick_check_with():
    quick_with = L.choice(
        "quick-with",
        question="After this program, what is true about the file?\n\n```python\nwith open(\"lessons/py-files/poem.txt\") as f:\n    text = f.read()\nprint(len(text))\n```",
        options={
            "open": "It is still open, because `f` still exists",
            "deleted": "It is gone: `read` takes the text out of it",
            "closed": "It is closed again, right after the indented line",
        },
        correct="closed",
        why={
            "open": "`with` takes care of the file for you. What does it do after its indented lines? See the table of new words in [Reading a text file](/course/open/py-files?lang=en).",
            "deleted": "Reading a file never changes it. See the idea in [Reading a text file](/course/open/py-files?lang=en).",
            "closed": "`with` closes the file after its indented lines, also when something goes wrong. `text` keeps what was read.",
        },
    )
    quick_with
    return (quick_with,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the file of a lesson in this course, every cell is a function with `@app.cell` above it. That line puts a decorator to work: it hands the function below it to marimo, which turns it into a cell. Later, in PyTorch, `@torch.no_grad()` above a function does for the whole function what `with torch.no_grad():` does for a block. Some decorators, like these two, are made by a call first, with parentheses; the call gives the decorator. A decorator adds something to a function, without changing the function itself.

    **An everyday example.** Gift wrapping. The present inside stays the same, but what you hand over now has paper and a ribbon around it. A decorator wraps a function in the same way: the function stays as it is, and the wrapping adds a little.
    """)
    return


@app.cell(hide_code=True)
def predict_loud():
    guess_loud = L.guess(
        "loud",
        question="What do you think this program shows?\n\n```python\ndef loud(fn):\n    def wrapper():\n        return f\"{fn()}!\"\n    return wrapper\n\n@loud\ndef hello():\n    return \"hi\"\n\nprint(hello())\n```",
        options={
            "plain": "`hi`",
            "error": "It stops: `hello` is no longer a function",
            "loud": "`hi!`",
        },
        reveal="**`hi!`.** `@loud` hands `hello` to `loud`, and from then on the name `hello` stands for `wrapper`. Calling `hello()` runs `wrapper`, which calls the old `hello` and adds `\"!\"`.",
    )
    guess_loud
    return (guess_loud,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A function is a value, just like a number or a list. You can give it to another function, and a function can make a new function and give it back. A decorator does both.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **function as a value** | the name of a function without parentheses is the function itself; you can pass it on, like any value | `loud(hello)` |
    | **inner function** | a function defined inside another function; it can use the names of the outer function, such as `fn` | `def wrapper():` |
    | **decorator** | a function that gets a function, and gives back a new function that wraps it | `def loud(fn):` |
    | **`@`** | on the line above a `def`: `@loud` means `hello = loud(hello)`, right after the `def` | `@loud` |

    With parentheses, `hello()` calls the function and gives back its result; without them, `hello` is the function itself. A decorator needs the function itself. And `wrapper` keeps the names it uses from the decorator, such as `fn`, even after the decorator has finished: that is how a call of the new `hello` can still call the old one.
    """)
    return


@app.cell(hide_code=True)
def picture_same():
    mo.vstack([
        draw.marked_lines([("With @", [("@loud", "1"), ("def hello():", ""), ("    return \"hi\"", "")]),
                           ("Without @: the same", [("def hello():", ""), ("    return \"hi\"", ""), ("hello = loud(hello)", "1")])],
                          {"1": "the decorator gets the function, and the name now stands for what it gives back"},
                          title="What @ does"),
        mo.md(r"""
        Both pieces of code do the same thing. `@loud` is a short way to write `hello = loud(hello)` right after the `def`: the name `hello` now stands for the new function that `loud` gave back.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A decorator gives back the new function itself: `return wrapper`, without parentheses. With `return wrapper()`, Python calls it right away, and the decorator gives back its result instead of the function. And the sign `@` has a second job: later in the course, `@` between two arrays, as in `x @ W`, multiplies them (matrix multiplication). The decorator's `@` always starts a line, right above a `def`."),
        mo.md(r"""
        ### Recipe

        To write a decorator and use it:

        1. Write a function that gets a function: `def loud(fn):`.
        2. Inside it, write a new function, `wrapper`, that calls `fn` and does something with what it gives back.
        3. End the decorator with `return wrapper`, without parentheses.
        4. Put `@loud` on the line above a `def`. From then on, the name of that function stands for `wrapper`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A decorator that doubles what a function gives back. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_decorator():
    # 1. a function that gets a function, fn
    def _double_result(fn):
        # 2. a new function that calls fn, and adds something
        def wrapper(x):
            return 2 * fn(x)

        # 3. give back the new function itself, without ()
        return wrapper

    # 4. @ above a def
    @_double_result
    def _add_one(x):
        return x + 1

    print(_add_one(3))

    # the same, without @
    def _add_two(x):
        return x + 2

    _twice_add_two = _double_result(_add_two)
    print(_add_two(3))
    print(_twice_add_two(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `8`, `5` and `10`. `_add_one(3)` runs `wrapper`: the old `_add_one` gives 4, and `wrapper` doubles it. Without `@`, `_double_result(_add_two)` gives a new function, which here gets a name of its own, so `_add_two` stays as it was.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_plus_ten():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Ten more

        Finish the decorator `plus_ten`: every function it wraps gives back 10 more. Replace the `...` in `wrapper`. The check puts `@plus_ten` above a function `square` that gives back `x * x`, and calls `square(3)`: that should be 19. It also tries other functions.
        """),
        L.task("Make `wrapper` give back what `fn` gives for `x`, plus 10."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plus_ten():
    box_plus_ten = L.code_box("plus-ten", lines=6)
    box_plus_ten
    return (box_plus_ten,)


@app.cell(hide_code=True)
def check_plus_ten(box_plus_ten):
    result_plus_ten = L.check("plus-ten", box_plus_ten.value)
    result_plus_ten
    return (result_plus_ten,)


@app.cell(hide_code=True)
def hints_plus_ten():
    L.hints(
        "plus-ten",
        [
            "Step 2 of the recipe, and `wrapper` in the worked example: it calls `fn`, and does something with the result.",
            "Call `fn` with `x`, and add 10 to what it gives back.",
            "`return fn(___) + ___`",
        ],
        why="`return fn(x) + 10` calls the wrapped function with `x`, and gives back its result plus 10. `@plus_ten` above `square` makes `square(3)` give 9 + 10, which is 19.",
    )
    return


@app.cell(hide_code=True)
def text_same_as():
    mo.md(r"""
    ### Exercise 2. What @ stands for
    """)
    return


@app.cell(hide_code=True)
def choice_same_as_cell():
    choice_same_as = L.choice(
        "same-as",
        question="Which line, right after the `def`, does the same as `@shout` above it?\n\n```python\n@shout\ndef greet():\n    return \"hello\"\n```",
        options={
            "same": "`greet = shout(greet)`",
            "called": "`greet = shout(greet())`",
            "swapped": "`shout = greet(shout)`",
        },
        correct="same",
        why={
            "called": "`greet()` with parentheses calls `greet`, and gives its result, `\"hello\"`. Does a decorator get a result, or the function itself?",
            "same": "`@shout` hands the function `greet` itself to `shout`, and the name `greet` then stands for what `shout` gives back.",
            "swapped": "The decorator is the one that gets a function. Which of the two is the decorator here? Look at the picture again.",
        },
    )
    choice_same_as
    return (choice_same_as,)


@app.cell(hide_code=True)
def text_exercise_no_negatives():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. No negatives

        Write the decorator `no_negatives`; the box has its `def` line, so write the rest under it. The functions it wraps never give back a number below 0. When the result is below 0, `wrapper` gives back 0 instead; every other result stays as it is. Replace the `...`. The check puts `@no_negatives` above a function `shift` that gives back `x - 5`, and calls `shift(8)` and `shift(2)`: that should give 3 and 0. It also tries other functions.
        """),
        L.task("Make `no_negatives` give back a `wrapper` that turns every result below 0 into 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_no_negatives():
    box_no_negatives = L.code_box("no-negatives", lines=9)
    box_no_negatives
    return (box_no_negatives,)


@app.cell(hide_code=True)
def check_no_negatives(box_no_negatives):
    result_no_negatives = L.check("no-negatives", box_no_negatives.value)
    result_no_negatives
    return (result_no_negatives,)


@app.cell(hide_code=True)
def hints_no_negatives():
    L.hints(
        "no-negatives",
        [
            "Steps 2 and 3 of the recipe, as in `_double_result`: a `wrapper` inside, and `return wrapper` at the end.",
            "In `wrapper`, first store what `fn(x)` gives back. If that is below 0, give back 0; otherwise give back the result itself.",
            "In `wrapper`: `result = fn(x)`, then `if result < 0:` with `return ___` under it.",
        ],
        why="`wrapper` calls `fn(x)` once and keeps the result. Below 0, it gives back 0; otherwise the result itself. `return wrapper` at the end gives the new function back, so `@no_negatives` can put it in place of `shift`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Functions](/course/open/py-functions?lang=en) you called functions; now a function is also a value that you pass on, and a decorator wraps it. In [Reading a text file](/course/open/py-files?lang=en), `with` wrapped a block of lines in opening and closing a file; a decorator wraps a whole function in the same spirit. Your `no_negatives` does what a ReLU does later in the course: in a neural network, it turns every result below 0 into 0. The next lesson draws your numbers as plots.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same_as, result_no_negatives, result_plus_ten):
    L.wrap_up(result_plus_ten, choice_same_as, result_no_negatives)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
