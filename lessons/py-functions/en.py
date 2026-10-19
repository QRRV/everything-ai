# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-functions. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-functions", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Repeating: for loops](/course/open/py-for?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="What does this program show?\n\n```python\ncount = 0\nfor word in [\"a\", \"bb\", \"ccc\"]:\n    if len(word) > 1:\n        count = count + 1\nprint(count)\n```",
        options={
            "three": "`3`",
            "two": "`2`",
            "one": "`1`",
        },
        correct="two",
        why={
            "three": "The `if` only adds 1 for a word longer than 1, and `a` is not. See the worked example in [Repeating: for loops](/course/open/py-for?lang=en).",
            "two": "`bb` and `ccc` are longer than 1, so `count` grows twice.",
            "one": "The loop looks at every word, and the `if` checks each one. Which words are longer than 1? See the recipe in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="What does this program show?\n\n```python\nfor i in range(1, 4):\n    print(i * 10)\n```",
        options={
            "three": "`10`, `20` and `30`, each on its own line",
            "four": "`10`, `20`, `30` and `40`, each on its own line",
            "zero": "`0`, `10` and `20`, each on its own line",
        },
        correct="three",
        why={
            "three": "`range(1, 4)` gives 1, 2 and 3, and each round shows that number times 10.",
            "four": "`range` stops just before its stop, so 4 is not part of it. See the idea in [Repeating: for loops](/course/open/py-for?lang=en).",
            "zero": "With two numbers, `range` starts at the first one, here 1. See the idea in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the last lessons you wrote the same kind of code again and again: a total, a count. A function puts such code under a name. Then one short line does the whole job, for any list you give it. Bigger programs are built from functions, and later in the course, a program that learns is a function too: numbers go in, an answer comes out.

    **An everyday example.** A coffee machine. You press a button and choose a size; the machine does all the steps inside, and a cup of coffee comes out. You do not have to know the steps each time: the button is enough. The size you choose is what you hand to it, and the cup is what comes back.
    """)
    return


@app.cell(hide_code=True)
def predict_call():
    guess_call = L.guess(
        "call",
        question="What do you think this program shows?\n\n```python\ndef double(x):\n    return x * 2\n\nprint(double(5) + 1)\n```",
        options={
            "twelve": "`12`",
            "ten": "`10`",
            "eleven": "`11`",
        },
        reveal="**`11`.** `double(5)` runs the function with `x` set to 5, and `return x * 2` gives back 10. Then `print` shows 10 plus 1. The lines of a function only run when you call it.",
    )
    guess_call
    return (guess_call,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        You write a function once, with `def`. After that you *call* it as often as you like, each time with other values.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **function** | code under a name, that you can use again and again | `double` |
        | **def** | starts a function: `def`, its name, the arguments in parentheses, and a colon; the indented lines under it are the function | `def double(x):` |
        | **argument** | a value you hand to a function when you call it; inside the function, it has the name from the `def` line | the `5` in `double(5)` |
        | **return** | gives the result back to where the function was called, and ends the function | `return x * 2` |
        | **call** | use a function: its name, with the arguments in parentheses | `double(5)` |
        | **None** | Python's word for nothing; what a function gives back when it has no `return` | `print(None)` shows `None` |

        A function can get several arguments, with commas between them. The values go to the names in order: with `def area(width, height):`, the call `area(3, 4)` gives `width` the value 3 and `height` the value 4.

        The names you make inside a function, such as `total`, exist only while the function runs, so they never clash with names elsewhere. And `print` is not the same as `return`: `print` only shows a value under the cell, while `return` hands it back, so you can keep it in a name or calculate with it.
        """),
        L.watch_out("Python's own documentation calls the names in the `def` line *parameters*. In this course, that word is kept for the numbers a learning program adjusts, so we say *argument* for both."),  # termen:ok
    ])
    return


@app.cell(hide_code=True)
def picture_call():
    mo.vstack([
        draw.flow(
            [("Where you call it", ["y = double(5)", "y gets 10"]),
             ("The function double", ["x = 5", "return x * 2"])],
            forward=["5 goes in,\nas x"],
            back=["10 comes back"],
            title="A call, and what comes back"),
        mo.md(r"""
        The call hands 5 to the function, where it is called `x`. The function runs its lines, and `return` sends 10 back to the place of the call, where `y` gets it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A function that only uses `print` gives nothing back: its result is `None`, Python's word for nothing. And `return` ends the function at once, so put it after the loop, indented once, in line with `for`, not inside the loop. In the box, press Backspace until `return` lines up with `for`."),
        mo.md(r"""
        ### Recipe

        To make and use a function:

        1. Write `def`, a name, the arguments in parentheses, and a colon: `def average(numbers):`.
        2. Under it, indented, write the steps, with the names of the arguments.
        3. End with `return` and the result.
        4. Call the function with values, such as `average([4, 8, 6])`, and keep the result in a name or show it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A function that gives the average of a list of numbers. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_average():
    # 1. def, the name, the argument, and a colon
    def _average(numbers):
        # 2. indented: the steps, with the name of the argument
        total = 0
        for hours in numbers:
            total = total + hours
        # 3. return: give the result back
        return total / len(numbers)

    # 4. call it, here twice, with two lists
    _sleep = _average([7, 6.5, 8, 7.5, 6])
    print(f"Average sleep: {_sleep} hours")
    print(_average([10, 20]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `Average sleep: 7.0 hours` and `15.0`. The same lines ran twice, once for each list; only the argument was different. `total` and `hours` exist only inside the function. The function's own name starts with `_` because it is only used in this cell, just like any other name.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_area():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The area of a rectangle

        Finish the function `area`: it gets a `width` and a `height`, and gives back the area, the width times the height. Replace the `...`. The check calls your function with 3 and 4, and with other sizes.
        """),
        L.task("Make `area(width, height)` give back the area of a rectangle."),
    ])
    return


@app.cell(hide_code=True)
def exercise_area():
    box_area = L.code_box("area", lines=4)
    box_area
    return (box_area,)


@app.cell(hide_code=True)
def check_area(box_area):
    result_area = L.check("area", box_area.value)
    result_area
    return (result_area,)


@app.cell(hide_code=True)
def hints_area():
    L.hints(
        "area",
        [
            "Step 3 of the recipe: the function ends with `return` and the result.",
            "The result is the width times the height. Inside the function, those values are called `width` and `height`.",
            "`return ___ * ___`, indented under the `def` line.",
        ],
        why="`return width * height` gives back the width times the height. Because it uses the arguments' names, it is right for every rectangle: `area(3, 4)` gives `12`, and `area(2.5, 3)` gives `7.5`.",
    )
    return


@app.cell(hide_code=True)
def text_none():
    mo.md(r"""
    ### Exercise 2. print or return?
    """)
    return


@app.cell(hide_code=True)
def choice_none_cell():
    choice_none = L.choice(
        "none",
        question="What does this program show?\n\n```python\ndef greet(name):\n    print(f\"Hi {name}\")\n\nresult = greet(\"Ada\")\nprint(result)\n```",
        options={
            "twice": "`Hi Ada`, and under it `Hi Ada` again",
            "once": "Only `Hi Ada`",
            "none": "`Hi Ada`, and under it `None`",
        },
        correct="none",
        why={
            "twice": "`greet` shows `Hi Ada` with `print`, but it has no `return`, so it gives nothing back. Look at the watch-out above the recipe again.",
            "none": "The call shows `Hi Ada`. `greet` has no `return`, so `result` gets `None`, and the last line shows that.",
            "once": "The last line also shows something: the value of `result`. What does a function without `return` give back? Look at the watch-out above the recipe again.",
        },
    )
    choice_none
    return (choice_none,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Counting long words

        Finish the function `count_long`: it gets a list `words` and a number `n`, and gives back how many words have more than `n` letters. Replace the `...` with a count in a loop, as in the lesson on for loops, and a `return`. The check calls your function with `["sun", "thunder", "drizzle"]` and 5, and with other lists and numbers.
        """),
        L.task("Make `count_long(words, n)` give back the number of words with more than `n` letters."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-long", lines=8)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-long", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-long",
        [
            "Look at the worked example: a name that starts at 0, a loop over the list, and `return` after the loop.",
            "In the loop, an `if` whose condition is `True` for a word with more than `n` letters adds 1 to the count. Use `n`, not the 5 from the check's call.",
            "`count = 0`, `for word in words:`, `if len(word) > n:`, `count = count + 1`, and at the end, indented once, ___.",
        ],
        why="The count starts at 0, the loop looks at every word, and `if len(word) > n:` adds 1 for each long word. `return count` stands after the loop, in line with `for`, so it gives back the count once every word has been looked at.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Repeating: for loops](/course/open/py-for?lang=en) you wrote a count once, for one list; as a function, it works for any list and any number. From here on, many exercises ask for a function, and the check calls it with several inputs. Later in the course, a program that learns is itself a function: numbers go in, and an answer comes out. In the next lesson, a function even calls itself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_none, result_area, result_count):
    L.wrap_up(result_area, choice_none, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
