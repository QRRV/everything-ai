# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-variables. Route: course/route/lessen/stam-1-basis.yaml. Checks, the program in the picture and the worked example: checks.py.
# termen:allow program

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-variables", "en")
    C = L.checks   # the program in the picture, its names after each line, and the worked example


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your first program](/course/open/py-first-program?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_text():
    quick_text = L.choice(
        "quick-text",
        question="What does `print(\"3 + 4\")` show under the cell?",
        options={
            "seven": "`7`",
            "text": "`3 + 4`",
            "error": "An error",
        },
        correct="text",
        why={
            "seven": "Between quotes, `3 + 4` is text, a string, and Python shows text exactly as it is, without calculating. See [Your first program](/course/open/py-first-program?lang=en).",
            "text": "Between quotes, it is a string, and Python shows it exactly as it is. In this lesson you calculate with numbers without quotes.",
            "error": "Text between quotes is fine for `print`. See [Your first program](/course/open/py-first-program?lang=en).",
        },
    )
    quick_text
    return (quick_text,)


@app.cell(hide_code=True)
def quick_check_comment():
    quick_comment = L.choice(
        "quick-comment",
        question="What does Python do with a line that starts with `#`?",
        options={
            "show": "It shows the line under the cell.",
            "error": "It stops with an error.",
            "skip": "It skips the line.",
        },
        correct="skip",
        why={
            "show": "Only `print` shows something under the cell. A line with `#` is a comment, for people. See [Your first program](/course/open/py-first-program?lang=en).",
            "error": "A comment is allowed anywhere: it is a note for people. See [Your first program](/course/open/py-first-program?lang=en).",
            "skip": "A line with `#` is a comment: a note for people, which Python skips.",
        },
    )
    quick_comment
    return (quick_comment,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far your programs showed text that you typed yourself. But AI is mostly calculation with numbers: lots of numbers, again and again. For that you need names, to keep results under. With a name you keep a result and use it again later, without typing the number again.

    **An everyday example.** A label on a jar in the kitchen: *sugar*. A recipe says "add a spoon from the sugar jar", without saying how much sugar is in it. And when the jar is refilled, the label stays and the contents change. It is only a picture, but it comes close to what a name does in Python.
    """)
    return


@app.cell(hide_code=True)
def predict_last():
    guess_last = L.guess(
        "last-value",
        question="What do you think this program shows?\n\n```python\nprice = 2\nprice = 5\nprint(price)\n```",
        options={
            "two": "`2`",
            "five": "`5`",
            "both": "`2` and `5`",
        },
        reveal="**`5`.** A name holds one value at a time. Line 1 gives `price` the value 2; line 2 gives it a new value, 5, and the 2 is gone. `print` shows the value the name has at that moment.",
    )
    guess_last
    return (guess_last,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **variable** | a name for a value; you give it a value with `=` | `price = 2.5` |
    | **int** | a whole number, short for *integer* | `3`, `-12` |
    | **float** | a number with a decimal point; Python always writes a point, never a comma | `2.5`, `3.0` |

    And the symbols to calculate with:

    | Symbol | You say | Example | Gives |
    |---|---|---|---|
    | `+` | plus | `7 + 2` | `9` |
    | `-` | minus | `7 - 2` | `5` |
    | `*` | times | `7 * 2` | `14` |
    | `/` | divided by | `7 / 2` | `3.5` |
    | `**` | to the power of | `7 ** 2` | `49` |

    Dividing with `/` always gives a float, even when it comes out whole: `6 / 2` gives `3.0`. A power is a number multiplied by itself: `2 ** 3` is `2 * 2 * 2`, so `8`. And as in math, `*` and `/` go before `+` and `-`; with parentheses you change that: `(1 + 2) * 3` gives `9`.

    Below is a program with names. For each line you see the names with their values after that line ran, and what the line shows under the cell.
    """)
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        draw.program_trace(C.STEPS, C.SHOWN, code_label="Code", names=C.NAMES, names_label="Names after this line",
                           output_label="Under the cell", title="Names and their values, line by line"),
        mo.md(r"""
        A name exists from the line that gives it a value; the name that changes on a line is in bold. Line 3 calculates with the values `price` and `cups` have at that moment, and only line 4 shows something under the cell.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("In Python, `=` does not mean \"is equal to\" as in math. It means: give the name on the left the value on the right. That is why `price = 5` right after `price = 2` is no contradiction: the name simply gets a new value."),
        mo.md(r"""
        ### Recipe

        To calculate with names:

        1. Give each number you start with a name, with `=`.
        2. Write the calculation with those names, and give the result a name too.
        3. Show the result with `print`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        Two pizzas at 9.5 each, and 2.5 for delivery. What do you pay?
        """),
        mo.md(f"```python\n{C.EXAMPLE['en']}\n```"),
        mo.md(r"""
        This program shows:
        """),
        mo.md(f"```\n{C.EXAMPLE_SHOWS}```"),
        mo.md(r"""
        `count` is an int, a whole number; `pizza` and `delivery` are floats. Python does `*` before `+`: first `9.5 * 2` is `19.0`, then plus `2.5` is `21.5`. Want to try it yourself? Copy the code into the scratchpad and run it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_example_power():
    mo.vstack([
        mo.md(r"""
        ### A second example, with a power

        Three dice, a red, a blue and a green one, each with 6 sides. How many different throws are there? 6 for the red die, times 6 for the blue, times 6 for the green: 6 multiplied by itself 3 times, a power.
        """),
        mo.md(f"```python\n{C.POWER['en']}\n```"),
        mo.md(r"""
        This program shows:
        """),
        mo.md(f"```\n{C.POWER_SHOWS}```"),
        mo.md(r"""
        `sides ** dice` is 6 to the power 3: `6 * 6 * 6`, so 216 throws.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_coffee():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Coffee

        Three cups of coffee, at 2.5 each. In the box, give the price the name `price` and the number of cups the name `cups`, store the total in `total`, and show it with `print`. Replace each `...`.
        """),
        L.task("Show the total price of three cups at 2.5 each."),
    ])
    return


@app.cell(hide_code=True)
def exercise_coffee():
    box_coffee = L.code_box("coffee", lines=5)
    box_coffee
    return (box_coffee,)


@app.cell(hide_code=True)
def check_coffee(box_coffee):
    result_coffee = L.check("coffee", box_coffee.value)
    result_coffee
    return (result_coffee,)


@app.cell(hide_code=True)
def hints_coffee():
    L.hints(
        "coffee",
        [
            "Step 1 of the recipe: `price` gets the price of one cup, and `cups` the number of cups.",
            "Step 2: the total is the price times the number of cups. Times is `*`.",
            "In the worked example, the total is `pizza * count + delivery`. Here there is no delivery.",
        ],
        why="Each number gets a name; the calculation uses the names, and the result gets a name too. `print(total)` then shows 7.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lock():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A code lock

        A lock opens with a code of 4 digits, and each digit can be 0 to 9. How many different codes are there? With 1 digit there are 10 codes, with 2 digits 10 times 10. The box already has `digits = 4`; work out the number of codes and call it `codes`.
        """),
        L.task("Show how many codes of 4 digits there are."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lock():
    box_lock = L.code_box("lock", lines=4)
    box_lock
    return (box_lock,)


@app.cell(hide_code=True)
def check_lock(box_lock):
    result_lock = L.check("lock", box_lock.value)
    result_lock
    return (result_lock,)


@app.cell(hide_code=True)
def hints_lock():
    L.hints(
        "lock",
        [
            "Each extra digit multiplies the number of codes by 10. How many times do you multiply by 10 for 4 digits?",
            "Four 10s multiplied together, one for each digit, is a power: 10 to the power 4.",
            "Look at the dice example: there, `sides ** dice` gave the number of throws. Here each digit has 10 choices, and there are `digits` of them.",
        ],
        why="`codes = 10 ** digits` gives 10000: 10 choices for each of the 4 digits. `10 * 10 * 10 * 10` is just as right.",
    )
    return


@app.cell(hide_code=True)
def text_divide():
    mo.md(r"""
    ### Exercise 3. Dividing
    """)
    return


@app.cell(hide_code=True)
def choice_divide_cell():
    choice_divide = L.choice(
        "divide",
        question="What does `print(8 / 4)` show?",
        options={
            "float": "`2.0`",
            "whole": "`2`",
            "comma": "`2,0`",
        },
        correct="float",
        why={
            "whole": "Dividing with `/` gives a float, even when it comes out whole. Look at the line under the table of symbols again.",
            "comma": "Python writes a decimal point, never a comma, also in what it shows. Look at the table of new words again.",
            "float": "Dividing with `/` always gives a float, even when it comes out whole: `8 / 4` gives `2.0`, with a point.",
        },
    )
    choice_divide
    return (choice_divide,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Your first program](/course/open/py-first-program?lang=en) you showed text; now you show numbers that Python works out. From here on you meet names everywhere: later in the course, a program that learns from examples keeps what it learns under names too. In the next lesson you see what names do in a notebook, where every name belongs to one cell.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_divide, result_coffee, result_lock):
    L.wrap_up(result_coffee, result_lock, choice_divide)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
