# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-if. Route: course/route/lessen/stam-1-basis.yaml. Checks and the worked example: checks.py.
# termen:allow variables, program

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-if", "en")
    C = L.checks   # the worked example, and what it shows for two amounts


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Variables and numbers](/course/open/py-variables?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_assign():
    quick_assign = L.choice(
        "quick-assign",
        question="What does the line `price = 5` do in Python?",
        options={
            "give": "It gives the name `price` the value 5.",
            "ask": "It checks whether `price` is 5.",
            "show": "It shows 5 under the cell.",
        },
        correct="give",
        why={
            "give": "One `=` gives the name on the left the value on the right.",
            "ask": "In Python, `=` does not mean \"is equal to\". See the watch-out in [Variables and numbers](/course/open/py-variables?lang=en).",
            "show": "Only `print` shows something under the cell. See [Variables and numbers](/course/open/py-variables?lang=en).",
        },
    )
    quick_assign
    return (quick_assign,)


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="What does this program show?\n\n```python\na = 3\nb = a * 2\nprint(b + 1)\n```",
        options={
            "six": "`6`",
            "word": "`b + 1`",
            "seven": "`7`",
        },
        correct="seven",
        why={
            "six": "`b` is 6, but `print` shows `b + 1`. See the recipe in [Variables and numbers](/course/open/py-variables?lang=en).",
            "word": "Without quotes, Python calculates what is between the parentheses. See [Variables and numbers](/course/open/py-variables?lang=en).",
            "seven": "`b` gets 3 times 2, which is 6, and `print` shows 6 plus 1.",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far, your programs did the same thing every time. But a useful program makes choices: a spam filter puts a message in the spam folder only if it looks like spam, and a game says "You win!" only if your score is high enough. Later in the course, a program that learns from examples makes such choices too. In this lesson, you make your program choose, with `if` and `else`.

    **An everyday example.** "If it rains, take an umbrella; otherwise, take your sunglasses." You look outside, and the answer to one question, *does it rain?*, decides what you do. A program does the same: it asks a question that is either true or false, and runs different lines for each answer.
    """)
    return


@app.cell(hide_code=True)
def predict_compare():
    guess_compare = L.guess(
        "compare",
        question="What do you think `print(3 > 5)` shows?",
        options={
            "false": "`False`",
            "three": "`3`",
            "error": "An error",
        },
        reveal="**`False`.** `3 > 5` asks: is 3 greater than 5? The answer is no, and Python writes that as `False`. A comparison always gives `True` or `False`.",
    )
    guess_compare
    return (guess_compare,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A comparison is a question with two possible answers: `True` or `False`. After `if`, that answer decides which lines run.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **bool** | a value that is either `True` or `False`, with a capital letter | `True` |
    | **comparison** | compares two values and gives a bool | `3 < 5` gives `True` |
    | **condition** | the comparison after `if`; its answer decides which lines run | `spent >= 10` |
    | **indent** | spaces at the start of a line; the indented lines belong to the `if` or `else` above them | two or four spaces, then `print("Hi")` |
    | **if** | runs the indented lines under it only when the condition after it is `True`; the line ends with a colon | `if spent >= 10:` |
    | **else** | runs its indented lines when the condition was `False` | `else:` |

    The symbols to compare with:

    | Symbol | You say | Example | Gives |
    |---|---|---|---|
    | `<` | less than | `3 < 5` | `True` |
    | `>` | greater than | `3 > 5` | `False` |
    | `<=` | less than or equal to | `5 <= 5` | `True` |
    | `>=` | greater than or equal to | `4 >= 5` | `False` |
    | `==` | equal to | `5 == 5` | `True` |
    | `!=` | not equal to | `5 != 5` | `False` |

    How many spaces you indent with does not matter, as long as all lines under the same `if` get the same number. Python programs usually use four; the box in a lesson uses two.
    """)
    return


@app.cell(hide_code=True)
def picture_if():
    mo.vstack([
        draw.if_else("if spent >= 10:", ['print("Free cookie!")', "cookies = 1"], ['print("No cookie this time.")', "cookies = 0"],
                     ["print(cookies)"], yes_label="True", no_label="False", title="Two paths, one of which runs"),
        mo.md(r"""
        If the condition is `True`, Python runs the lines on the left; if it is `False`, the lines on the right. Only one of the two paths runs, and after it the program goes on with the next line that is not indented.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("One `=` gives a name a value; two, `==`, ask whether two values are equal. And do not forget the colon at the end of the `if` and `else` lines. After a colon, press Enter: the box indents the next line by itself. To go back to the left edge, for `else:`, press Backspace."),
        mo.md(r"""
        ### Recipe

        To let a program choose:

        1. Think of the condition: a comparison that is `True` in one case and `False` in the other.
        2. Write `if`, the condition, and a colon.
        3. Under it, indented, write the lines that should run only when the condition is `True`.
        4. Write `else:` at the left edge, right under `if`, and under it, indented, the lines for when the condition is `False`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        A bakery gives you a free cookie when you spend 10 or more. The comments with `#` say which step of the recipe each part is.
        """),
        mo.md(f"```python\n{C.EXAMPLE['en']}\n```"),
        mo.md(r"""
        This program shows:
        """),
        mo.md(f"```\n{C.EXAMPLE_SHOWS['en']}```"),
        mo.md(f"""
        `12.5 >= 10` is `True`, so the two lines under `if` run, and the lines under `else` are skipped. With `spent = {C.OTHER_SPENT}`, the condition is `False`, and the program shows:
        """),
        mo.md(f"```\n{C.EXAMPLE_OTHER['en']}```"),
        mo.md(r"""
        The last line, `print(cookies)`, is not indented, so it runs in both cases. Want to try it yourself? Copy the code into the scratchpad and change the amount.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_below():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Below zero?

        Replace the `...` with an `if` and an `else`: when `temperature` is less than 0, the program shows `Below zero`, and otherwise `Zero or above`. The check also tries other temperatures, such as 0 and 5.
        """),
        L.task("Show `Below zero` or `Zero or above`, depending on `temperature`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_below():
    box_below = L.code_box("below-zero", lines=6)
    box_below
    return (box_below,)


@app.cell(hide_code=True)
def check_below(box_below):
    result_below = L.check("below-zero", box_below.value)
    result_below
    return (result_below,)


@app.cell(hide_code=True)
def hints_below():
    L.hints(
        "below-zero",
        [
            "Step 1 of the recipe: which comparison is `True` when the temperature is below zero? Look at the table of symbols.",
            "The condition is `temperature < 0`. After it comes a colon, and under it, indented, the line with `Below zero`.",
            "`if temperature ___ 0:`, with `print(\"Below zero\")` indented under it, then `else:` at the left edge, with `print(\"Zero or above\")` indented under it.",
        ],
        why="`temperature < 0` is `True` for -3 and -0.5, and `False` for 0 and 5. Under `if` comes the line for below zero, under `else` the other one; the colon and the spaces in front tell Python which lines belong together.",
    )
    return


@app.cell(hide_code=True)
def text_boundary():
    mo.md(r"""
    ### Exercise 2. Exactly on the edge
    """)
    return


@app.cell(hide_code=True)
def choice_boundary_cell():
    choice_boundary = L.choice(
        "boundary",
        question="What does `print(7 > 7)` show?",
        options={
            "true": "`True`",
            "false": "`False`",
            "seven": "`7`",
        },
        correct="false",
        why={
            "true": "`>` asks whether the left value is greater, and 7 is not greater than itself. Look at the table of symbols again: which symbol includes \"equal to\"?",
            "false": "7 is not greater than 7, so the answer is `False`. `7 >= 7` would be `True`.",
            "seven": "A comparison gives `True` or `False`, never a number. Look at the word *comparison* in the table again.",
        },
    )
    choice_boundary
    return (choice_boundary,)


@app.cell(hide_code=True)
def text_exercise_ticket():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A ticket

        A museum ticket costs 5 for children under 12, and 9 for everyone else. Replace the `...` with an `if` and an `else` that give `price` the right value; the last line shows it. The check also tries other ages, such as 8 and 12.
        """),
        L.task("Give `price` the value 5 or 9, depending on `age`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ticket():
    box_ticket = L.code_box("ticket", lines=8)
    box_ticket
    return (box_ticket,)


@app.cell(hide_code=True)
def check_ticket(box_ticket):
    result_ticket = L.check("ticket", box_ticket.value)
    result_ticket
    return (result_ticket,)


@app.cell(hide_code=True)
def hints_ticket():
    L.hints(
        "ticket",
        [
            "Step 1 of the recipe: which comparison is `True` for children under 12? Is 12 itself under 12?",
            "In the worked example, the lines under `if` and `else` gave `cookies` a value. Here they give `price` a value.",
            "`if age ___ 12:`, with `price = 5` indented under it, then `else:`, with `price = 9` indented under it.",
        ],
        why="`age < 12` is `True` for 8 and 11, and `False` for 12 and 15. Under `if`, `price` gets 5; under `else`, 9. `print(price)` is not indented, so it runs in both cases.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Variables and numbers](/course/open/py-variables?lang=en) every line of a program ran; now a condition decides which lines run. Comparisons come back throughout the course: later, a program that learns compares a number with a limit to choose between two answers. In the next lesson you keep many values together, in a list.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_boundary, result_below, result_ticket):
    L.wrap_up(result_below, choice_boundary, result_ticket)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
