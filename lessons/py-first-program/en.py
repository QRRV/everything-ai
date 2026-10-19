# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-first-program. Route: course/route/lessen/stam-1-basis.yaml. Checks and the program in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-first-program", "en")
    C = L.checks   # the program in the picture, and what it shows after each line


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How a lesson works](/course/open/start-how-this-course-works?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="You run a code cell. Where does its result appear?",
        options={
            "start": "On the start page.",
            "under": "Right under the code.",
            "window": "In a new window.",
        },
        correct="under",
        why={
            "under": "Right under the code. That is where your own programs show their results too.",
            "start": "The start page shows the route and your progress, not the results of cells. Run the first cell of [How a lesson works](/course/open/start-how-this-course-works?lang=en) again and see where its result appears.",
            "window": "Everything stays in the lesson itself. Run the first cell of [How a lesson works](/course/open/start-how-this-course-works?lang=en) again and see where its result appears.",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def quick_check_scratchpad():
    quick_scratchpad = L.choice(
        "quick-scratchpad",
        question="You want to try out a bit of code without changing the lesson. Where do you do that?",
        options={
            "box": "In the box of an exercise.",
            "start": "On the start page.",
            "scratchpad": "In the scratchpad.",
        },
        correct="scratchpad",
        why={
            "scratchpad": "The scratchpad is for trying things out; the lesson stays as it is.",
            "box": "What you type in an exercise box is checked as your answer. See *Trying something out* in [How a lesson works](/course/open/start-how-this-course-works?lang=en).",
            "start": "The start page has no place for code. See *Trying something out* in [How a lesson works](/course/open/start-how-this-course-works?lang=en).",
        },
    )
    quick_scratchpad
    return (quick_scratchpad,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A computer can do a lot, but it does nothing by itself: it only carries out instructions. To make it do something, you write those instructions down. From this lesson on you do that yourself, in Python. Everything you build in this course is made that way, from a program that counts letters to a program that learns from examples.

    **An everyday example.** A recipe. It is a list of steps, and a cook carries them out one by one, from top to bottom. Swap two steps, and you get something else. And the cook does exactly what the recipe says: if the recipe says "add salt" twice, the salt goes in twice.
    """)
    return


@app.cell(hide_code=True)
def predict_quotes():
    guess_quotes = L.guess(
        "no-quotes",
        question="`print(\"Hello\")` shows `Hello` under the cell. What do you think happens with `print(Hello)`, without the quotes?",
        options={
            "same": "It shows `Hello` too.",
            "error": "Python stops with an error.",
            "nothing": "It shows nothing.",
        },
        reveal="**Python stops with an error.** Without quotes, Python reads `Hello` as a word it must already know, like `print`, and it does not know that word. With quotes, `\"Hello\"` is just text, which Python shows exactly as it is.",
    )
    guess_quotes
    return (guess_quotes,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **program** | a list of instructions for the computer, which carries them out one by one, from top to bottom | the code in a cell |
    | **print** | an instruction that shows what is between its parentheses, on a new line under the cell | `print("Hello")` |
    | **string** | a piece of text between quotes; Python shows it exactly as it is, and does not try to understand it | `"Hello"` |
    | **comment** | a line that starts with `#`; Python skips it, so you can write notes for people | `# a note` |

    Below is a program with three instructions. Python carries out the first line, then the second, then the third.
    """)
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        draw.program_trace(C.STEPS["en"], C.SHOWN["en"], code_label="Code",
                           output_label="What the line shows under the cell", title="A program, line by line"),
        mo.md(r"""
        Python runs the lines from top to bottom, and each `print` adds its text as a new line under the cell, in the order of the code. The quotes do not appear: they only tell Python where the text starts and ends.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Python is exact. `Print` with a capital P is not the same as `print`, and quotes must be straight, like `\"`, not curly, like `“`. Typing in a box gives straight quotes; curly ones come from text copied out of a document or a web page."),
        mo.md(r"""
        ### Recipe

        To show lines of text:

        1. Write `print`, with parentheses after it: `print()`.
        2. Between the parentheses, put the text between quotes: `print("Hello")`.
        3. Write one `print` for each line, in the order in which the lines should appear.
        4. Run the cell, and compare what appears with what you meant.

        ### Worked example

        A shopping list of three lines. The comments with `#` say which step of the recipe each part is. Run the cell.
        """),
    ])
    return


@app.cell
def example_shopping():
    # 1 and 2. print, and the text in quotes inside the parentheses
    print("Shopping list")
    # 3. one print for each line, in the order of the lines
    print("bread")
    print("cheese")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        4. Under the cell appear `Shopping list`, `bread` and `cheese`, each on its own line, in the order of the code. The comments do not appear: Python skips them.
        """),
        L.task("Move `print(\"cheese\")` above `print(\"bread\")` in the cell above, and run the cell again with **Shift + Enter**. To move a line: click in it, and press **Alt + ↑** (on a Mac **Option + ↑**)."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_tools():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Three lines

        This course uses three tools, which you will meet one by one: `Python`, `NumPy` and `PyTorch`. Write a program in the box that shows these three names, each on its own line, in this order. The check reads what your program shows.
        """),
        L.task("Show `Python`, `NumPy` and `PyTorch`, each on its own line, in this order."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tools():
    box_tools = L.code_box("tools", lines=4)
    box_tools
    return (box_tools,)


@app.cell(hide_code=True)
def check_tools(box_tools):
    result_tools = L.check("tools", box_tools.value)
    result_tools
    return (result_tools,)


@app.cell(hide_code=True)
def hints_tools():
    L.hints(
        "tools",
        [
            "Look at step 3 of the recipe: one `print` for each line you want to see.",
            "The first line of your program can be `print(\"Python\")`. The other two look the same, with other text.",
            "`print(\"Python\")`, then `print(\"NumPy\")`, then a third line with ___.",
        ],
        why="Three instructions, one for each line, in the order in which the lines should appear. The capitals count: Python shows text exactly as you write it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_countdown():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Put it in order

        The program in the box should count down: `3`, `2`, `1` and then `Go!`. The four lines are all there, but in the wrong order. Do not change the lines themselves; only move them, the way you moved a line in the worked example.
        """),
        L.task("Put the four lines in the order of a countdown."),
    ])
    return


@app.cell(hide_code=True)
def exercise_countdown():
    box_countdown = L.code_box("countdown", lines=5)
    box_countdown
    return (box_countdown,)


@app.cell(hide_code=True)
def check_countdown(box_countdown):
    result_countdown = L.check("countdown", box_countdown.value)
    result_countdown
    return (result_countdown,)


@app.cell(hide_code=True)
def hints_countdown():
    L.hints(
        "countdown",
        [
            "Python runs from top to bottom. Which line should run first?",
            "A countdown starts at 3, so `print(\"3\")` goes at the top.",
            "`print(\"Go!\")` comes last.",
        ],
        why="The same four instructions, now in the order in which the lines should appear: 3, 2, 1, Go!",
    )
    return


@app.cell(hide_code=True)
def text_quotes():
    mo.md(r"""
    ### Exercise 3. Which program?
    """)
    return


@app.cell(hide_code=True)
def choice_quotes_cell():
    choice_quotes = L.choice(
        "quotes",
        question="Which program shows `Hello` on one line, and `world` on the line under it?",
        options={
            "one": "`print(\"Hello world\")`",
            "two": "`print(\"Hello\")`, and under it `print(\"world\")`",
            "bare": "`print(Hello)`, and under it `print(world)`",
        },
        correct="two",
        why={
            "two": "Two instructions give two lines, and the text between quotes is shown exactly as it is.",
            "one": "One `print` shows one line. Look at step 3 of the recipe again.",
            "bare": "Without quotes, Python reads `Hello` as a word it must already know, like `print`, and stops with an error. Look at step 2 of the recipe again.",
        },
    )
    choice_quotes
    return (choice_quotes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [How a lesson works](/course/open/start-how-this-course-works?lang=en) you ran a cell with `print` in it without knowing what it was; now you know. Every program in this course runs this way: line by line, from top to bottom. In the next lesson you give a value a name, and calculate with numbers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_quotes, result_countdown, result_tools):
    L.wrap_up(result_tools, result_countdown, choice_quotes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
