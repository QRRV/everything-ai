# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-while. Route: course/route/lessen/stam-1-basis.yaml. Checks and the loop in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-while", "en")
    C = L.checks   # the loop in the picture, check by check


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
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="What does this program show?\n\n```python\ntotal = 0\nfor n in [2, 3, 4]:\n    total = total + n\nprint(total)\n```",
        options={
            "nine": "`9`",
            "four": "`4`",
            "lines": "`2`, `5` and `9`, each on its own line",
        },
        correct="nine",
        why={
            "nine": "Each round adds `n` to `total`: 2, then 5, then 9. The `print` is not indented, so it runs once, after the loop.",
            "four": "`total` does not just get the last item: each round, it gets what it already was plus `n`. See the picture in [Repeating: for loops](/course/open/py-for?lang=en).",
            "lines": "The `print` is at the left edge, after the loop, so it runs once. See the recipe in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Which numbers does `range(3)` give?",
        options={
            "one-three": "1, 2 and 3",
            "zero-two": "0, 1 and 2",
            "zero-three": "0, 1, 2 and 3",
        },
        correct="zero-two",
        why={
            "one-three": "With one number, `range` starts at 0. See the idea in [Repeating: for loops](/course/open/py-for?lang=en).",
            "zero-three": "`range` stops just before its stop. See the idea in [Repeating: for loops](/course/open/py-for?lang=en).",
            "zero-two": "With one number, `range` starts at 0 and stops just before 3.",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A `for` loop goes through a list you already have. But sometimes you do not know in advance how many rounds you need: you want to go on until something happens. Save until you have enough; try again until it works. Later in the course, a program that learns keeps practicing until its mistakes are small enough. That is a `while` loop.

    **An everyday example.** "Stir until the sauce gets thick." The recipe does not say how many times to stir. Before each stir you look: is it thick yet? No? Stir again. Yes? Stop.
    """)
    return


@app.cell(hide_code=True)
def predict_double():
    guess_double = L.guess(
        "double",
        question="What do you think this program shows?\n\n```python\nn = 1\nwhile n < 20:\n    n = n * 2\nprint(n)\n```",
        options={
            "sixteen": "`16`",
            "thirty-two": "`32`",
            "twenty": "`20`",
        },
        reveal="**`32`.** The condition is checked before every round. When `n` is 16, `16 < 20` is still `True`, so one more round makes it 32. Then `32 < 20` is `False`, and the loop stops. You see each condition in the picture below.",
    )
    guess_double
    return (guess_double,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    `while` runs its indented lines again and again, as long as its condition is `True`. Before every round, it looks at the condition once more.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **while** | runs the indented lines under it again and again, as long as the condition after it is `True`; it looks at the condition before every round | `while n < 20:` |
    | **break** | stops a loop at once, also a `for` loop; Python goes on with the first line after the loop | `break` |
    | **endless loop** | a loop whose condition never becomes `False`, so it never stops | |

    Something in the indented lines has to change what the condition looks at, or the loop goes on forever. Does a cell keep running? Click its stop button, ■. In the box of an exercise, the check stops your code after 2 seconds.
    """)
    return


@app.cell(hide_code=True)
def picture_checks():
    mo.vstack([
        mo.md("```python\nn = 1\nwhile n < 20:\n    n = n * 2\nprint(n)\n```"),
        draw.program_trace(C.PICTURE_LINES["en"], C.PICTURE_SHOWN, code_label="Condition before the round", names=C.PICTURE_NAMES,
                           names_label="n after it", output_label="Under the cell", title="The loop above, one condition at a time"),
        mo.md(r"""
        Each round doubles `n`, until the condition before the next round gives `False`. Only then does the program go on to `print(n)`, which shows 32.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A `while` loop stops only when its condition becomes `False`. If nothing in the loop changes what the condition looks at, it never does. And `break` only runs when Python gets to it: put it inside an `if`, so the loop stops at the right moment."),
        mo.md(r"""
        ### Recipe

        To repeat until something happens:

        1. Before the loop, give the names in the condition their starting value.
        2. Write `while`, the condition to go on, and a colon: `while savings < 100:`.
        3. Under it, indented, write one round. One of its lines must change what the condition looks at.
        4. To stop early, from inside the loop, write `break` under an `if`.
        5. After the loop, at the left edge, use the result: it runs once.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You save 15 dollars a week. After how many weeks do you have at least 100? And in a list of temperatures, which is the first one below zero? The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_saving():
    # 1. before the loop: the names in the condition get a starting value
    _savings = 0
    _weeks = 0
    # 2. while, the condition to go on, and a colon
    while _savings < 100:
        # 3. one round: this changes what the condition looks at
        _savings = _savings + 15
        _weeks = _weeks + 1
    # 5. after the loop: the result
    print(f"After {_weeks} weeks: {_savings} saved")

    _temps = [4, 2, -1, 3, -5]
    _first_frost = "none"
    for _t in _temps:
        if _t < 0:
            _first_frost = _t
            # 4. break: stop the loop at once, under an if
            break
    print(f"First frost: {_first_frost}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `After 7 weeks: 105 saved` and `First frost: -1`. After 6 weeks there is 90, and `90 < 100` is still `True`, so one more round makes it 105. In the second loop, `break` stops the `for` loop at -1, so 3 and -5 are never looked at.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_double():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Doubling

        The number of bacteria doubles every hour. It starts at 1. After how many hours are there at least `target`, which is 1000? Replace the two `...`: the condition after `while`, and one round, in which `count` doubles and `hours` goes up by 1. The check also tries other targets.
        """),
        L.task("Make `hours` the number of hours until `count` is at least `target`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_double():
    box_double = L.code_box("double", lines=7)
    box_double
    return (box_double,)


@app.cell(hide_code=True)
def check_double(box_double):
    result_double = L.check("double", box_double.value)
    result_double
    return (result_double,)


@app.cell(hide_code=True)
def hints_double():
    L.hints(
        "double",
        [
            "Step 2 of the recipe: the loop should go on as long as `count` is still below `target`. Which comparison is that?",
            "In one round, two names change: `count` doubles, and `hours` goes up by 1. Both lines are indented.",
            "`while count ___ target:`, then indented the line that doubles `count`, and `hours = hours + ___`.",
        ],
        why="`while count < target:` goes on as long as there are fewer than 1000. Each round doubles `count` and adds 1 to `hours`. After 10 rounds `count` is 1024, the condition gives `False`, and `print(hours)` shows 10.",
    )
    return


@app.cell(hide_code=True)
def text_break():
    mo.md(r"""
    ### Exercise 2. Where break stops
    """)
    return


@app.cell(hide_code=True)
def choice_break_cell():
    choice_break = L.choice(
        "break",
        question="What does this program show?\n\n```python\nfor n in [3, 8, 2, 9]:\n    if n > 5:\n        break\n    print(n)\n```",
        options={
            "three-two": "`3` and `2`, each on its own line",
            "eight": "`8`",
            "three": "`3`",
        },
        correct="three",
        why={
            "three-two": "`break` does not skip one item: it stops the whole loop. Look at the word *break* in the table again.",
            "eight": "`break` comes before `print`, so at 8 the loop stops before `print` can show 8. Look at the order of the lines again.",
            "three": "3 is not greater than 5, so it is shown. At 8, `break` stops the whole loop, before its `print`.",
        },
    )
    choice_break
    return (choice_break,)


@app.cell(hide_code=True)
def text_exercise_long():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The first long word

        Find the first word in `words` with more than 5 letters, and put it in `first_long`. Replace the `...` with an `if` and, inside it, the line for `first_long` and a `break`. The check also tries other lists.
        """),
        L.task("Put the first word with more than 5 letters in `first_long`, and stop the loop there."),
    ])
    return


@app.cell(hide_code=True)
def exercise_long():
    box_long = L.code_box("first-long", lines=8)
    box_long
    return (box_long,)


@app.cell(hide_code=True)
def check_long(box_long):
    result_long = L.check("first-long", box_long.value)
    result_long
    return (result_long,)


@app.cell(hide_code=True)
def hints_long():
    L.hints(
        "first-long",
        [
            "Look at the second loop of the worked example: a name for the result before the loop, an `if`, and under it the line for the result and a `break`.",
            "`len(word)` gives the number of characters, here letters. Which comparison is `True` for a word with more than 5 letters?",
            "`if len(word) ___ 5:`, and under it, indented, the line that gives `first_long` the word, and then ___.",
        ],
        why="`if len(word) > 5:` is `False` for `sun` and `rain`, and `True` for `thunder`. Then `first_long` gets `thunder`, and `break` stops the loop, so `drizzle` never replaces it.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Repeating: for loops](/course/open/py-for?lang=en) a loop went through a whole list; `while` goes on until its condition is `False`, and `break` stops any loop early. Later in the course, a program that learns repeats its practice until its mistakes are small enough, or for a fixed number of rounds. In the next lesson you put code under a name of its own, as a function.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_break, result_double, result_long):
    L.wrap_up(result_double, choice_break, result_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
