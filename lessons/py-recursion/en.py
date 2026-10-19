# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-recursion. Route: course/route/lessen/stam-1-basis.yaml. Checks and the calls in the picture: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-recursion", "en")
    C = L.checks   # the calls in the picture, and what each gives back


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
def quick_check_call():
    quick_call = L.choice(
        "quick-call",
        question="What does this program show?\n\n```python\ndef triple(x):\n    return x * 3\n\nprint(triple(2) + 1)\n```",
        options={
            "six": "`6`",
            "seven": "`7`",
            "nine": "`9`",
        },
        correct="seven",
        why={
            "six": "`triple(2)` gives 6, but that is not all `print` shows. See the guess in [Functions](/course/open/py-functions?lang=en).",
            "seven": "`triple(2)` gives back 6, and `print` shows 6 plus 1.",
            "nine": "The argument is 2, so `x * 3` is 6, not 3 times 3. See the picture in [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_call
    return (quick_call,)


@app.cell(hide_code=True)
def quick_check_none():
    quick_none = L.choice(
        "quick-none",
        question="What does this program show?\n\n```python\ndef show(x):\n    print(x)\n\nresult = show(4)\nprint(result)\n```",
        options={
            "none": "`4`, and under it `None`",
            "twice": "`4`, and under it `4` again",
            "once": "Only `4`",
        },
        correct="none",
        why={
            "none": "`show` prints 4, but has no `return`, so `result` gets `None`.",
            "twice": "`show` uses `print`, not `return`, so it gives nothing back. See the watch-out in [Functions](/course/open/py-functions?lang=en).",
            "once": "The last line shows the value of `result` too. See the watch-out in [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_none
    return (quick_none,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Some problems hold a smaller copy of themselves: a folder can hold files, and other folders. To count all its files, you count the files right there, plus the files in each folder inside it: the same question, only smaller. A function that calls itself solves such problems in a few lines. Later in the course, a program that learns can split its examples into smaller groups and treat each group in the same way.

    **An everyday example.** Russian nesting dolls. How many dolls are there? Open the big one: it is one doll, plus all the dolls inside it. That is the same question, for a smaller doll. The smallest doll does not open, and there the counting stops.
    """)
    return


@app.cell(hide_code=True)
def predict_countdown():
    guess_countdown = L.guess(
        "countdown",
        question="What do you think this program shows?\n\n```python\ndef countdown(n):\n    if n == 0:\n        print(\"Go!\")\n    else:\n        print(n)\n        countdown(n - 1)\n\ncountdown(3)\n```",
        options={
            "three": "Only `3`",
            "all": "`3`, `2`, `1` and `Go!`, each on its own line",
            "go": "Only `Go!`",
        },
        reveal="**`3`, `2`, `1` and `Go!`.** `countdown(3)` shows 3 and calls `countdown(2)`, which shows 2 and calls `countdown(1)`, and so on. `countdown(0)` shows `Go!` and calls nothing more, so it all stops there.",
    )
    guess_countdown
    return (guess_countdown,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## The idea

    A function may call itself, as long as each call works on a smaller piece, and the smallest piece is answered directly.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **recursion** | a function that calls itself, each time on a smaller piece of the problem | `return n + sum_to(n - 1)` |
    | **base case** | the smallest case, whose answer the function gives back without calling itself; it stops the recursion | `if n == 0:` and `return 0` |

    Each call has its own names: the `n` of one call is not the `n` of the next. A slice works on a string just as on a list: `text[1:]` is the text without its first character, a smaller piece. `""` is the empty text, with no characters at all; `==` compares texts too, so `text == ""` is `True` only for the empty text. And `+` glues two texts together: `"ab" + "c"` is `"abc"`. Without a base case, or when the calls never get smaller, the function never stops; Python gives up after about a thousand calls inside each other, with an error.
    """)])
    return


@app.cell(hide_code=True)
def picture_calls():
    mo.vstack([
        mo.md("```python\ndef sum_to(n):\n    if n == 0:\n        return 0\n    return n + sum_to(n - 1)\n```"),
        draw.program_trace(C.PICTURE_LINES["en"], C.PICTURE_BACK, code_label="The call", output_label="Gives back",
                           title="sum_to(3), call by call"),
        mo.md(r"""
        The calls go down, each with a number one smaller, until the base case gives back 0. Then the answers come back up: 1 + 0 is 1, 2 + 1 is 3, and 3 + 3 is 6.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Every function that calls itself needs a base case that gives back an answer without calling again, and every call must get closer to it. If not, the function never stops. Put the base case first, at the top of the function."),
        mo.md(r"""
        ### Recipe

        To let a function call itself:

        1. Base case: with `if`, catch the smallest case, and `return` its answer.
        2. Otherwise, call the function itself on a smaller piece: `n - 1`, or `text[1:]`.
        3. Combine that answer with the part you set aside, `n` or `text[0]`, and `return` the result.
        4. Check that every call gets closer to the base case.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two functions that call themselves: a power, such as 2 to the power 5, and the number of times the letter `a` is in a text. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_recursion():
    def _power(number, exponent):
        # 1. base case: anything to the power 0 is 1
        if exponent == 0:
            return 1
        # 2. and 3. the power one smaller, times the number once more
        return number * _power(number, exponent - 1)

    def _count_a(text):
        # 1. base case: an empty text has no a
        if text == "":
            return 0
        # 2. the rest of the text, one character shorter
        rest = _count_a(text[1:])
        # 3. add 1 if the first character is an a
        if text[0] == "a":
            return rest + 1
        return rest

    # 4. each call is one smaller: exponent - 1, or text[1:]
    print(_power(2, 5))
    print(_count_a("banana"))
    # every power on the way, from 2 to the power 0 up to 5
    for _e in range(6):
        print(f"2 to the power {_e} is {_power(2, _e)}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `32` and `3`, and then every power on the way: each one is 2 times the one before. `_power(2, 5)` is 2 times `_power(2, 4)`, and so on down to `_power(2, 0)`, which is 1. `_count_a("banana")` looks at `b` and asks the same question for `anana`, then for `nana`, all the way down to the empty text.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_factorial():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The factorial

        The factorial of a number is that number times every whole number below it, down to 1: the factorial of 4 is `4 * 3 * 2 * 1`, which is 24. The factorial of 0 is 1. Finish `factorial(n)` with a base case and a call to itself. The check calls it with 4, and with other numbers, also 0.
        """),
        L.task("Make `factorial(n)` give back the factorial of `n`, by calling itself."),
    ])
    return


@app.cell(hide_code=True)
def exercise_factorial():
    box_factorial = L.code_box("factorial", lines=6)
    box_factorial
    return (box_factorial,)


@app.cell(hide_code=True)
def check_factorial(box_factorial):
    result_factorial = L.check("factorial", box_factorial.value)
    result_factorial
    return (result_factorial,)


@app.cell(hide_code=True)
def hints_factorial():
    L.hints(
        "factorial",
        [
            "Step 1 of the recipe: what is the smallest case, and what is its answer? The task says what the factorial of 0 is.",
            "The factorial of 4 is 4 times the factorial of 3. So the factorial of `n` is `n` times the factorial of ___.",
            "`if n == 0:` with `return 1` under it, and then `return n * factorial(___)`.",
        ],
        why="`if n == 0: return 1` is the base case. Otherwise `return n * factorial(n - 1)`: the number itself times the factorial of the number one smaller. `factorial(4)` becomes `4 * 3 * 2 * 1 * 1`, which is 24.",
    )
    return


@app.cell(hide_code=True)
def text_sum_four():
    mo.md(r"""
    ### Exercise 2. Follow the calls
    """)
    return


@app.cell(hide_code=True)
def choice_sum_four_cell():
    choice_sum_four = L.choice(
        "sum-four",
        question="What does this program show?\n\n```python\ndef f(n):\n    if n == 1:\n        return 1\n    return n + f(n - 1)\n\nprint(f(4))\n```",
        options={
            "four": "`4`",
            "twenty-four": "`24`",
            "ten": "`10`",
        },
        correct="ten",
        why={
            "four": "`f(4)` does not stop at 4: it adds `f(3)`, which adds `f(2)`, and so on. Look at the picture with the calls again.",
            "twenty-four": "The function adds, `+`; it does not multiply. Read the last line of the function again.",
            "ten": "`f(4)` is 4 + `f(3)`, which is 4 + 3 + 2 + 1, so 10. `f(1)` is the base case.",
        },
    )
    choice_sum_four
    return (choice_sum_four,)


@app.cell(hide_code=True)
def text_exercise_reverse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A text backwards

        Finish `reverse(text)`: it gives back the text backwards, so `reverse("abc")` gives `"cba"`. The reverse of a text is the reverse of the rest, `text[1:]`, followed by the first character. The empty text is the base case: backwards, it is still empty. The check also tries other texts.
        """),
        L.task("Make `reverse(text)` give back the text backwards, by calling itself."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reverse():
    box_reverse = L.code_box("reverse", lines=6)
    box_reverse
    return (box_reverse,)


@app.cell(hide_code=True)
def check_reverse(box_reverse):
    result_reverse = L.check("reverse", box_reverse.value)
    result_reverse
    return (result_reverse,)


@app.cell(hide_code=True)
def hints_reverse():
    L.hints(
        "reverse",
        [
            "Step 1 of the recipe: the base case is the empty text, `\"\"`. What does it give back?",
            "Look at `_count_a` in the worked example: it calls itself on `text[1:]`. Here, the first character, `text[0]`, goes after that result.",
            "`if text == \"\":` with `return \"\"` under it, and then `return reverse(___) + ___`.",
        ],
        why="`if text == \"\": return \"\"` stops the calls. Otherwise `return reverse(text[1:]) + text[0]`: the rest backwards, then the first character. `reverse(\"abc\")` becomes `reverse(\"bc\") + \"a\"`, which is `\"cb\" + \"a\"`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Functions](/course/open/py-functions?lang=en) a function did its work in one go; now it hands a smaller piece of the work to itself. A loop can do many of the same jobs, and in Python a loop is often the simpler choice. Recursion shines where the problem itself is nested, such as folders in folders. In the next lesson you keep values under names of their own in a dictionary.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sum_four, result_factorial, result_reverse):
    L.wrap_up(result_factorial, choice_sum_four, result_reverse)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
