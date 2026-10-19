# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-imports. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-imports", "en")


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
def quick_check_greet():
    quick_greet = L.choice(
        "quick-greet",
        question="What does this program show?\n\n```python\ndef greet(name):\n    return f\"Hi {name}\"\n\nmessage = greet(\"Ada\")\nprint(message)\n```",
        options={
            "hi-ada": "`Hi Ada`",
            "none": "`None`",
            "hi-name": "`Hi name`",
        },
        correct="hi-ada",
        why={
            "hi-ada": "`greet(\"Ada\")` gives back the f-string with `Ada` filled in, and `message` keeps it.",
            "none": "`greet` has a `return`, so it does give something back. See the watch-out in [Functions](/course/open/py-functions?lang=en).",
            "hi-name": "Inside the function, `name` has the value of the argument. See the picture in [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_greet
    return (quick_greet,)


@app.cell(hide_code=True)
def quick_check_inside():
    quick_inside = L.choice(
        "quick-inside",
        question="What happens when this program runs?\n\n```python\ndef double(x):\n    result = x * 2\n    return result\n\ndouble(4)\nprint(result)\n```",
        options={
            "eight": "It shows `8`",
            "error": "It stops: Python does not know `result` on the last line",
            "four": "It shows `4`",
        },
        correct="error",
        why={
            "eight": "`result` exists only while `double` runs. See the idea in [Functions](/course/open/py-functions?lang=en).",
            "error": "`result` is made inside the function and exists only while it runs; outside it, Python does not know the name.",
            "four": "`result` is never 4, and outside the function it does not exist at all. See the idea in [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_inside
    return (quick_inside,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You do not have to write everything yourself. Python comes with hundreds of modules full of ready-made code: `math` for square roots and pi, `random` for chance, and many more. Later in the course, NumPy and PyTorch are modules too, and you bring them in the same way. And chance matters for programs that learn: a seed makes "random" repeatable, so an experiment gives the same result again.

    **An everyday example.** A toolbox. You do not forge a new hammer each time you need one: you take it out of the box. `import` brings the box into your program, and the dot takes one tool out of it: `math.sqrt`.
    """)
    return


@app.cell(hide_code=True)
def predict_sqrt():
    guess_sqrt = L.guess(
        "sqrt",
        question="What do you think this program shows?\n\n```python\nimport math\nprint(math.sqrt(16))\n```",
        options={
            "four": "`4`",
            "sixteen": "`16`",
            "float": "`4.0`",
        },
        reveal="**`4.0`.** `math.sqrt` is the square root from the module `math`: the number that, times itself, gives 16. It always gives back a float, so Python shows `4.0`.",
    )
    guess_sqrt
    return (guess_sqrt,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A module is a file with ready-made code. `import` brings it into your program; after that, you use what is in it with the module's name and a dot.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **module** | a file with ready-made code, such as `math` or `random` | |
    | **import** | brings a module into your program; it goes at the top | `import math` |
    | **dot** | takes a name out of a module | `math.sqrt(16)` gives `4.0` |
    | **from ... import** | brings in one name, so you can use it without the module's name | `from math import pi` |
    | **seed** | the starting point of the random numbers: the same seed gives the same numbers again | `random.seed(42)` |

    In `math`: `math.sqrt(x)` gives the square root, `math.pi` is pi, 3.14159..., and `math.floor(7.8)` rounds down to `7`. In `random`: `random.randint(1, 6)` gives a whole number from 1 up to and including 6, and `random.choice(items)` picks one item from a list. The numbers of `random` are not truly random: they come from a fixed calculation that starts at the seed.
    """)
    return


@app.cell(hide_code=True)
def picture_module():
    mo.vstack([
        draw.flow(
            [("Your program", ["import math", "math.sqrt(16)"]),
             ("The module math", ["sqrt", "pi", "floor", "..."])],
            forward=["asks math\nfor sqrt(16)"],
            back=["4.0 comes back"],
            title="A call to a module"),
        mo.md(r"""
        `math.sqrt(16)` is a call, just like a call to your own function: the dot says that `sqrt` lives in `math`. The function in the module does the work, and the result comes back to your program.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("After `import math`, every name from the module needs `math.` in front: `sqrt(16)` on its own stops with an error, because Python does not know `sqrt`. And `random.randint(1, 6)` includes 6, while `range(1, 6)` stops before 6."),
        mo.md(r"""
        ### Recipe

        To use code from a module:

        1. At the top, write `import` and the module's name: `import math`.
        2. Use a name from it with the module's name and a dot: `math.sqrt(16)`.
        3. Do you use one name often? `from math import pi` brings it in, and then you write just `pi`.
        4. For chance: first `random.seed(...)` if you want the same numbers again, then `random.randint(a, b)` or `random.choice(items)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A few things from `math`, and dice rolls with a seed. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_modules():
    # 1. import at the top
    import math
    import random
    from math import pi
    # 2. the module's name, a dot, and the name inside it
    print(math.sqrt(2))
    print(math.floor(7.8))
    print(math.pi)
    # 3. a name brought in with from: no math. in front
    print(pi)
    # 4. a seed, and then three dice rolls
    random.seed(7)
    _rolls = []
    for _i in range(3):
        _rolls.append(random.randint(1, 6))
    print(_rolls)
    # the same seed again gives the same rolls
    random.seed(7)
    _again = []
    for _i in range(3):
        _again.append(random.randint(1, 6))
    print(_again)

    # 4. inside a function: the same seed gives the same pick
    def _coin(seed):
        random.seed(seed)
        return random.choice(["heads", "tails"])

    print(_coin(5))
    print(_coin(5))
    return math, pi, random


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `1.4142135623730951`, `7`, twice `3.141592653589793`, twice `[3, 2, 4]`, and twice `tails`. The two lists of rolls are the same, because both start at seed 7; without the seed, they would usually be different each time you run the cell. `_coin` sets the seed inside the function, so the same seed gives the same pick. An `import` makes a name, just like `=`: this cell makes `math`, `random` and `pi`. Another cell can use `math.sqrt(25)` right away, but `import math` in another cell breaks *one name, one cell*; try things out in the scratchpad. The exercise boxes run on their own, so each box has its own `import` line.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_circle():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The area of a circle

        Finish `circle_area(r)`: the area of a circle with radius `r` is pi times `r` squared. Use `math.pi`; the box already starts with `import math`. The check calls it with 3, and with other sizes.
        """),
        L.task("Make `circle_area(r)` give back pi times `r` squared, with `math.pi`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_circle():
    box_circle = L.code_box("circle", lines=5)
    box_circle
    return (box_circle,)


@app.cell(hide_code=True)
def check_circle(box_circle):
    result_circle = L.check("circle", box_circle.value)
    result_circle
    return (result_circle,)


@app.cell(hide_code=True)
def hints_circle():
    L.hints(
        "circle",
        [
            "Step 2 of the recipe: pi is in the module `math`. How do you write it, with the module's name?",
            "`r` squared is `r ** 2`, or `r * r`.",
            "`return math.___ * r ** 2`",
        ],
        why="`math.pi * r ** 2` takes the exact pi from `math`, times `r` squared. For `r = 1`, that is pi itself: 3.14159...",
    )
    return


@app.cell(hide_code=True)
def text_import_name():
    mo.md(r"""
    ### Exercise 2. Without the module's name
    """)
    return


@app.cell(hide_code=True)
def choice_import_name_cell():
    choice_import_name = L.choice(
        "import-name",
        question="What happens when this program runs?\n\n```python\nimport math\nprint(sqrt(9))\n```",
        options={
            "float": "It shows `3.0`",
            "whole": "It shows `3`",
            "error": "It stops: Python does not know `sqrt`",
        },
        correct="error",
        why={
            "error": "`import math` brings in the module, not its names: `sqrt` lives in `math`, so it needs `math.` in front.",
            "float": "Look at how the call is written here, and at the watch-out above the recipe.",
            "whole": "`math.sqrt` would give a float. But look at how the call is written here, and at the watch-out above the recipe.",
        },
    )
    choice_import_name
    return (choice_import_name,)


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A repeatable pick

        Finish `pick(items, seed)`: it sets the seed with `random.seed`, using its argument `seed`, and then picks one item from `items` with `random.choice`. With the same seed, it always picks the same item. The check calls it with `["rock", "paper", "scissors"]` and seed 1, and with other lists and seeds.
        """),
        L.task("Make `pick(items, seed)` pick an item that is the same every time for the same seed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick", lines=6)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick",
        [
            "Step 4 of the recipe, and `_coin` in the worked example: first the seed, then the pick.",
            "On the first line, set the seed with the argument `seed`; on the second, give back one item picked from `items`.",
            "`random.seed(___)`, then `return random.choice(___)`",
        ],
        why="`random.seed(seed)` sets the starting point, so `random.choice(items)` after it always picks the same item for the same seed. That is how an experiment with chance can be repeated.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Functions](/course/open/py-functions?lang=en) you wrote your own functions; a module is a collection of functions that others wrote, and `math.sqrt(16)` is a call, just like yours. Later in the course, NumPy and PyTorch are imported the same way, and a seed makes training repeatable. In the next lesson you put your own code in a module.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_import_name, result_circle, result_pick):
    L.wrap_up(result_circle, choice_import_name, result_pick)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
