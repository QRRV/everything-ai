# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-names-in-cells. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow variables, function

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-names-in-cells", "en")


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
def quick_check_area():
    quick_area = L.choice(
        "quick-area",
        question="What does this program show?\n\n```python\nwidth = 4\nheight = 3\narea = width * height\nprint(area)\n```",
        options={
            "seven": "`7`",
            "twelve": "`12`",
            "word": "`area`",
        },
        correct="twelve",
        why={
            "seven": "`*` is times, not plus. See the table of symbols in [Variables and numbers](/course/open/py-variables?lang=en).",
            "twelve": "`area` gets the value of 4 times 3, and `print(area)` shows that value.",
            "word": "Without quotes, `print(area)` shows the value of the name, not the word. See [Variables and numbers](/course/open/py-variables?lang=en).",
        },
    )
    quick_area
    return (quick_area,)


@app.cell(hide_code=True)
def quick_check_power():
    quick_power = L.choice(
        "quick-power",
        question="What does `print(2 ** 3)` show?",
        options={
            "eight": "`8`",
            "six": "`6`",
            "nine": "`9`",
        },
        correct="eight",
        why={
            "eight": "`**` is to the power of: 2 times 2 times 2 is 8.",
            "six": "`**` is not times. See the table of symbols in [Variables and numbers](/course/open/py-variables?lang=en).",
            "nine": "That is 3 to the power 2, the other way around. See the table of symbols in [Variables and numbers](/course/open/py-variables?lang=en).",
        },
    )
    quick_power
    return (quick_power,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far, each program was in a box of its own. A notebook has many cells, and they work together: you can make a name in one cell and use it in another. That is handy. The numbers go in one cell and the calculation in the next, and when you change a number, the calculation runs again by itself. But it only works with one rule, and this lesson is about that rule.

    **An everyday example.** A shared kitchen, with two jars on the shelf that both say *sugar*, and different sugar in each. A recipe says "a spoon from the sugar jar". Which jar? Nobody can tell. A notebook has the same problem when two cells make the same name, and it does not guess.
    """)
    return


@app.cell(hide_code=True)
def predict_two():
    guess_two = L.guess(
        "two-cells",
        question="One cell has `total = 3`, and another cell has `print(total)`. In a third cell you type `total = 5`, and you run it. What do you think happens?",
        options={
            "five": "`print(total)` now shows `5`.",
            "both": "`total` is 3 in one cell and 5 in the other.",
            "stops": "Your new cell does not run, with a message that `total` is also made in another cell.",
        },
        reveal="**Your new cell does not run.** The notebook does not guess which `total` you mean. Under your cell it says that `total` is also made in another cell, and `print(total)` keeps showing 3. That is the rule of this lesson: one name, one cell.",
    )
    guess_two
    return (guess_two,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A cell *makes* a name when it gives it a value with `=`. Other cells can then use that name. The notebooks of this course run on *marimo*, a program that keeps track of which cell makes which name. That is how it knows what to do when you change a cell: it runs every cell that uses one of its names again. Where a cell is on the page does not matter.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **marimo** | the program that runs the notebooks of this course | |
    | **one name, one cell** | the rule of a notebook: each name is made in one cell only; other cells may use it, but not make it again | `price = 2.5` in one cell, `total = price * 3` in another |
    | **`_` name** | a name that starts with `_`; it exists only inside its own cell, so other cells cannot see it, and another cell can make its own `_` name spelled the same way | `_months = 12` |

    When a cell makes a name that another cell already makes, marimo does not run it. Under the cell you see *This cell redefines variables from other cells*, with the name and the other cell that makes it too. *Variables* is the word from the last lesson for names. The next time the notebook opens, neither cell runs, and both show the message. You may also see a button *Fix: Wrap in a function*; leave it alone, because here the fix is another name. And a cell that uses a `_` name from another cell stops with an error: Python does not know that name there.
    """)
    return


@app.cell(hide_code=True)
def picture_cells():
    mo.vstack([
        draw.notebook_cells([
            ("One name, one cell", ["price = 2.5", "total = price * 3", "print(total)"], ["price", "total"],
             "✓ all three cells run"),
            ("The same name in two cells", ["total = 2.5 * 3", "total = 4 * 2"], [None],
             "✗ marimo stops: total is made twice"),
            ("Names that start with _", ["_total = 2.5 * 3\nprint(_total)", "_total = 4 * 2\nprint(_total)"], [None],
             "✓ each cell has its own _total"),
        ], title="Names across cells"),
        mo.md(r"""
        On the left, `price` and `total` each come from one cell, and an arrow shows which cell uses them. In the middle, two cells make `total`, so marimo cannot know which one counts; on the right, each `_total` stays inside its own cell.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Want to give a name a new value, as in the last lesson? Do that in the cell that makes it. A second cell with `price = price * 2` makes `price` again, and then marimo does not run it."),
        mo.md(r"""
        ### Recipe

        To use names across cells:

        1. Make each name in one cell only. Other cells may use it; they do not make it again.
        2. Does only this cell need a name, for a step in between? Start it with `_`.
        3. Does marimo say *This cell redefines variables from other cells*? Find the other cell that makes the same name. Did you mean the name that cell makes? Then remove the line in your cell. Otherwise, use another name in your cell, or put `_` in front of it if only your cell needs it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A room costs 750 a month. What does it cost in a year? The comments with `#` say which step of the recipe each part is.
    """)
    return


@app.cell
def example_rent():
    # 1. this cell makes the name monthly_rent; other cells may use it
    monthly_rent = 750
    return (monthly_rent,)


@app.cell
def example_year(monthly_rent):
    # 2. only this cell needs these names, so they start with _
    _months = 12
    # monthly_rent comes from the cell above; this cell does not make it again
    _year_cost = monthly_rent * _months
    print(_year_cost)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        The second cell shows 9000. `monthly_rent` comes from the first cell, and `_months` and `_year_cost` stay inside the second one, so another cell could have its own `_months`.

        3. Suppose the second cell also had the line `monthly_rent = 800`. Then two cells make `monthly_rent`, and you get the message from step 3. The fix: leave out that line, or give the new rent another name, such as `new_rent`.
        """),
        L.task("Change `750` to `800` in the first cell, and run it with **Shift + Enter**. Watch the second cell: it runs again by itself, because it uses `monthly_rent`."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_trip():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A train trip

        Four friends take the train to the beach and back. A one-way ticket costs 12.5. The cell below makes `ticket_price`. In the box under it, make `trip_cost`: what the four friends pay in total, there and back, and show it. Use `ticket_price` in your calculation; do not make it again. The check runs your box with the `ticket_price` of the cell above, just as a cell would.

        When the check says *Correct*, change `12.5` to `15` in that cell and run it: your box follows the new price.
        """),
        L.task("Make `trip_cost` in the box, with `ticket_price` from the cell above."),
    ])
    return


@app.cell
def data_trip():
    ticket_price = 12.5
    return (ticket_price,)


@app.cell(hide_code=True)
def exercise_trip():
    box_trip = L.code_box("trip_cost", lines=4)
    box_trip
    return (box_trip,)


@app.cell(hide_code=True)
def check_trip(box_trip, ticket_price):
    try:
        _inputs = {"ticket_price": ticket_price}   # the learner may change it, so the check uses the current price
    except NameError:                              # the cell with ticket_price is gone: check with the lesson's own price
        _inputs = None
    result_trip = L.check("trip_cost", box_trip.value, inputs=_inputs)
    result_trip
    return (result_trip,)


@app.cell(hide_code=True)
def hints_trip():
    L.hints(
        "trip_cost",
        [
            "Step 1 of the recipe: `ticket_price` is already made in the cell above. Use it in the box; do not make it again.",
            "Each friend needs two tickets: one there and one back. And there are four friends.",
            "In the worked example, the second cell had `_year_cost = monthly_rent * _months`. Here: `trip_cost = ticket_price * ___ * ___`, and `print(trip_cost)` under it.",
        ],
        why="`trip_cost = ticket_price * 4 * 2` gives 100, and with another price it follows by itself. With names for the steps in between it works just as well: `_friends = 4`, `_tickets = 2`, and then `trip_cost = ticket_price * _friends * _tickets`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Which cells can share a notebook?

        Six pairs of cells; each pair is in a notebook of its own. What happens with each pair? Look at the picture and at the new words.
        """),
        L.task("Put each pair of cells in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    sort_pairs = L.sort("cell-pairs")
    sort_pairs
    return (sort_pairs,)


@app.cell(hide_code=True)
def check_pairs(sort_pairs):
    result_pairs = L.check("cell-pairs", sort_pairs)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "cell-pairs",
        [
            "For each pair: which names does each cell make with `=`, and which names does it only use?",
            "A name without `_` may be made in one cell only. A name with `_` stays inside the cell that makes it.",
            "`price = price * 2` in cell 2 makes `price` again, just as cell 1 does.",
        ],
        why="**Both cells run:** `price` and `cost` (cell 2 uses `price` and makes `cost`), and the two `_total` (each cell has its own). **marimo stops:** the two `total`, and the two `price` (cell 2 makes `price` again). **Cell 2 does not know the name:** `_n` and `_rate` stay inside cell 1.",
    )
    return


@app.cell(hide_code=True)
def text_fix():
    mo.md(r"""
    ### Exercise 3. The message
    """)
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    choice_fix = L.choice(
        "fix",
        question="In a notebook, your cell has `total = 10`. Under your cell, marimo says: *This cell redefines variables from other cells.* What do you do?",
        options={
            "move": "Move your cell to the top of the notebook.",
            "again": "Run your cell once more.",
            "rename": "Use another name in your cell, such as `my_total`, or `_total` if only your cell needs it.",
        },
        correct="rename",
        why={
            "move": "Where a cell is on the page does not matter: two cells still make `total`. Look at step 3 of the recipe.",
            "again": "Running it again changes nothing: two cells still make `total`. Look at step 3 of the recipe.",
            "rename": "Then each name comes from one cell again, and both cells run.",
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Variables and numbers](/course/open/py-variables?lang=en), every program was in one box; now names go from cell to cell. From here on, the worked examples use `_` names for their steps in between, so you can copy their form without clashes. In the next lesson you do more with text.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, result_pairs, result_trip):
    L.wrap_up(result_trip, result_pairs, choice_fix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
