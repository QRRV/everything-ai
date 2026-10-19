# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-tuples-sets. Route: course/route/lessen/stam-1-basis.yaml. Checks and the rows in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-tuples-sets", "en")
    C = L.checks   # the numbers in the picture, kept in four ways


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Lists](/course/open/py-lists?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question="What does this program show?\n\n```python\nletters = [\"a\", \"b\", \"c\", \"d\"]\nprint(letters[:2])\n```",
        options={
            "three": "`['a', 'b', 'c']`",
            "end": "`['c', 'd']`",
            "two": "`['a', 'b']`",
        },
        correct="two",
        why={
            "three": "A slice stops just before its second number, so index 2 is not part of it. See the watch-out in [Lists](/course/open/py-lists?lang=en).",
            "end": "With nothing before the colon, a slice starts at the beginning. See the idea in [Lists](/course/open/py-lists?lang=en).",
            "two": "`[:2]` starts at the beginning and stops just before index 2: the first two items.",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def quick_check_append():
    quick_append = L.choice(
        "quick-append",
        question="What does this program show?\n\n```python\nnums = [1, 2]\nnums.append(3)\nprint(len(nums))\n```",
        options={
            "three": "`3`",
            "two": "`2`",
            "list": "`[1, 2, 3]`",
        },
        correct="three",
        why={
            "three": "`append` adds 3 at the end, so the list has three items, and `len` gives that number.",
            "two": "`append` changes the list itself, before `len` counts. See the recipe in [Lists](/course/open/py-lists?lang=en).",
            "list": "`len` does not show the list: it gives the number of items. See the idea in [Lists](/course/open/py-lists?lang=en).",
        },
    )
    quick_append
    return (quick_append,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A list keeps any values in order, and you can change it. Sometimes you need something else: values that belong together and must stay as they are, each value just once, or all values in order. Python has a way for each. Later in the course, the size of a table of numbers is kept as a tuple, the different words of a text as a set, and sorting finds the best answers.

    **An everyday example.** A date: day, month and year belong together, and you do not change one of them later. A guest list: each name is on it once, however often someone says they are coming. And a results list after a race: the times in order, fastest first.
    """)
    return


@app.cell(hide_code=True)
def predict_set():
    guess_set = L.guess(
        "set",
        question="What do you think this program shows?\n\n```python\ncolors = [\"red\", \"blue\", \"red\", \"red\"]\nprint(len(set(colors)))\n```",
        options={
            "four": "`4`",
            "two": "`2`",
            "one": "`1`",
        },
        reveal="**`2`.** `set(colors)` keeps each value once: `\"red\"` and `\"blue\"`. `len` counts what is in the set, so 2.",
    )
    guess_set
    return (guess_set,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A tuple is like a list that cannot change, a set keeps each value once, and `sorted` puts values in order.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **tuple** | values in order, between parentheses; once made, it cannot change | `point = (3, 4)` |
    | **set** | values that are each there once, in no fixed order; `set(...)` makes one from a list | `set(["a", "b", "a"])` keeps `"a"` and `"b"` |
    | **sorted** | gives a new list with the values in order, from small to large | `sorted([3, 1, 2])` gives `[1, 2, 3]` |

    A tuple works like a list for reading: `point[0]`, a slice and `len(point)` all work. But `point[0] = 5` stops the program with an error, and a tuple has no `append`. `x, y = point` gives each value of the tuple its own name, in order. A set has no places, so it has no index, and Python shows its values between curly braces, in any order. With small whole numbers it often happens to show them from small to large; with words, the order can differ each time. `sorted` works on a list, a tuple and a set, and always gives a list. Words are sorted from A to Z, and every word with a capital comes before every word with a small letter: `"Zoe"` before `"ada"`.
    """)
    return


@app.cell(hide_code=True)
def picture_rows():
    mo.vstack([
        draw.value_rows(C.picture_rows({"list": "in order, can change", "tuple": "in order, cannot change",
                                        "set": "each value once, no order", "sorted": "a new list, in order"}),
                        title="The same numbers, kept in four ways"),
        mo.md(r"""
        A list and a tuple keep the order, and both 3s; only the list can change. The set keeps each value once and has no order: that is why its values are in a bag, without places. `sorted` gives a new list, from small to large.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`sorted(times)` gives a new list and leaves `times` as it was. There is also `times.sort()`: it changes the list itself and gives nothing back, just like `append`. So never write `times = times.sort()`: then `times` becomes `None`, Python's word for nothing."),
        mo.md(r"""
        ### Recipe

        To pick the right way to keep values:

        1. Values that belong together and stay as they are: a tuple, `point = (3, 4)`. Take a value with `point[0]`, or give each its own name: `x, y = point`.
        2. Each value only once: make a set from the list, `set(words)`. `len(set(words))` gives the number of different values.
        3. All values in order: `sorted(...)` gives a new list, from small to large.
        4. The smallest or largest few: sort first, then take a slice, such as `sorted(scores)[:3]` or `sorted(scores)[-3:]`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A screen, the visitors of a website, and the ages in a group. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_tuples_sets():
    # 1. a tuple: the width and height of a screen belong together
    _screen = (1920, 1080)
    print(_screen)
    print(_screen[0])
    # or give each value its own name
    _width, _height = _screen
    print(f"{_width} by {_height}")
    # 2. a set: each visitor once
    _visitors = ["ana", "ben", "ana", "cas", "ben"]
    print(len(set(_visitors)))
    # 3. sorted: a new list, in order, here of the set
    _names = sorted(set(_visitors))
    print(_names)
    _ages = [34, 19, 52, 27]
    print(sorted(_ages))
    print(_ages)
    # 4. sort, then a slice: the two youngest, and the two oldest
    _youngest = sorted(_ages)[:2]
    print(_youngest)
    _oldest = sorted(_ages)[-2:]
    print(_oldest)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear nine lines. Python shows a tuple between parentheses: `(1920, 1080)`. Five visits, but only three different visitors: in the set, `ana` and `ben` are there once. `sorted(set(_visitors))` gives them as a list, from A to Z. `sorted(_ages)` gives a new list in order, and the line after it shows that `_ages` itself has not changed. The last two lines take the first two and the last two of the sorted list: the two youngest and the two oldest.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unique():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Each word once

        Make `unique`: every word of `words` once, as a list from A to Z. Replace the `...`. The check also tries another list.
        """),
        L.task("Make `unique`, so the program shows `['fog', 'rain', 'sun']`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unique():
    box_unique = L.code_box("unique", lines=4)
    box_unique
    return (box_unique,)


@app.cell(hide_code=True)
def check_unique(box_unique):
    result_unique = L.check("unique", box_unique.value)
    result_unique
    return (result_unique,)


@app.cell(hide_code=True)
def hints_unique():
    L.hints(
        "unique",
        [
            "Two steps: first each word only once, then in order. Which steps of the recipe are those?",
            "A set keeps each word once, but has no order. Which instruction from the recipe turns it into a list, in order?",
            "`unique = ___(set(words))`",
        ],
        why="`set(words)` keeps `rain`, `sun` and `fog` once each. `sorted(...)` turns that set into a list from A to Z: `['fog', 'rain', 'sun']`.",
    )
    return


@app.cell(hide_code=True)
def text_tuple_change():
    mo.md(r"""
    ### Exercise 2. Changing a tuple
    """)
    return


@app.cell(hide_code=True)
def choice_tuple_change_cell():
    choice_tuple_change = L.choice(
        "tuple-change",
        question="What happens when this program runs?\n\n```python\npoint = (3, 4)\npoint[0] = 5\nprint(point)\n```",
        options={
            "five": "It shows `(5, 4)`",
            "same": "It shows `(3, 4)`",
            "error": "Python stops with an error on the second line",
        },
        correct="error",
        why={
            "error": "A tuple cannot change once it is made. So Python stops at `point[0] = 5`, and the `print` never runs.",
            "five": "That is what a list would do. Look at the word *tuple* in the table again.",
            "same": "Python never quietly skips a line. Look at the word *tuple* in the table again: what happens when you try to change one?",
        },
    )
    choice_tuple_change
    return (choice_tuple_change,)


@app.cell(hide_code=True)
def text_exercise_podium():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The podium

        `times` holds the times of five runners, in seconds. The lower the time, the faster. Make `podium`: the three fastest times, fastest first. Replace the `...`. The check also tries another race.
        """),
        L.task("Make `podium`, so the program shows `[11.8, 11.9, 12.0]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_podium():
    box_podium = L.code_box("podium", lines=4)
    box_podium
    return (box_podium,)


@app.cell(hide_code=True)
def check_podium(box_podium):
    result_podium = L.check("podium", box_podium.value)
    result_podium
    return (result_podium,)


@app.cell(hide_code=True)
def hints_podium():
    L.hints(
        "podium",
        [
            "Step 4 of the recipe: sort first, then take a slice.",
            "`sorted(times)` puts the lowest time first. Which slice takes the first three?",
            "`podium = sorted(times)[___]`",
        ],
        why="`sorted(times)` gives `[11.8, 11.9, 12.0, 12.4, 13.1]`, and `[:3]` takes the first three of that list: the three fastest.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Lists](/course/open/py-lists?lang=en) you kept values in order and changed them. A tuple keeps them in order and keeps them fixed, a set keeps each value once, and `sorted` puts them in order. Later in the course, the size of a table of numbers is a tuple, such as `(3, 4)` for 3 rows and 4 columns. In the next lesson you build a new list from an old one, in a single line.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_tuple_change, result_podium, result_unique):
    L.wrap_up(result_unique, choice_tuple_change, result_podium)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
