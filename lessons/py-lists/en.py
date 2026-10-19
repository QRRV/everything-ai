# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-lists. Route: course/route/lessen/stam-1-basis.yaml. Checks and the list in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-lists", "en")
    C = L.checks   # the list in the picture, and the slice it marks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Text: strings](/course/open/py-strings?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_last():
    quick_last = L.choice(
        "quick-last",
        question="What does this program show?\n\n```python\nword = \"python\"\nprint(word[-1])\n```",
        options={
            "p": "`p`",
            "o": "`o`",
            "n": "`n`",
        },
        correct="n",
        why={
            "p": "`p` is at index 0, the first character. A negative index counts from the end. See the picture in [Text: strings](/course/open/py-strings?lang=en).",
            "o": "That is the character before the last one, `word[-2]`. See the picture in [Text: strings](/course/open/py-strings?lang=en).",
            "n": "`-1` counts from the end: the last character, `n`.",
        },
    )
    quick_last
    return (quick_last,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="What does this program show?\n\n```python\nname = \"Ada\"\nprint(f\"Hi {name}\")\n```",
        options={
            "filled": "`Hi Ada`",
            "braces": "`Hi {name}`",
            "word": "`Hi name`",
        },
        correct="filled",
        why={
            "filled": "In an f-string, Python fills in the value of each name between braces.",
            "braces": "The braces stay only when the `f` before the quote is missing. See the recipe in [Text: strings](/course/open/py-strings?lang=en).",
            "word": "Between braces, Python fills in the value of the name, not the name itself. See [Text: strings](/course/open/py-strings?lang=en).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    AI works with lots of data: the temperatures of a whole year, all the words of a book, a thousand scores. You cannot give each value a name of its own. A list keeps many values together, in order, under one name. From the next lessons on, you go through lists value by value, and later in the course a program learns from long lists of examples.

    **An everyday example.** A playlist. The songs are in order: song 1, song 2, song 3. You can play one song by its number, play songs 2 to 4, or add a new song at the end. A list in Python works much like that. One difference: a playlist counts from 1, and Python from 0.
    """)
    return


@app.cell(hide_code=True)
def predict_slice():
    guess_slice = L.guess(
        "slice",
        question="What do you think this program shows?\n\n```python\nscores = [7, 9, 4, 8, 6]\nprint(scores[1:3])\n```",
        options={
            "three": "`[9, 4, 8]`",
            "two": "`[9, 4]`",
            "first": "`[7, 9, 4]`",
        },
        reveal="**`[9, 4]`.** `scores[1:3]` starts at index 1, which is `9`, and stops just before index 3. So you get the items at index 1 and 2. You see it in the picture below.",
    )
    guess_slice
    return (guess_slice,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A list keeps values in order. Places work just as in a string: counted from 0, and with `-1` for the last one.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **list** | values in order, between square brackets, with a comma between them | `scores = [7, 9, 4, 8, 6]` |
    | **item** | one value in a list; you take it by its index | `scores[0]` gives `7` |
    | **slice** | a piece of a list, from one index up to, not including, another | `scores[1:3]` gives `[9, 4]` |
    | **append** | adds one item at the end of a list; it is written after the list's name, with a dot | `scores.append(5)` |

    A slice may leave out a number: `scores[:2]` starts at the beginning, and `scores[2:]` goes on to the end. A negative number counts from the end, as with a string: `scores[-2:]` gives the last two. `len(scores)` gives the number of items, as with a string. And when Python shows a list of strings, it puts single quotes around each one, even if you typed double ones: `['bread', 'milk']`.
    """)
    return


@app.cell(hide_code=True)
def picture_list():
    mo.vstack([
        draw.list_boxes([str(_x) for _x in C.PICTURE], index_label="index", back_label="from the end", span=C.PICTURE_SPAN,
                        span_label="scores[1:3]", title="A list, and a slice of it"),
        mo.md(r"""
        Above each item is its index, and below it the index from the end. The slice `scores[1:3]` takes the items at index 1 and 2; index 3 is where it stops, and that item is not part of it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A slice stops just before its second number: `scores[1:3]` gives 2 items, not 3. That way, `scores[:2]` gives exactly the first 2. And `append` changes the list itself and gives nothing back, so write `scores.append(5)` on a line of its own, not `scores = scores.append(5)`."),
        mo.md(r"""
        ### Recipe

        To work with a list:

        1. Put the values in a list: `scores = [7, 9, 4, 8, 6]`.
        2. Take one item by its index: `scores[0]` for the first, `scores[-1]` for the last.
        3. Take a piece with a slice: `scores[start:stop]`; the item at `stop` is not included. A negative number counts from the end.
        4. Add an item at the end with `append`: `scores.append(5)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Hours of sleep from Monday to Friday. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_sleep():
    # 1. the values in a list, Monday to Friday
    _sleep = [7, 6.5, 8, 7.5, 6]
    # 2. one item: the first and the last day
    print(f"Monday {_sleep[0]}, Friday {_sleep[-1]}")
    # 3. a piece: Tuesday to Thursday, from index 1 up to, not including, 4
    _middle = _sleep[1:4]
    print(_middle)
    # and the last two days, counted from the end
    print(_sleep[-2:])
    # 4. one more item at the end: Saturday
    _sleep.append(9)
    print(f"{len(_sleep)} days: {_sleep}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Under the cell appear four lines. The slice `_sleep[1:4]` gives the three days in the middle, `[6.5, 8, 7.5]`, and `_sleep[-2:]` the last two, `[7.5, 6]`. After `append`, the list has 6 items, and the new one is at the end.
        """),
        L.task("Change `_sleep[1:4]` into `_sleep[:2]` in the cell above, and run it. Which days do you get now?"),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_top():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The first three

        Make `top`: the first three scores of `scores`, as a list. Replace the `...` with a slice. The check also tries another list.
        """),
        L.task("Make `top` with a slice, so the program shows `[7, 9, 4]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top():
    box_top = L.code_box("top", lines=4)
    box_top
    return (box_top,)


@app.cell(hide_code=True)
def check_top(box_top):
    result_top = L.check("top", box_top.value)
    result_top
    return (result_top,)


@app.cell(hide_code=True)
def hints_top():
    L.hints(
        "top",
        [
            "Step 3 of the recipe: a slice is `scores[start:stop]`. At which index is the first score?",
            "The slice starts at 0. It stops just before its second number: which number gives exactly three items?",
            "`top = scores[0:___]`",
        ],
        why="`scores[:3]` (or `scores[0:3]`) starts at index 0 and stops just before 3: the items at 0, 1 and 2, three in all.",
    )
    return


@app.cell(hide_code=True)
def text_last_two():
    mo.md(r"""
    ### Exercise 2. From the end
    """)
    return


@app.cell(hide_code=True)
def choice_last_two_cell():
    choice_last_two = L.choice(
        "last-two",
        question="What does this program show?\n\n```python\nnums = [5, 6, 7, 8]\nprint(nums[-2:])\n```",
        options={
            "three": "`[6, 7, 8]`",
            "two": "`[7, 8]`",
            "one": "`[8]`",
        },
        correct="two",
        why={
            "three": "`-1` is the last item. Which one is `-2`? Look at the row from the end in the picture again.",
            "two": "`-2` is `7`, the second item from the end, and with nothing after the colon the slice goes on to the end.",
            "one": "With nothing after the colon, a slice goes on to the end. Look at the line under the table of new words again.",
        },
    )
    choice_last_two
    return (choice_last_two,)


@app.cell(hide_code=True)
def text_exercise_cart():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The shopping cart

        Replace the `...` with one line that adds `"apples"` at the end of `cart`. The last line shows the cart. The check also tries another cart.
        """),
        L.task("Add `\"apples\"` at the end of `cart`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cart():
    box_cart = L.code_box("cart", lines=4)
    box_cart
    return (box_cart,)


@app.cell(hide_code=True)
def check_cart(box_cart):
    result_cart = L.check("cart", box_cart.value)
    result_cart
    return (result_cart,)


@app.cell(hide_code=True)
def hints_cart():
    L.hints(
        "cart",
        [
            "Step 4 of the recipe: which instruction adds an item at the end of a list?",
            "`append` is written after the list's name, with a dot, and the new item goes between its parentheses.",
            "`cart.append(___)`, with the text between quotes.",
        ],
        why="`cart.append(\"apples\")` adds the string `\"apples\"` at the end of the list. `append` changes `cart` itself, so the line has no `cart =` in front.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Text: strings](/course/open/py-strings?lang=en) you took characters from a string by their index; a list works the same way, but its items can be anything: numbers, strings, even other lists. In the next lesson you go through a list item by item, with a loop. Later, NumPy keeps numbers in a row much like this, and calculates with them much faster.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_last_two, result_cart, result_top):
    L.wrap_up(result_top, choice_last_two, result_cart)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
