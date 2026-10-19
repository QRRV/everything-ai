# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-strings. Route: course/route/lessen/stam-1-basis.yaml. Checks and the word in the picture: checks.py.
# termen:allow variables, program

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-strings", "en")
    C = L.checks   # the word in the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Variables and numbers](/course/open/py-variables?lang=en) and [Names in a notebook](/course/open/py-names-in-cells?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_calc():
    quick_calc = L.choice(
        "quick-calc",
        question="What does this program show?\n\n```python\nwidth = 4\nlength = width * 3\nprint(length - 2)\n```",
        options={
            "ten": "`10`",
            "twelve": "`12`",
            "word": "`length - 2`",
        },
        correct="ten",
        why={
            "ten": "`length` gets 4 times 3, which is 12, and `print` shows 12 minus 2.",
            "twelve": "`print` shows `length - 2`, not `length`. See the recipe in [Variables and numbers](/course/open/py-variables?lang=en).",
            "word": "Without quotes, Python calculates what is between the parentheses. See [Variables and numbers](/course/open/py-variables?lang=en).",
        },
    )
    quick_calc
    return (quick_calc,)


@app.cell(hide_code=True)
def quick_check_cells():
    quick_cells = L.choice(
        "quick-cells",
        question="Two cells in the same notebook both need a name for a step in between. Which name can they both make?",
        options={
            "plain": "`step`",
            "underscore": "`_step`",
            "capital": "`Step`",
        },
        correct="underscore",
        why={
            "plain": "A name without `_` may be made in one cell only. See [Names in a notebook](/course/open/py-names-in-cells?lang=en).",
            "capital": "`Step` is just another name without `_`, so it may be made in one cell only. See [Names in a notebook](/course/open/py-names-in-cells?lang=en).",
            "underscore": "A name that starts with `_` stays inside its own cell, so each cell can have its own `_step`.",
        },
    )
    quick_cells
    return (quick_cells,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A lot of AI works with text: an AI assistant reads your question and writes an answer. Often, a program has to take text apart: how long is it, which letter comes first, which comes last? And it has to put text together, such as an answer with a number in it. In this lesson you do both, with strings.

    **An everyday example.** A crossword puzzle. You count the squares to know how long the word is, and a crossing word may tell you that the third letter is an *n*: one place in the word. Python does the same with a string: it counts the characters, and finds one character by its place. One difference: the puzzle counts its squares from 1, and Python counts from 0.
    """)
    return


@app.cell(hide_code=True)
def predict_place():
    guess_place = L.guess(
        "second-letter",
        question="What do you think this program shows?\n\n```python\nword = \"banana\"\nprint(word[1])\n```",
        options={
            "b": "`b`",
            "a": "`a`",
            "error": "An error: there is no place 1.",
        },
        reveal="**`a`.** Python counts places from 0. `word[0]` is the first character, `b`, so `word[1]` is the second one, `a`. You see it in the picture below.",
    )
    guess_place
    return (guess_place,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A string is a row of characters. Each character has a place, and Python counts the places from 0.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **str** | Python's name for a string: text between quotes | `"banana"` |
    | **character** | one letter, digit, space or other symbol in a string, such as `!` | `"b"`, `" "` |
    | **len** | gives the number of characters in a string, its length | `len("banana")` gives `6` |
    | **index** | the place of a character, counted from 0; you write it between square brackets after the string or its name | `word[0]` is the first character |
    | **f-string** | a string with an `f` before the quote; Python fills in the value of each name between braces `{ }` | `f"Hello {name}"` |

    A negative index counts from the end: `word[-1]` is the last character, `word[-2]` the one before it. Between the braces of an f-string you may also write a calculation or an index, such as `{len(word)}` or `{word[0]}`.
    """)
    return


@app.cell(hide_code=True)
def picture_string():
    mo.vstack([
        draw.string_boxes(C.WORD, index_label="index", back_label="from the end", title="The places in a string"),
        mo.md(r"""
        Above each character is its index, counted from 0; below it, the index counted from the end. `banana` has 6 characters, so its last index is 5, one less than its length.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("The first character is `word[0]`, not `word[1]`. And a string of 6 characters has no place 6: with `word[6]`, Python stops with an error. For the last character, write `word[-1]`."),
        mo.md(r"""
        ### Recipe

        To take text apart and put it together:

        1. Put the text in a name: `word = "banana"`.
        2. Count its characters with `len(word)`.
        3. Pick one character by its index: `word[0]` for the first, `word[-1]` for the last.
        4. Build new text with an f-string: an `f` before the quote, and the names between braces, as in `f"{word} has {len(word)} characters"`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A sentence about the city Utrecht: how many characters its name has, and its first and last character. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_city():
    # 1. the text in a name
    _city = "Utrecht"
    # 2. the number of characters
    _length = len(_city)
    # 3. the first and the last character
    _first = _city[0]
    _last = _city[-1]
    # 4. new text with an f-string
    _sentence = f"{_city} has {_length} characters, from {_first} to {_last}."
    print(_sentence)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Under the cell appears `Utrecht has 7 characters, from U to t.` Python filled in the value of each name between braces. The capital `U` stays a capital: a string keeps every character exactly as it is.
        """),
        L.task("Change `\"Utrecht\"` into the name of your own town or city, and run the cell again."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_initials():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Initials

        Make the initials of `first_name` and `last_name` in the box: the first character of each, right next to each other. Replace the `...`. The check also tries other names, so take the letters from the names. Grace Hopper (1906–1992) was a computer pioneer: she built one of the first programs that turn code that people can read into the instructions a computer runs.
        """),
        L.task("Make `initials` from the first character of each name, so the program shows `GH`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_initials():
    box_initials = L.code_box("initials", lines=5)
    box_initials
    return (box_initials,)


@app.cell(hide_code=True)
def check_initials(box_initials):
    result_initials = L.check("initials", box_initials.value)
    result_initials
    return (result_initials,)


@app.cell(hide_code=True)
def hints_initials():
    L.hints(
        "initials",
        [
            "Step 3 of the recipe: which index has the first character of a string?",
            "`first_name[0]` is the first character of `first_name`. You need the same for `last_name`.",
            "Put the two characters together in an f-string, each between braces, with nothing in between: `f\"{___}{___}\"`.",
        ],
        why="`first_name[0]` is `G` and `last_name[0]` is `H`; the f-string `f\"{first_name[0]}{last_name[0]}\"` puts them right next to each other.",
    )
    return


@app.cell(hide_code=True)
def text_spaces():
    mo.md(r"""
    ### Exercise 2. Counting characters
    """)
    return


@app.cell(hide_code=True)
def choice_spaces_cell():
    choice_spaces = L.choice(
        "spaces",
        question="What does `print(len(\"hi there\"))` show?",
        options={
            "seven": "`7`",
            "two": "`2`",
            "eight": "`8`",
        },
        correct="eight",
        why={
            "seven": "`len` counts every character, not just the letters. Look at the word *character* in the table again.",
            "two": "`len` counts characters, not words. Look at the word *len* in the table again.",
            "eight": "`len` counts every character: 7 letters and the space between them.",
        },
    )
    choice_spaces
    return (choice_spaces,)


@app.cell(hide_code=True)
def text_exercise_caption():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A caption

        Make a caption for a map in `caption`: the city in `city_name`, a space, and between parentheses its number of characters. For `Groningen`, that is `Groningen (9)`. Let `len` do the counting, inside an f-string: the check also tries another city.
        """),
        L.task("Make `caption` with an f-string, so the program shows `Groningen (9)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_caption():
    box_caption = L.code_box("caption", lines=4)
    box_caption
    return (box_caption,)


@app.cell(hide_code=True)
def check_caption(box_caption):
    result_caption = L.check("caption", box_caption.value)
    result_caption
    return (result_caption,)


@app.cell(hide_code=True)
def hints_caption():
    L.hints(
        "caption",
        [
            "Step 4 of the recipe: an `f` before the quote, and between braces what Python should fill in.",
            "The caption has two parts that Python fills in: the name of the city, and its length. The space and the parentheses are plain text in the f-string.",
            "`f\"{city_name} (___)\"`: what goes in the parentheses, between braces?",
        ],
        why="`f\"{city_name} ({len(city_name)})\"` fills in `Groningen` and the 9 that `len` counts. The space and the parentheses stay as they are.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Your first program](/course/open/py-first-program?lang=en) a string was only text to show; now you measure it, pick characters from it and build new text with names in it. Places counted from 0 come back in later lessons, and later in the course a program splits text into pieces and counts them before it can learn from it. In the next lesson your program makes a choice, with `if`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spaces, result_caption, result_initials):
    L.wrap_up(result_initials, choice_spaces, result_caption)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
