# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-comprehensions. Route: course/route/lessen/stam-1-basis.yaml. Checks and the parts in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-comprehensions", "en")
    C = L.checks   # the loop and the line in the picture, part by part


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Dictionaries](/course/open/py-dicts?lang=en) and [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="What does this program show?\n\n```python\nages = {\"Ada\": 36}\nages[\"Bo\"] = 12\nprint(len(ages))\n```",
        options={
            "two": "`2`",
            "one": "`1`",
            "error": "Nothing: Python stops with an error, because `\"Bo\"` is not a key yet",
        },
        correct="two",
        why={
            "two": "`ages[\"Bo\"] = 12` adds a new pair, so the dictionary has two pairs.",
            "one": "The second line adds a pair; it does not replace `\"Ada\"`. See the recipe in [Dictionaries](/course/open/py-dicts?lang=en).",
            "error": "Looking up a missing key stops the program, but giving a new key a value with `=` adds a pair. See the recipe in [Dictionaries](/course/open/py-dicts?lang=en).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def quick_check_sorted():
    quick_sorted = L.choice(
        "quick-sorted",
        question="What does this program show?\n\n```python\nprint(sorted(set([3, 1, 3])))\n```",
        options={
            "twice": "`[1, 3, 3]`",
            "set": "`{1, 3}`",
            "once": "`[1, 3]`",
        },
        correct="once",
        why={
            "twice": "The set comes first, and it keeps each value once. See the recipe in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "set": "`sorted` always gives a list, also when you give it a set. See the idea in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "once": "The set keeps 1 and 3 once each, and `sorted` turns it into a list, in order.",
        },
    )
    quick_sorted
    return (quick_sorted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You often make a new list from an old one: each price with tax added, or only the warm days. With a `for` loop that takes a few lines: an empty list, the loop, and `append`. A comprehension says the same thing in one line that reads almost like a sentence. In code about data and AI you see them everywhere.

    **An everyday example.** A teacher says: "Write down the name of everyone who scored 8 or more." That one sentence describes the whole new list: what goes on it (a name), who to look at (everyone), and who counts (8 or more). A comprehension has those same three parts.
    """)
    return


@app.cell(hide_code=True)
def predict_double():
    guess_double = L.guess(
        "double",
        question="What do you think this program shows?\n\n```python\nprint([n * 2 for n in [1, 2, 3]])\n```",
        options={
            "doubled": "`[2, 4, 6]`",
            "twice": "`[1, 2, 3, 1, 2, 3]`",
            "total": "`12`",
        },
        reveal="**`[2, 4, 6]`.** For each `n` in the list, the comprehension works out `n * 2` and puts it in a new list. The square brackets around it make that new list.",
    )
    guess_double
    return (guess_double,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A comprehension builds a new list in one line: for each item of an old list, what goes in the new one, and, if you want, a condition for which items count.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **comprehension** | a new list in one line: in square brackets, what goes in for one item, then `for`, a name and the old list | `[n * 2 for n in nums]` |
    | **if** at the end | keeps only the items for which the condition is `True` | `[t for t in temps if t > 20]` |

    The name after `for` is the item of this round, just as in a `for` loop, and you use it in front: `[word[0] for word in words]` gives the first letter of each word. The old list stays as it is. With curly braces, and a key and a value in front, a comprehension builds a dictionary: `{word: word[0] for word in words}`.
    """)
    return


@app.cell(hide_code=True)
def picture_parts():
    mo.vstack([
        draw.comprehension_parts(C.PICTURE_LOOP, C.PICTURE_LINE,
                                 {"1": "what goes in the new list, for one item",
                                  "2": "for each item of the old list",
                                  "3": "only when the condition is True"},
                                 loop_label="With a loop", line_label="The same list, in one line",
                                 title="A loop, and the same list as a comprehension"),
        mo.md(r"""
        Both make the same list: for each temperature above 20, how many degrees above 20 it is. The numbers show where each part of the loop goes. What goes in the list, the last line of the loop, comes first; then the `for`, and then the `if`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("In a comprehension, what goes in the list comes first, before `for`, and the `if` comes last; there is no colon at the end, as there is after a `for` line. And keep it short: if a comprehension gets hard to read, a normal `for` loop is better. A comprehension is for building a list or a dictionary, not for showing things with `print`."),
        mo.md(r"""
        ### Recipe

        To build a new list from an old one, in one line:

        1. Open square brackets: they make the new list.
        2. Write what goes in the new list, for one item: `t - 20`.
        3. Then `for`, a name for one item, `in`, and the old list: `for t in temps`.
        4. Only some items? Add `if` and a condition at the end: `if t > 20`.
        5. A dictionary instead? Use curly braces, with a key and a value in front: `{w: len(w) for w in words}`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Temperatures in degrees Celsius, and a few words. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_comprehensions():
    _celsius = [0, 10, 25, 30, -5]
    # 1. to 3. square brackets, what goes in for one item, and for: each temperature in Fahrenheit
    _fahrenheit = [_c * 9 / 5 + 32 for _c in _celsius]
    print(_fahrenheit)
    # 4. an if at the end: only the days above 20
    _warm = [_c for _c in _celsius if _c > 20]
    print(_warm)
    # 4. also over a dictionary: the days without rain
    _rain = {"mon": 3, "tue": 0, "wed": 0, "thu": 5}
    _dry = [_d for _d in _rain if _rain[_d] == 0]
    print(_dry)
    # 5. curly braces, a key and a value: each word with its length
    _words = ["sun", "cloud", "rain"]
    _lengths = {_w: len(_w) for _w in _words}
    print(_lengths)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear four lines. The first list has one number for every temperature: 0 degrees Celsius is 32.0 in Fahrenheit. The second keeps only 25 and 30, because the `if` is `True` only for them. The third goes through the keys of `_rain` and keeps the days whose value is 0. The last line is a dictionary, with each word as a key and its length as the value.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lengths():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Word lengths

        Make `lengths`: the length of each word in `words`, as a list. Replace the `...` with a comprehension. The check also tries another list.
        """),
        L.task("Make `lengths` in one line, so the program shows `[5, 2, 4]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lengths():
    box_lengths = L.code_box("lengths", lines=4)
    box_lengths
    return (box_lengths,)


@app.cell(hide_code=True)
def check_lengths(box_lengths):
    result_lengths = L.check("lengths", box_lengths.value)
    result_lengths
    return (result_lengths,)


@app.cell(hide_code=True)
def hints_lengths():
    L.hints(
        "lengths",
        [
            "Look at the first comprehension of the worked example: what goes in the new list is in front of `for`.",
            "What goes in the new list is the length of one word. Which instruction gives the length of a string?",
            "`lengths = [___ for word in words]`",
        ],
        why="`[len(word) for word in words]` goes through `words`; for each word it works out `len(word)` and puts that in the new list: 5, 2 and 4.",
    )
    return


@app.cell(hide_code=True)
def text_dict_comp():
    mo.md(r"""
    ### Exercise 2. A dictionary in one line
    """)
    return


@app.cell(hide_code=True)
def choice_dict_comp_cell():
    choice_dict_comp = L.choice(
        "dict-comp",
        question="What does this program show?\n\n```python\nwords = [\"hi\", \"hello\"]\nprint({w: len(w) for w in words})\n```",
        options={
            "list": "`[2, 5]`",
            "dict": "`{'hi': 2, 'hello': 5}`",
            "set": "`{2, 5}`",
        },
        correct="dict",
        why={
            "list": "Square brackets would make a list, but these are curly braces with a colon. Look at step 5 of the recipe again.",
            "dict": "Curly braces with `key: value` in front make a dictionary: each word as a key, and its length as the value.",
            "set": "Look at what is in front of `for`: there is a colon in it. Look at step 5 of the recipe again.",
        },
    )
    choice_dict_comp
    return (choice_dict_comp,)


@app.cell(hide_code=True)
def text_exercise_passed():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Who passed?

        `scores` gives each student's score. A score of 5.5 or more is a pass. Make `passed`: the names of everyone who passed, as a list. Replace the `...` with a comprehension with an `if`. The check also tries other scores.
        """),
        L.task("Make `passed` in one line, so the program shows `['Ada', 'Cas', 'Dex']`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_passed():
    box_passed = L.code_box("passed", lines=4)
    box_passed
    return (box_passed,)


@app.cell(hide_code=True)
def check_passed(box_passed):
    result_passed = L.check("passed", box_passed.value)
    result_passed
    return (result_passed,)


@app.cell(hide_code=True)
def hints_passed():
    L.hints(
        "passed",
        [
            "A `for` over a dictionary goes through its keys, here the names. What goes in the new list: the name, or the score?",
            "Step 4 of the recipe, and the third comprehension of the worked example: the `if` comes at the end. The score of a name is `scores[name]`. Which comparison includes exactly 5.5?",
            "`passed = [name for name in scores if ___]`",
        ],
        why="`for name in scores` goes through the names. `if scores[name] >= 5.5` keeps only the names whose score is 5.5 or more, and `name` in front puts those names in the list. Dex has exactly 5.5, so `>=` keeps Dex too.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Repeating: for loops](/course/open/py-for?lang=en) you added up and counted with a loop. With `append` from [Lists](/course/open/py-lists?lang=en), the same kind of loop can build a new list, as in the picture; a comprehension does that in one line, and with curly braces it builds a [dictionary](/course/open/py-dicts?lang=en). Later in the course, code that prepares data is full of them, for example to turn every word of a text into its number. In the next lesson you learn to read the error messages Python gives, and to find a bug.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dict_comp, result_lengths, result_passed):
    L.wrap_up(result_lengths, choice_dict_comp, result_passed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
