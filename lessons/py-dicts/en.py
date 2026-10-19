# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-dicts. Route: course/route/lessen/stam-1-basis.yaml. Checks and the dictionary in the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-dicts", "en")
    C = L.checks   # the dictionary in the picture, and the key it looks up


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
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="What does this program show?\n\n```python\nfor name in [\"Ada\", \"Bo\"]:\n    print(f\"Hi {name}\")\n```",
        options={
            "each": "`Hi Ada` and `Hi Bo`, each on its own line",
            "one-line": "`Hi Ada Bo`, on one line",
            "word": "`Hi name`, twice",
        },
        correct="each",
        why={
            "one-line": "The `print` runs once in every round, and each `print` starts a new line. See the guess in [Repeating: for loops](/course/open/py-for?lang=en).",
            "each": "Each round, `name` gets the next item, and the f-string fills it in.",
            "word": "Between braces, the f-string fills in the value of `name`, and that is another item each round. See the idea in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="What does this program show?\n\n```python\ntotal = 0\nfor n in [3, 8, 5]:\n    if n > 4:\n        total = total + n\nprint(total)\n```",
        options={
            "all": "`16`",
            "last": "`5`",
            "big": "`13`",
        },
        correct="big",
        why={
            "all": "The `if` only adds the numbers greater than 4, and 3 is not. See the worked example in [Repeating: for loops](/course/open/py-for?lang=en).",
            "big": "8 and 5 are greater than 4, so `total` becomes 8 + 5.",
            "last": "`total` keeps what it already had, and each round adds to it. See the picture in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far you found an item by its place: `scores[0]`. But often you want to find something by a name: the price of pears, or how many times a word is in a text. A dictionary keeps values under names of their own, so you can look them up. Later in the course, a program that reads text keeps a dictionary that gives each word its own number.

    **An everyday example.** The contacts in your phone. You look up a name, and you get the phone number that goes with it. You never need to know at which place in the list that name is. And each name is there only once: save a new number under the same name, and the old number is gone.
    """)
    return


@app.cell(hide_code=True)
def predict_replace():
    guess_replace = L.guess(
        "replace",
        question="What do you think this program shows?\n\n```python\nprices = {\"apple\": 0.5, \"pear\": 0.75}\nprices[\"apple\"] = 0.6\nprint(prices[\"apple\"])\n```",
        options={
            "old": "`0.5`",
            "new": "`0.6`",
            "both": "`0.5` and `0.6`",
        },
        reveal="**`0.6`.** The second line gives the key `\"apple\"` a new value. A key is in a dictionary only once, so the old value, 0.5, is gone.",
    )
    guess_replace
    return (guess_replace,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A dictionary keeps pairs: a key, and the value that goes with it. You look up a value by its key, not by its place.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **dictionary** | pairs of a key and a value, between curly braces; a colon between each key and its value, and a comma between the pairs | `prices = {"apple": 0.5, "pear": 0.75}` |
        | **key** | what you look up a value by; each key is in a dictionary only once | `prices["pear"]` gives `0.75` |
        | **in** | checks whether a key is in a dictionary, and gives `True` or `False` | `"plum" in prices` gives `False` |

        With `=` you change the value under a key, or add a new pair: `prices["plum"] = 0.3`. Looking up a key that is not there stops the program with an error, so check first with `in` when you are not sure. `{}` is an empty dictionary, to fill later, and `len(prices)` gives the number of pairs. When Python shows a dictionary, it puts single quotes around strings: `{'apple': 0.5}`.
        """),
        L.watch_out("Later in the course, *key* also means something else, in a program that reads text; that lesson says what. Here, a key is what you look a value up by."),
    ])
    return


@app.cell(hide_code=True)
def picture_dict():
    mo.vstack([
        mo.md("```python\nprices = {\"apple\": 0.5, \"pear\": 0.75, \"plum\": 0.3}\nprint(prices[\"pear\"])\n```"),
        draw.dict_boxes([(f'"{_k}"', str(_v)) for _k, _v in C.PICTURE.items()], key_label="key", value_label="value",
                        mark=list(C.PICTURE).index(C.PICTURE_MARK), mark_label=f'prices["{C.PICTURE_MARK}"]',
                        title="A dictionary, and one lookup"),
        mo.md(r"""
        Each key points to its value. `prices["pear"]` finds the key `"pear"` and gives the value next to it, 0.75. Where the pair is in the dictionary does not matter.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A key is in a dictionary only once: giving it a value again replaces the old one. And a key must be exactly the same: `\"Pear\"` with a capital is a different key from `\"pear\"`. Looking up a key that is not there stops your program, so check with `in` first when you are not sure."),
        mo.md(r"""
        ### Recipe

        To work with a dictionary:

        1. Make it with curly braces, and pairs `key: value` with a comma between them: `prices = {"apple": 0.5, "pear": 0.75}`.
        2. Look up a value with its key in square brackets: `prices["pear"]`.
        3. Change a value, or add a new pair, with `=`: `prices["plum"] = 0.3`.
        4. Not sure a key is there? Check with `in` first: `if "plum" in prices:`.
        5. Go through all pairs with `for`: in `for fruit in prices:`, `fruit` gets each key, and `prices[fruit]` gives its value.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The stock of a small shop: how many of each product are there? The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_stock():
    # 1. the dictionary: each product, with how many there are
    _stock = {"bread": 4, "milk": 6}
    # 2. look up a value by its key
    print(_stock["milk"])
    # 3. add a new pair: a delivery of eggs
    _stock["eggs"] = 12
    # what was sold today
    _sold = ["milk", "apples", "milk"]
    for _item in _sold:
        # 4. check with in whether the key is there
        if _item in _stock:
            # 3. change the value under a key: one less
            _stock[_item] = _stock[_item] - 1
        else:
            print(f"No {_item} in the shop")
    # 5. go through all pairs: _item gets each key
    for _item in _stock:
        print(f"{_item}: {_stock[_item]}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Under the cell appear five lines. `_stock["milk"]` gives `6`. `milk` was sold twice, so it goes down to 4. `apples` is not a key, so the `else` runs. The last loop shows each key with its value, in the order the pairs were added: `eggs` came last.
        """),
        L.task("Add `\"bread\"` to the list `_sold` in the cell above, and run it. What does the last loop show for bread now?"),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_capital():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A capital

        `capitals` gives the capital city of each country. Make `city` the capital of the country in `country`. Replace the `...`. The check also tries another country.
        """),
        L.task("Look up the capital of `country` in `capitals`, so the program shows `Tokyo`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_capital():
    box_capital = L.code_box("capital", lines=5)
    box_capital
    return (box_capital,)


@app.cell(hide_code=True)
def check_capital(box_capital):
    result_capital = L.check("capital", box_capital.value)
    result_capital
    return (result_capital,)


@app.cell(hide_code=True)
def hints_capital():
    L.hints(
        "capital",
        [
            "Step 2 of the recipe: you look up a value with its key in square brackets. Which name holds the key here?",
            "`country` holds the key, `\"Japan\"`. Which name goes between the brackets, so it also works when `country` is another country?",
            "`city = capitals[___]`",
        ],
        why="`capitals[country]` looks up the value under the key in `country`: for `\"Japan\"` that is `\"Tokyo\"`, and for `\"Peru\"` it is `\"Lima\"`. Without quotes, `country` is a name, and Python uses its value.",
    )
    return


@app.cell(hide_code=True)
def text_keys():
    mo.md(r"""
    ### Exercise 2. A loop over a dictionary
    """)
    return


@app.cell(hide_code=True)
def choice_keys_cell():
    choice_keys = L.choice(
        "keys",
        question="What does this program show?\n\n```python\nages = {\"Ada\": 36, \"Bo\": 12}\nfor name in ages:\n    print(name)\n```",
        options={
            "values": "`36` and `12`, each on its own line",
            "pairs": "`Ada 36` and `Bo 12`, each on its own line",
            "keys": "`Ada` and `Bo`, each on its own line",
        },
        correct="keys",
        why={
            "values": "A loop over a dictionary does not give its values. Look at step 5 of the recipe again: what does the name after `for` get?",
            "keys": "A loop over a dictionary goes through its keys: `name` is `\"Ada\"`, then `\"Bo\"`. `ages[name]` would give the values.",
            "pairs": "The name after `for` gets one thing each round, not a pair. Look at step 5 of the recipe again.",
        },
    )
    choice_keys
    return (choice_keys,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Counting words

        Count how many times each word is in `words`. `counts` starts empty, as `{}`. Replace the `...` in the loop with an `if` and an `else`: is the word already a key in `counts`? Then its count goes up by 1. If not, it gets a new pair, with count 1. Inside the loop, `else:` lines up with its `if`: in the box, press Backspace once before you type it. The check also tries another list.
        """),
        L.task("Make `counts` give each word in `words` the number of times it is there."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count",
        [
            "Look at the first loop of the worked example: an `if` with `in` checks whether the item is a key, and the `else` handles the other case.",
            "Is the word already a key in `counts`? Then its new count is the old count plus 1. If not, the word gets a new pair, with count 1.",
            "`if word in counts:`, with the line that adds 1 to the count under it; then `else:`, and under it ___.",
        ],
        why="The first time a word comes by, `word in counts` is `False`, so the `else` gives it a new pair with 1. The next time, it is `True`, and its count goes up by 1. `rain` comes by three times: 1, 2, 3.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Lists](/course/open/py-lists?lang=en) you found an item by its place; in a dictionary you find a value by its key. With a `for` loop from [Repeating: for loops](/course/open/py-for?lang=en), you counted every word in one go. Later in the course, a program that reads text keeps a dictionary just like `counts`, and one that gives each word its own number. In the next lesson you meet tuples and sets, two more ways to keep values together, and how to sort a list.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keys, result_capital, result_count):
    L.wrap_up(result_capital, choice_keys, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
