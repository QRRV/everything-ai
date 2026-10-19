# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-classes. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-classes", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Dictionaries](/course/open/py-dicts?lang=en) and [Functions](/course/open/py-functions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="What does this program show?\n\n```python\npoint = {\"x\": 3, \"y\": 4}\nprint(point[\"y\"])\n```",
        options={
            "four": "`4`",
            "three": "`3`",
            "key": "`y`",
        },
        correct="four",
        why={
            "four": "`point[\"y\"]` looks up the value under the key `\"y\"`: 4.",
            "three": "3 is the value under `\"x\"`. See the recipe in [Dictionaries](/course/open/py-dicts?lang=en).",
            "key": "Square brackets give the value under a key, not the key itself. See the picture in [Dictionaries](/course/open/py-dicts?lang=en).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def quick_check_area():
    quick_area = L.choice(
        "quick-area",
        question="What does this program show?\n\n```python\ndef area(w, h):\n    return w * h\n\nprint(area(2, 5) + area(1, 1))\n```",
        options={
            "ten": "`10`",
            "twelve": "`12`",
            "eleven": "`11`",
        },
        correct="eleven",
        why={
            "ten": "That is only the first call. `print` shows the sum of both calls. See the guess in [Functions](/course/open/py-functions?lang=en).",
            "twelve": "Each call multiplies its own two arguments: 2 times 5, and 1 times 1. See the idea in [Functions](/course/open/py-functions?lang=en).",
            "eleven": "`area(2, 5)` gives back 10 and `area(1, 1)` gives back 1, so the sum is 11.",
        },
    )
    quick_area
    return (quick_area,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Data often comes in bundles. A bank account has an owner and a balance, and you put money in or take it out. Later in the course, a layer of a neural network has its numbers and a way to calculate with them. A class keeps such data together with the functions that work on it. In PyTorch, every kind of layer, and every model, is a class.

    **An everyday example.** The plan of a house, and the houses built from it. One plan, many houses: each house is built the same way, but each has its own furniture inside. A class is the plan; an object is one house.
    """)
    return


@app.cell(hide_code=True)
def predict_dog():
    guess_dog = L.guess(
        "dog",
        question="What do you think this program shows?\n\n```python\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n\na = Dog(\"Rex\")\nb = Dog(\"Bo\")\nprint(a.name)\n```",
        options={
            "bo": "`Bo`",
            "rex": "`Rex`",
            "both": "`Rex` and `Bo`",
        },
        reveal="**`Rex`.** `Dog(\"Rex\")` makes a new dog and gives it the name `\"Rex\"`; `Dog(\"Bo\")` makes another one, with its own name. `a.name` is the name of the dog in `a`.",
    )
    guess_dog
    return (guess_dog,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A class is a plan for a new kind of value: its data, and the functions that work on that data. Each object you make from it has its own data.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **class** | a plan for a new kind of value: its data, and the functions that work on it | `class Account:` |
    | **object** | one value made from a class; each object has its own data | `a = Account("Ada")` |
    | **`__init__`** | the method that runs when an object is made; it gives the object its starting data | `def __init__(self, owner):` |
    | **self** | inside a method: the object itself; `self.balance` is that object's own data | `self.balance = 0` |
    | **attribute** | a piece of an object's data, reached with a dot | `a.balance` |
    | **method** | a function inside a class; you call it on an object, with a dot | `a.deposit(50)` |

    `Account("Ada")` makes a new object, and runs `__init__` with `self` as that new object and `owner` as `"Ada"`. You never pass `self` yourself: Python does that. In the same way, `a.deposit(50)` runs `deposit` with `self` as `a`. By convention, the name of a class starts with a capital.
    """)
    return


@app.cell(hide_code=True)
def picture_objects():
    mo.vstack([
        draw.class_objects(["class Account", "__init__(self, owner)", "deposit(self, amount)"],
                           [("a = Account(\"Ada\")", ["owner: 'Ada'", "balance: 50"]),
                            ("b = Account(\"Bo\")", ["owner: 'Bo'", "balance: 0"])],
                           class_label="The class: the plan", objects_label="Two objects, each with its own data",
                           title="One class, two objects"),
        mo.md(r"""
        Both objects come from the same plan, with the same methods. But each has its own `owner` and its own `balance`: after `a.deposit(50)`, only `a` has 50.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Every method gets `self` as its first argument, and reaches the object's data through it. Inside a method, `balance` without `self.` in front is a new name that disappears when the method ends; `self.balance` stays with the object."),
        mo.md(r"""
        ### Recipe

        To make your own kind of value:

        1. Write `class`, a name with a capital, and a colon: `class Account:`.
        2. Indented under it, `def __init__(self, ...):` gives a new object its starting data, with `self.name = ...`.
        3. Add methods, `def deposit(self, amount):`, that work on the object's data through `self`.
        4. Make objects by calling the class, `a = Account("Ada")`; call methods with a dot, `a.deposit(50)`; read data with a dot, `a.balance`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A bank account with an owner and a balance. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_account():
    # 1. the class: a plan, with a capital
    class _Account:
        # 2. __init__ gives each new object its starting data
        def __init__(self, owner):
            self.owner = owner
            self.balance = 0
            self.deposits = []

        # 3. methods work on the object's own data, through self
        def deposit(self, amount):
            self.balance = self.balance + amount
            self.deposits.append(amount)

        def count_deposits(self):
            return len(self.deposits)

        def report(self):
            return f"{self.owner} has {self.balance} dollars"

    # 4. make objects, call methods, read data
    _a = _Account("Ada")
    _b = _Account("Bo")
    _a.deposit(50)
    _a.deposit(25)
    print(_a.balance)
    print(_b.balance)
    print(_a.report())
    print(_a.deposits)
    print(_a.count_deposits())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `75`, `0`, `Ada has 75 dollars`, `[50, 25]` and `2`. Both deposits went into Ada's account; Bo's balance is still 0, because each object has its own `balance`. Each account also keeps its own list, `self.deposits`: `deposit` adds to it with `append`, and `count_deposits` gives back its length. The name of the class starts with `_` because it is only used in this cell, just like any other name.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counter():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A counter

        Finish the class `Counter`: a new counter starts at 0 in `self.count`, and each call of `add` adds 1 to it. Replace the two `...`. The check makes a counter, calls `add` twice and reads `count`: that should be 2. It also tries more, such as two counters at once.
        """),
        L.task("Make `Counter` start at 0 and count up by 1 with each `add`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counter():
    box_counter = L.code_box("counter", lines=8)
    box_counter
    return (box_counter,)


@app.cell(hide_code=True)
def check_counter(box_counter):
    result_counter = L.check("counter", box_counter.value)
    result_counter
    return (result_counter,)


@app.cell(hide_code=True)
def hints_counter():
    L.hints(
        "counter",
        [
            "Step 2 of the recipe: what is the count of a counter that has counted nothing yet?",
            "In `add`, the object's count gets a new value: what it already was, plus 1. Reach it through `self`.",
            "`self.count = 0` in `__init__`, and in `add`, `self.count = ___ + 1`.",
        ],
        why="`self.count = 0` gives each new counter its own count, starting at 0. `self.count = self.count + 1` in `add` adds 1 to that object's count, and to no other.",
    )
    return


@app.cell(hide_code=True)
def text_two_objects():
    mo.md(r"""
    ### Exercise 2. Two objects
    """)
    return


@app.cell(hide_code=True)
def choice_two_objects_cell():
    choice_two_objects = L.choice(
        "two-objects",
        question="What does this program show?\n\n```python\nclass Lamp:\n    def __init__(self):\n        self.on = False\n\n    def switch(self):\n        self.on = True\n\nkitchen = Lamp()\nhall = Lamp()\nkitchen.switch()\nprint(hall.on)\n```",
        options={
            "true": "`True`",
            "false": "`False`",
            "error": "It stops: `hall` was never switched",
        },
        correct="false",
        why={
            "true": "`switch` changes `self.on` of the object it is called on. Which object was that? Look at the picture again.",
            "false": "`kitchen.switch()` changes only `kitchen`'s own `on`. `hall` still has the `False` it got from `__init__`.",
            "error": "`hall` got its own `on` in `__init__`, when it was made. Look at step 2 of the recipe again.",
        },
    )
    choice_two_objects
    return (choice_two_objects,)


@app.cell(hide_code=True)
def text_exercise_playlist():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A playlist

        Finish the class `Playlist`: each playlist has a name and its own list of songs. `add` puts a song at the end of that list, and `length` gives back how many songs it has. Replace the three `...`. The check makes a playlist, adds two songs and calls `length`: that should be 2. It also tries more, such as two playlists at once.
        """),
        L.task("Make `Playlist` keep its own songs, add them, and give back its length."),
    ])
    return


@app.cell(hide_code=True)
def exercise_playlist():
    box_playlist = L.code_box("playlist", lines=11)
    box_playlist
    return (box_playlist,)


@app.cell(hide_code=True)
def check_playlist(box_playlist):
    result_playlist = L.check("playlist", box_playlist.value)
    result_playlist
    return (result_playlist,)


@app.cell(hide_code=True)
def hints_playlist():
    L.hints(
        "playlist",
        [
            "Step 2 of the recipe: a new playlist has no songs yet. What does an empty list look like?",
            "In `add`, the song goes at the end of the object's list, `self.songs`. In `length`, give back the length of that list.",
            "`self.songs = []`, `self.songs.append(___)`, and `return len(___)`.",
        ],
        why="`self.songs = []` in `__init__` gives each playlist its own empty list. `self.songs.append(song)` adds a song to that list, and `return len(self.songs)` gives back how many it holds.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Functions](/course/open/py-functions?lang=en) a function got its data as arguments; a method gets its object as `self`, and finds the data there. In [Dictionaries](/course/open/py-dicts?lang=en) you kept values under keys; an object keeps them as attributes, together with the methods that use them. Later in the course, every layer of a network is an object made from such a class. In the next lesson you make `+` and `print` work on your own objects.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_two_objects, result_counter, result_playlist):
    L.wrap_up(result_counter, choice_two_objects, result_playlist)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
