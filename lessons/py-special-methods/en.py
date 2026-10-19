# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-special-methods. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-special-methods", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Classes and objects](/course/open/py-classes?lang=en) and [Text: strings](/course/open/py-strings?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grow():
    quick_grow = L.choice(
        "quick-grow",
        question="What does this program show?\n\n```python\nclass Box:\n    def __init__(self, size):\n        self.size = size\n\n    def grow(self):\n        self.size = self.size * 2\n\nb = Box(3)\nb.grow()\nb.grow()\nprint(b.size)\n```",
        options={
            "six": "`6`",
            "three": "`3`",
            "twelve": "`12`",
        },
        correct="twelve",
        why={
            "six": "That is after one `grow`. `b.grow()` runs twice, and each time it changes `b`'s own `size`. See the recipe in [Classes and objects](/course/open/py-classes?lang=en).",
            "three": "`grow` changes `self.size`, and `self` is `b`, so `b.size` changes. See the watch-out in [Classes and objects](/course/open/py-classes?lang=en).",
            "twelve": "`Box(3)` starts with size 3; each `grow` doubles `b`'s own `size`: 6, then 12.",
        },
    )
    quick_grow
    return (quick_grow,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="What does this program show?\n\n```python\nx = 2\ny = 5\nprint(f\"({x}, {y})\")\n```",
        options={
            "values": "`(2, 5)`",
            "names": "`(x, y)`",
            "braces": "`({x}, {y})`",
        },
        correct="values",
        why={
            "values": "In an f-string, each `{...}` is replaced by its value; the parentheses and the comma are just text.",
            "names": "Python does not show the names between `{}`; it fills in their values. See the idea in [Text: strings](/course/open/py-strings?lang=en).",
            "braces": "The `f` in front makes it an f-string, so the `{...}` are filled in. See the recipe in [Text: strings](/course/open/py-strings?lang=en).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Python's own values work with symbols: `2 + 3` adds, and `print` shows a number or a list nicely. Your own objects cannot do that yet: `print` shows something like `<__main__.Account object at 0x7f3a2c>`, Python's own text for an object, with its class and where it is in memory. And `+` stops with an error. Special methods fix that. Later, in PyTorch, you call a model like a function, `model(x)`, and that works through a special method too.

    **An everyday example.** The power button on a remote control. You press it the same way on every remote, but what happens depends on the device: the TV turns on, or the radio. `+` is such a button, and your class decides what it does.
    """)
    return


@app.cell(hide_code=True)
def predict_money():
    guess_money = L.guess(
        "money",
        question="What do you think this program shows?\n\n```python\nclass Money:\n    def __init__(self, cents):\n        self.cents = cents\n\n    def __add__(self, other):\n        return Money(self.cents + other.cents)\n\n    def __str__(self):\n        return f\"{self.cents} cents\"\n\ntotal = Money(250) + Money(75)\nprint(total)\n```",
        options={
            "text": "`Money(250) + Money(75)`",
            "error": "It stops: Python cannot add two objects",
            "cents": "`325 cents`",
        },
        reveal="**`325 cents`.** `+` between two `Money` objects runs `__add__`, which makes a new `Money` with 250 + 75 cents. `print` then runs `__str__` of that new object, to get the text it shows.",
    )
    guess_money
    return (guess_money,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A special method has a name with two underscores on each side. You do not call it yourself: Python calls it when you use `+`, `print`, or call an object. You already know one: `__init__`, which Python runs when you make an object.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **special method** | a method with two underscores on each side of its name; Python calls it for you | `def __add__(self, other):` |
    | **`__add__`** | runs for `+`: `a + b` runs `a.__add__(b)`, with `self` as `a` and `other` as `b` | `def __add__(self, other):` |
    | **`__str__`** | gives back the text that `print` and an f-string show for the object | `def __str__(self):` |
    | **`__call__`** | runs when you call the object like a function: `double(21)` runs `double.__call__(21)` | `def __call__(self, x):` |

    `__add__` usually gives back a new object, and leaves `a` and `b` as they were, just like `2 + 3` leaves 2 and 3 alone. `__str__` must give back a string with `return`. Programmers often call these methods *dunder methods*: dunder is short for double underscore.
    """)
    return


@app.cell(hide_code=True)
def picture_calls():
    mo.vstack([
        draw.dict_boxes([("a + b", "a.__add__(b)"), ("print(a)", "a.__str__()"), ("f\"{a}\"", "a.__str__()"), ("double(21)", "double.__call__(21)")],
                        key_label="What you write", value_label="What Python runs",
                        title="What calls your methods"),
        mo.md(r"""
        On the left is what you write; on the right is the method Python runs for it. The object on the left of `+` is `self`; the one on the right is `other`. `print` and an f-string both ask the object for its text, through `__str__`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`__str__` must give back the text with `return`. With `print` inside `__str__` instead, the method gives back nothing, and `print(a)` stops with an error, because it gets no text."),
        mo.md(r"""
        ### Recipe

        To make `+`, `print` and calling work on your objects:

        1. In your class, write a method whose name has two underscores on each side; `self` is still its first argument.
        2. For `+`: `def __add__(self, other):`, and give back a new object made from both: `return Money(self.cents + other.cents)`.
        3. For `print` and f-strings: `def __str__(self):`, and give back a string, often an f-string: `return f"{self.cents} cents"`.
        4. For calling: `def __call__(self, x):`, and give back what the call gives. Then an object of that class works like a function: `double(21)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Money that adds up and shows itself, and a multiplier that you call like a function. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_special():
    # 1. methods whose names have two underscores on each side
    class _Money:
        def __init__(self, cents):
            self.cents = cents

        # 2. + gives back a new object, made from self and other
        def __add__(self, other):
            return _Money(self.cents + other.cents)

        # 3. print and f-strings get the text from __str__
        def __str__(self):
            return f"{self.cents} cents"

    class _Scale:
        def __init__(self, factor):
            self.factor = factor

        # 4. calling the object runs __call__
        def __call__(self, x):
            return self.factor * x

    _a = _Money(250)
    _b = _Money(75)
    _total = _a + _b
    print(_total)
    print(_a)
    print(f"Together: {_total}")
    _double = _Scale(2)
    print(_double(21))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `325 cents`, `250 cents`, `Together: 325 cents` and `42`. `_a + _b` made a new object, so `_a` still has its 250 cents. The f-string used `__str__` too. And `_double(21)` ran `__call__`, with `x` as 21.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vec():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A pair that adds up

        A `Vec` is a pair of numbers, `x` and `y`. Finish `__add__`, which gives back a new `Vec`: its `x` is the `x` of both pairs added up, and the same for `y`. And finish `__str__`, which gives back the text `(x, y)`. Replace the two `...`. The check makes `u = Vec(1, 2)` and `v = Vec(3, 4)`, and shows `u + v` with an f-string: that should be `(4, 6)`. It also tries more, such as whether `u` and `v` stay the same after `u + v`.
        """),
        L.task("Make `+` give a new `Vec`, and `print` show it as `(4, 6)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vec():
    box_vec = L.code_box("vec", lines=11)
    box_vec
    return (box_vec,)


@app.cell(hide_code=True)
def check_vec(box_vec):
    result_vec = L.check("vec", box_vec.value)
    result_vec
    return (result_vec,)


@app.cell(hide_code=True)
def hints_vec():
    L.hints(
        "vec",
        [
            "Steps 2 and 3 of the recipe, and `_Money` in the worked example: `__add__` makes a new object, and `__str__` gives back an f-string.",
            "In `__add__`, the new `x` is `self.x` plus `other.x`, and the same for `y`. In `__str__`, put the object's own `x` and `y` between parentheses.",
            "`return Vec(___, ___)` in `__add__`, and `return f\"({___}, {___})\"` in `__str__`.",
        ],
        why="`return Vec(self.x + other.x, self.y + other.y)` makes a new pair from both, and leaves `self` and `other` as they were. `return f\"({self.x}, {self.y})\"` gives back the text that `print` and f-strings show.",
    )
    return


@app.cell(hide_code=True)
def text_add_order():
    mo.md(r"""
    ### Exercise 2. Which one is self?
    """)
    return


@app.cell(hide_code=True)
def choice_add_order_cell():
    choice_add_order = L.choice(
        "add-order",
        question="What does this program show?\n\n```python\nclass Word:\n    def __init__(self, text):\n        self.text = text\n\n    def __add__(self, other):\n        return Word(f\"{other.text}-{self.text}\")\n\nw = Word(\"ice\") + Word(\"cream\")\nprint(w.text)\n```",
        options={
            "left-first": "`ice-cream`",
            "right-first": "`cream-ice`",
            "error": "It stops: `Word` has no `__str__`",
        },
        correct="right-first",
        why={
            "left-first": "That would put `self.text` first. In `a + b`, which object is `self`, and which is `other`? Look at the picture again.",
            "right-first": "`Word(\"ice\") + Word(\"cream\")` runs `__add__` with `self` as the ice word and `other` as the cream word. The method puts `other.text` first: `cream-ice`.",
            "error": "`print(w.text)` shows the attribute `text`, which is already a string. `__str__` is only needed for `print(w)` itself.",
        },
    )
    choice_add_order
    return (choice_add_order,)


@app.cell(hide_code=True)
def text_exercise_line():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A line you can call

        A straight line has a slope `w` and a starting value `b`: for each `x` it gives `w * x + b`. Finish `__call__`, so that you can call a line like a function, and `__str__`, which gives back its formula, such as `y = 2x + 1`. Replace the two `...`. The check makes `line = Line(2, 1)`, calls `line(3)`, which should give 7, and shows `Line(2, 1)` with an f-string: `y = 2x + 1`. It also tries more, such as two lines at once.
        """),
        L.task("For `line = Line(2, 1)`, make `line(3)` give 7, and `print(line)` show `y = 2x + 1`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_line():
    box_line = L.code_box("line", lines=11)
    box_line
    return (box_line,)


@app.cell(hide_code=True)
def check_line(box_line):
    result_line = L.check("line", box_line.value)
    result_line
    return (result_line,)


@app.cell(hide_code=True)
def hints_line():
    L.hints(
        "line",
        [
            "Step 4 of the recipe, and `_Scale` in the worked example: `__call__` gives back what the call should give.",
            "In `__call__`, use the line's own `self.w` and `self.b`, and the `x` it gets. In `__str__`, put `self.w` and `self.b` in an f-string.",
            "`return self.w * ___ + ___` in `__call__`, and `return f\"y = {___}x + {___}\"` in `__str__`.",
        ],
        why="`return self.w * x + self.b` works out the line at `x`, with this line's own numbers. `return f\"y = {self.w}x + {self.b}\"` gives back its formula, so `print(line)` shows `y = 2x + 1`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Classes and objects](/course/open/py-classes?lang=en) Python ran `__init__` for you when you made an object; now you know it does the same with `__add__`, `__str__` and `__call__`. Your `Line` is already a tiny model: two numbers, `w` and `b`, and you call it on an `x`. Later in the course, you call a PyTorch model in just that way, `model(x)`, and the model learns good numbers for itself. In the next lesson a class builds on another class: inheritance.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add_order, result_line, result_vec):
    L.wrap_up(result_vec, choice_add_order, result_line)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
