# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-inheritance. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-inheritance", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Classes and objects](/course/open/py-classes?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tally():
    quick_tally = L.choice(
        "quick-tally",
        question="What does this program show?\n\n```python\nclass Tally:\n    def __init__(self):\n        self.n = 0\n\n    def up(self):\n        self.n = self.n + 1\n\na = Tally()\nb = Tally()\na.up()\na.up()\nb.up()\nprint(a.n)\n```",
        options={
            "three": "`3`",
            "two": "`2`",
            "one": "`1`",
        },
        correct="two",
        why={
            "three": "Each object keeps its own `n`. Which object did `b.up()` change? See the picture in [Classes and objects](/course/open/py-classes?lang=en).",
            "two": "`a.up()` runs twice and changes only `a`'s own `n`; `b.up()` changes `b`'s.",
            "one": "`up` runs on `a` more than once. Count the calls with `a.` in front again, and see the recipe in [Classes and objects](/course/open/py-classes?lang=en).",
        },
    )
    quick_tally
    return (quick_tally,)


@app.cell(hide_code=True)
def quick_check_hello():
    quick_hello = L.choice(
        "quick-hello",
        question="What does this program show?\n\n```python\nclass Pet:\n    def __init__(self, name):\n        self.name = name\n\n    def hello(self):\n        return f\"I am {self.name}\"\n\np = Pet(\"Mo\")\nprint(p.hello())\n```",
        options={
            "value": "`I am Mo`",
            "word": "`I am name`",
            "code": "`I am self.name`",
        },
        correct="value",
        why={
            "value": "`Pet(\"Mo\")` gives the object the name `\"Mo\"`; in the f-string, `{self.name}` is replaced by it.",
            "word": "In an f-string, `{self.name}` is replaced by the value of the object's attribute. See the worked example in [Classes and objects](/course/open/py-classes?lang=en).",
            "code": "The part between `{}` is worked out, not shown as it is. Which value did `__init__` store in `self.name`? See the idea in [Classes and objects](/course/open/py-classes?lang=en).",
        },
    )
    quick_hello
    return (quick_hello,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Often a new class is almost the same as one you already have. You do not want to copy the whole class and change one line: then you have two copies to keep in step. A subclass builds on the class you have, and you write only what is different. In PyTorch, every model you write starts like that: `class MyModel(nn.Module):`, built on `nn.Module`, PyTorch's parent class for models, with `super().__init__()` as the first line of its `__init__`.

    **An everyday example.** An electric bike is a bike: it has the wheels, the brakes and the pedals of a bike, plus a motor. Nobody designs new wheels for it: the maker takes a bike and adds a motor.
    """)
    return


@app.cell(hide_code=True)
def predict_dog():
    guess_dog = L.guess(
        "dog",
        question="What do you think this program shows?\n\n```python\nclass Pet:\n    def __init__(self, name):\n        self.name = name\n\n    def speak(self):\n        return f\"{self.name} says hi\"\n\nclass Dog(Pet):\n    def fetch(self):\n        return \"got it\"\n\nd = Dog(\"Rex\")\nprint(d.speak())\n```",
        options={
            "speak": "`Rex says hi`",
            "no-speak": "It stops: `Dog` has no `speak`",
            "no-init": "It stops: `Dog` has no `__init__`",
        },
        reveal="**`Rex says hi`.** `class Dog(Pet):` builds `Dog` on `Pet`. A `Dog` gets `__init__` and `speak` from `Pet`, and adds `fetch` of its own.",
    )
    guess_dog
    return (guess_dog,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A subclass is a class built on another class, its parent. It gets every method of the parent, and adds new methods or changes a few.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **subclass** | a class built on another class; it gets all the methods of that class | `class Dog(Pet):` |
    | **parent class** | the class a subclass is built on; it goes between the parentheses | `Pet` |
    | **inheritance** | a subclass getting the methods of its parent class | `Dog("Rex").speak()` |
    | **override** | a method in the subclass with the same name as one in the parent: the subclass's own version wins | `def sound(self):` in `Dog` |
    | **`super()`** | inside a method: the parent class, so you can run its version of a method | `super().__init__(name)` |

    When you call a method on an object, Python looks for it in the object's own class first, and then in the parent class. So a method of the parent that calls `self.sound()` gets the subclass's `sound`, if it has one. And `super()` passes `self` for you, just like a dot does: `super().__init__(name)` gets only `name`.
    """)
    return


@app.cell(hide_code=True)
def picture_tree():
    mo.vstack([
        draw.class_objects(["class Pet", "__init__(self, name)", "sound(self)", "speak(self)"],
                           [("class Dog(Pet)", ["sound(self)"]),
                            ("class Parrot(Pet)", ["__init__(self, name, word)", "sound(self)"])],
                           class_label="The parent class", objects_label="Two subclasses: only what they add or change",
                           title="Two classes built on Pet"),
        mo.md(r"""
        Each arrow means: built on. `Dog` writes only `sound`; for `__init__` and `speak`, Python looks in `Pet`. `Parrot` also writes its own `__init__`, which first runs `Pet`'s with `super()`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Once a subclass has its own `__init__`, the parent's `__init__` no longer runs by itself. Forget `super().__init__(...)`, and the object misses what the parent sets up, such as `self.name`."),
        mo.md(r"""
        ### Recipe

        To build a class on another class:

        1. Put the parent class between parentheses after the new class's name: `class Dog(Pet):`. The subclass gets every method of `Pet`.
        2. To change a method, override it: write a method with the same name in the subclass. Its own version wins.
        3. Does the subclass need its own `__init__`? Run the parent's first, `super().__init__(name)`, and then set what is new: `self.word = word`.
        4. Use the subclass like any class: `Dog("Rex").speak()` finds `speak` in `Pet`, and `sound` in `Dog`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Pets that make their own sound. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_pets():
    class _Pet:
        def __init__(self, name):
            self.name = name

        def sound(self):
            return "hm"

        def speak(self):
            return f"{self.name} says {self.sound()}"

    # 1. the parent class between parentheses
    class _Dog(_Pet):
        # 2. override: the same name, its own version
        def sound(self):
            return "woof"

    class _Parrot(_Pet):
        # 3. its own __init__: first the parent's, then what is new
        def __init__(self, name, word):
            super().__init__(name)
            self.word = word

        def sound(self):
            return self.word

    # 4. use the subclasses like any class
    print(_Pet("Mo").speak())
    print(_Dog("Rex").speak())
    print(_Parrot("Kiki", "hello").speak())
    print(_Parrot("Kiki", "hello").name)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `Mo says hm`, `Rex says woof`, `Kiki says hello` and `Kiki`. `speak` is written only once, in `_Pet`, but it calls `self.sound()`, and so each pet gets its own sound. The parrot has a name because its `__init__` runs `_Pet`'s first, with `super()`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_greeter():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A greeter in Dutch

        `Greeter` greets in English: `Greeter("Ada").greet()` gives `Hello, Ada!`. Under it, `DutchGreeter` is built on `Greeter`. Replace its `...` with one method, so that it greets in Dutch: `Hallo, Ada!`. Leave `Greeter` as it is. The check makes `DutchGreeter("Ada")` and calls `greet()`: that should be `Hallo, Ada!`. It also checks that `Greeter` still says Hello.
        """),
        L.task("Make `DutchGreeter(\"Ada\").greet()` give `Hallo, Ada!`, by changing only `DutchGreeter`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_greeter():
    box_greeter = L.code_box("greeter", lines=16)
    box_greeter
    return (box_greeter,)


@app.cell(hide_code=True)
def check_greeter(box_greeter):
    result_greeter = L.check("greeter", box_greeter.value)
    result_greeter
    return (result_greeter,)


@app.cell(hide_code=True)
def hints_greeter():
    L.hints(
        "greeter",
        [
            "Step 2 of the recipe, and `_Dog` in the worked example: which method of `Greeter` gives the word Hello?",
            "`greet` asks `self.greeting()` for the word. Override `greeting` in `DutchGreeter`, and let it give back the Dutch word.",
            "`def greeting(self):`, indented under `class DutchGreeter(Greeter):`, and under it `return ___`.",
        ],
        why="`DutchGreeter` overrides `greeting` with `return \"Hallo\"`. It gets `__init__` and `greet` from `Greeter`, and `greet` calls `self.greeting()`: for a `DutchGreeter`, that is the new version.",
    )
    return


@app.cell(hide_code=True)
def text_which_switch():
    mo.md(r"""
    ### Exercise 2. Which version runs?
    """)
    return


@app.cell(hide_code=True)
def choice_which_switch_cell():
    choice_which_switch = L.choice(
        "which-switch",
        question="What does this program show?\n\n```python\nclass Lamp:\n    def __init__(self):\n        self.on = False\n\n    def switch(self):\n        self.on = True\n\nclass BrokenLamp(Lamp):\n    def switch(self):\n        self.on = False\n\na = BrokenLamp()\na.switch()\nprint(a.on)\n```",
        options={
            "error": "It stops: `BrokenLamp` has no `__init__`",
            "on": "`True`",
            "off": "`False`",
        },
        correct="off",
        why={
            "error": "`BrokenLamp` is built on `Lamp`, so it gets every method of `Lamp`, `__init__` too. Look at step 1 of the recipe again.",
            "on": "That is what `switch` of `Lamp` would do. Which class does Python look in first for `a.switch()`?",
            "off": "`a` is a `BrokenLamp`, so `a.switch()` runs `BrokenLamp`'s own version, which keeps `self.on` at `False`. `__init__` comes from `Lamp`.",
        },
    )
    choice_which_switch
    return (choice_which_switch,)


@app.cell(hide_code=True)
def text_exercise_savings():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A savings account

        `Account` has an owner, a balance that starts at 0, and `deposit`. `Savings` is built on it, with an interest rate of its own, `rate`. Replace the two `...`: in `__init__`, let `Account` set up the owner and the balance; in `add_interest`, add the interest, the balance times `self.rate`, to the balance. The check makes `Savings("Ada", 0.1)`, deposits 100 and adds the interest once: the balance should be 110. It also tries more, such as whether `Account` still works on its own.
        """),
        L.task("Make `Savings` call the `__init__` of `Account` with `super()`, and add interest to its balance."),
    ])
    return


@app.cell(hide_code=True)
def exercise_savings():
    box_savings = L.code_box("savings", lines=16)
    box_savings
    return (box_savings,)


@app.cell(hide_code=True)
def check_savings(box_savings):
    result_savings = L.check("savings", box_savings.value)
    result_savings
    return (result_savings,)


@app.cell(hide_code=True)
def hints_savings():
    L.hints(
        "savings",
        [
            "Step 3 of the recipe, and `_Parrot` in the worked example: first the parent's `__init__`, then what is new.",
            "In `__init__`, pass `owner` on to the `__init__` of `Account`. In `add_interest`, the new balance is the old balance plus the balance times `self.rate`.",
            "`super().__init__(___)` in `__init__`, and in `add_interest`, `self.balance = self.balance + ___`.",
        ],
        why="`super().__init__(owner)` runs the `__init__` of `Account`, which sets `self.owner` and `self.balance = 0`. `self.deposit(self.balance * self.rate)` adds the interest, with `deposit` from `Account`; `self.balance = self.balance + self.balance * self.rate` does the same.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Classes and objects](/course/open/py-classes?lang=en) you wrote a class from scratch; now a class can build on another one, and you write only what is different. Later in the course, you build every model this way: `class MyModel(nn.Module):`, with `super().__init__()` first, and your own layers after it. The next lesson is about generators: functions that hand out values one at a time.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which_switch, result_greeter, result_savings):
    L.wrap_up(result_greeter, choice_which_switch, result_savings)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
