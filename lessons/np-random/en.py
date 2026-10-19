# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson np-random. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-random", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Using code from modules](/course/open/py-imports?lang=en) and [Sums and means along an axis](/course/open/np-axis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_randint():
    quick_randint = L.choice(
        "quick-randint",
        question="Which numbers can `random.randint(1, 6)` give?",
        options={
            "five": "1, 2, 3, 4 and 5",
            "six": "1, 2, 3, 4, 5 and 6",
            "zero": "0, 1, 2, 3, 4, 5 and 6",
        },
        correct="six",
        why={
            "five": "`randint` does include its second number. See the idea in [Using code from modules](/course/open/py-imports?lang=en).",
            "six": "`random.randint(1, 6)` gives a whole number from 1 up to and including 6: a die.",
            "zero": "It starts at its first number, 1. See the idea in [Using code from modules](/course/open/py-imports?lang=en).",
        },
    )
    quick_randint
    return (quick_randint,)


@app.cell(hide_code=True)
def quick_check_named():
    quick_named = L.choice(
        "quick-named",
        question="In `m.sum(axis=0)`, what is `axis=0`?",
        options={
            "named": "A value with a name: the name `axis`, then `=` and the value",
            "new": "A new name `axis` that the program makes, with the value 0",
            "compare": "A check whether `axis` is 0",
        },
        correct="named",
        why={
            "named": "Between the parentheses, `axis=0` gives `sum` the value 0, under the name `axis`.",
            "new": "It makes no name in your program: it only goes into `sum`. See the table of new words in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "compare": "A check would be `==`, with two signs. See the table of new words in [Sums and means along an axis](/course/open/np-axis?lang=en).",
        },
    )
    quick_named
    return (quick_named,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model starts with random numbers, before it has learned anything, and it sees its examples in a random order. So two runs give slightly different models. To find a mistake, or to compare two ideas fairly, you want to run the same thing again, with the same random numbers: a seed does that. When you reproduce a research paper later in the course, the seed is part of the recipe. But one seed shows only one run: to see how much the results change, try several seeds.

    **An everyday example.** In many games, you can type in a seed for a new world: anyone who types the same seed gets the same world.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same",
        question="NumPy makes random numbers with a random number generator, or RNG. Two RNGs are both made with seed 42, and each is asked for three dice rolls. What do you think happens?",
        options={
            "same": "Both give the same three rolls",
            "different": "They give different rolls",
            "continue": "The second one goes on where the first one stopped",
        },
        reveal="**Both give the same three rolls.** Each RNG starts at its own seed. With the same seed, they make the same numbers, in the same order. Only when you ask the same RNG again do you get new numbers.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    NumPy makes random numbers with a random number generator, RNG for short; in code, it is usually called `rng`. You make one with a seed, and then ask it for numbers: whole numbers, floats, or a random order. The numbers look random, but they follow entirely from the seed.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **random number generator (RNG)** | makes numbers that look random but follow from its seed; NumPy makes one with `np.random.default_rng` | `rng = np.random.default_rng(42)` |
    | **`rng.integers(low, high, size=n)`** | n whole numbers from `low` up to, but not including, `high` | `rng.integers(1, 7, size=3)`: three dice rolls |
    | **`rng.random(size=n)`** | n floats from 0 up to, but not including, 1 | `rng.random(size=3)` |
    | **`rng.permutation(items)`** | the items in a random order, as a new array | `rng.permutation(["a", "b", "c"])` |
    | **`size=(rows, columns)`** | random numbers in a matrix of that shape | `rng.random(size=(2, 3))` |

    `size=3` is a value with a name, as `axis=0` in [Sums and means along an axis](/course/open/np-axis?lang=en). The seed is the starting point, as with `random.seed` in [Using code from modules](/course/open/py-imports?lang=en). An RNG keeps going: each time you ask it, you get the next numbers; making it again with the same seed starts over. And an array can also hold text, such as names: then it shows them between quotes, and its dtype says how long the longest text is, such as `<U3`.
    """)
    return


@app.cell(hide_code=True)
def picture_seeds():
    mo.vstack([
        draw.value_rows(C.picture_rows({"first": "a new RNG, seed 7", "same": "another new one, seed 7: the same rolls",
                                        "asked": "first RNG again", "again": "the first RNG, asked again: it goes on",
                                        "other": "seed 8: different rolls"}),
                        title="Six dice rolls at a time, from RNGs"),
        mo.md(r"""
        Each row is six dice rolls, from `integers(1, 7, size=6)`. The first two come from two new RNGs with seed 7, so they are the same, in the same order. The third row asks the first RNG again: it goes on, so the rolls are new. The last row comes from an RNG with seed 8: different rolls. NumPy works these rolls out when the lesson opens; with the same version of NumPy, they come out the same on every computer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`rng.integers(1, 6)` never gives a 6: its second number is not included. Python's `random.randint(1, 6)` does include it."),
        mo.md(r"""
        ### Recipe

        To make random numbers you can repeat:

        1. Make one random number generator with a seed, before you ask for numbers: `rng = np.random.default_rng(42)`. In a function that gets a seed, make it inside the function.
        2. Ask it for numbers: `rng.integers`, `rng.random` or `rng.permutation`, with `size=` for how many, or for the shape.
        3. Use that one RNG for everything random, in the same order every time.
        4. To get the same numbers again, start over: make the RNG again with the same seed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Dice, names in a random order, a small matrix of floats, and a function that rolls one die from a seed. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_random():
    import numpy as np

    # 1. one RNG, with a seed
    _rng = np.random.default_rng(7)
    # 2. and 3. ten dice rolls: from 1 up to, but not including, 7
    print(_rng.integers(1, 7, size=10))
    # 2. and 3. five names in a random order
    print(_rng.permutation(["Ada", "Bo", "Cas", "Dex", "Eva"]))
    # 2. and 3. floats from 0 up to 1, in a matrix of shape (2, 3)
    print(_rng.random(size=(2, 3)))
    # 4. a new RNG with the same seed starts over
    _again = np.random.default_rng(7)
    print(_again.integers(1, 7, size=10))


    # 1. in a function that gets a seed: the RNG is made inside, with that seed
    def _one_roll(seed):
        return np.random.default_rng(seed).integers(1, 7)


    print(_one_roll(7), _one_roll(7), _one_roll(8))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first line has ten rolls, from 1 to 6. The names come in a different order from the list, and the matrix has six floats between 0 and 1. The fourth line has the same ten rolls as the first: the new RNG, `_again`, starts over at seed 7. The function makes a new RNG each time, from the seed it gets, so seed 7 gives the same roll twice, and seed 8 usually another one. Run the cell again: every line stays the same, because the seeds are fixed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_number():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One number from 1 to 10
    """)
    return


@app.cell(hide_code=True)
def choice_number_cell():
    choice_number = L.choice(
        "one-number",
        question="`rng` is a random number generator. Which line gives one whole number from 1 up to and including 10?",
        options={
            "short": "`rng.integers(1, 10)`",
            "zero": "`rng.integers(0, 10)`",
            "ten": "`rng.integers(1, 11)`",
        },
        correct="ten",
        why={
            "short": "The second number is not included, so this never gives a 10. Look at the *Watch out* box above the recipe again.",
            "zero": "That starts at 0, so it gives 0 to 9. Where should it start, and where should it stop?",
            "ten": "From 1 up to, but not including, 11: 1 to 10.",
        },
    )
    choice_number
    return (choice_number,)


@app.cell(hide_code=True)
def text_exercise_rolls():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Twenty rolls you can repeat

        Make a random number generator with seed 2026, ask it for twenty dice rolls at once, and call them `rolls`: a vector of twenty whole numbers from 1 to 6. Replace the `...`. The last line is `rolls`, so the check sees it.
        """),
        L.task("Make `rolls`: twenty dice rolls from a random number generator with seed 2026."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rolls():
    box_rolls = L.code_box("rolls", lines=6)
    box_rolls
    return (box_rolls,)


@app.cell(hide_code=True)
def check_rolls(box_rolls):
    result_rolls = L.check("rolls", box_rolls.value)
    result_rolls
    return (result_rolls,)


@app.cell(hide_code=True)
def hints_rolls():
    L.hints(
        "rolls",
        [
            "Steps 1 and 2 of the recipe: first the RNG, then the numbers.",
            "The rolls come from `rng.integers`. Its second number is not included: which number makes a 6 possible? `size=20` asks for twenty at once.",
            "`rng = np.random.default_rng(___)`, then `rolls = rng.integers(___, ___, size=20)`.",
        ],
        why="`np.random.default_rng(2026)` makes an RNG that starts at seed 2026. `rng.integers(1, 7, size=20)` asks it for twenty whole numbers from 1 to 6. With the same seed and the same version of NumPy, anyone gets these same twenty rolls.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shuffled():
    mo.vstack([
        mo.md(f"""
        ### Exercise 3. A random order you can repeat

        Finish `shuffled(items, seed)`: it makes a random number generator with `seed`, and gives back `items` in a random order. The same seed must give the same order every time. Replace the `...`. The check calls `shuffled(["Ada", "Bo", "Cas", "Dex", "Eva"], 7)`: that should give `{C.shuffled_example()}`. That is a different order from the one in the worked example, with the same seed: there, the RNG had rolled ten dice first, so it had already gone on.
        """),
        L.task("Make `shuffled(items, seed)` give back the items in a random order that is the same for the same seed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shuffled():
    box_shuffled = L.code_box("shuffled", lines=7)
    box_shuffled
    return (box_shuffled,)


@app.cell(hide_code=True)
def check_shuffled(box_shuffled):
    result_shuffled = L.check("shuffled", box_shuffled.value)
    result_shuffled
    return (result_shuffled,)


@app.cell(hide_code=True)
def hints_shuffled():
    L.hints(
        "shuffled",
        [
            "Steps 1 and 2 of the recipe, inside the function: an RNG, then a random order.",
            "Make the RNG with the seed the function gets, not with a number of your own. For a random order, look at the table of new words.",
            "`rng = np.random.default_rng(___)`, and then `return rng.___(items)`.",
        ],
        why="`np.random.default_rng(seed)` starts the RNG at the seed the check passes in, so the same seed gives the same order. `rng.permutation(items)` gives the items back in that random order, as a new array.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Using code from modules](/course/open/py-imports?lang=en) `random.seed` made the random numbers of Python repeatable; a NumPy random number generator does the same, and gives whole arrays at once. In [Sums and means along an axis](/course/open/np-axis?lang=en) a value with a name, `axis=0`, chose the direction; here `size=` chooses how many numbers, or their shape. Later in the course, in *Sampling: let the model write*, your language model picks its next letter with an RNG. In the next lesson, *An image is a grid of numbers*, you look at a picture as an array.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_number, result_rolls, result_shuffled):
    L.wrap_up(choice_number, result_rolls, result_shuffled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
