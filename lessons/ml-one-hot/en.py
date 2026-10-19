# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-one-hot. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-one-hot", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Selecting parts of an array](/course/open/np-indexing?lang=en) and [Classification: predicting a class](/course/open/ml-classification?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. What does `m[1]` give?",
        options={
            "column": "`array([2, 5])`",
            "right": "`array([4, 5, 6])`",
            "number": "`2`",
        },
        correct="right",
        why={
            "column": "That is the column `m[:, 1]`. What does one index between the brackets select? See the table of new words in [Selecting parts of an array](/course/open/np-indexing?lang=en).",
            "number": "One index on a matrix gives more than one number. See the table of new words in [Selecting parts of an array](/course/open/np-indexing?lang=en).",
            "right": "One index selects a whole row, counted from 0: row 1 is the second row.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_label_kind():
    quick_label_kind = L.choice(
        "quick-label-kind",
        question="Which of these labels is a class written as a number?",
        options={
            "right": "The bus line someone takes, such as line 3",
            "temperature": "The temperature tomorrow, such as 14 degrees",
            "price": "The price of a house, such as 300 thousand euros",
        },
        correct="right",
        why={
            "temperature": "14.5 degrees makes sense too: values in between mean something, so this label is a number. See step 4 of the recipe in [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "price": "A price in between makes sense, and close is almost right: this label is a number. See step 4 of the recipe in [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "right": "There is no line 2.5, and line 3 is not three times line 1: the numbers are only names, so the bus lines are classes.",
        },
    )
    quick_label_kind
    return (quick_label_kind,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model calculates with numbers, so a class must become numbers too. But the class numbers 0, 1 and 2 bring something false along: an order, and amounts. To a model, "bird" is then twice "dog". Give each class a place of its own instead, in a vector with one place per class. Every class is then equally far from every other class, and a model can give each class its own number, such as a score or a probability.

    **An everyday example.** A ballot paper has one box for each party, and you mark exactly one. Nobody reads the paper as "party 3 is three times party 1".
    """)
    return


@app.cell(hide_code=True)
def predict_hesitate():
    guess_hesitate = L.guess(
        "hesitate",
        question="A model learns the labels 0 = cat, 1 = dog and 2 = bird as plain numbers. For a photo it cannot decide between cat and bird. What do you think it predicts?",
        options={
            "dog": "About 1: a dog",
            "either": "0 or 2, at random",
            "unsure": "A number that means 'unsure'",
        },
        reveal="**About 1: a dog.** Halfway between 0 and 2 lies 1, so a model that hesitates between cat and bird predicts the one class it did not consider. With a place per class, it can say 0.5 for cat and 0.5 for bird, and 0 for dog.",
    )
    guess_hesitate
    return (guess_hesitate,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Fix an order for the $K$ classes once, such as cat, dog, bird, and number them from 0. The **one-hot** vector of class $c$ has $K$ places, a 1 at place $c$ and 0 everywhere else: cat is $(1, 0, 0)$, dog $(0, 1, 0)$ and bird $(0, 0, 1)$. "Hot" is the one place that is on.

        In NumPy, `np.eye(K)` is the identity matrix: $K$ rows of $K$ numbers, with 1s on the diagonal. Row $c$ of it is exactly the one-hot vector of class $c$. A list or array of row numbers between square brackets selects those rows, in that order: `np.eye(3)[[1, 0]]` gives the rows of dog and cat. So `np.eye(K)[labels]` makes the one-hot vectors of all labels at once. The way back is `.argmax()`: the place of the 1 is the class.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **one-hot vector**, **one-hot encoding** | | a class as a vector with one place per class: a 1 at the place of the class, 0 everywhere else | `[0, 1, 0]` |
        | $K$ | K | the number of classes | `k` |
        | **identity matrix** | | a square matrix with 1s on the diagonal and 0 elsewhere; row $c$ is the one-hot vector of class $c$ | `np.eye(3)` |
        | **rows by number** | | an array or list of row numbers between square brackets selects those rows, in that order | `np.eye(3)[[1, 0]]` |
        """),
        L.watch_out("The order of the classes is a choice you make once, and then keep everywhere. $(0, 1, 0)$ only means dog as long as everyone agrees that place 1 is dog."),
        draw.array_grids([("labels", [["1", "0", "2", "1"]], ["shape (4,)"]),
                          ("np.eye(3)[labels]", [["0", "1", "0"], ["1", "0", "0"], ["0", "0", "1"], ["0", "1", "0"]], ["shape (4, 3)", "columns: cat, dog, bird"],
                           {(0, 1), (1, 0), (2, 2), (3, 1)})],
                         title="One row per label, one column per class"),
        mo.md(r"""
        Four labels, dog, cat, bird, dog, become four rows. Each row has its 1 in the column of its class.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To turn classes into one-hot vectors, and back:

    1. Fix the order of the $K$ classes, and number them from 0.
    2. For one label: a vector of $K$ zeros, with a 1 at the place of the label.
    3. For many labels at once: `np.eye(K)[labels]`, one row per label.
    4. Back to the class: `.argmax()`, the place of the 1; for many rows, `.argmax(axis=1)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three classes, cat, dog and bird, and four labels: dog, cat, bird, dog. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_one_hot():
    import numpy as np

    # 1. the order: 0 = cat, 1 = dog, 2 = bird
    _classes = ["cat", "dog", "bird"]
    _labels = np.array([1, 0, 2, 1])
    # 2. one label, dog, by hand: 3 places, with a 1 at place 1
    _dog = []
    for _i in range(3):
        if _i == 1:
            _dog.append(1)
        else:
            _dog.append(0)
    print(_dog)
    # the same vector is row 1 of the identity matrix
    print(np.eye(3))
    # 3. all labels at once
    _onehot = np.eye(3)[_labels]
    print(_onehot)
    # 4. back: the place of the 1 in each row
    print(_onehot.argmax(axis=1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_dog` is `[0, 1, 0]`: three places, with the 1 at place 1, the place of dog. `np.eye(3)` has the one-hot vectors of cat, dog and bird as its rows, and row 1 is that same vector. `np.eye(3)[_labels]` picks row 1, 0, 2 and 1 again: the four one-hot vectors, with a 1 in each row. NumPy writes `1.` and `0.`, because the identity matrix holds floats. `.argmax(axis=1)` gives back the place of the 1 in each row, `[1 0 2 1]`: the labels you started with.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_long():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How long?
    """)
    return


@app.cell(hide_code=True)
def choice_long_cell():
    choice_long = L.choice(
        "how-long",
        question="A model reads handwritten digits, 0 to 9. How many numbers does the one-hot vector of one label have?",
        options={
            "one": "1",
            "nine": "9",
            "right": "10",
            "four": "4: enough to write 9 in zeros and ones",
        },
        correct="right",
        why={
            "one": "That is the label itself, the class number. How many places does a one-hot vector have? See *The idea*.",
            "nine": "Count the digits from 0 to 9: how many classes are there? See *The idea*.",
            "four": "That is writing the number in binary, with several places on. A one-hot vector has one place per class, and only one is on. See *The idea*.",
            "right": "Ten classes, so ten places: nine zeros and a single 1.",
        },
    )
    choice_long
    return (choice_long,)


@app.cell(hide_code=True)
def text_exercise_one():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One label, by hand

        Finish `one_hot(label, k)`: it gets a class number and the number of classes, and gives back the one-hot vector as a list, with plain Python. Replace the `...`. The check calls `one_hot(1, 3)`: that should give `[0, 1, 0]`. It also tries other labels and other numbers of classes.
        """),
        L.task("Make `one_hot` give back a list of `k` numbers, with a 1 at place `label`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one():
    box_one = L.code_box("one-hot", lines=8)
    box_one
    return (box_one,)


@app.cell(hide_code=True)
def check_one(box_one):
    result_one = L.check("one-hot", box_one.value)
    result_one
    return (result_one,)


@app.cell(hide_code=True)
def hints_one():
    L.hints(
        "one-hot",
        [
            "Step 2 of the recipe: a place for every class, all 0, except the place of the label.",
            "A loop over the places `i` from 0 up to, not including, `k`. At which place do you add a 1, and how do you test for it?",
            "`for i in range(k):`, then `if i == ___:` with `vector.append(1)`, and under `else:` `vector.append(___)`.",
        ],
        why="The loop goes over all `k` places; only at `i == label` does it add a 1. For `one_hot(1, 3)`: places 0, 1 and 2 give 0, 1 and 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Count with one-hot vectors

        Five handwritten digits have the labels in `labels`. Make their one-hot vectors with NumPy, and add them up per class, so that `counts` holds ten numbers: how often each digit appears. Replace the `...`. The last line is `counts`, so the check sees it. The check also tries other labels.
        """),
        L.task("Put how often each of the ten digits appears in `counts`, by adding up the one-hot vectors."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-classes", lines=6)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-classes", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-classes",
        [
            "Step 3 of the recipe gives one row per label. In which direction do you add up, to get one number per class?",
            "With ten digits, `np.eye` needs how many rows? Then sum each column, as in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "`counts = np.eye(___)[labels].sum(axis=___)`.",
        ],
        why="`np.eye(10)[labels]` is a matrix of 5 rows and 10 columns; `.sum(axis=0)` adds up each column, one count per digit: `[1, 0, 0, 2, 0, 0, 0, 1, 0, 1]`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Classification: predicting a class](/course/open/ml-classification?lang=en), classes written as numbers were only names; a one-hot vector gives each class its own place, so nothing false comes along. In [Selecting parts of an array](/course/open/np-indexing?lang=en) you selected one row by its number; now a whole list of row numbers selects many rows at once. Later in the course, a model for many classes gives one probability per class, and those probabilities should come close to the one-hot vector of the label. In the next lesson, *Many classes: softmax*, you turn a model's scores into such probabilities.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long, result_count, result_one):
    L.wrap_up(choice_long, result_one, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
