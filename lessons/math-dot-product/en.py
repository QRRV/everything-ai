# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-dot-product. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-dot-product", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en) and [Calculating without loops](/course/open/np-vectorization?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_two():
    quick_two = L.choice(
        "quick-two",
        question="A model with two features predicts price = 2.5 × area + 10 × rooms + 30. What does it predict for an area of 100 and 4 rooms?",
        options={
            "added": "146.5",
            "no-rooms": "280",
            "right": "320",
        },
        correct="right",
        why={
            "added": "That adds the weights to the features. What does × do? See exercise 3 of [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "no-rooms": "That is 2.5 × 100 + 30. What do the rooms add? See exercise 3 of [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "right": "2.5 × 100 + 10 × 4 + 30 = 250 + 40 + 30 = 320: each feature times its own weight, plus the bias term.",
        },
    )
    quick_two
    return (quick_two,)


@app.cell(hide_code=True)
def quick_check_times():
    quick_times = L.choice(
        "quick-times",
        question="What does `np.array([1, 2, 3]) * np.array([4, 5, 6])` give?",
        options={
            "right": "`array([4, 10, 18])`",
            "sum": "`32`",
            "plus": "`array([5, 7, 9])`",
        },
        correct="right",
        why={
            "sum": "That adds the products up as well. Does `*` do that? See the table of new words in [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "plus": "That is `+`. What does `*` do with the numbers in the same place? See [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "right": "With two arrays of the same shape, `*` multiplies the numbers in the same place: 1 × 4, 2 × 5 and 3 × 6.",
        },
    )
    quick_times
    return (quick_times,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A house has more than an area: rooms, a garden. A model gives each feature its own weight, multiplies each feature by its weight, and adds it all up. With three features you can still write that out; a model for text has thousands of features, and you cannot. The dot product is the short way to write it, and NumPy works it out in one call.

    **An everyday example.** A receipt: 3 apples at 0.50, 2 loaves at 2.50 and 1 carton of milk at 1.20. The total is each quantity times its price, added up: 1.50 + 5.00 + 1.20 = 7.70.
    """)
    return


@app.cell(hide_code=True)
def predict_receipt():
    guess_receipt = L.guess(
        "receipt",
        question="`quantities = np.array([3, 2, 1])` and `prices = np.array([0.5, 2.5, 1.2])`. Which line do you think gives the total on the receipt, 7.7, as one number?",
        options={
            "times": "`quantities * prices`",
            "sums": "`quantities.sum() * prices.sum()`",
            "right": "`(quantities * prices).sum()`",
        },
        reveal="**`(quantities * prices).sum()`.** `quantities * prices` gives the three amounts, `array([1.5, 5. , 1.2])`, one per product, and `.sum()` adds them up to 7.7. `quantities.sum() * prices.sum()` is 6 × 4.2 = 25.2: every quantity times every price. Multiply the pairs, then add up: that is the dot product.",
    )
    guess_receipt
    return (guess_receipt,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Put the weights in one vector, and the features of an example in another, in the same order. The dot product multiplies the numbers in the same place and adds up the products. A model with $d$ features predicts:
        """),
        L.formula(r"[[pred:\hat{y}]] = [[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]] = [[param:w_1]] x_1 + [[param:w_2]] x_2 + \dots + [[param:w_d]] x_d + [[param:b]]"),
        mo.md(r"""
        With one feature, $\mathbf{w} \cdot \mathbf{x}$ is just $w$ times $x$, the model you know. With more, it is the weighted sum of all features, written in short.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **dot product**, $\mathbf{w} \cdot \mathbf{x}$ | w dot x | multiply the numbers in the same place of two vectors of the same length, and add up the products: one number | `np.dot(w, x)` |
        | $\mathbf{w}$ | w | the vector of weights, one per feature | `w` |
        | $d$ | d | the number of features: the length of $\mathbf{x}$ and of $\mathbf{w}$ | `len(x)` |
        | $x_1$, $w_1$ | x one, w one | the first feature and its weight | `x[0]`, `w[0]` |
        """),
        mo.Html(C.picture([2, 15, 20], [110, 5, 1], {"w": "w", "x": "x", "product": "w × x", "sum": "sum", "alt": "A dot product box by box. The weights 2, 15 and 20 stand above the features 110, 5 and 1. Below them are the products: 220, 75 and 20. Added up, they make 315."})),
        mo.md(r"""
        A house of 110 m², with 5 rooms and a garden (1 = yes), and the <span class="c-param">weights</span> 2, 15 and 20. Each weight times its own feature, and the products added up: the dot product, 315.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make a prediction with many features:

    1. Put the weights in one vector, and the features of the example in another, in the same order.
    2. Multiply the numbers in the same place.
    3. Add up the products: that is the dot product, `np.dot(w, x)`.
    4. Add the bias term.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The prediction for the house in the picture, with the bias term $b = 30$: first step by step, then with `np.dot` in one call. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_dot():
    import numpy as np

    # 1. weights and features, in the same order: area, rooms, garden
    _w = np.array([2, 15, 20])
    _x = np.array([110, 5, 1])
    _b = 30
    # 2. the numbers in the same place, multiplied
    print(_w * _x)
    # 3. added up: the dot product, with a loop place by place, and with np.dot
    _total = 0
    for _i in range(len(_w)):
        _total = _total + _w[_i] * _x[_i]
    print(_total, (_w * _x).sum(), np.dot(_w, _x))
    # 4. plus the bias term: the prediction
    print(np.dot(_w, _x) + _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_w * _x` gives the three products, `[220  75  20]`. The loop adds them up place by place; `.sum()` and `np.dot` do the same, and all three give 315: `np.dot` does steps 2 and 3 at once. With the bias term, the model predicts 345 thousand euros. With 300 features, the code would be exactly the same.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hand():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A dot product by hand
    """)
    return


@app.cell(hide_code=True)
def choice_hand_cell():
    choice_hand = L.choice(
        "by-hand",
        question="What is the dot product of $(2, 0, 1)$ and $(3, 7, 5)$?",
        options={
            "products": "$(6, 0, 5)$",
            "right": "11",
            "all": "18",
        },
        correct="right",
        why={
            "products": "Those are the products. What is the last step of a dot product? See the recipe.",
            "all": "That adds up all six numbers. What happens to the numbers in the same place first? See the recipe.",
            "right": "2 × 3 + 0 × 7 + 1 × 5 = 6 + 0 + 5 = 11: one number.",
        },
    )
    choice_hand
    return (choice_hand,)


@app.cell(hide_code=True)
def text_exercise_grade():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Predict a grade

        A model predicts a student's grade from two features: hours of study and hours of sleep, like the students in [Examples, features and labels](/course/open/ml-vocabulary?lang=en). The box has the weights `w`, the bias term `b` and the features `x` of one student. Put the prediction in `y_pred`. Replace the `...`. The last line is `y_pred`, so the check sees it. The check also tries another student.
        """),
        L.task("Put the predicted grade in `y_pred`, with a dot product."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grade():
    box_grade = L.code_box("grade", lines=8)
    box_grade
    return (box_grade,)


@app.cell(hide_code=True)
def check_grade(box_grade):
    result_grade = L.check("grade", box_grade.value)
    result_grade
    return (result_grade,)


@app.cell(hide_code=True)
def hints_grade():
    L.hints(
        "grade",
        [
            "The recipe, steps 2 to 4: multiply the pairs, add up, add the bias term.",
            "NumPy does steps 2 and 3 in one call. Which function is it? See step 3 of the recipe.",
            "`y_pred = np.dot(___, ___) + ___`.",
        ],
        why="`np.dot(w, x)` is 0.5 × 6 + 0.3 × 8 = 5.4, and `+ b` makes 6.4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. What np.dot does

        Finish `dot(a, b)`: the dot product of two lists of the same length, in plain Python, without NumPy. It gives back one number. Replace the `...`. The check calls `dot([1, 2, 3], [4, 5, 6])`: that should give `32`. It also tries other lists.
        """),
        L.task("Make `dot` give back the dot product, with a loop."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loop():
    box_loop = L.code_box("dot-loop", lines=7)
    box_loop
    return (box_loop,)


@app.cell(hide_code=True)
def check_loop(box_loop):
    result_loop = L.check("dot-loop", box_loop.value)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "dot-loop",
        [
            "Start a total at 0, and go through the places one by one.",
            "`range(len(a))` gives every place. What do you add to the total at each place?",
            "`total = total + a[i] * ___`, and after the loop `return ___`.",
        ],
        why="The loop goes through every place `i`, multiplies `a[i]` by `b[i]`, and adds that to `total`. For `[1, 2, 3]` and `[4, 5, 6]`: 4 + 10 + 18 = 32.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en) a model with two features added up each feature times its weight; the dot product is that weighted sum, for any number of features. In [Calculating without loops](/course/open/np-vectorization?lang=en), `*` multiplied the numbers in the same place; `np.dot` also adds them up. In the next lesson, *A whole batch at once: matrix multiplication*, you work out the dot products of many examples at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hand, result_grade, result_loop):
    L.wrap_up(choice_hand, result_grade, result_loop)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
