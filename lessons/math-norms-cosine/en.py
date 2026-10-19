# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-norms-cosine. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-norms-cosine", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Many features: the dot product](/course/open/math-dot-product?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="What is the dot product of $(2, 3)$ and $(4, -1)$?",
        options={
            "products": "$(8, -3)$",
            "right": "5",
            "sign": "11",
        },
        correct="right",
        why={
            "products": "Those are the products. What is the last step of a dot product? See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "sign": "Look at the second product: 3 times −1. Which sign does it get? See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "right": "2 × 4 + 3 × (−1) = 8 − 3 = 5.",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def quick_check_np_dot():
    quick_np_dot = L.choice(
        "quick-np-dot",
        question="`a` and `b` are arrays of 3 numbers each. What does `np.dot(a, b)` give?",
        options={
            "right": "One number",
            "three": "An array of 3 numbers",
            "matrix": "A 3 by 3 matrix",
        },
        correct="right",
        why={
            "three": "That is `a * b`. What does `np.dot` do with those products? See the worked example in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "matrix": "A dot product multiplies the numbers in the same place, not every number with every number. See *The idea* in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "right": "`np.dot` multiplies the numbers in the same place and adds the products up: one number.",
        },
    )
    quick_np_dot
    return (quick_np_dot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model sees everything as vectors: a house, a student, later in the course a word or a whole text. To compare two of them, you want to know how far apart they are, and whether they point the same way. Search engines find texts about the same topic by comparing the directions of their vectors.

    **An everyday example.** Two friends walk from the same corner, both to the north-east. One stops after 1 km, the other walks on for 5 km. They are 4 km apart, but they went exactly the same way.
    """)
    return


@app.cell(hide_code=True)
def predict_same_way():
    guess_same_way = L.guess(
        "same-way",
        question="Two vectors: $\\mathbf{a} = (1, 1)$ and $\\mathbf{b} = (5, 5)$. What do you think is true?",
        options={
            "close": "They are close together, and point the same way",
            "far-same": "They are far apart, but point the same way",
            "far-other": "They are far apart, and point different ways",
        },
        reveal="**Far apart, but the same way.** $\\mathbf{b}$ is $\\mathbf{a}$ times 5: the same direction, five times as long. The distance between their heads is about 5.7, but their cosine similarity, which only looks at the direction, is exactly 1.",
    )
    guess_same_way
    return (guess_same_way,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Think of a vector as an arrow from the origin, the point (0, 0), to the point its numbers give. Its norm is its length: the square root of the sum of its squares. If you know Pythagoras from school: it is his rule, with as many squares as the vector has numbers. That is also the square root of the dot product of the vector with itself.
        """),
        L.formula(r"\lVert \mathbf{x} \rVert = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2} = \sqrt{\mathbf{x} \cdot \mathbf{x}}"),
        mo.md(r"""
        The distance between two vectors is the norm of their difference, $\lVert \mathbf{a} - \mathbf{b} \rVert$: the length of the straight line between their heads. The cosine similarity, written $\cos(\mathbf{a}, \mathbf{b})$, looks only at the direction: the dot product, divided by the two norms, so that the lengths drop out.
        """),
        L.formula(r"\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}"),
        mo.md(r"""
        It is 1 when $\mathbf{a}$ and $\mathbf{b}$ point the same way, 0 when they are at right angles, and −1 when they point opposite ways. The name comes from the cosine of the angle between them, from geometry; you do not need that here, as the formula works it out.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **norm** | | the length of a vector: the square root of the sum of its squares; never negative | `np.linalg.norm(x)` |
        | **distance** | | how far apart two vectors are: the norm of their difference | `np.linalg.norm(a - b)` |
        | **cosine similarity** | | the dot product divided by the product of the two norms: 1 for the same direction, 0 at right angles, −1 for opposite directions | `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` |
        | $\lVert \mathbf{x} \rVert$ | the norm of x | the norm of the vector $\mathbf{x}$ | `np.linalg.norm(x)` |
        """),
        L.watch_out("The cosine similarity ignores length: (1, 1) and (100, 100) have a cosine similarity of 1. When the size matters, such as the area of two houses, use the distance."),
        mo.md(r"""
        Move the slider: it turns $\mathbf{b}$ away from $\mathbf{a}$. $\mathbf{a}$ is 4 long, $\mathbf{b}$ 3, and the dashed line joins their heads: the distance. The other chart shows the cosine similarity for every angle; drag across it to set the angle.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_angle_cell():
    def _frame(angle):
        distance, similarity = C.measures(angle, 3)
        return (C.picture(angle, 3, {"title": "Two vectors", "every": "For every angle", "x": "angle (°)", "y": "cosine similarity",
                                 "alt": f"Two arrows from the origin: a, 4 long, and b, 3 long, with {L.number(angle, 0)} degrees between them. A dashed line joins their heads: the distance. A second chart shows the cosine similarity for every angle from 0 to 180 degrees: it falls from 1 through 0 at 90 degrees to −1, slowly near 0 and 180 degrees and fastest around 90."}, "angle"),
                f"""With {L.number(angle, 0)}° between them, and $\\mathbf{{b}}$ 3 long: the dashed line, the distance between the heads, is <span class="ea-num">{L.number(distance, 2)}</span>, and the cosine similarity is <span class="ea-num">{L.number(similarity, 2)}</span>.""")

    explore_angle = L.explore("angle", "angle", _frame, start=45, low=0, high=180, step=15, role="", task="At which angle is the cosine similarity 0.5?")
    explore_angle
    return


@app.cell(hide_code=True)
def text_between():
    mo.md(r"""
    At 60° the cosine similarity is 0.5. It does not fall evenly with the angle: from 0° to 60° it loses half, and from 60° to 90° the other half.

    In the next block the angle stays at 45°, and you only make $\mathbf{b}$ longer or shorter. The other chart shows both numbers for every length: the distance dashed, as the line between the heads, and the cosine similarity solid; drag across it to set the length.
    """)
    return


@app.cell(hide_code=True)
def explore_length_cell():
    def _frame(length):
        distance, similarity = C.measures(45, length)
        return (C.picture(45, length, {"title": "Two vectors", "every": "For every length of b", "x": "length of b", "distance": "distance", "similarity": "cosine similarity",
                                 "alt": f"Two arrows from the origin: a, 4 long, and b, {L.number(length, 1)} long, with {L.number(45, 0)} degrees between them. A dashed line joins their heads: the distance. A second chart shows, for every length of b from 1 to 5, the distance, dashed, which is lowest near 3, and the cosine similarity, solid, which stays 0.71."}, "length"),
                f"""With {L.number(45, 0)}° between them, and $\\mathbf{{b}}$ {L.number(length, 1)} long: the dashed line, the distance between the heads, is <span class="ea-num">{L.number(distance, 2)}</span>, and the cosine similarity is <span class="ea-num">{L.number(similarity, 2)}</span>.""")

    explore_length = L.explore("length", "length", _frame, start=4, low=1, high=5, step=0.5, role="", task="At which length of $\\mathbf{b}$ is the distance smallest?")
    explore_length
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Only the distance changes. It is smallest at a length of about 3, where it is 2.83, and not at 4, where $\mathbf{b}$ is as long as $\mathbf{a}$: at 45°, the head of $\mathbf{b}$ comes closest to that of $\mathbf{a}$ when $\mathbf{b}$ is a bit shorter than $\mathbf{a}$, and moves away again when $\mathbf{b}$ gets shorter still. The cosine similarity stays 0.71 at every length: it only sees the direction.

    ### Recipe

    To work out the cosine similarity of two vectors:

    1. Work out their dot product: `np.dot(a, b)`.
    2. Work out both norms: `np.linalg.norm(a)` and `np.linalg.norm(b)`.
    3. Divide the dot product by the product of the norms, with brackets around that product.
    4. Read it: close to 1 is the same direction, close to 0 at right angles, close to −1 opposite.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three texts, each as a vector: how often each uses the words "ball", "goal" and "vote". Text `_b` is long and about the same thing as `_a`; text `_c` is about an election. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_cosine():
    import numpy as np

    _a = np.array([3, 2, 0])
    _b = np.array([12, 8, 1])
    _c = np.array([0, 1, 4])
    # 1. the dot products
    print(np.dot(_a, _b), np.dot(_a, _c))
    # 2. the norms: of _a step by step, then all three with np.linalg.norm
    _total = 0
    for _i in range(len(_a)):
        _total = _total + _a[_i] ** 2
    print(_total, _total ** 0.5)
    print(np.linalg.norm(_a), np.linalg.norm(_b), np.linalg.norm(_c))
    # 3. divide by the product of the norms
    print(np.dot(_a, _b) / (np.linalg.norm(_a) * np.linalg.norm(_b)))
    print(np.dot(_a, _c) / (np.linalg.norm(_a) * np.linalg.norm(_c)))
    # and the distances, to compare
    print(np.linalg.norm(_a - _b), np.linalg.norm(_a - _c))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The loop adds 9 + 4 + 0 = 13 for `_a`, and its square root, 3.61, is what `np.linalg.norm` gives. The cosine similarity of `_a` and `_b` is 0.998, almost 1: the same words in about the same mix, only four times as many. `_a` and `_c` get 0.13: hardly alike. The distance tells another story: `_c` is 5.10 from `_a`, closer than `_b` at 10.86, only because `_b` is a longer text. To find texts about the same thing, the cosine similarity is the better measure. Word counts are never negative, so here it never drops below 0; the angle slider at 180° shows −1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_norm():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A norm by hand
    """)
    return


@app.cell(hide_code=True)
def choice_norm_cell():
    choice_norm = L.choice(
        "norm",
        question="What is the norm of the vector $(2, -3, 6)$?",
        options={
            "no-root": "49",
            "sum": "5",
            "right": "7",
        },
        correct="right",
        why={
            "no-root": "That is the sum of the squares. Which step comes after it? See the formula in *The idea*.",
            "sum": "That adds up the numbers themselves. What happens to each number first? See the formula in *The idea*.",
            "right": "2² + (−3)² + 6² = 4 + 9 + 36 = 49, and the square root of 49 is 7. Squaring makes the −3 count as much as a 3.",
        },
    )
    choice_norm
    return (choice_norm,)


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A similar taste in music

        How many songs two people played this week: pop, jazz and rock. `a` is the first person, `b` the second. Put the cosine similarity of `a` and `b` in `similarity`. Replace the `...`. The last line is `similarity`, so the check sees it. The check also tries another second person.
        """),
        L.task("Put the cosine similarity of `a` and `b` in `similarity`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("cosine", lines=7)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("cosine", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "cosine",
        [
            "The recipe has the steps: the dot product, the two norms, and a division.",
            "The dot product goes on top. What goes below the line, and what does Python need so that it divides by all of it?",
            "`similarity = np.dot(a, b) / (___ * ___)`.",
        ],
        why="`np.dot(a, b)` is 10 × 4 + 2 × 1 + 5 × 2 = 52. The norms are about 11.36 and 4.58, and their product is about 52.05, so the cosine similarity is about 0.999. The brackets make Python multiply the norms before it divides.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A distance without NumPy

        Finish `distance(a, b)`. It gets two lists of numbers of the same length, and gives back the distance between them. Replace the `...`. The check calls `distance([1, 2], [4, 6])`: that should give `5.0`. It also tries other lists, some with negative numbers.
        """),
        L.task("Make `distance` give back the distance between `a` and `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_distance():
    box_distance = L.code_box("distance-loop", lines=6)
    box_distance
    return (box_distance,)


@app.cell(hide_code=True)
def check_distance(box_distance):
    result_distance = L.check("distance-loop", box_distance.value)
    result_distance
    return (result_distance,)


@app.cell(hide_code=True)
def hints_distance():
    L.hints(
        "distance-loop",
        [
            "The distance is the norm of the difference: subtract place by place, square, add up, and take the square root.",
            "A loop over the places, `for i in range(len(a)):`, with a total that grows. What do you add to it for place `i`?",
            "`total = total + (a[i] - b[i]) ** ___` in the loop, and after it `return total ** ___`.",
        ],
        why="Start with `total = 0`. For each place, `(a[i] - b[i]) ** 2` is the squared difference; the loop adds them up, and `** 0.5` takes the square root. For the example: 3² + 4² = 25, and the square root of 25 is `5.0`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Many features: the dot product](/course/open/math-dot-product?lang=en) you multiplied two vectors place by place and added up. The square root of the dot product of a vector with itself is its length, the norm; the norm of a difference is a distance; and the dot product divided by two norms says whether two vectors point the same way. Later in the course, words become vectors, and words with a similar meaning get a high cosine similarity. In the next lesson, *Rates multiply: the chain rule*, you see how rates of change multiply.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_norm, result_cosine, result_distance):
    L.wrap_up(choice_norm, result_cosine, result_distance)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
