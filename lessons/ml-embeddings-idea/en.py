# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-embeddings-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-embeddings-idea", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One-hot vectors for classes](/course/open/ml-one-hot?lang=en) and [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_onehot():
    quick_onehot = L.choice(
        "quick-onehot",
        question="The classes cat, dog and bird are written as the one-hot vectors $(1, 0, 0)$, $(0, 1, 0)$ and $(0, 0, 1)$. Which two classes lie closest together?",
        options={
            "pets": "Cat and dog: they are the two pets",
            "right": "None: all three lie equally far apart",
            "order": "Cat and bird: their 1s are furthest apart, so their vectors are the most alike",
        },
        correct="right",
        why={
            "pets": "The vectors know nothing about pets: cat and dog differ in two places, just like cat and bird. See [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).",
            "order": "Each pair of vectors differs in exactly two places; where the 1 sits does not matter. See [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).",
            "right": "Each pair differs in two places, so every distance is $\\sqrt{2}$, and every cosine similarity is 0. One-hot vectors say which class something is, and nothing about which classes are alike. Today you meet vectors that do.",
        },
    )
    quick_onehot
    return (quick_onehot,)


@app.cell(hide_code=True)
def quick_check_angle():
    quick_angle = L.choice(
        "quick-angle",
        question="What is the cosine similarity of $(1, 0)$ and $(0, 2)$?",
        options={
            "right": "0: the two arrows stand at right angles",
            "half": "0.5: one is twice as long as the other",
            "two": "2: the dot product of the two",
        },
        correct="right",
        why={
            "half": "The lengths drop out of a cosine similarity: it only looks at the directions. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "two": "The dot product is $1 \\cdot 0 + 0 \\cdot 2 = 0$, and the cosine similarity divides it by the two norms. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "right": "The dot product is 0, so the cosine similarity is 0, whatever the lengths: one arrow points right, the other up. Today you measure how alike two images are in just this way.",
        },
    )
    quick_angle
    return (quick_angle,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To compare things, a computer needs numbers for them. One-hot vectors make every class equally far from every other. Pixels help only a little: two images of the same digit, written by two people, share only part of their pixels. You want numbers in which things that are alike lie close together, and things that differ lie far apart.

    **An everyday example.** On a map, two numbers say where a town is: how far east, and how far north. Towns that lie close on the map lie close in reality, and the map lets you see at a glance which towns lie next to each other. A model can learn such a map of meaning.
    """)
    return


@app.cell(hide_code=True)
def predict_ones():
    guess_ones = L.guess(
        "where-ones",
        question="The course's digit reader turns every image into 2 numbers, a point on a map, just before it decides which digit it is. Where do you think the points of the 126 test images of a 1 end up?",
        options={
            "spread": "All over the map: every 1 is written differently",
            "right": "Along one line out from the middle: some close, some far, all the same way",
            "point": "All on one point: to the model they are all the same digit",
        },
        reveal="**Along one line out from the middle.** Nine in ten of the 1s lie up and to the right, in a narrow wedge, some close to the middle and some far from it. Every digit gets its own direction. How far out an image lies says how sure the model is, not which digit it is.",
    )
    guess_ones
    return (guess_ones,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **embedding** is a list of numbers that a model gives a thing, such as a word, an image or a sound, so that things that are alike get numbers that are alike. The course's digit reader makes one for every image. It is a small network like the MLP of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), with other layers at the start that a later lesson explains, and the course trained it on the 6,000 training digits. Just before it decides which digit it sees, it turns the image into 2 numbers: its embedding. Real models use hundreds or thousands of numbers; this one uses 2, so that you can draw them. Each embedding is then a point, and all of them together make an **embedding map**.

        On the map, every digit gets its own direction from the middle. You measure how alike two embeddings are by the angle between them: their cosine similarity. Two images of the same digit have a cosine similarity of 0.95 on average, two images of different digits −0.11. On their 784 pixels, the same pairs get 0.53 and 0.37, a much smaller difference.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **embedding** | | a list of numbers that a model gives a thing, so that things that are alike get numbers that are alike | `E[0]`, the 2 numbers of image 0 |
        | **embedding map** | | the embeddings drawn as points, so that you see which things lie close together | `plt.scatter(E[:, 0], E[:, 1])` |
        """),
        L.watch_out("In everyday computer use, embedding a video means putting it into a web page. In AI, an embedding is a list of numbers. And the numbers mean nothing on their own: what counts is how the embeddings compare with each other. Trained again with another seed, the model draws another map, with the digits in other directions; with only 2 numbers, two digits sometimes even share one ray. The course picked a training run whose map shows ten clear rays."),
        mo.md(r"""
        The picture: the embeddings of the first 300 test digits, each drawn as its own digit, at its 2 numbers. Ten rays come out of the middle, one for each digit.
        """),
        mo.Html(C.map_picture({"title": "The embeddings of 300 test digits", "x": "number 1", "y": "number 2",
                               "alt": "Digits drawn at their 2 numbers. They form ten rays out of the middle, one per digit: 1s up and to the right, 4s to the right, 7s to the left, 0s down and to the right, and so on; 4s and 9s lie next to each other."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Finding the image most like a given one, with embeddings:

    1. Get the embedding of every image: here, `C.embeddings()` runs the course's digit reader on the 1,000 test digits and gives back `E`, one row of 2 numbers per image, and their `labels`.
    2. Compare two embeddings with their cosine similarity: close to 1 means alike, 0 or less means not alike.
    3. Compare the embedding of the given image with that of every other image, and keep the one with the largest cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Test image 0, a 7, compared with the next four, and then with all others. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_embeddings():
    import numpy as np

    # 1. the embeddings of the 1,000 test digits, from the course's model
    _E, _labels = C.embeddings()
    print(_E.shape, _labels[:5])
    print(round(_E[0, 0], 1), round(_E[0, 1], 1), _labels[0])
    # 2. the cosine similarity of image 0 with images 1 to 4
    for _j in range(1, 5):
        _cos = np.dot(_E[0], _E[_j]) / (np.linalg.norm(_E[0]) * np.linalg.norm(_E[_j]))
        print(_j, _labels[_j], round(_cos, 3))
    # 3. the image most like image 0: the largest cosine similarity of all others
    _best = -2
    _best_j = -1
    for _j in range(1, len(_E)):
        _cos = np.dot(_E[0], _E[_j]) / (np.linalg.norm(_E[0]) * np.linalg.norm(_E[_j]))
        if _cos > _best:
            _best = _cos
            _best_j = _j
    print(_best_j, _labels[_best_j], round(_best, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_E` has 1,000 rows of 2 numbers, and the first five labels are 7, 2, 1, 0 and 4. Image 0, the 7, has the embedding $(-168.9, 23.0)$: far to the left, where the 7s are on the map. Against the 2, its cosine similarity is 0.319; against the 1, −0.384; against the 0, −0.662; and against the 4, −0.999, the opposite direction. The image most like it is image 26, another 7, with a cosine similarity of 1.0, rounded to four decimals.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Close to the middle, far from it
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "same-direction",
        question="On the map, one 4 lies far from the middle, another one close to it, in the same direction. What is the cosine similarity of their embeddings?",
        options={
            "half": "About 0.5: one is much longer than the other",
            "zero": "About 0: they lie far apart",
            "right": "About 1: they point the same way",
        },
        correct="right",
        why={
            "half": "The lengths drop out of the cosine similarity: it divides by both norms.",
            "zero": "Far apart is the distance. The cosine similarity looks at the directions only, and these are the same.",
            "right": "The same direction gives a cosine similarity close to 1, whatever the lengths. That is why it fits this map: the direction says which digit, the length only how sure the model is.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_similar():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The most similar row

        Make `most_similar(E, i)` give the number `j` of the row of `E` most like row `i`, by step 3 of the recipe: the largest cosine similarity, for a `j` other than `i`. `E` is a NumPy array with one embedding per row. The loop is there: fill in what happens for each `j`. The check calls `most_similar(np.array([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), 0)`, which should give `1`: row 1 points almost the same way as row 0, though it is ten times as long. It tries other tables too, also with 3 numbers per row.
        """),
        L.task("Make `most_similar` give the row most like row `i`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_similar():
    box_similar = L.code_box("most-similar", lines=13)
    box_similar
    return (box_similar,)


@app.cell(hide_code=True)
def check_similar(box_similar):
    result_similar = L.check("most-similar", box_similar.value)
    result_similar
    return (result_similar,)


@app.cell(hide_code=True)
def hints_similar():
    L.hints(
        "most-similar",
        [
            "Inside the loop, the same as in step 3 of the worked example, with `E` and `i` instead of `_E` and `0`.",
            "First `if j != i:`, so that row `i` is not compared with itself; inside it, the cosine similarity of `E[i]` and `E[j]`.",
            "`cos = np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))`, and then `if cos > ___:` keep `best = cos` and `best_j = ___`.",
        ],
        why="In the loop: `if j != i:`, then `cos = np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))`, and `if cos > best:` with `best = cos` and `best_j = j`. In the example, row 1 has a cosine similarity of 0.995 with row 0, row 2 0.707, and row 3 −1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_map():
    mo.md(r"""
    ### Exercise 3. Reading the map
    """)
    return


@app.cell(hide_code=True)
def choice_map_cell():
    choice_map = L.choice(
        "most-like-4",
        question="Look at the map. The embeddings of which digit are most like those of the 4s?",
        options={
            "right": "The 9s",
            "zero": "The 0s",
            "seven": "The 7s",
        },
        correct="right",
        why={
            "zero": "The 0s lie on the other side of the 4s, further away: a 4 and a 0 have a cosine similarity of 0.66 on average.",
            "seven": "The 7s point almost the opposite way: a 4 and a 7 have a cosine similarity of −0.93 on average.",
            "right": "The 9s lie right next to the 4s: a 4 and a 9 have a cosine similarity of 0.82 on average, two 4s 0.95. And they look alike: both have a closed or nearly closed top and a stroke down on the right. Nobody told the model that 4s and 9s look alike; it learned that from the images.",
        },
    )
    choice_map
    return (choice_map,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The vectors of [One-hot vectors for classes](/course/open/ml-one-hot?lang=en) make every class equally far from every other; an embedding is learned, so that its directions mean something. To compare embeddings, you use the cosine similarity of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), and the map is a scatter plot as in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en). In the next lesson, *Convolution: one small filter slides over the image*, you see how networks such as the course's digit reader look at an image. Later, words get embeddings too: that is how a language model knows which words are alike.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, result_similar, choice_map):
    L.wrap_up(choice_direction, result_similar, choice_map)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
