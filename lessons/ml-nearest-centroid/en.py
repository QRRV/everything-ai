# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-nearest-centroid. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-nearest-centroid", "en")
    C = L.checks   # the penguins and k-NN of ml-knn, the course's nearest centroid, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) and [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_knn():
    quick_knn = L.choice(
        "quick-knn",
        question="k-NN has 206 training penguins, and you ask it about one new penguin. What does it work out?",
        options={
            "right": "The distance from the new penguin to each of the 206 training penguins",
            "means": "The distance from the new penguin to the mean of each species",
            "formula": "Nothing much: it learned a formula from the training penguins beforehand",
        },
        correct="right",
        why={
            "means": "That is what today's model does. k-NN keeps every training penguin, and looks for the k nearest among all of them. See *The idea* in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "formula": "k-NN learns no parameters: it keeps the training penguins as they are, and does its work when you ask. See *The idea* in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "All 206 distances, then the k nearest, then their vote: for every penguin you ask about. Today, a model that does its work before you ask.",
        },
    )
    quick_knn
    return (quick_knn,)


@app.cell(hide_code=True)
def quick_check_centroid():
    quick_centroid = L.choice(
        "quick-centroid",
        question="In k-means, what is the centroid of a cluster?",
        options={
            "middle": "The example that lies most in the middle of the cluster",
            "right": "The mean of the examples in the cluster, one mean for each feature",
            "first": "The example the cluster started from",
        },
        correct="right",
        why={
            "middle": "A centroid need not be one of the examples: it is their mean, one number for each feature, and usually no example lies exactly there. See *The idea* in [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "first": "k-means starts from some examples, but every move step puts each centroid at the mean of its examples. See *The idea* in [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "right": "The mean of its examples, feature by feature. Today, each species gets such a mean, and with the species already known, one step is enough.",
        },
    )
    quick_centroid
    return (quick_centroid,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    k-NN keeps every training example and does its work when you ask: for each new penguin, 206 distances. With a million training examples, that is a million distances for every single question, and all million examples have to stay in memory. Many models work the other way round: they do the work once, before any question, and keep only what they learned. The simplest of them is the nearest centroid: one mean for each class, worked out beforehand. It answers in a moment and keeps almost nothing. What it gives up for that is the subject of this lesson.

    **An everyday example.** To tell whether a coin in your hand is a euro or a pound, you do not compare it with every coin you have ever held. You remember what a typical euro and a typical pound look like, and you see which one it is closest to.
    """)
    return


@app.cell(hide_code=True)
def predict_mistakes():
    guess_mistakes = L.guess(
        "mistakes",
        question="The 206 training penguins and the 68 validation penguins of the k-NN lesson, by bill length and body mass. k-NN with k = 15 got 2 of the 68 wrong. The nearest centroid gives each penguin the species whose mean is nearest. How many does it get wrong?",
        options={
            "fewer": "0 or 1: the means of whole species are steadier than 15 neighbors",
            "right": "7",
            "many": "About 30",
        },
        reveal="**7**, against 2 for k-NN with k = 15, and 6 with k = 1. Four of them are light Gentoos, on the wrong side of a straight line. With bill length and flipper length, which tell the species apart better, it gets 2 wrong, as k-NN does at its best.",
    )
    guess_mistakes
    return (guess_mistakes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Lazy and eager.** k-NN does nothing when it gets its training set: it keeps it, and does its work when you ask, so its work grows with the training set at every question. That is **lazy learning**. **Eager learning** does the work before any question: it turns the training set into a model, and then predicting is quick. Most models of this course are eager: linear regression keeps a weight for each feature and a bias, and no training example at all.

        **The nearest centroid.** The **nearest centroid classifier** is eager. It learns one thing: for every class, the mean of its training examples, its centroid, as in k-means. Here the classes are known, so there is nothing to repeat: one mean per class, and it is done. To classify a new example, it works out the distance to each centroid and gives back the class of the nearest. For the penguins, it keeps 3 centroids of 2 numbers: 6 numbers, against the 412 numbers and 206 species that k-NN keeps. And it works out 3 distances for each question, instead of 206.

        **Straight boundaries.** The points that are equally far from two centroids lie on a straight line: the line at right angles to the line between the two centroids, through the point halfway between them. So between two classes, the boundary of the nearest centroid is always straight, and with three classes, three straight pieces meet in one point. The boundary of k-NN can bend around the penguins; this one cannot. Here that costs: the boundary between Chinstraps and Gentoos lies at about 4,400 g, and lighter Gentoos fall on the wrong side of it. When every class is one round clump, a straight boundary is all you need; when a class lies in a bent band, or in two clumps, it is not.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **lazy learning** | learning that keeps the training examples and does its work when asked, such as k-NN | |
        | **eager learning** | learning that turns the training examples into a model before any question, such as linear regression | |
        | **nearest centroid classifier** | a classifier that learns the mean of each class, and gives a new example the class of the nearest mean | `nc_predict(names, means, x)` |
        """),
        L.watch_out("As for k-NN, the distance only works on standardized features: without standardizing, body mass in grams decides almost alone. Standardize with the mean and standard deviation of the training set, before you take the means of the classes."),
        mo.md(r"""
        The picture: the 206 training penguins by bill length and body mass, a circle for an Adelie, a triangle for a Chinstrap and a square for a Gentoo; here color and shape mean the species. Left, the three centroids, large and ringed, and the <span class="c-pred">boundaries</span> of the nearest centroid; right, the boundary of k-NN with k = 15, from [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
        """),
    ])
    return


@app.cell(hide_code=True)
def figure_compare():
    mo.Html(C.compare_picture({"nc": "Nearest centroid", "knn": "k-NN, k = 15", "x": "bill length (mm)", "y": "body mass (g)",
                               "alt": "Two charts of the 206 training penguins by bill length and body mass. Left: three straight blue lines meet near 44 mm and 4,330 g, and split the chart into the areas of the three centroids: Adelie at 39.1 mm and 3,723 g, Chinstrap at 48.6 mm and 3,686 g, Gentoo at 47.6 mm and 5,096 g. Right: the boundary of k-NN with k = 15 bends between the penguins."}))
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    A new penguin of your own: drag the dot over the plane, or move it with the arrow keys, bill length from left to right and body mass from bottom to top. The chart beside it shows the penguin, the three centroids, and a line to each; the line to the nearest is solid.
    """)
    return


@app.cell(hide_code=True)
def plane_penguin_cell():
    _names = ["Adelie", "Chinstrap", "Gentoo"]

    def _frame(_b, _m):
        _d = C.distances(_b, _m)
        _nc, _knn = C.nc_says(_b, _m), C.knn_says(_b, _m)
        _same = f"k-NN with k = 15 works out 206 distances, and agrees: {_knn}." if _knn == _nc else f"k-NN with k = 15 works out 206 distances, and says **{_knn}**."
        _said = (f"A penguin of {_b} mm and {C.grouped(_m, L.lang)} g. Its distance to each centroid: "
                 + ", ".join(f"{_n} {L.number(_v, 2)}" for _n, _v in zip(_names, _d))
                 + f". The nearest centroid says **{_nc}**. {_same}")
        _alt = (f"The three centroids and their straight boundaries, and a new penguin at {_b} mm and {_m} g with a line to each "
                f"centroid; the line to the {_nc} centroid is solid.")
        return C.plane_picture(_b, _m, {"title": "Your penguin and the centroids", "x": "bill length (mm)",
                                         "y": "body mass (g)", "alt": _alt}), _said

    _plane = L.plane("penguin", "bill length (mm)", "mass (g)", _frame, C.BILLS, C.MASSES, start=(40, 3600), role="",
                     task="Find a penguin for which the nearest centroid and k-NN disagree. Where on the plane does that happen?")
    _plane
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    They disagree only near a boundary, on this plane at three spots. A penguin of 44 mm and 4,000 g is a little nearer to the Chinstrap centroid than to the Adelie centroid, 0.95 against 0.98, while most of its 15 nearest penguins are Adelies. And at 36 mm and 5,600 g, and at 38 mm and 5,200 g, heavy penguins with short bills, the nearest centroid says Gentoo and k-NN says Adelie. Everywhere else both say the same; k-NN just worked out 206 distances for it, and the nearest centroid 3.

    ### Recipe

    To classify with the nearest centroid:

    1. Standardize the features with the mean and standard deviation of the training set, the new example too.
    2. Learn, once: for each class, the mean of its training examples, `X_train[y_train == name].mean(axis=0)`.
    3. For a new example, work out its distance to each centroid: `np.linalg.norm(means - x, axis=1)`.
    4. Give back the class of the nearest: `names[np.argmin(dists)]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the k-NN lesson, split and standardized in the same way. The cell learns the three centroids, asks about the second validation penguin, the one that k = 1 got wrong in the k-NN lesson, counts the mistakes on all 68, and compares what the two models keep. Run the cell.
    """)
    return


@app.cell
def example_centroid():
    import numpy as np
    import pandas as pd

    # 1. the penguins with all measurements, shuffled and split, standardized with the training set
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    _train, _val = _order[:206], _order[206:274]
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
    _X_train, _y_train = (_X[_train] - _mean) / _std, _y[_train]
    _X_val, _y_val = (_X[_val] - _mean) / _std, _y[_val]

    # 2. learn: the mean of each species
    _names = np.unique(_y_train)
    _means = []
    for _name in _names:
        _means.append(_X_train[_y_train == _name].mean(axis=0))
    _means = np.array(_means)
    print(_names)
    print(_means)

    # 3. and 4. the second validation penguin: its distance to each centroid, and the nearest
    _dists = np.linalg.norm(_means - _X_val[1], axis=1)
    print(_X[_val[1]], _y_val[1])
    print(_dists, _names[np.argmin(_dists)])

    # the mistakes on all 68 validation penguins, and what the two models keep
    _wrong = 0
    for _i in range(len(_X_val)):
        if _names[np.argmin(np.linalg.norm(_means - _X_val[_i], axis=1))] != _y_val[_i]:
            _wrong = _wrong + 1
    print(_wrong)
    print(_X_train.shape, _means.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The three centroids, standardized: the Adelies have short bills and are light, the Chinstraps have long bills and are light, and the Gentoos are heavy. The second validation penguin, 43.5 mm and 3,400 g, is a Chinstrap, but its distance to the Adelie centroid, 0.915, is a little smaller than to the Chinstrap centroid, 1.017: the nearest centroid says Adelie. k-NN with k = 5 got it right, from the Chinstraps among its neighbors. On all 68, the nearest centroid gets 7 wrong. It keeps 3 × 2 = 6 numbers, where k-NN keeps 206 × 2 = 412 and the species of all 206 penguins.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_centroids():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Learn the centroids

        `centroids(X_train, y_train)` should give back the names of the classes and their centroids, one row each, as step 2 of the recipe. Replace the `...` in the loop with one line. The check calls `centroids(X_train, y_train)` with the ten made-up penguins of the k-NN lesson, already standardized, and tries a set with three measurements and a set with two species.
        """),
        L.task("Put the mean of the training examples of each class, one mean for each feature, at the end of `means`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_centroids():
    box_centroids = L.code_box("centroids", lines=9)
    box_centroids
    return (box_centroids,)


@app.cell(hide_code=True)
def check_centroids(box_centroids):
    result_centroids = L.check("centroids", box_centroids.value)
    result_centroids
    return (result_centroids,)


@app.cell(hide_code=True)
def hints_centroids():
    L.hints(
        "centroids",
        [
            "Step 2 of the recipe, as in the worked example. Which rows of `X_train` belong to the class `name`?",
            "Compare `y_train` with `name` to get a mask that picks those rows. Then take the mean down the rows, one for each feature: which `axis` is that?",
            "`means.append(X_train[___].mean(axis=___))`",
        ],
        why="`X_train[y_train == name]` keeps the rows of one class; their mean down the rows is its centroid. For the four made-up Adelies, that is -0.475 and -0.75.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The nearest centroid

        `nc_predict(names, means, x)` gets the names and the centroids that `centroids` gives back, and one new example `x`, and gives back the class of the nearest centroid, as steps 3 and 4 of the recipe. Replace the two `...`. The check calls `nc_predict(names, means, np.array([0.2, -0.9]))` with the centroids of the ten made-up penguins: that should give `"Chinstrap"`. It also tries five other points.
        """),
        L.task("Make `nc_predict` give back the class whose centroid is nearest to `x`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("nc-predict", lines=6)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("nc-predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "nc-predict",
        [
            "Steps 3 and 4 of the recipe: one distance for each centroid, then the class of the smallest.",
            "`means - x` takes `x` from every row at once. `np.linalg.norm` with an `axis` gives one length for each row; which function gives the position of the smallest? Look that position up in `names`.",
            "`dists = np.linalg.norm(means - x, axis=___)` and `return names[np.___(dists)]`.",
        ],
        why="The distances from `np.array([0.2, -0.9])` to the three centroids are 0.691, 0.559 and 2.037; the smallest is at position 1, and `names[1]` is Chinstrap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clumps():
    mo.md(r"""
    ### Exercise 3. A class in two clumps
    """)
    return


@app.cell(hide_code=True)
def choice_clumps_cell():
    choice_clumps = L.choice(
        "two-clumps",
        question="The examples of class A lie in two clumps of the same size, far apart, one on the left and one on the right; those of class B lie in one clump in the middle, between them. What does the nearest centroid do?",
        options={
            "two": "It gives class A two centroids, one for each clump, and gets almost everything right",
            "one": "The centroid of A lies in one of its clumps, so it only gets the other clump wrong",
            "right": "The centroid of A lies between its clumps, close to B, so it gets many examples of both classes wrong",
        },
        correct="right",
        why={
            "two": "It learns one mean for each class, however its examples lie. See *The idea*.",
            "one": "The mean of two clumps of the same size lies halfway between them, not in one of them. See the quick check on the centroid.",
            "right": "Both centroids lie in the middle, close together, so the straight boundary between them cuts through B: one whole clump of A and about half of B fall on the wrong side. k-NN asks the neighbors and has no trouble here: that is what keeping every training example buys.",
        },
    )
    choice_clumps
    return (choice_clumps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The centroid is the one of [Groups without labels: k-means](/course/open/ml-kmeans?lang=en), now with the classes known, so one step is enough. The distance and the standardizing are those of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), but k-NN is lazy and keeps every example, and the nearest centroid is eager and keeps one mean per class. That makes it quick and small, with straight boundaries; k-NN pays with work at every question for boundaries that bend. The decision boundary of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) is straight too, but logistic regression finds its line by gradient descent on the loss; the nearest centroid simply puts it halfway between two means. In the next lesson, *The widest street: the maximum-margin classifier*, a straight boundary is chosen in another way: as far as possible from the nearest examples of both classes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_clumps, result_centroids, result_predict):
    L.wrap_up(result_centroids, result_predict, choice_clumps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
