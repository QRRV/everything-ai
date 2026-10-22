# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-knn. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-knn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en) and [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="What is the distance between the points (1, 2) and (4, 6)?",
        options={
            "steps": "7",
            "right": "5",
            "square": "25",
        },
        correct="right",
        why={
            "steps": "That is 3 across plus 4 up, as if you walk along the streets. The distance is the straight line. See *The idea* in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "square": "That is 9 + 16: the square of the distance. Take its square root. See *The idea* in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "right": "3 across and 4 up: the square root of 9 + 16 is 5, the norm of the difference.",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def quick_check_validation():
    quick_validation = L.choice(
        "quick-validation",
        question="You fit a curve of each degree from 1 to 9. On which examples do you compare them to choose the degree?",
        options={
            "train": "The training set: the curves learned from it",
            "right": "The validation set",
            "test": "The test set",
        },
        correct="right",
        why={
            "train": "On the training set the most flexible curve always looks best. See *The idea* in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "test": "The test set measures the chosen model once, at the end; a set that helped choose gives a score that is too optimistic. See *Why this matters* in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "right": "Examples the curves did not train on, kept apart from the test set.",
        },
    )
    quick_validation
    return (quick_validation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The look-up model of [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en) gave a new house the price of the closest training house. The same idea can classify: give a new penguin the species of the most similar penguin you measured. But one neighbor can be a fluke. Ask several and let them vote, and one fluke is outvoted. How many you ask is a hyperparameter.

    **An everyday example.** To guess which language is spoken in a village you have never heard of, you look at the villages around it. The single nearest village can be an exception; the five nearest are a safer bet. Take every village in the country, and you only learn the most common language of the country.
    """)
    return


@app.cell(hide_code=True)
def predict_best_k():
    guess_best_k = L.guess(
        "best-k",
        question="k-NN asks the k training penguins most like a new one, and takes the species most of them have. Of 68 validation penguins, which k do you think gets the fewest wrong?",
        options={
            "one": "k = 1: the single most similar penguin",
            "fifteen": "k = 15",
            "all": "k = 205: all 206 training penguins but one",
        },
        reveal="**k = 15**, with 2 of the 68 wrong. With k = 1, k-NN gets 6 wrong: it follows every single penguin, even one that looks like another species. With k = 205 it gets 39 wrong: almost every training penguin votes, so it calls every penguin an Adelie, the most common species of the training set.",
    )
    guess_best_k
    return (guess_best_k,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **k-nearest neighbors**, or **k-NN**, classifies a new example in three steps: work out its distance to every training example, take the k with the smallest distance, its **nearest neighbors**, and let them vote. The class most of them have wins: **majority voting**. k-NN learns no parameters; it keeps the training examples and does its work when it predicts.

        With k = 1, k-NN is the look-up model for classes: every training example is its own nearest neighbor, so it gets them all right, and the quirks of single penguins decide for the penguins around them: it overfits. A larger k smooths that out, until with k as large as the training set it always says the most common class: it underfits. So you choose k on the validation set.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **k-nearest neighbors**, **k-NN** | | a classifier that gives a new example the class most of its k nearest training examples have | `knn_predict(X_train, y_train, x, k)` |
        | **nearest neighbors** | | the training examples with the smallest distance to the new example | `nearest` |
        | **majority voting** | | taking the answer that comes up most often among several | |
        | `np.linalg.norm(A - x, axis=1)` | | the distance from `x` to every row of `A`: one number per row | `dists` |
        | `np.argsort(a, kind="stable")` | arg sort | the positions of the elements of `a` from the smallest to the largest, equal ones in their order: for 3, 1 and 2, the positions 1, 2 and 0 | `nearest` |
        | `np.unique(a, return_counts=True)` | | the different values in `a`, sorted, and how often each one occurs | `names, counts` |
        """),
        L.watch_out("The distance adds up the squared differences in all features, so a feature with large numbers counts for more. Body mass is in grams, thousands; bill length in millimeters, tens. Without standardizing, the distance is almost only body mass. So standardize first, with the mean and standard deviation of the training set, as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en)."),
        mo.md(r"""
        The picture shows the 206 training penguins of the worked example below, by bill length and body mass: a circle is an Adelie, a triangle a Chinstrap and a square a Gentoo; here color and shape mean the species. The <span class="c-pred">blue line</span> is the decision boundary of k-NN: on either side of it, k-NN predicts a different species. The other chart counts, for every k, the <span class="c-loss">mistakes</span> on the 68 validation penguins, with k on a log scale. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        k = int(k)
        m = C.mistakes()[C.KS.index(k)]
        alt = (f"The first chart: the 206 training penguins by bill length and body mass, a color and shape per species. Adelies have short bills, Gentoos are heavy, Chinstraps have long bills and are light. With k = {k}, "
               + ("there is no blue line: k-NN calls every point an Adelie." if C.species_in_picture(k) == 1 else
                  "a blue line separates the areas where k-NN predicts each species; it bends around single penguins for a small k and is smoother for a larger one.")
               + f" The second chart: the validation mistakes for every k, 6 at k = 1, 2 at k = 15 and 39 at k = 205; at k = {k}, {m}.")
        said = f"""With k = {k}: <span class="c-loss ea-num">{m}</span> of the 68 validation penguins wrong."""
        if C.species_in_picture(k) == 1:
            said += " Every penguin is called an Adelie."
        return (C.picture(k, {"title": f"Decision boundary, k = {k}", "x": "bill length (mm)", "y": "body mass (g)",
                              "every": "Mistakes for every k", "mistakes": "mistakes (of 68)", "alt": alt}), said)

    explore_k = L.explore("k", "k", _frame, start=1, values=C.KS, role="hyper",
                          task="Move k from 1 to 205. Where is the boundary most ragged, and which k makes the fewest mistakes?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    At k = 1 the boundary is most ragged: it bends around single penguins that sit among another species, and k-NN gets 6 of the 68 wrong. It gets smoother as k grows, and k = 15 makes the fewest mistakes: 2. After that the Chinstraps, the smallest group with 44 of the 206, lose more and more votes: at k = 205 the boundary is gone, every penguin is an Adelie, and 39 are wrong.

    ### Recipe

    To classify a new example with k-NN:

    1. Standardize the features with the mean and standard deviation of the training set, the new example too.
    2. Work out the distance from the new example to every training example: `np.linalg.norm(X_train - x, axis=1)`.
    3. Take the positions of the k nearest: `np.argsort(dists, kind="stable")[:k]`.
    4. Count their votes, `np.unique(y_train[nearest], return_counts=True)`, and give back the class with the most.
    5. To choose k: do this for every validation example with each value of k, count the mistakes, and keep the k with the fewest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the picture: the 342 with all measurements, shuffled with a seed and split into 206 to train on, 68 to choose on and 68 for the test, which stay out of sight. Then steps 1 to 4 for the second validation penguin, with k = 5. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_knn():
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
    print(_X[_val[1]], _y_val[1])
    # 2. the distance from this validation penguin to all 206 training penguins
    _dists = np.linalg.norm(_X_train - _X_val[1], axis=1)
    # 3. the positions of the 5 nearest, and how far away they are
    _nearest = np.argsort(_dists, kind="stable")[:5]
    print(_dists[_nearest])
    # 4. their species, the votes, and the species with the most
    print(_y_train[_nearest])
    _names, _counts = np.unique(_y_train[_nearest], return_counts=True)
    print(_names, _counts)
    print(_names[np.argmax(_counts)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The second validation penguin, 43.5 mm and 3400 g, is a Chinstrap. Its nearest training penguin is an Adelie, so k = 1 would get it wrong; of its 5 nearest, 3 are Chinstraps, and the vote gets it right. `np.unique` sorts the species and counts them, 2 and 3; `np.argmax` gives position 1, and `_names[1]` is Chinstrap. On a tie it takes the first: the species first in the alphabet.

    The second and third neighbors were measured exactly the same, 42.5 mm and 3350 g, so they are exactly as far away; `kind="stable"` keeps such penguins in the order they have in the training set, so k-NN gives the same answer on every computer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own k-NN

        Finish `knn_predict(X_train, y_train, x, k)`. It gets the standardized training examples, one row each, their classes, one new example `x` and `k`, and gives back the class most of the k nearest have. Step 2 of the recipe, `dists`, is already there: replace the two `...` with steps 3 and 4. The check calls `knn_predict(X_train, y_train, np.array([0.2, -0.9]), 3)` with ten made-up penguins, already standardized: that should give `"Adelie"`. It also tries other points and other values of k.
        """),
        L.task("Make `knn_predict` give back the species most of the k nearest training penguins have."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("knn-predict", lines=8)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("knn-predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "knn-predict",
        [
            "Steps 3 and 4 of the recipe, as in the worked example, with `k` instead of 5.",
            "`np.argsort(dists, kind=\"stable\")` gives every position, the nearest first: which part of it do you keep? After `np.unique`, `np.argmax(counts)` is a position: in which of the two arrays do you look it up?",
            "`nearest = np.argsort(dists, kind=\"stable\")[___]` and `return names[___]`.",
        ],
        why="With k = 3, the three nearest to `np.array([0.2, -0.9])` are a Chinstrap and two Adelies: `names` is `['Adelie' 'Chinstrap']`, `counts` is `[2 1]`, `np.argmax(counts)` is 0, and `names[0]` is Adelie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Choose k

        Now with flipper length and body mass instead of bill length: `X_train` and `y_train` are the 206 training penguins, `X_val` and `y_val` the 68 validation penguins, split and standardized as in the worked example. In this box, `knn_predict` is ready: the course's version of your function from exercise 1. In the loop, add 1 to `wrong` for every validation penguin that k-NN gets wrong; then put the k with the fewest mistakes in `best`. Replace the two `...`. The last line is `best`, so the check sees it. The check also tries other values of k.
        """),
        L.task("Put the k with the fewest mistakes on the validation set in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose-k", lines=10)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose-k", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose-k",
        [
            "Step 5 of the recipe, with `np.argmin` as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "In the loop: if the prediction for `X_val[i]` is not `y_val[i]`, add 1 to `wrong`. After the loop, `np.argmin(mistakes)` gives a position: which k is at that position in `ks`?",
            "`if knn_predict(X_train, y_train, X_val[i], k) != ___:`, under it `wrong = wrong + 1`, and `best = ks[___]`.",
        ],
        why="The mistakes for the ten values of k are 15, 16, 19, 17, 13, 13, 11, 13, 17 and 39. The fewest are at position 6, and `ks[6]` is 41.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.md(r"""
    ### Exercise 3. k = 1 on the training set
    """)
    return


@app.cell(hide_code=True)
def choice_train_cell():
    choice_train = L.choice(
        "k1-train",
        question="With k = 1, bill length and body mass, k-NN gets 6 of the 68 validation penguins wrong. How many of its own 206 training penguins does it get wrong?",
        options={
            "six": "About as many as on the validation set: some 18 of the 206",
            "right": "None",
            "more": "More than with k = 15",
        },
        correct="right",
        why={
            "six": "That would hold for penguins it never saw. Which training penguin is nearest to a training penguin itself? See *The idea*.",
            "more": "Which training penguin is nearest to a training penguin itself, and how far away is it? See *The idea*.",
            "right": "Every training penguin is its own nearest neighbor, at distance 0, so k = 1 gives back its own species. With k = 15 it gets 17 of them wrong. As with the look-up model, a perfect score on the training set says nothing about new penguins.",
        },
    )
    choice_train
    return (choice_train,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The look-up model of [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en) was k-NN with k = 1 for a number, the price, instead of a class; now several neighbors vote, and k is a hyperparameter you choose on the validation set, as the degree in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en). The distance is the one from [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), and it only works on standardized features. In the next lesson, *Choosing more reliably: cross-validation*, you see that 68 validation penguins are not many to choose with, and how every penguin can take a turn in the validation set.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_predict, result_choose, choice_train):
    L.wrap_up(result_predict, result_choose, choice_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
