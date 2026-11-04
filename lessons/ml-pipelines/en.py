# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-pipelines. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-pipelines", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en) and [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_inside():
    quick_inside = L.choice(
        "quick-inside",
        question="In 5-fold cross-validation, which examples should feature selection choose its columns from?",
        options={
            "all": "All examples, once, before the folds",
            "right": "In every fold, the examples that fold trains on",
            "val": "In every fold, the examples that fold is measured on",
        },
        correct="right",
        why={
            "all": "Then the examples each fold is measured on helped choose the columns: on pure noise that gave 0.96. See *The idea* in [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en).",
            "val": "Those are the examples the fold measures on: they must not help choose anything. See the recipe in [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en).",
            "right": "Whatever is learned from data is part of training, so every fold learns it again, from its own training examples.",
        },
    )
    quick_inside
    return (quick_inside,)


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="The distance between two penguins, from their bill length and depth in millimeters, flipper length in millimeters and body mass in grams, without standardizing. Which measurement decides it?",
        options={
            "all": "All four equally",
            "right": "Body mass",
            "bill": "Bill depth",
        },
        correct="right",
        why={
            "all": "Each difference counts in its own unit, and grams are in the thousands. See *The idea* in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "bill": "Bill depth differs by a few millimeters between penguins, body mass by hundreds of grams. See *The idea* in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "Body mass differs by hundreds of grams, the others by millimeters, so the distance is almost only body mass. That is why k-nearest neighbors standardizes first.",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) you standardized the penguins with the mean and standard deviation of the training set, and in [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en) you chose the columns again in every fold. By hand, every step that learns from data is a few lines in every fold, and one forgotten line leaks. Take the species of a penguin from its four measurements, with the 5 nearest neighbors, on 274 penguins: 5-fold cross-validation gives 0.759 without standardizing, and 0.985 with standardizing learned inside every fold. It would be good if the steps and the model were one thing, which every fold learns again by itself.

    **An everyday example.** Before every flight, a pilot goes through a checklist: the same steps, in the same order, every time. Nobody trusts their memory for that, however good they are, because one forgotten step is enough.
    """)
    return


@app.cell(hide_code=True)
def predict_scale_first():
    guess_scale_first = L.guess(
        "scale-first",
        question="Standardize all 274 penguins first, with their own mean and standard deviation, and then run the cross-validation with the model alone. Standardizing inside every fold gave 0.985. What do you think this gives?",
        options={
            "higher": "Much higher, as on the noise of the previous lesson",
            "right": "About the same",
            "lower": "Lower",
        },
        reveal="**The same, 0.985.** The mean and standard deviation of all 274 penguins hardly differ from those of the 219 or 220 a fold trains on, so this leak is tiny. Choosing 20 columns of 1000 was a huge one. A pipeline makes both impossible, so you never have to find out which kind you have.",
    )
    guess_scale_first
    return (guess_scale_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **pipeline** is a fixed chain of steps with a model at the end, which you fit and use as one model. Every step before the model learns something from the data and then changes the data with it: `StandardScaler()` learns the mean and standard deviation of every feature and standardizes with them, `SelectKBest(k=20)` learns which 20 features to keep and keeps them. In scikit-learn such a step has `fit`, which learns, and `transform`, which changes the data with what was learned.

        `pipe.fit(X, y)` goes through the steps in order: each step learns from the data it gets, changes that data, and passes it on, and the model learns from what arrives at the end. `pipe.predict(X_new)` goes through the same steps, but only changes the data, with what was learned from `X`, and then predicts. That is exactly the rule of the previous lesson: everything is learned from the training examples, and new examples are only changed with it.

        Cross-validation then needs no loop of yours. `cross_val_score(pipe, X, y, cv=KFold(5))` makes a fresh copy of the whole pipeline for every fold, fits it on that fold's training examples and measures it on the others, and gives the 5 scores. Every step is learned again in every fold, so no step can leak, however many steps there are.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **pipeline** | a fixed chain of steps that learn from data, with a model at the end, fitted and used as one model | `make_pipeline(StandardScaler(), KNeighborsClassifier())` |
        | `StandardScaler()` | standardizing as a step: `fit` learns the mean and standard deviation of every feature, `transform` standardizes with them | |
        | `SelectKBest(k=20)` | feature selection as a step: keeps the 20 features whose means differ most between the labels, compared with the spread within each label | |
        | `KFold(5)` | 5 folds, cut in order, as you did by hand; the examples are already shuffled | |
        | `cross_val_score(pipe, X, y, cv=KFold(5))` | cross-validation in one line: fits a fresh copy of `pipe` in every fold and gives each fold's score | |
        """),
        L.watch_out("A pipeline protects only the steps that are in it. Whatever you do to `X` before you give it to the pipeline, such as standardizing all of it, happens once, on all the examples, and is in every fold."),
        mo.md(r"""
        The picture shows k-nearest neighbors on the 274 training penguins with all four measurements. On the left, the accuracy on each of the 5 folds for one k: filled dots with `StandardScaler()` as the first step of the pipeline, open dots without. On the right, the mean over the folds for k from 1 to 51, the same two ways. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        k = int(k)
        scaled, raw = C.fold_scores(k, True), C.fold_scores(k, False)
        alt = (f"The first chart: the accuracy on each of the 5 folds with k = {k}: with standardizing between {L.number(min(scaled))} and {L.number(max(scaled))}, without between {L.number(min(raw))} and {L.number(max(raw))}. "
               + "The second chart: the mean accuracy for k from 1 to 51; with standardizing between 0.96 and 0.99, without between 0.74 and 0.78, for every k. "
               + f"With k = {k}: {L.number(C.mean(scaled), 3)} and {L.number(C.mean(raw), 3)}.")
        said = f"""k = {k}: with standardizing in the pipeline **{L.number(C.mean(scaled), 3)}**, without **{L.number(C.mean(raw), 3)}**."""
        return (C.picture(k, {"title": f"Every fold, k = {k}", "fold": "fold", "score": "accuracy",
                              "every": "Mean over the folds", "k": "k", "mean": "mean accuracy",
                              "scaled": "with StandardScaler", "raw": "without", "alt": alt}), said)

    explore_k = L.explore("k", "k", _frame, start=5, values=C.KS, role="hyper",
                          task="Move k from 1 to 51. Is there a k for which the model without standardizing does as well, in any fold?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Without standardizing, the mean stays between 0.74 and 0.78 whatever k is, and in every fold and for every k the pipeline with `StandardScaler()` does better: the neighbors were chosen almost by body mass alone. With it, the mean lies between 0.96 and 0.99. k = 5 and k = 7 come out best, 0.985, and differ by less than one penguin: the folds cannot tell them apart.

    ### Recipe

    To measure a model with steps that learn from data:

    1. Put the steps in order, the model last: `pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))`.
    2. Measure it with cross-validation, on the same folds for every model you compare: `scores = cross_val_score(pipe, X, y, cv=KFold(5))`, and take `scores.mean()`.
    3. Compare pipelines, or values of a hyperparameter, by that mean, as in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).
    4. Fit the pipeline you chose on all the training examples, `pipe.fit(X, y)`, and measure it once on the test set, with `pipe.predict(X_test)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The 342 penguins with all four measurements, shuffled with seed 0: the first 274 to train and choose on, the other 68 kept apart as the test set. The four measurements are the features, the species is the label. Steps 1 to 4 for 5-nearest neighbors, first without standardizing, then with `StandardScaler()` as the first step; `np.round` shows the 5 fold scores with 3 decimals, and `round(x, 3)` rounds one number to 3 decimals. At the end, `_pipe[0]` is the first step of the fitted pipeline, and its `mean_` is the mean it learned for every feature. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_pipeline():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:274]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _X, _y = np.array(_train[_features]), np.array(_train["species"])
    # 2. the same 5 folds for every model
    _cv = KFold(5)
    # without standardizing: the model alone
    _scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), _X, _y, cv=_cv)
    print(np.round(_scores, 3), round(_scores.mean(), 3))
    # 1. the steps in order, the model last
    _pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    _scores = cross_val_score(_pipe, _X, _y, cv=_cv)
    print(np.round(_scores, 3), round(_scores.mean(), 3))
    # 4. fitted on all 274 training penguins: what the first step learned
    _pipe.fit(_X, _y)
    print(np.round(_pipe[0].mean_, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Without standardizing, the folds score between 0.727 and 0.8, 0.759 on average; with `StandardScaler()` in the pipeline, three folds get every penguin right and the mean is 0.985. In each of the 5 fits, the scaler learned the mean and standard deviation of that fold's 219 or 220 training penguins only. Fitted on all 274, it learned a mean of 44 mm for bill length, 17.1 mm for bill depth, 201.4 mm for flipper length and 4235.1 g for body mass, and it standardizes every new penguin with those, as in step 4 of the recipe of [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_noise():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Feature selection as a step

        The box draws the table of noise of [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en) with seed 6: 50 people, 1000 columns of random numbers and a random label. Replace the two `...`: a pipeline with `SelectKBest(k=20)` as its step and `KNeighborsClassifier(n_neighbors=5)` as its model, and its scores with `cross_val_score`, on the folds `cv=KFold(5)`. The check looks at the mean on the last line, and also tries another table.
        """),
        L.task("Let a pipeline choose the 20 columns inside every fold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_noise():
    box_noise = L.code_box("noise-pipeline", lines=15)
    box_noise
    return (box_noise,)


@app.cell(hide_code=True)
def check_noise(box_noise):
    result_noise = L.check("noise-pipeline", box_noise.value)
    result_noise
    return (result_noise,)


@app.cell(hide_code=True)
def hints_noise():
    L.hints(
        "noise-pipeline",
        [
            "Steps 1 and 2 of the worked example, with `SelectKBest(k=20)` in place of `StandardScaler()`, and `X` and `y` in place of `_X` and `_y`.",
            "`make_pipeline` takes the steps in order, the model last. `cross_val_score` takes the pipeline, all of `X`, all of `y`, and the folds.",
            "`pipe = make_pipeline(SelectKBest(k=20), ___)` and `scores = cross_val_score(pipe, X, y, cv=___)`.",
        ],
        why="`SelectKBest(k=20)` is the first step, so in every fold `cross_val_score` fits it on that fold's 40 training people only, and the neighbors get only those 20 columns. The columns never saw the 10 people they are measured on.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Fit once, measure once

        Step 4 of the recipe. The box reads the penguins as in the worked example, with the 274 training penguins in `X_train` and `y_train` and the 68 test penguins in `X_test` and `y_test`. Replace the three `...`: the pipeline of `StandardScaler()` and `KNeighborsClassifier(n_neighbors=5)`, its fit on the training penguins, and in `accuracy` the share of the test penguins it puts in the right species.
        """),
        L.task("Fit the pipeline on the training penguins and measure it once on the test penguins."),
    ])
    return


@app.cell(hide_code=True)
def exercise_test():
    box_test = L.code_box("test-once", lines=18)
    box_test
    return (box_test,)


@app.cell(hide_code=True)
def check_test(box_test):
    result_test = L.check("test-once", box_test.value)
    result_test
    return (result_test,)


@app.cell(hide_code=True)
def hints_test():
    L.hints(
        "test-once",
        [
            "Step 1 and the end of the worked example, with `X_train` and `y_train`; then predict for the test penguins.",
            "`pipe.fit` takes the training penguins and their species. `pipe.predict(X_test)` gives a species for every test penguin: what do you compare it with, and how do you get the share that is the same?",
            "`pipe.fit(X_train, ___)` and `accuracy = np.mean(pipe.predict(___) == y_test)`.",
        ],
        why="`pipe.fit` learns the mean, the standard deviation and the neighbors from the training penguins only. `pipe.predict(X_test)` standardizes the test penguins with what it learned and predicts, so the test penguins are measured as new penguins would be.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scaled_first():
    mo.md(r"""
    ### Exercise 3. Standardized before the pipeline
    """)
    return


@app.cell(hide_code=True)
def choice_scaled_first_cell():
    choice_scaled_first = L.choice(
        "scaled-first",
        question="A colleague standardizes all 274 training penguins with their own mean and standard deviation, and then gives them to `cross_val_score` with a pipeline of only `KNeighborsClassifier(n_neighbors=5)`. Is that score honest?",
        options={
            "refit": "Yes: `cross_val_score` fits the model again in every fold",
            "right": "Not quite: the penguins each fold is measured on helped make the mean and standard deviation; `StandardScaler()` belongs in the pipeline",
            "labels": "Yes: standardizing does not look at the labels, so nothing can leak",
        },
        correct="right",
        why={
            "refit": "It fits the model again, but not the standardizing, which happened once, before the folds. Which penguins did the mean come from?",
            "labels": "Without the labels the leak is small: on the penguins the score is the same, 0.985. But the penguins each fold is measured on still helped make the mean and standard deviation, and with other steps or fewer examples that can matter. See the guess in *Why this matters*.",
            "right": "Here it makes no difference, 0.985 both ways, but you cannot know that in advance. With `StandardScaler()` as the first step, every fold learns its own mean and standard deviation from its own training penguins.",
        },
    )
    choice_scaled_first
    return (choice_scaled_first,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A pipeline is the rule of [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en) built into the code: every step learns from the training part only. Its first step here is the standardizing of [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) and [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), and `cross_val_score` does the folds of [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en) in one line, with the `fit` and `predict` of [The same model with scikit-learn](/course/open/ml-sklearn?lang=en). In the next lesson, *Project: which model is best? A fair comparison*, you put the models of this module side by side, each in its own pipeline, on the same folds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scaled_first, result_noise, result_test):
    L.wrap_up(result_noise, result_test, choice_scaled_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
