# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-data-leakage. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-data-leakage", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en) and [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fold():
    quick_fold = L.choice(
        "quick-fold",
        question="5-fold cross-validation on 50 examples, 10 in each fold. In the round of fold 3, which examples does the model train on?",
        options={
            "all": "All 50",
            "right": "The 40 examples of the other four folds",
            "fold": "The 10 examples of fold 3",
        },
        correct="right",
        why={
            "all": "Then it would also train on the 10 examples it is measured on in this round. See the recipe in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "fold": "Those are the examples it is measured on in this round. See the recipe in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "right": "It trains on the other folds and is measured on fold 3, so every example is checked once by a model that did not train on it.",
        },
    )
    quick_fold
    return (quick_fold,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="A model trained on standardized features. With which mean and standard deviation do you standardize a new example?",
        options={
            "own": "Its own",
            "right": "Those of the training set",
            "all": "Those of the training set and the new examples together",
        },
        correct="right",
        why={
            "own": "One example has no spread of its own, and the model learned its weights for the training set's scale. See step 4 of the recipe in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "all": "The model learned its weights on the training set's scale, and that scale should not change with the examples you happen to predict. See step 4 of the recipe in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "right": "The mean and standard deviation are learned from the training set, like the weights, and stay as they are for every new example.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Cross-validation checks every example once, with a model that did not train on it. That only works if nothing about those examples got into the model in another way. Take 50 people and give each of them 1000 numbers drawn at random between 0 and 1, and a label of 0 or 1, also drawn at random, half of each. Nothing in those numbers can predict the label: on new people, any model is right about half the time. With 1000 columns and only 50 people, it seems wise to keep the most useful columns first: the 20 whose mean differs most between the people with label 1 and the people with label 0. Then 5-fold cross-validation, with the 5 nearest neighbors of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) on those 20 columns.

    **An everyday example.** Around 1904 in Berlin, a horse called Clever Hans seemed to do sums: asked what 3 plus 2 is, he tapped his hoof five times. At the end of 1904 the psychologist Oskar Pfungst found that Hans got it right only when the person asking knew the answer and Hans could see them; in 1907 he described it in a book. Hans stopped tapping when they relaxed, without them noticing that they gave it away. The answer leaked through the one asking the question.
    """)
    return


@app.cell(hide_code=True)
def predict_noise():
    guess_noise = L.guess(
        "noise-score",
        question="On these random numbers, what share of the 50 people does the cross-validation get right?",
        options={
            "half": "About 0.5: there is nothing to learn",
            "some": "About 0.7",
            "right": "Almost all of them",
        },
        reveal="**0.96**, 48 of the 50 people. Yet the same 20 columns and 5 neighbors get 52.1% of 1000 new people right: guessing. The 20 columns were chosen with the labels of all 50 people, so also with the labels of the 10 people each fold is measured on. The answer leaked in, as with Hans.",
    )
    guess_noise
    return (guess_noise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Data leakage** is information getting into the training of a model that it will not have when it predicts for real: most often, information about the examples it is measured on. The score then measures something that new examples do not bring along, and comes out too good. It happens in three common ways.

        1. **A choice made with the examples you measure on.** Here it is **feature selection**, choosing which features a model gets: the 20 columns were chosen with the labels of all 50 people. The same goes for the mean and standard deviation for standardizing, and for the values that fill in gaps: whatever is learned from data is part of training, so it is learned from the training examples only, and in cross-validation again in every fold. Hyperparameters too: the score you chose them with comes out a little too good, which is why the test set stays apart until the very end.
        2. **Target leakage**: a feature that is only known after the label, or is made from it, such as whether a patient got the medicine for the very disease the model is to predict.
        3. **The same example, or the same person, on both sides**: a row that is in the table twice, or several photos of one patient, some to train on and some to measure on. A **group split** keeps every group whole, on one side: split by patient, not by photo.

        Why does it work so well on noise? Among 1000 columns of random numbers, some differ between the two labels by chance alone, and the best 20 differ the most. Chosen on all 50 people, those columns also differ between the labels of the 10 people a fold is measured on, by that same chance, and the neighbors find it. Chosen on the 40 training people of the fold, they say nothing about the other 10.

        Sayash Kapoor and Arvind Narayanan (2023) gathered what others had found: data leakage in 294 published papers, in 17 fields of science that use machine learning.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **data leakage** | information in training that the model will not have when it predicts for real, such as information about the examples it is measured on | |
        | **feature selection** | choosing which features a model gets | `select(X[train], y[train], 20)` |
        | **target leakage** | a feature that is only known after the label, or is made from it | |
        | `KNeighborsClassifier(n_neighbors=5)` | scikit-learn's k-nearest neighbors with k = 5: `fit` keeps the training examples, and `predict` gives each new example the label most of its 5 nearest neighbors have | |
        | **group split** | a split that keeps each group of examples that belong together, such as the photos of one patient, on one side | `GroupKFold(n_splits=5)` |
        """),
        L.watch_out("Cross-validation does not protect you by itself. It checks the model that trains in each fold, not the steps before the split: a step done once on all the examples is inside every fold. Shuffling or more folds do not help either."),
        mo.md(r"""
        The picture shows tables of noise like this one, with 10 to 10000 columns. On the left, for one table: the 20 kept columns, largest gap first. A filled dot is the gap between the means of the two labels among its 50 people, what the selection saw; an open grey dot is the gap in the same column among 1000 new people. On the right, for ten such tables, the score of cross-validation with the columns chosen on all 50 people (filled dots) and inside each fold (open dots); the dashed line is guessing, 0.5, and the large dots are the table on the left. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_columns_cell():
    def _frame(n):
        n = int(n)
        leaky, honest = C.MEASURED[True][n][C.EXAMPLE], C.MEASURED[False][n][C.EXAMPLE]
        seen, new = C.chosen_gaps(n)
        alt = (f"The first chart: the {min(n, C.KEEP)} kept columns of a table of noise with {n} columns, largest gap first. Among its 50 people the gaps between the means of the two labels run from {L.number(seen[0])} down to {L.number(seen[-1])}; among 1000 new people they are all below {L.number(max(new))}. "
               + "The second chart: for 10 to 10000 columns, the cross-validation scores of ten tables. With the columns chosen on all 50 people they climb from about 0.5 to between 0.92 and 0.98; with the columns chosen inside each fold they stay between 0.3 and 0.66, around guessing. "
               + f"With {n} columns, this table scores {L.number(leaky)} and {L.number(honest)}.")
        said = f"""{n} columns: chosen on all 50 people, cross-validation says **{L.number(leaky)}**; chosen inside each fold, **{L.number(honest)}**. The kept columns differ by {L.number(seen.mean(), 3)} on average between the labels of the 50, and by {L.number(new.mean(), 3)} among new people."""
        return (C.picture(n, {"title": "The 20 kept columns", "rank": "kept column, largest gap first", "gap": "gap between the means",
                              "every": "Ten tables of noise", "columns": "columns", "score": "cross-validation score",
                              "leaky": "chosen on all 50", "honest": "chosen inside each fold", "alt": alt}), said)

    explore_columns = L.explore("columns", "columns", _frame, start=20, values=C.COLUMNS, role="",
                                task="Move the number of columns from 10 to 10000. What happens to the gaps of the kept columns, and to the two scores?")
    explore_columns
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With 10 or 20 columns the selection keeps all of them, so both scores are the same, around 0.5. The more columns there are to choose from, the larger the gaps of the best 20 by chance: on average 0.075 among 20 columns, 0.21 among 1000 and 0.28 among 10000, while among new people they stay around 0.01. The score with the columns chosen on all 50 people climbs with them, to between 0.82 and 1 from 1000 columns on. Chosen inside each fold, the scores stay between 0.3 and 0.66: guessing, with the spread that 50 people give.

    ### Recipe

    To measure a model honestly with cross-validation:

    1. Split first: cut the examples into folds, and keep the test set apart, before anything is learned from them.
    2. In every fold, learn every step that learns from data from the training examples of that fold only: the features you keep, the mean and standard deviation for standardizing, the values that fill in gaps.
    3. Apply those steps, as they were learned, to the examples the fold is measured on, and measure.
    4. For every feature, ask: would I know this at the moment I predict? If not, leave it out.
    5. When examples belong together, such as the photos of one patient, split by group: scikit-learn's `GroupKFold(n_splits=5)` gives the folds, with `split(X, y, groups)`.
    6. Does a score look too good to be true? Measure it on examples that neither the model nor its steps have seen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The table of noise of *Why this matters*: 50 people and 1000 columns, drawn with seed 6, and 1000 new people from the same generator. `_select` keeps the k columns with the largest gap between the means of the two labels; `np.argsort` gives the positions of the gaps from small to large, so the last k are the positions of the largest. The cross-validation is that of [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en), counting the people it gets right, with scikit-learn's `KNeighborsClassifier` for the 5 nearest neighbors you counted by hand in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), but the columns are chosen once, before the folds, with all 50 people: it breaks step 2 of the recipe. Then the same steps on all 50 people, measured on the new people as in step 6. The comments with `#` say what each part does. Run the cell.
    """)
    return


@app.cell
def example_leak():
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    _rng = np.random.default_rng(6)
    _X = _rng.random(size=(50, 1000))
    _y = _rng.permutation([0, 1] * 25)
    _X_new = _rng.random(size=(1000, 1000))
    _y_new = _rng.permutation([0, 1] * 500)


    def _select(X, y, k):
        # the k columns whose mean differs most between label 1 and label 0
        gap = abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
        return np.argsort(gap)[-k:]


    # the leak: the columns are chosen once, with the labels of all 50 people
    _cols = _select(_X, _y, 20)
    _rows = np.arange(50)
    _right = 0
    for _f in range(5):
        _val = _rows[10 * _f:10 * (_f + 1)]
        _train = np.concatenate([_rows[:10 * _f], _rows[10 * (_f + 1):]])
        _model = KNeighborsClassifier(n_neighbors=5).fit(_X[_train][:, _cols], _y[_train])
        _right = _right + np.sum(_model.predict(_X[_val][:, _cols]) == _y[_val])
    print(_right / 50)
    # step 6: the same steps on all 50 people, measured on 1000 new people
    _model = KNeighborsClassifier(n_neighbors=5).fit(_X[:, _cols], _y)
    print(np.mean(_model.predict(_X_new[:, _cols]) == _y_new))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Cross-validation says 0.96; the new people get 0.521, a hair above guessing. `_X[_train][:, _cols]` first takes the rows of the training people, then the kept columns: `[:, _cols]` keeps the columns whose positions are in `_cols`, as `X[rows]` keeps rows. Every model in the folds trains only on its own 40 people, as cross-validation should; the leak is in the line before the loop. With the columns chosen inside each fold, this table scores 0.52. In exercise 1 you write that for another table.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Inside the folds

        The box draws another table of noise, with seed 2, and has the cross-validation of the worked example, with `select` as a function. Only the columns are left: replace the `...` in the loop so that every fold chooses its own 20 columns, from its own training people. The check runs your code and looks at the score on the last line; it also tries another table.
        """),
        L.task("Choose the columns inside every fold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("inside-folds", lines=21)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("inside-folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "inside-folds",
        [
            "The worked example chose the columns once, before the loop, with `_X` and `_y`. Here every fold chooses them again, inside the loop: with which rows of `X`, and which labels?",
            "`train` holds the positions of the 40 people this fold trains on. `X[train]` are their rows; which labels belong with them?",
            "`cols = select(X[___], y[___], 20)`.",
        ],
        why="Inside the loop, `X[train]` and `y[train]` are the 40 people the fold trains on. Columns chosen from them know nothing about the 10 people of `val`, so the score on those 10 is honest, as it will be for new people.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_known():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Known on 1 September?

        A school wants to predict on 1 September which students will pass their final exam in June, so that it can help the others in time. For the students of earlier years it has the columns below, and whether they passed. Which may be a feature, and which leak?
        """),
        L.task("Put every column in its group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_known():
    sort_known = L.sort("known-then")
    sort_known
    return (sort_known,)


@app.cell(hide_code=True)
def check_known(sort_known):
    result_known = L.check("known-then", sort_known)
    result_known
    return (result_known,)


@app.cell(hide_code=True)
def text_exercise_patient():
    mo.md(r"""
    ### Exercise 3. Ten photos of every patient
    """)
    return


@app.cell(hide_code=True)
def choice_patient_cell():
    choice_patient = L.choice(
        "same-patient",
        question="A model learns to spot a skin disease on photos: 1000 photos of 100 patients, ten of each. The photos are split at random, 800 to train on and 200 to test on, and the model gets 95% of the test photos right. In another hospital it gets 70% right. What went wrong first?",
        options={
            "few": "200 test photos are too few to say anything",
            "right": "Photos of the same patients are on both sides: the model can recognize the patient instead of the disease",
            "folds": "They should have used cross-validation instead of one test set",
        },
        correct="right",
        why={
            "few": "On 200 photos, 95% against 70% is far more than chance. What do the test photos have in common with the training photos?",
            "folds": "Cross-validation with the photos split at random has the same problem in every fold. Whose photos are in each fold?",
            "right": "With ten photos of every patient, almost every test photo has photos of the same patient in training: the same skin, the same light. The test measured patients the model had already seen. A group split by patient, such as `GroupKFold`, keeps all photos of a patient on one side.",
        },
    )
    choice_patient
    return (choice_patient,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The leak hid in the folds of [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en). The rule is the one of [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) for the mean and standard deviation, now for every step that learns from data; and as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), a score is only honest about the examples nothing was chosen with. Language models have a leak of their own: test questions that are on the internet end up in their training text; *Keeping test data out of training data* is about that. In the next lesson, *Every step in every fold: pipelines*, scikit-learn repeats every step that learns from data inside each fold for you, without a loop of your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_patient, result_folds, result_known):
    L.wrap_up(result_folds, result_known, choice_patient)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
