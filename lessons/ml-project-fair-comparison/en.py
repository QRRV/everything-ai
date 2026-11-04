# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-project-fair-comparison. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-project-fair-comparison", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en) and [Project: predict something real, end to end](/course/open/ml-project-tabular?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_refit():
    quick_refit = L.choice(
        "quick-refit",
        question="What does `cross_val_score(pipe, X, y, cv=KFold(5))` do with the steps of `pipe` in every fold?",
        options={
            "once": "It fits them once, on all of `X`, before the folds",
            "right": "It fits a fresh copy of them on that fold's training examples",
            "asis": "It uses them as they were fitted last",
        },
        correct="right",
        why={
            "once": "Then the examples each fold is measured on would have helped fit them. See *The idea* in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "asis": "Every fold starts from a fresh copy, so nothing a fold learned carries over to the next. See *The idea* in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "right": "Every step is learned again in every fold, from that fold's training examples, so no step can leak.",
        },
    )
    quick_refit
    return (quick_refit,)


@app.cell(hide_code=True)
def quick_check_dummy():
    quick_dummy = L.choice(
        "quick-dummy",
        question="What does `DummyClassifier()` predict for every new example?",
        options={
            "random": "A class drawn at random",
            "right": "The most common class of the training set",
            "near": "The class of the nearest training example",
        },
        correct="right",
        why={
            "random": "It does not draw anything: it learns one class and always says that one. See the recipe in [Project: predict something real, end to end](/course/open/ml-project-tabular?lang=en).",
            "near": "That is 1-nearest neighbors, a real model. See the recipe in [Project: predict something real, end to end](/course/open/ml-project-tabular?lang=en).",
            "right": "It is the baseline: what you get without looking at the features. A model has to beat it to be worth anything.",
        },
    )
    quick_dummy
    return (quick_dummy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    This module brought three models: [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en), [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en) and [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en). Before them came [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) and [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en). Which one do you take for a new question? Here: is a penguin male or female, from its four measurements? From the outside the two look alike, and 333 of the penguins have a known sex. A comparison only says something if every model gets the same chance: the same penguins, the same folds, the steps it needs, a baseline next to them, and a test set that decides nothing until the very end.

    **An everyday example.** A fair race has the same track, the same distance and the same start for every runner. And whoever wins one race by a hundredth of a second is not shown to be faster: run it again and the order may change.
    """)
    return


@app.cell(hide_code=True)
def predict_best():
    guess_best = L.guess(
        "best-model",
        question="Five models on the same five folds of 266 penguins: logistic regression, k-nearest neighbors, a tree of depth 3, a random forest and gradient boosting. Which do you think gets the highest cross-validation score for the sex of a penguin?",
        options={
            "boosting": "Gradient boosting",
            "forest": "The random forest",
            "right": "One of the simplest: logistic regression or k-nearest neighbors",
        },
        reveal="**k-nearest neighbors, 0.913**, then logistic regression 0.906, the forest 0.902, boosting 0.895 and the tree 0.864; always saying the most common sex gets 0.451. But with 53 penguins in a fold, one penguin is 0.019, and the first four lie within 0.02 of each other: one penguin per fold.",
    )
    guess_best
    return (guess_best,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        There is no new idea today: you measure several models the way the previous lessons measured one, and make sure the comparison is fair.

        1. **The same examples and the same folds.** One `cv = KFold(5)` for every model, so that every model trains and is measured on exactly the same penguins in every fold.
        2. **Every model with the steps it needs**, in its own pipeline: logistic regression and k-nearest neighbors standardize first; trees ask one feature at a time, so the scale does not matter to them.
        3. **A baseline**: `DummyClassifier()`, which always says the most common sex. Every model must beat it clearly.
        4. **Fold by fold.** In every fold two models saw the same training penguins and are measured on the same others, so the difference per fold says more than two means that are each a little lucky or unlucky.
        5. **Differences against the spread.** With 53 penguins in a fold, one penguin is 0.019. A lead smaller than how much a score moves from fold to fold, or between two ways of cutting the folds, is not a lead you can trust.
        6. **The test set once.** Choose the model and its hyperparameters with cross-validation alone, and give every model the same care. Then fit the one you chose on all 266 penguins and measure it once on the 67 test penguins.
        """),
        L.watch_out("The more models and settings you try, the more the best cross-validation score is luck too: it was chosen for being the highest. That is why the test set comes last, once, for one model, and why its number is the one you report."),
        mo.md(r"""
        The picture shows the five models, a row each. On the left, an open dot for the accuracy on each of the 5 folds and a filled dot for their mean. On the right, the mean for nine ways of cutting the same 266 penguins into five folds: way 1 is `KFold(5)`, in order, ways 2 to 9 shuffle first, with `KFold(5, shuffle=True, random_state=0)` to `random_state=7`. The filled dot is the way on the left. The baseline, around 0.5, is left out, so that the models can be told apart. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_cuts_cell():
    def _frame(cut):
        cut = int(cut)
        names = {"logistic": "logistic regression", "knn": "k-NN", "tree": "tree", "forest": "forest", "boosting": "boosting"}
        listed = ", ".join(f"{names[n]} {L.number(C.mean(C.MEASURED[n][cut - 1]), 3)}" for n in C.MODELS)
        most = max(C.right(n, cut) for n in C.MODELS)
        tied = [names[n] for n in C.MODELS if C.right(n, cut) == most]
        best = tied[0] if len(tied) == 1 else ", ".join(tied[:-1]) + " and " + tied[-1]
        alt = (f"The first chart: for each of the five models a row with its accuracy on the 5 folds of way {cut}, between {L.number(min(min(C.MEASURED[n][cut - 1]) for n in C.MODELS))} and {L.number(max(max(C.MEASURED[n][cut - 1]) for n in C.MODELS))}, and their mean. "
               + "The second chart: for each model the mean of all nine ways; k-NN lies between 0.91 and 0.92, logistic regression and the forest between 0.89 and 0.91, boosting between 0.88 and 0.90, the tree between 0.83 and 0.89. "
               + f"Way {cut}: {listed}.")
        said = f"""Way {cut}, the mean accuracy: {listed}. The most penguins right: **{best}**, {most} of the 266."""
        return (C.picture(cut, {"title": f"Every fold, way {cut}", "score": "accuracy", "every": "The mean, nine ways",
                                "mean": "mean accuracy", "alt": alt, **names}), said)

    explore_cuts = L.explore("cuts", "way of cutting", _frame, start=1, values=C.CUTS, role="",
                             task="Move through the nine ways of cutting the folds. Which model is on top each time, and by how much?")
    explore_cuts
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    k-NN gets the most penguins right in seven of the nine ways, one to four of the 266 more than the next model; in ways 2 and 3 it is level with others. Logistic regression and the forest lie between 0.89 and 0.91, boosting between 0.88 and 0.90, and the tree moves the most, from 0.83 to 0.89: which penguins it trains on changes its questions. Four penguins of 266 is 0.015, and the mean of k-NN itself moves that much from one way of cutting to another. So k-NN is a little ahead, and the four best are close.

    ### Recipe

    To compare models fairly:

    1. Keep the test set apart, and compare on the rest.
    2. Make one `cv = KFold(5)` for every model.
    3. Put every model in a pipeline with the steps it needs, and add `DummyClassifier()` as the baseline.
    4. Measure each with `cross_val_score(model, X, y, cv=cv)`; compare the means, and compare the folds one by one.
    5. Choose hyperparameters the same way, on the same folds, and give every model the same care.
    6. Choose one model. Fit it on all the comparison examples and measure it once on the test set; report that number next to the baseline.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The 333 penguins with a known sex, shuffled with seed 0: 266 to compare on, the other 67 kept apart. Steps 1 to 4 of the recipe for the six models, in a dictionary from name to model: the tree with depth 3, the others with scikit-learn's default settings. `HistGradientBoostingClassifier` is `HistGradientBoostingRegressor` for classes: again small trees one after another, each on what the trees before it still get wrong. For every model the mean over the folds, and with `.min()` and `.max()` the lowest and highest fold; `round(x, 3)` rounds a number to 3 decimals. `random_state=0` fixes the random draws of the tree, the forest and boosting. The comments with `#` say which step of the recipe each part is. Run the cell: the forest and boosting take a few seconds.
    """)
    return


@app.cell
def example_compare():
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["sex"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    # 1. 266 penguins to compare on; the other 67 are the test set
    _compare = _df.iloc[_order[:266]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _X, _y = np.array(_compare[_features]), np.array(_compare["sex"])
    # 2. one way of cutting the folds, for every model
    _cv = KFold(5)
    # 3. every model with the steps it needs, and the baseline
    _models = {
        "baseline": DummyClassifier(),
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "forest": RandomForestClassifier(random_state=0),
        "boosting": HistGradientBoostingClassifier(random_state=0),
    }
    # 4. the same folds for all: the mean, and the lowest and highest fold
    for _name, _model in _models.items():
        _scores = cross_val_score(_model, _X, _y, cv=_cv)
        print(_name, round(_scores.mean(), 3), round(_scores.min(), 3), round(_scores.max(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The baseline gets 0.451, less than half. In each fold it says the more common sex of that fold's training penguins, and the two sexes are almost equally common: fold 5 holds 68% males, so the other four folds hold more females, and it says female for all of fold 5. Every model beats it by far. The best four lie between 0.895 and 0.913, and each of their folds lies somewhere between 0.83 and 0.98. The tree, with its three questions, gets 0.864, and its worst fold 0.774. Which of the best four is best, these numbers cannot say; the next exercise looks fold by fold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Fold by fold

        The box reads the 266 penguins as in the worked example, in `X` and `y`, and makes `cv = KFold(5)`. Replace the two `...`: the 5 fold scores of logistic regression and of k-nearest neighbors with k = 5, each in its pipeline with `StandardScaler()` first, on the folds `cv`. The last line gives the difference per fold.
        """),
        L.task("Compare logistic regression and k-NN fold by fold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("fold-by-fold", lines=21)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("fold-by-fold", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "fold-by-fold",
        [
            "Two lines like step 4 of the worked example, one for each model, with `X`, `y` and `cv`.",
            "`cross_val_score` takes the model, here a pipeline, then `X`, `y` and the folds. The pipelines are those of the worked example.",
            "`logistic = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=___)`, and the same for `KNeighborsClassifier(n_neighbors=5)`.",
        ],
        why="With `cv=cv` both models are measured on exactly the same penguins in each fold, so `knn - logistic` says per fold how many penguins more or fewer k-NN gets right.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Choose k, then test once

        Steps 5 and 6 of the recipe for k-nearest neighbors. The box has the same `X`, `y` and `cv`, and the 67 test penguins in `X_test` and `y_test`. The loop tries nine values of k and keeps in `best` the one with the highest mean cross-validation score. Replace the three `...`: that score for this `k`, the pipeline with `best`, and in `accuracy` its share right on the test penguins. In a full comparison you would tune the depth of the tree and the settings of the other models in the same way, on the same folds; here you do it for k only.
        """),
        L.task("Choose k with cross-validation, and measure the pipeline with that k once on the test set."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose-k", lines=28)
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
            "The score is the line of exercise 1 for k-NN, with `n_neighbors=k`, and its mean. The test is exercise 2 of [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "`cross_val_score` gives 5 scores; `if score > top` needs one number. After the loop, `best` is the k to use; the test penguins come in only on the last line with `...`.",
            "`score = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X, y, cv=cv).___()`, `pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=___))` and `accuracy = np.mean(pipe.predict(X_test) == ___)`.",
        ],
        why="Every k is judged on the same folds of the 266 penguins, and the test penguins play no part in the choice. Only the pipeline with the chosen k sees them, once, so its accuracy there is a fair measurement of a model chosen this way.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test_pick():
    mo.md(r"""
    ### Exercise 3. Choosing on the test set?
    """)
    return


@app.cell(hide_code=True)
def choice_test_pick_cell():
    choice_test_pick = L.choice(
        "test-pick",
        question="A colleague measures all five models of the worked example once on the 67 test penguins: k-NN with k = 5, the tree and boosting get 0.866, logistic regression 0.851 and the forest 0.821. They want to take the tree now: it ties for first and is the simplest. What do you say?",
        options={
            "agree": "Good idea: the test set is the most honest measurement there is",
            "right": "No: one penguin is 0.015 here, so these differ by one to three penguins, and a test set you choose with is no longer an honest measurement",
            "forest": "Take any of them, but drop the forest: it clearly does worst on new penguins",
        },
        correct="right",
        why={
            "agree": "It is honest for one model chosen beforehand. Choosing the best of five on it makes that score the best of five tries, a little too good, like the 0.932 of exercise 2.",
            "forest": "0.821 against 0.866 is three penguins of 67, and in cross-validation the forest was among the best four. Would you drop it for three penguins?",
            "right": "Choose with cross-validation, on the 266 penguins, and use the test set once, for the one model you chose. Its number is then a fair measurement; the five numbers here mostly show how much 67 penguins can move.",
        },
    )
    choice_test_pick
    return (choice_test_pick,)


@app.cell(hide_code=True)
def text_report():
    mo.md(r"""
    ### Exercise 4. The report
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Write down for the penguin researchers, in three or four sentences, which model you would use to predict the sex of a penguin, and why: with the baseline, the cross-validation scores, how large the differences are, and the number on the test set.",
        model_answer="We compared five models and a baseline on the same five folds of 266 penguins. Always saying the most common sex gets 0.45; k-nearest neighbors on the standardized measurements gets 0.91, logistic regression and the random forest 0.90 to 0.91, boosting 0.89 and a small tree 0.86. The best four differ by about one penguin per fold, less than one model's folds differ from each other, so k-NN is only a little ahead. We chose k = 9 with cross-validation, and on the 67 test penguins it gets 0.88, which is the number to expect for new penguins.",
        points=[
            "the baseline next to the models: about 0.45 against 0.86 to 0.91",
            "that the best four are close: about one penguin per fold",
            "how the model and its k were chosen: with cross-validation, without the test set",
            "the number on the test set, measured once: 0.88",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** You measured the models of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en), [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en) and [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en) side by side, each in its pipeline from [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en), next to the baseline of [Always beat a simple baseline](/course/open/ml-baselines?lang=en). In the research branch, *Fair comparisons* asks the same of a paper: did the new method and the old ones get the same data, the same care and a baseline? In the next lesson, *Groups without labels: k-means*, there are no labels at all: the penguins have to show their groups by themselves.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_pick, explain_report, result_choose, result_folds):
    L.wrap_up(result_folds, result_choose, choice_test_pick, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
