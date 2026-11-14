# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-grid-random-search. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-grid-random-search", "en")
    C = L.checks   # the tries of the grid and of the random search on the machine runs, measured once, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en) and one about [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hyper():
    quick_hyper = L.choice(
        "quick-hyper",
        question="k-NN with `n_neighbors=15`. Who sets the 15?",
        options={
            "fit": "`fit` learns it from the training examples",
            "right": "You do, before training: it is a hyperparameter",
            "data": "It follows from the number of training examples",
        },
        correct="right",
        why={
            "fit": "`fit` of k-NN keeps the training examples and learns no k: it uses the k you gave it. A number you choose before training is a hyperparameter. See *The idea* in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).",
            "data": "Nothing in the data fixes k. A good k depends on the data, but you choose it, before training. See *The idea* in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).",
            "right": "You choose it, and the training loss cannot choose it for you. Today you search for a good k and a second setting at the same time.",
        },
    )
    quick_hyper
    return (quick_hyper,)


@app.cell(hide_code=True)
def quick_check_folds():
    quick_folds = L.choice(
        "quick-folds",
        question="`scores = cross_val_score(pipe, X, y, cv=KFold(5))`. Which number do you compare between two settings?",
        options={
            "max": "`scores.max()`, the best fold",
            "first": "`scores[0]`, the first fold",
            "right": "`scores.mean()`, the mean of the 5 folds",
        },
        correct="right",
        why={
            "max": "The best fold is the luckiest of the five. The mean uses every example once for validation. See the recipe of [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "first": "One fold is one validation set again, with all its chance. The mean of the 5 is what cross-validation buys you. See the recipe of [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "right": "The mean of the 5 folds. Today you work it out for many settings and keep the best.",
        },
    )
    quick_folds
    return (quick_folds,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Most models have more than one setting you choose yourself. k-NN has its k and the way it measures a distance; a random forest has the number of trees, how deep they grow and how many features each split may look at. Trying one setting at a time misses how they work together, and trying every combination of many values soon takes too long: 5 values of 4 settings are 625 combinations, and every one is a whole cross-validation. With a limited number of tries, which combinations do you try?

    **An everyday example.** You test a new cake recipe and can bake 9 cakes. The oven temperature matters a lot, the shelf in the oven hardly. Bake 3 temperatures on each of 3 shelves, and you have tried only 3 temperatures. Bake 9 cakes at 9 different temperatures, each on any shelf, and you have tried 9.
    """)
    return


@app.cell(hide_code=True)
def predict_values_cell():
    guess_values = L.guess(
        "values-guess",
        question="A search tries 3 values of k with each of 3 values of p: 9 tries. How many different values of k does it test?",
        options={
            "nine": "9",
            "six": "6",
            "right": "3",
        },
        reveal="**3.** Every value of k is tried 3 times, once with each p. If p hardly matters, as for the machines in this lesson, 6 of the 9 tries tell you almost nothing new about k. 9 tries drawn at random test up to 9 values of k.",
    )
    guess_values
    return (guess_values,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Two settings.** k-NN from [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) has two settings here: k, the number of neighbors, and p, how it measures a distance. With p = 2 it is the straight-line distance of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en): the square root of the sum of the squared differences. With p = 1 it adds up the sizes of the differences without squaring them. Values between 1 and 2 lie between the two, and above 2 the largest difference counts more and more. The **search space** is every combination you allow: here k from 1 to 300 and p from 1 to 4. Every try is a cross-validation as in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en), on the 7,000 training runs of [When one class is rare](/course/open/ml-imbalanced?lang=en), and its score is the mean AUC of [Every threshold at once](/course/open/ml-roc-auc?lang=en) over the 5 folds, because failures are rare.

        **Grid search.** Write down a few values of every setting and try every combination: 3 values of k and 3 of p give $3 \cdot 3 = 9$ tries. On a plane with k across and p up, the tries form a **grid** of rows and columns.

        **Random search.** Draw every try at random from the search space, k and p anew each time. With 9 tries you test up to 9 values of k (k is a whole number, so a value can come up twice) and 9 of p.

        **A log scale for k.** From k = 1 to 2 the model changes a lot; from 201 to 202 hardly. So draw k evenly on a log scale, as in [Log scales and power laws](/course/open/math-log-scales?lang=en): draw its logarithm evenly between $\log_{10} 1 = 0$ and $\log_{10} 300 = 2.48$, and take 10 to that power. Then 1 to 10 and 10 to 100 get about as many tries; the grid spreads its values of k in the same way. `rng.uniform(low, high, n)` draws n numbers evenly between `low` and `high`, as `rng.random` of [Random numbers and seeds](/course/open/np-random?lang=en) does between 0 and 1. You write the draw in exercise 2.

        **One setting matters.** James Bergstra and Yoshua Bengio (2012) noticed that usually only a few of the settings matter much. Here that is k: the AUC rises from 0.716 at k = 1 to 0.952 near k = 60, and then slowly falls. p hardly matters: at every k of the grids the values of p differ by at most 0.01 in AUC, and k moves it from 0.716 to 0.952. A grid with 3 values per setting tests only 3 values of k, however many values of p it adds.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **search space** | every combination of settings a search may try: a range or a list of values for every setting | |
        | **grid search** | trying every combination of a few values chosen for every setting | two loops, one inside the other |
        | **random search** | trying combinations drawn at random from the search space | `rng.uniform(low, high, n)` |
        | `p` of k-NN | how k-NN measures a distance: 2 is the straight line, 1 the sum of the sizes of the differences, and above 2 the largest difference counts more and more | `KNeighborsClassifier(n_neighbors=k, p=p)` |
        | `scoring="roc_auc"` | lets `cross_val_score` give the AUC of every fold instead of the share it gets right | `cross_val_score(pipe, X, y, cv=KFold(5), scoring="roc_auc")` |
        """),
        L.watch_out("Search on the training part only. The test set measures the winner once, at the end; if it helps you choose, its score is no longer an honest estimate for new runs."),
        mo.md(r"""
        The slider sets the number of tries, the same for both searches. Left, every try on the plane of k and p: open dots are the grid, filled dots the random search, and a ring marks the best try of each. Right, the AUC of every try against its k; the gray line is the AUC with p = 2 for every k. k is on a log scale. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_budget_cell():
    def _frame(n):
        said = C.budget_said(n, L.lang)
        alt = ("Left: the tries of a grid, in rows and columns, and of a random search, spread unevenly, on a plane with k "
               "on a log scale across and p up. Right: the AUC of every try against its k, beside a line that rises "
               "from 0.72 at k = 1 to 0.95 near k = 60 and then slowly falls. " + said)
        return (C.budget_picture(n, {"plane": "Tries", "score": "AUC of every try", "p": "p (power)", "alt": alt}), said)

    explore_budget = L.explore("tries", "number of tries", _frame, start=9, values=C.BUDGETS, role="hyper",
                               task="Go from 4 to 25 tries. Which search finds the higher AUC, and how many values of k does each test?")
    explore_budget
    return


@app.cell(hide_code=True)
def text_after_budget():
    mo.md(r"""
    With 4 tries the grid wins: its k = 72 happens to lie near the top of the gray line. From 9 tries on, the random search finds 0.952 at k = 64, and the grid gets no further than 0.947: its values of k lie beside the top, such as 116, 147 and 54. More tries in a grid do not always help: 16 tries find less than 9. The random search tests 4, 9, 15 and 23 different values of k. One draw can be lucky: the course ran 20 random searches of 9 tries, each with its own seed, and their best AUC lay between 0.939 and 0.952; 12 of the 20 beat the grid of 9.

    The gain is small here. Measured once on the 3,000 test runs, the winner of the random search of 9 tries has AUC 0.960, and that of the grid 0.962: a difference of a few thousandths in cross-validation can be chance. Random search pays off with more settings: a grid of 3 values of 4 settings takes 81 tries and still tests only 3 values of each; 81 random tries draw 81 values of each.

    ### Recipe

    To search the settings of a model:

    1. Choose the settings and a range for every one. Use a log scale for a setting that works by factors, such as k, a learning rate or the strength of regularization.
    2. Choose a budget: how many tries you can afford, each a whole cross-validation.
    3. Draw the tries at random from the ranges. A grid is fine for one or two settings with a few values.
    4. Score every try with the mean over the same folds, with a metric that fits the problem.
    5. If the best try lies at the edge of a range, widen that range and search again.
    6. Fit the winner on the whole training part and measure it once on the test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads the runs and splits them as in [When one class is rare](/course/open/ml-imbalanced?lang=en). `_score(k, p)` gives the mean AUC of 5 folds for k-NN with those settings, in a pipeline with scaling as in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en). The cell makes the grid of 9 tries with two loops, draws 9 tries at random, scores all 18 and prints the best of each search with its AUC. 18 cross-validations take about ten seconds. Run the cell.
    """)
    return


@app.cell
def example_search():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _numbers = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    _X = np.array(_df[_numbers])
    _y = np.array(_df["Machine failure"])
    _order = np.random.default_rng(0).permutation(len(_y))
    _train = _order[3000:]                                 # the 7,000 training runs; the 3,000 test runs wait

    def _score(k, p):
        _pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, p=p))
        return cross_val_score(_pipe, _X[_train], _y[_train], cv=KFold(5), scoring="roc_auc").mean()

    _grid = []
    for _k in [3, 17, 116]:
        for _p in [1.5, 2.5, 3.5]:
            _grid.append((_k, _p))

    _rng = np.random.default_rng(0)
    _random = []
    for _ in range(9):
        _k = int(10 ** _rng.uniform(0, np.log10(300)))     # k evenly on a log scale, from 1 to 300
        _p = round(_rng.uniform(1, 4), 2)
        _random.append((_k, _p))

    for _name, _tries in [("grid", _grid), ("random", _random)]:
        _scores = []
        for _k, _p in _tries:
            _scores.append(_score(_k, _p))
        _i = np.argmax(_scores)                            # the position of the best try
        print(_name, _tries[_i], round(_scores[_i], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The grid finds AUC 0.947 at k = 116 and p = 1.5, the random search 0.952 at k = 64 and p = 1.53. The grid tested k = 3, 17 and 116; the random search 9 values of k, from 1 to 137.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_grid():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Every combination

        `grid(ks, ps)` gets a list of values of k and a list of values of p, and should give every combination as a list of pairs `(k, p)`, as in the worked example. Replace the `...`. The check calls `grid(ks, ps)` with `ks = [3, 17, 116]` and `ps = [1.5, 2.5, 3.5]`, and tries three more pairs of lists; the order of the pairs does not matter.
        """),
        L.task("Let `grid` give every combination of a value of k and a value of p."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grid():
    box_grid = L.code_box("grid", lines=7)
    box_grid
    return (box_grid,)


@app.cell(hide_code=True)
def check_grid(box_grid):
    result_grid = L.check("grid", box_grid.value)
    result_grid
    return (result_grid,)


@app.cell(hide_code=True)
def hints_grid():
    L.hints(
        "grid",
        [
            "Every value of `ks` must meet every value of `ps`. What do you need inside a loop over `ks`?",
            "A second loop, over `ps`, inside the first. In it, append the pair of the two current values to `tries`.",
            "`for k in ks:`, then one step further in `for p in ps:`, then one step further again `tries.append((___, ___))`.",
        ],
        why="Two loops, one inside the other, make every combination: 3 values of k times 3 of p are 9 tries. Every value of k comes back 3 times; with 4 settings of 3 values each, there would be 81 tries.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Evenly on a log scale

        `log_uniform(rng, low, high, n)` should draw n numbers between `low` and `high`, evenly on a log scale: draw their logarithms evenly between `np.log10(low)` and `np.log10(high)` with `rng.uniform`, and turn them back into numbers. Replace the three `...`. The check calls `log_uniform(np.random.default_rng(0), 1, 300, 9)`, and three more with other ranges, such as from 0.0001 to 0.1, a range for a learning rate.
        """),
        L.task("Let `log_uniform` draw n numbers evenly on a log scale."),
    ])
    return


@app.cell(hide_code=True)
def exercise_log():
    box_log = L.code_box("log-uniform", lines=6)
    box_log
    return (box_log,)


@app.cell(hide_code=True)
def check_log(box_log):
    result_log = L.check("log-uniform", box_log.value)
    result_log
    return (result_log,)


@app.cell(hide_code=True)
def hints_log():
    L.hints(
        "log-uniform",
        [
            "Two steps: draw the logarithms evenly, then go back from every logarithm to its number.",
            "The logarithms lie between the logarithm of `low` and that of `high`, with base 10. The logarithm of a number is the exponent of 10 that gives it; going back is taking 10 to that exponent.",
            "`logs = rng.uniform(np.log10(___), np.log10(___), n)` and `return 10 ** ___`.",
        ],
        why="The logarithms of 1 and 300 are 0 and 2.48. Drawn evenly between them, every factor 10 gets about as many draws, and `10 ** logs` turns them back into numbers from 1 to 300. With `low=0.0001` and `high=0.1` it draws learning rates in the same way.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_few():
    mo.md(r"""
    ### Exercise 3. Four settings, one that matters
    """)
    return


@app.cell(hide_code=True)
def choice_few_cell():
    choice_few = L.choice(
        "few-matter",
        question="You can afford 16 tries to search four settings. You expect that only one of them really matters, but not which one. What do you do?",
        options={
            "grid": "A grid with 2 values per setting: 2 · 2 · 2 · 2 = 16 tries, every combination",
            "right": "A random search: every one of the 16 tries draws new values of every setting",
            "one": "Search one setting at a time, 4 tries each, with the others fixed",
        },
        correct="right",
        why={
            "grid": "Then every setting gets only 2 values, the one that matters too: 16 tries to compare 2 values of it. A random search draws a new value of every setting in every try, so the one that matters gets up to 16 different values as well.",
            "one": "Then every setting gets 4 values, but each search fixes the other three, so you miss how they work together, and the one that matters gets only 4 values.",
            "right": "Whichever setting matters, it gets up to 16 different values (a whole number such as k can come up twice). That is the point of Bergstra and Bengio: when few settings matter, a random search wastes fewer tries than a grid.",
        },
    )
    choice_few
    return (choice_few,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Every try is the cross-validation of [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en), in one line with the pipeline of [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en): the scaling is learned again in every fold. k and p are hyperparameters as in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en): the AUC of [Every threshold at once](/course/open/ml-roc-auc?lang=en) on held-out folds chooses them, never the training loss. The draw of k uses the log scale of [Log scales and power laws](/course/open/math-log-scales?lang=en). In the next lesson, *Let the computer choose: AutoML*, a search spends its tries more cleverly still: it gives many candidates a little data and goes on with the best.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_few, result_grid, result_log):
    L.wrap_up(result_grid, result_log, choice_few)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
