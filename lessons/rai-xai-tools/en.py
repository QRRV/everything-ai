# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-xai-tools. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-xai-tools", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Shapley values](/course/open/rai-shapley-values?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="In the last lesson, the four Shapley values of the person aged 41 added up to 0.226. What is 0.226?",
        options={
            "right": "The chance minus the background's average",
            "both": "What education_num and age get between them",
            "chance": "This person's chance of yes",
        },
        correct="right",
        why={
            "right": "Right: 0.451 minus 0.225. That is Shapley's efficiency, and every shap explainer in this lesson keeps it: whatever it samples or fits, its values add up to the prediction minus the average over its background.",
            "both": "education_num and age get +0.181 and +0.087, together 0.268, and hours_per_week and capital_gain take 0.042 off. 0.226 is the person's chance, 0.451, minus the background's average, 0.225. See [Shapley values](/course/open/rai-shapley-values?lang=en).",
            "chance": "This person's chance is 0.451. The four values share out the gap between that chance and the background's average, 0.225: 0.226. See [Shapley values](/course/open/rai-shapley-values?lang=en).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_game():
    quick_game = L.choice(
        "quick-game",
        question="Against 200 people with education_num 13 or more, education_num got −0.027 instead of +0.181. Why?",
        options={
            "right": "The background is part of the game",
            "random": "Each run draws new random orders",
            "trained": "The forest was trained on those people",
        },
        correct="right",
        why={
            "right": "Right: the same person and the same forest, compared with other people. In the tools of this lesson the background is one argument, and some of them cut it down with no more than a log message, or play another game when you leave it out.",
            "random": "The last lesson went through all 24 orders, so nothing was left to chance once the background was drawn. Compared with people with 13 or more, 13 is the lowest: the background is part of the game. See [Shapley values](/course/open/rai-shapley-values?lang=en).",
            "trained": "The forest stayed the same; only the rows that fill in the unknown columns changed. Compared with people with 13 or more, 13 is the lowest. See [Shapley values](/course/open/rai-shapley-values?lang=en).",
        },
    )
    quick_game
    return (quick_game,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The bank wants to train its forest on all 14 columns of UCI Adult and give everyone in the test half, 24,421 people, Shapley values. Exactly, as in the last lesson, that is $2^{14}$ = 16,384 coalitions for one person, and with 200 background rows the forest has to give a chance for 3,276,800 rows for that one person. So you reach for a library. shap and InterpretML do what you built by hand, much faster, DiCE does what the next lesson builds, and all three make choices for you: which rows fill in, how many, which game, how many samples. Some of these choices are printed nowhere, and each can change the answer.

    **An everyday example.** A camera on automatic picks the shutter time, the focus and the light for you. Two cameras on automatic take two different photos of the same street, and when a photo comes out dark, you look up what the camera picked. Do the same with a library for explanations: find out what it picked before you trust its numbers.
    """)
    return


@app.cell(hide_code=True)
def predict_kept():
    guess_kept = L.guess(
        "kept",
        question="You give `shap.Explainer` the whole training half, 24,421 rows, as background. How many of them does it use to fill in the unknown columns?",
        options={
            "all": "All 24,421",
            "thousand": "1,000",
            "hundred": "100",
            "ten": "10",
        },
        reveal="**100.** shap turns a table you give as background into a masker, `maskers.Independent(background)`, with `max_samples=100`: \"If data has more than max_samples then shap.utils.sample is used to subsample the dataset.\" That draws 100 rows without replacement and with `random_state=0`, so the same 100 every run, and shap says so only in a log message. This is read from the code of shap 0.54.1; the course does not run shap.",
    )
    guess_kept
    return (guess_kept,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Three libraries.** This lesson shows the calls of three libraries next to what you built, and does not run them: the course does not install them, and the numbers in its cells come from the course's own code, computed the way you did by hand. It was written for shap 0.54.1 (October 2026), InterpretML 0.7.8 (March 2026; the packages `interpret` and `interpret-core`) and dice-ml 0.12 (July 2025). Name the version when you report a value: shap changed defaults in version 0.47, and code from an older tutorial can give other numbers.

        **shap and its background.** The usual call:

        ```python
        import shap

        explainer = shap.Explainer(model, background)   # model(rows): the chance of yes for each row
        explanation = explainer(rows)                   # rows: the people to explain
        explanation.base_values                         # the average answer over the background
        explanation.values                              # a Shapley value for each person and column
        ```

        The background is what you filled in from in the last lesson. A table there becomes a **masker**, shap's object that hides the unknown columns by filling them in from background rows, and that masker keeps 100 rows, as in the guess. Its documentation gives a rule of thumb: "Normally about 1, 10, 100, or 1000 background samples are reasonable choices." Then `algorithm="auto"` picks how to compute. For a model that is just a function, it picks `"exact"` up to 10 columns and `"permutation"` above that, so the same call runs one algorithm on 8 columns of Adult and another on 14. With the four columns of the last lesson shap would do what you did, by its code over 100 of your 200 rows. Which 100 matters: in 10 draws of 100 of those 200 rows, education_num gets from +0.164 to +0.196, against +0.181 with all 200. The slider below draws backgrounds of other sizes.

        **Orders at random.** Above 10 columns, shap's `PermutationExplainer` takes orders at random, as Štrumbelj and Kononenko did, and averages what each column adds when it joins. By its code it "guarantees local accuracy (additivity) by iterating completely through an entire permutation of the features in both forward and reverse directions (antithetic sampling)": each order is also run backward, and in every order the columns share out the whole gap, so the estimates always add up. Its `max_evals`, how many times it may ask the model, is 500 by default. The orders are random, so two runs give slightly different numbers: with 10 random orders for the person of the last lesson, education_num gets from +0.157 to +0.205 in 10 runs. You write this method in exercise 1.

        **KernelSHAP: LIME with the Shapley kernel.** Lundberg and Lee showed that a line like LIME's gives the Shapley values when you choose its parts their way. Write each coalition as a row of 0s and 1s, one for each column, and fit a straight line from those rows to the values of the coalitions, with no penalty on its weights. Let a coalition with $s$ of the $M$ columns known count with the **Shapley kernel**

        $$\pi(s) = \frac{M - 1}{\binom{M}{s}\, s\, (M - s)}.$$

        For the empty and the full coalition that is infinite: they must come out exactly, which makes the line start at the background's average and end at the prediction. Then, in their words, "the Shapley values from game theory can be computed using weighted linear regression." That is **KernelSHAP**:

        ```python
        explainer = shap.KernelExplainer(model, background)
        values = explainer.shap_values(rows, nsamples="auto", l1_reg="num_features(10)")   # its defaults
        ```

        About the background its documentation says: "For small problems, this background dataset can be the whole training set, but for larger problems consider using a single reference value or using the ``kmeans`` function to summarize the dataset." With `nsamples="auto"` it asks for $2M + 2048$ coalitions, but never more than the $2^M - 2$ between the empty and the full one: up to 11 columns that is all of them; above 11 it samples. With your four columns it takes all 14. Two defaults change the answer without telling you. Since version 0.47, `l1_reg="num_features(10)"` "selects a fixed number of top features": a feature selection before the line keeps at most about 10 columns, and the others get 0. And the coalitions are drawn with NumPy's random numbers, with no seed to pass, so two runs on many columns differ slightly. I. Elizabeth Kumar and colleagues count KernelSHAP among the interventional methods: it fills in from rows as they are, as you did. Exercise 2 is the kernel.

        **TreeSHAP: from the trees themselves.** For forests and boosted trees shap has `TreeExplainer`. Lundberg and colleagues list as the first contribution of their paper "A polynomial time algorithm" for "optimal explanations based on game theory". By their 2019 preprint, the time grows with the number of trees, their leaves and their depth, not with $2^M$; with data, it also grows with the number of background rows. **TreeSHAP** can play two games:

        ```python
        shap.TreeExplainer(forest)                     # no data: "tree_path_dependent"
        shap.TreeExplainer(forest, data=background)    # data: "interventional", over 100 of its rows
        ```

        The default, `feature_perturbation="auto"`, picks "interventional" when you pass data: the game of the last lesson. Without data, "no background dataset is required and the approach is to just follow the trees and use the number of training examples that went down each leaf to represent the background distribution." Kumar and colleagues count TreeSHAP of that kind among the conditional methods. The default has moved: the 2019 preprint of Lundberg and colleagues says "By default, TreeExplainer computes conditional expectations using tree traversal", the documentation of shap says the default changed from "interventional" to "auto" in 0.47, and in 0.54.1 it depends on whether you pass data. The worked example plays both games by hand.

        **InterpretML: a model that explains itself.** InterpretML builds the Explainable Boosting Machine of [Additive models](/course/open/rai-additive-models?lang=en):

        ```python
        from interpret import show
        from interpret.glassbox import ExplainableBoostingClassifier

        ebm = ExplainableBoostingClassifier()          # defaults include interactions="3x", random_state=42
        ebm.fit(X_train, y_train)
        show(ebm.explain_global())                     # the curve of each term
        show(ebm.explain_local(X_test[:5], y_test[:5]), 0)
        ebm.eval_terms(X_test[:5])                     # the same local values, as numbers
        ```

        `interactions="3x"` asks for three times as many pairs of columns as there are columns: with Adult's 14 columns, 42 pairs. `eval_terms` gives "Term scores identical to the local explanation values obtained by calling ebm.explain_local(x)." Nothing is filled in and nothing is sampled when an EBM explains: its local explanation is read off its own terms, so it needs no background and gives the same numbers every time. What changes it is the model: another `random_state` or other `interactions` train another EBM. InterpretML also wraps shap: `interpret.blackbox.ShapKernel` calls KernelExplainer.

        **DiCE: what would have to change.** DiCE, by Ramaravind K. Mothilal, Amit Sharma and Chenhao Tan, looks for rows close to the person that get the other answer, the subject of the next lesson:

        ```python
        import dice_ml

        data = dice_ml.Data(dataframe=train_table, continuous_features=["age", "hours_per_week"],
                            outcome_name="income")
        model = dice_ml.Model(model=forest, backend="sklearn")    # without backend: "TF1", TensorFlow 1
        dice = dice_ml.Dice(data, model, method="random")         # the default method
        found = dice.generate_counterfactuals(person, total_CFs=4, desired_class="opposite",
                                              features_to_vary=["education_num", "hours_per_week"])
        found.visualize_as_dataframe(show_only_changes=True)
        ```

        Three choices change what it finds. `method` picks the search: "random", "genetic", "kdtree" or "gradient", and the random one starts from `random_seed=None`, so a new set each run. The weights for closeness and variety, such as `proximity_weight`, are "ignored by ['random', 'kdtree'] methods", although DiCE's own README passes them to the random method (by the code, there they most likely stop with an error). And `features_to_vary="all"`, the default, lets DiCE change anything: in the paper's own example for Adult, one of four answers changes the education to Masters, the age to 65, the marital status to married and the sex to male.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **masker** | shap's object that hides the unknown columns, by filling them in from background rows |
        | **KernelSHAP** | Lundberg and Lee's way to compute Shapley values for any model: a weighted linear regression on the coalitions |
        | **Shapley kernel** | the weight of a coalition in that regression; infinite for the empty and the full coalition |
        | **TreeSHAP** | Shapley values for trees in polynomial time, from the structure of the trees |
        | **InterpretML** | the library with the Explainable Boosting Machine, and wrappers around other explainers |
        | $M$ | the number of columns that vary |
        | $\binom{M}{s}$ | the number of coalitions with $s$ of the $M$ columns known; in Python `math.comb(M, s)` |
        """),
        L.watch_out("A library's defaults are choices: which background and how many rows, which game, how many samples, which columns may get a value. Name the version and these settings next to every value you report: two calls that differ in one of them can give different values for the same model and the same person."),
    ])
    return


@app.cell(hide_code=True)
def explore_size_cell():
    explore_size = L.explore("size", "rows in each background", lambda v: C.size_picture(v, "en"), start=100,
                             values=C.SIZES, role="hyper",
                             task="Move the slider down to 1, then up to 1000. How far apart do the ten values of education_num lie at each size?")
    mo.vstack([
        mo.md("**How many rows?**"),
        explore_size,
        mo.md(r"""
        The course's exact Shapley values for the person aged 41, each dot with its own background of the slider's number of rows from the training half, drawn with the seeds 0 to 9. With one row the background is a single other person, and the values say how this person differs from that one: 2 of the 10 had a capital gain, of 7,298 and 99,999 dollars, and capital_gain then gets −0.458 and −0.531. With 1000 rows age, hours_per_week and capital_gain nearly agree, and eight of the ten draws give education_num between +0.155 and +0.162, below the +0.181 of the 200 rows of the last lesson; the other two, +0.147 and +0.172, lie almost as far apart as the ends at 100 rows. Every dot is exact for its own background; what moves is the game.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To use a library for explanations:

    1. Write down the library and its version, such as shap 0.54.1: defaults change between versions.
    2. Say which game it plays: which rows fill in and how many (a table given to `shap.Explainer` or `TreeExplainer` becomes at most 100 rows by its code; `KernelExplainer` uses every row), or, for trees, whether you passed data.
    3. Say how it computes: exactly, from sampled orders, or by a regression on sampled coalitions. Fix the seed where there is one, and where there is none, run it twice.
    4. Check that the values add up to the prediction minus the base value.
    5. Look for defaults that change the answer, such as `l1_reg` in KernelSHAP and `features_to_vary` in DiCE.
    6. On a few columns, compare with your own exact computation.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forest, the person and the 200 background rows of the last lesson, in the two games of `TreeExplainer`. A fitted tree of scikit-learn keeps its structure in `tree_`: for each node, `children_left` and `children_right` give the two nodes below it (−1 at a leaf), `feature` and `threshold` give its question, `value[node, 0, 1] / value[node, 0].sum()` is its share of yes, and `weighted_n_node_samples` says how many training rows reached it, each counted as often as the tree's own sample drew it. `_down` calls itself for the node below, and so goes down the tree until it reaches a leaf. `_forest.estimators_` holds the 100 trees. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def example_trees():
    import itertools
    import math

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # 1. the forest, the person and the background of the last lesson
    _people = pd.read_csv("data/adult.csv.gz")
    _columns = ["age", "education_num", "hours_per_week", "capital_gain"]
    _X = _people[_columns].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _person = _X[_test][5]
    _background = _X[_train][np.random.default_rng(0).choice(len(_train), 200, replace=False)]
    print("chance of yes for this person:", _forest.predict_proba([_person])[0, 1].round(3))


    # 2. with data: fill in the unknown columns from the 200 rows, as in the last lesson
    def _with_data(known):
        _rows = np.where(np.array(known) == 1, _person, _background)
        return _forest.predict_proba(_rows)[:, 1].mean()


    # 3. without data: go down a tree; at a question about an unknown column take both branches,
    #    each counted by how many training rows went that way
    def _down(tree, node, known):
        if tree.children_left[node] == -1:
            return tree.value[node, 0, 1] / tree.value[node, 0].sum()     # a leaf: its share of yes
        _left, _right = tree.children_left[node], tree.children_right[node]
        _column = tree.feature[node]
        if known[_column] == 1:
            if _person[_column] <= tree.threshold[node]:
                return _down(tree, _left, known)
            return _down(tree, _right, known)
        _n = tree.weighted_n_node_samples
        return (_n[_left] * _down(tree, _left, known) + _n[_right] * _down(tree, _right, known)) / _n[node]


    def _without_data(known):
        _total = 0
        for _tree in _forest.estimators_:
            _total = _total + _down(_tree.tree_, 0, known)
        return _total / len(_forest.estimators_)


    # 4. the Shapley values of a game, by the formula of the last lesson
    def _shapley(v, i):
        _total = 0
        for _known in v:
            if _known[i] == 0:
                _with_i = list(_known)
                _with_i[i] = 1
                _s, _n = sum(_known), len(_known)
                _weight = math.factorial(_s) * math.factorial(_n - _s - 1) / math.factorial(_n)
                _total = _total + _weight * (v[tuple(_with_i)] - v[_known])
        return _total


    for _name, _game in [("with data:   ", _with_data), ("without data:", _without_data)]:
        _v = {}
        for _known in itertools.product([0, 1], repeat=4):
            _v[_known] = _game(_known)
        _line = _name + " average " + f"{_v[(0, 0, 0, 0)]:.3f}"
        for _i in range(4):
            _line = _line + "  " + _columns[_i] + f" {_shapley(_v, _i):+.3f}"
        print(_line)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Both games end at the person's own chance, 0.451, but start somewhere else: at the forest's average over the 200 rows, 0.225, or at the average of the trees' leaves by their training rows, 0.238, close to the share of yes in the training half, 0.237. education_num and age hardly move; hours_per_week gets −0.026 instead of −0.010, more than twice as much. Neither is wrong: they answer different questions. With data, the question is what knowing a column adds against these 200 people; without, it is what it adds along the paths of the trees, where at a question about an unknown column the person goes the ways the training rows in that node went. TreeSHAP gets the values of that second game by going down each tree once, without going through the 16 coalitions; this cell plays it by hand.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sampled():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Orders at random

        Write `sampled(v, i, orders)`, the permutation method: `v` is a dict from each coalition, a tuple of 0s and 1s, to its value, `i` is the column, and `orders` is a list of orders, each a tuple of column numbers, every column once, in the order in which they join: `(1, 0)` means column 1 joins first. Start `total` at 0. For each `order` in `orders`, start `known` with all 0s, `[0] * len(order)`, and let the columns `j` of the order join one at a time: take `before`, the value of `tuple(known)`, set `known[j]` to 1, and when `j` is `i`, add the value of `tuple(known)` minus `before` to `total`. Return the average, `total` divided by the number of orders, `len(orders)`. An order may come twice, as when you draw at random. The check first calls `sampled({(0, 0): 0, (1, 0): 2, (0, 1): 4, (1, 1): 10}, 0, [(1, 0), (1, 0), (0, 1)])`, the pub quiz of the last lesson with three orders drawn: Ash should get (6 + 6 + 2) / 3. The picture runs your function on the 16 values of the last lesson with the first 1 to 100 of 100 random orders.
        """),
        L.task("Write `sampled(v, i, orders)`: for each order, let the columns join one at a time, add to `total` what the value gains when column `i` joins, and return the average over the orders."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sampled():
    box_sampled = L.code_box("sampled", lines=12)
    box_sampled
    return (box_sampled,)


@app.cell(hide_code=True)
def check_sampled(box_sampled):
    result_sampled = L.check("sampled", box_sampled.value)
    result_sampled
    return (result_sampled,)


@app.cell(hide_code=True)
def hints_sampled():
    L.hints(
        "sampled",
        [
            "Each order is a little story: nothing is known, and the columns join one at a time. Only the step where column `i` joins counts, and it counts what the value gains at that step.",
            "Inside `for order in orders:` start `known = [0] * len(order)`, and inside `for j in order:` take `before = v[tuple(known)]` and then set `known[j] = 1`; if `j == i`, add `v[tuple(known)] - before` to `total`.",
            "`return total / ___`",
        ],
        why="`total = 0`, then `for order in orders:` with `known = [0] * len(order)` and `for j in order:`: `before = v[tuple(known)]`, `known[j] = 1`, and `if j == i:` `total = total + v[tuple(known)] - before`. After both loops, `return total / len(orders)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kernel():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The Shapley kernel

        Write `kernel_weight(m, s)`, Lundberg and Lee's Shapley kernel for a coalition with `s` of the `m` columns known: $m - 1$ divided by $\binom{m}{s}\, s\, (m - s)$. `math.comb(m, s)` gives $\binom{m}{s}$, the number of coalitions with `s` of the `m` columns. For the empty and the full coalition, with `s` 0 or `m`, the formula divides by 0: return `math.inf` there, Python's infinity. The check first calls `kernel_weight(4, 1)`: 3 / (4 · 1 · 3) = 0.25. The picture draws your weights for 4 and for 14 columns.
        """),
        L.task("Write `kernel_weight(m, s)`: `math.inf` for the empty and the full coalition, and otherwise (m − 1) divided by `math.comb(m, s) * s * (m - s)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kernel():
    box_kernel = L.code_box("kernel", lines=8)
    box_kernel
    return (box_kernel,)


@app.cell(hide_code=True)
def check_kernel(box_kernel):
    result_kernel = L.check("kernel", box_kernel.value)
    result_kernel
    return (result_kernel,)


@app.cell(hide_code=True)
def hints_kernel():
    L.hints(
        "kernel",
        [
            "Two cases: the empty and the full coalition weigh infinitely much, and every other coalition gets the formula.",
            "First `if s == 0 or s == m:` with `return math.inf`. Below it the formula: `m - 1` on top, and under the line the number of coalitions, `math.comb(m, s)`, times `s` times `m - s`.",
            "`return (m - 1) / (___ * s * (m - s))`",
        ],
        why="`if s == 0 or s == m:` `return math.inf`; then `return (m - 1) / (math.comb(m, s) * s * (m - s))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_kernel():
    mo.md(r"""
    The kernel weighs the coalitions with few or with many columns known most: with 4 columns 0.25, 0.125 and 0.25, and with 14 columns a coalition with 1 known counts 924 times as much as one with 7. With these weights, the regression on the 14 coalitions of four columns, between none and all, gives exactly the four Shapley values of the last lesson. With 14 columns there are 16,382 such coalitions, and KernelSHAP's default asks for 2,076 of them: then it samples.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_data():
    mo.md(r"""
    ### Exercise 3. With data or without

    In the worked example the same forest gave the same person two sets of values.
    """)
    return


@app.cell(hide_code=True)
def choice_data_cell():
    choice_data = L.choice(
        "data",
        question="Two calls explain the same forest for the same person: `shap.TreeExplainer(forest)` and `shap.TreeExplainer(forest, data=background)`. What differs?",
        options={
            "right": "The game, and with it the values",
            "exact": "Nothing: TreeSHAP is exact either way",
            "speed": "Only how fast the values come out",
        },
        correct="right",
        why={
            "right": "Right. Without data, TreeExplainer follows the trees and lets the training rows that went down each branch stand for the background; with data, it fills in from the rows you give, by its code 100 of them. In the worked example by hand: an average of 0.238 against 0.225, and hours_per_week −0.026 against −0.010.",
            "exact": "TreeSHAP is exact for the game it plays, and the two calls play different games: the trees' own training rows, or the rows you give. In the worked example by hand, hours_per_week gets −0.026 without data and −0.010 with. See *TreeSHAP* in *The idea*.",
            "speed": "They can take different times, but the values differ too, because the game differs: without data, the trees' own training rows stand for the background. In the worked example by hand, hours_per_week gets −0.026 without data and −0.010 with.",
        },
    )
    choice_data
    return (choice_data,)


@app.cell(hide_code=True)
def text_exercise_ten():
    mo.md(r"""
    ### Exercise 4. Which columns get a value?

    A default of KernelSHAP decides how many columns can get a value at all.
    """)
    return


@app.cell(hide_code=True)
def choice_ten_cell():
    choice_ten = L.choice(
        "ten",
        question="You explain one person with `shap.KernelExplainer` on a table with 30 columns, with the defaults of shap 0.54.1. How many columns can get a value other than 0?",
        options={
            "right": "At most about 10 of them",
            "all": "All 30, each with its share",
            "above": "Only those above 0.01",
        },
        correct="right",
        why={
            "right": "Right. Since version 0.47 the default is `l1_reg=\"num_features(10)\"`, which \"selects a fixed number of top features\": a feature selection before the regression keeps at most about 10 columns, and the others get 0 (read from shap's code, not run). When every column matters, set `l1_reg` yourself and say what you set.",
            "all": "Not with the defaults of 0.54.1: since 0.47, `l1_reg=\"num_features(10)\"` keeps at most about 10 columns, and the others get 0. See *KernelSHAP* in *The idea*.",
            "above": "There is no cut-off at 0.01. The default `l1_reg=\"num_features(10)\"` keeps at most about 10 columns by a feature selection, and the others get 0. See *KernelSHAP* in *The idea*.",
        },
    )
    choice_ten
    return (choice_ten,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Shapley values](/course/open/rai-shapley-values?lang=en) worked them out exactly for four columns, over all 16 coalitions. The libraries get there faster, by sampled orders, by a weighted regression or by the trees themselves, and each asks you to choose: which background and how many rows, data or no data, how many samples, which defaults. The Explainable Boosting Machine of [Additive models](/course/open/rai-additive-models?lang=en) needs none of that, since it explains itself. The next lesson, [What would have to change?](/course/open/rai-counterfactuals?lang=en), builds by hand what DiCE looks for: the smallest change that gives the other answer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_data, choice_ten, result_kernel, result_sampled):
    L.wrap_up(result_sampled, result_kernel, choice_data, choice_ten)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
