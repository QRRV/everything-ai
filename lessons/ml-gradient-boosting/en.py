# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-gradient-boosting. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-gradient-boosting", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How wrong is one prediction?](/course/open/ml-residuals?lang=en) and [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="A penguin weighs 4000 g, and a model predicts 4300 g. What is the residual?",
        options={
            "size": "300 g",
            "right": "−300 g",
            "pred": "4300 g",
        },
        correct="right",
        why={
            "size": "The residual keeps its sign, so it also says which way the prediction is off. See *The idea* in [How wrong is one prediction?](/course/open/ml-residuals?lang=en).",
            "pred": "That is the prediction itself. The residual says how far it is off. See *The idea* in [How wrong is one prediction?](/course/open/ml-residuals?lang=en).",
            "right": "The label minus the prediction: 4000 − 4300 = −300, negative because the prediction is too high.",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def quick_check_forest():
    quick_forest = L.choice(
        "quick-forest",
        question="What does each tree of a random forest learn from?",
        options={
            "fix": "From what the trees before it got wrong",
            "right": "From its own bootstrap sample, without looking at the other trees",
            "val": "From the validation set",
        },
        correct="right",
        why={
            "fix": "The trees of a forest do not look at each other: each grows on its own bootstrap sample. See *The idea* in [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
            "val": "The validation set is for choosing, not for learning. See the recipe in [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
            "right": "Each tree learns everything by itself, and the vote outvotes the mistakes they do not share.",
        },
    )
    quick_forest
    return (quick_forest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The trees of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en) are deep, and each learns the labels themselves, from its own bootstrap sample; the forest fights their variance. Now predict a number with the same two measurements: the body mass of a penguin from its bill length and bill depth. On the 68 validation penguins, a forest of 100 such trees is off by 447 g (the RMSE of [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en)), a line by 611 g, and always saying the mean by 801 g. A small tree, two questions deep, is off by 535 g: too simple, so it underfits. What if small trees worked in a row, each learning only what the trees before it still get wrong?

    **An everyday example.** A golfer rarely gets the ball in the hole with one stroke. The first stroke takes it most of the way. Every next stroke aims at what is left, from wherever the ball now lies, and close to the hole the strokes get small and careful. Nobody hits the whole distance again.
    """)
    return


@app.cell(hide_code=True)
def predict_rounds():
    guess_rounds = L.guess(
        "forty-two-trees",
        question="42 small trees in a row: each learns what the trees before it still get wrong, and counts for a tenth of what it predicts. How far off do you think they are?",
        options={
            "small": "About 535 g, like one small tree",
            "right": "Less than the forest's 447 g",
            "worse": "More than 535 g: small trees stay too simple",
        },
        reveal="**415 g**, closer than the forest of 100 deep trees, with 447 g. Each small tree on its own is too simple, but together, each fixing a part of what is left, they follow the penguins far better.",
    )
    guess_rounds
    return (guess_rounds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Boosting** builds an ensemble one model at a time: each new model learns from what the models before it still get wrong. **Gradient boosting** does that with small trees and residuals. It starts with the same prediction for every example: the mean of the labels. Then, round after round, it works out the residuals, the label minus the prediction; lets a small tree learn them; and adds a small part of what that tree predicts to the prediction. That part is the **learning rate**, as in gradient descent: with 0.1, every tree fixes a tenth of what it found. The model is the start plus 0.1 times what all its trees predict.

        The trees are **regression trees**: decision trees that predict a number. They ask the same threshold questions, but a leaf predicts the mean of its training examples instead of the most common class. Here those examples are residuals, so a leaf says how much to add to every penguin that lands in it.

        Why *gradient*? The derivative of the squared residual (y − ŷ)² with respect to the prediction ŷ is −2(y − ŷ): the residual times −2. So the residuals point downhill, and adding a small part of them is a step of gradient descent in which the predictions themselves move. The tree makes that step for new examples too, whose labels you do not know.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **boosting** | building an ensemble one model at a time, each learning what the ones before it still get wrong | |
        | **gradient boosting** | boosting with small trees that learn the residuals, and add a small part of what they predict | `GradientBoostingRegressor()` |
        | **regression tree** | a decision tree that predicts a number: each leaf predicts the mean of its training examples | `DecisionTreeRegressor(max_depth=2)` |
        | `learning_rate=0.1` | the part of what each tree predicts that is added | |
        | `n_estimators=42` | how many trees in a row | |
        """),
        L.watch_out("A forest and boosting both combine trees, but the other way around. A forest grows deep trees side by side and averages them: that lowers the variance. Boosting grows small trees one after another and adds them up: that lowers the bias of [Will more data help? Bias, variance and learning curves](/course/open/ml-bias-variance?lang=en), what a model gets wrong because it is too simple."),
        mo.md(r"""
        The picture shows boosting on the penguins. On the left, a dot for every validation penguin, across its real body mass and up the mass the first n trees predict: on the grey diagonal a prediction is exactly right, and the <span class="c-residual">stroke</span> to it is the residual. The small open grey dots are the training penguins. The other chart has the RMSE on the training set (grey) and on the validation set (<span class="c-loss">purple</span>) for 1 to 1000 trees, and the forest's 447 g as a dashed line. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    def _frame(n):
        n = int(n)
        b = C.boosted()
        trees = f"{n} {'tree' if n == 1 else 'trees'}"
        val, train = round(b["val"][n - 1]), round(b["train"][n - 1])
        alt = (f"The first chart: every validation penguin as a dot, across its real body mass from 2850 to 6000 g and up the mass that {trees} predict, with a stroke to the diagonal where the prediction would be exactly right. "
               + ("The predictions all lie close to the mean, 4243 g, in a few horizontal rows." if n < 5 else "The dots lie along the diagonal." if n <= 100 else "The training penguins lie on the diagonal, the validation penguins further from it.")
               + f" The second chart: the RMSE for 1 to 1000 trees; on the training set it falls all the way, to 158 g, on the validation set it falls to 415 g at 42 trees and then rises to 575 g. With {trees}: {val} g on the validation set, {train} g on the training set.")
        said = f"""{trees}: off by <span class="c-loss ea-num">{val} g</span> on the validation penguins, and by {train} g on the training penguins."""
        return (C.picture(n, {"title": f"Gradient boosting, {trees}", "x": "real body mass (g)", "y": "predicted (g)",
                              "every": "RMSE", "trees": "trees", "rmse": "RMSE (g)", "forest": "random forest", "alt": alt}), said)

    explore_rounds = L.explore("rounds", "trees", _frame, start=1, values=C.SIZES, role="hyper",
                               task="Move the number of trees from 1 to 1000. Where is the validation RMSE smallest, and what does the training RMSE do after that?")
    explore_rounds
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With 1 tree every prediction still lies close to the mean, in four rows: the tree has four leaves, and it moved the penguins only a tenth of the way. With 10 trees the dots turn towards the diagonal. At 42 trees the validation RMSE is smallest, 415 g, and between 30 and 60 trees it stays under 420 g. After that the training RMSE keeps falling, to 158 g at 1000 trees, but the validation RMSE rises again, to 575 g. Once the trees have learned what heavy penguins have in common, what is left in the residuals is the quirks of single training penguins, and the next trees learn those too. So the number of trees is a hyperparameter, chosen on the validation set.

    ### Recipe

    To fit gradient boosting with T trees and learning rate `rate`:

    1. Start every prediction at the mean of the labels: `pred = np.zeros(len(y)) + np.mean(y)`.
    2. Work out the residuals: `residuals = y - pred`.
    3. Let a small regression tree learn them: `DecisionTreeRegressor(max_depth=2).fit(X, residuals)`.
    4. Add a part of what it predicts, and keep the tree: `pred = pred + rate * tree.predict(X)`.
    5. Do steps 2 to 4 T times; choose T on the validation set.
    6. To predict for new examples: the mean plus `rate` times what every tree predicts for them.

    scikit-learn does it all: `GradientBoostingRegressor(n_estimators=T, learning_rate=rate, max_depth=2)`, with `fit` and `predict`. For a large table, `HistGradientBoostingRegressor()` is much faster: it first sorts every feature into at most 255 bins, a histogram, and only tries the thresholds between bins. It also handles missing values by itself.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the project, shuffled with the same seed: 206 to train on and 68 to choose on, with bill length and bill depth as features and body mass as label. Steps 1 to 6 for 42 trees, printing the training RMSE after the first three; then the questions of the first tree, which `export_text` writes out as in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en); then the validation RMSE of the 42 trees, of scikit-learn's model with the same settings, and of `HistGradientBoostingRegressor` with its own settings. `random_state=0` fixes the order in which a tree tries the features, so that you get the same trees. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_boosting():
    import math

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor, export_text

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    _features = ["bill_length_mm", "bill_depth_mm"]
    _X_train, _y_train = np.array(_train[_features]), np.array(_train["body_mass_g"])
    _X_val, _y_val = np.array(_val[_features]), np.array(_val["body_mass_g"])
    # 1. the same start for every penguin: the mean body mass of the training set
    _start = np.mean(_y_train)
    _pred = np.zeros(206) + _start
    _trees = []
    # 5. steps 2 to 4, 42 times
    for _i in range(42):
        # 2. the residuals: what the trees so far still get wrong
        _residuals = _y_train - _pred
        # 3. a small regression tree learns them
        _tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(_X_train, _residuals)
        # 4. a tenth of what it predicts is added, and the tree is kept
        _pred = _pred + 0.1 * _tree.predict(_X_train)
        _trees.append(_tree)
        if _i < 3:
            print(round(math.sqrt(np.mean((_y_train - _pred) ** 2))))
    print(export_text(_trees[0], feature_names=_features))
    # 6. the validation penguins: the start plus a tenth of what every tree predicts
    _pred_val = np.zeros(68) + _start
    for _tree in _trees:
        _pred_val = _pred_val + 0.1 * _tree.predict(_X_val)
    print(round(math.sqrt(np.mean((_y_val - _pred_val) ** 2))))
    # scikit-learn: steps 1 to 6 in one model
    _model = GradientBoostingRegressor(n_estimators=42, learning_rate=0.1, max_depth=2, random_state=0)
    _model.fit(_X_train, _y_train)
    print(round(math.sqrt(np.mean((_y_val - _model.predict(_X_val)) ** 2))))
    # the fast one for large tables, with its own settings
    _fast = HistGradientBoostingRegressor(random_state=0).fit(_X_train, _y_train)
    print(round(math.sqrt(np.mean((_y_val - _fast.predict(_X_val)) ** 2))))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The training RMSE goes from 821 g at the start to 770, 727 and 689 g: every tree takes a tenth of a step. The first tree asks first about bill depth. The 72 training penguins with a bill of at most 16.45 mm deep and longer than 38.2 mm, all of them Gentoos, are on average 848 g heavier than the mean, so this tree adds a tenth of that, 84.8 g, to each of them. Its leaves are means of residuals: `value: [848.48]` is how much heavier, not a body mass. After 42 trees the validation RMSE is 415 g, and scikit-learn's model with the same settings gives the same 415 g. `HistGradientBoostingRegressor` grows 100 trees of up to 31 leaves each, also with learning rate 0.1, and gets 398 g.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_round():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One round

        Steps 2 to 4 of the recipe, once. The box reads the training penguins as in the worked example, with `X` and `y`, and starts every prediction in `pred` at the mean. Replace the three `...`: the residuals, a regression tree of depth 2 that learns them, and the new `pred`, with `rate` as the learning rate. The check also tries another `rate`, and another start.
        """),
        L.task("Do one round of gradient boosting on `pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_round():
    box_round = L.code_box("one-round", lines=16)
    box_round
    return (box_round,)


@app.cell(hide_code=True)
def check_round(box_round):
    result_round = L.check("one-round", box_round.value)
    result_round
    return (result_round,)


@app.cell(hide_code=True)
def hints_round():
    L.hints(
        "one-round",
        [
            "Steps 2, 3 and 4 of the worked example, with `X`, `y` and `rate` instead of `_X_train`, `_y_train` and 0.1.",
            "A residual is the label minus the prediction. The tree learns `X` with the residuals as labels: which two things go between the parentheses of `fit`? And `pred` gets something added to it.",
            "`residuals = y - ___`, `tree = DecisionTreeRegressor(max_depth=2).fit(X, ___)` and `pred = pred + ___ * tree.predict(X)`.",
        ],
        why="`y - pred` is what is still wrong for each penguin. The tree learns it from the bill measurements, so it can predict it for any penguin, and `pred + rate * tree.predict(X)` moves every prediction a part `rate` of the way the tree says.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Predict

        Step 6 of the recipe in a function. `predict(start, trees, rate, X)` gets the start, the list of trees, the learning rate and the features of new examples, and gives back the prediction for each example: the start plus `rate` times what every tree predicts. Replace the two `...`. The check calls `predict(start, trees, 0.1, X_new)` with the start and the first three trees of the worked example, and four validation penguins: that should give `[4078.5, 4509.6, 4118.5, 4434.6]`. It also tries other trees and another `rate`.
        """),
        L.task("Make `predict` give the prediction of the start and all the trees."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict", lines=7)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict",
        [
            "Step 6 of the worked example, with `start`, `trees`, `rate` and `X`.",
            "Where does every prediction begin, before any tree? And in the loop, what does `pred` become: the prediction so far, plus what?",
            "`pred = np.zeros(len(X)) + ___` and `pred = pred + ___ * tree.predict(X)`.",
        ],
        why="Every prediction begins at `start`, and each tree adds `rate` times what it predicts. The trees are added up, not averaged: each learned what the trees before it left over.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_too_many():
    mo.md(r"""
    ### Exercise 3. Why can more trees make it worse?
    """)
    return


@app.cell(hide_code=True)
def choice_too_many_cell():
    choice_too_many = L.choice(
        "too-many",
        question="In the random forest, more trees made the vote steadier: with 100, all ten forests made the same 3 mistakes. Here the validation RMSE is smallest at 42 trees and rises to 575 g at 1000. Why can more trees make boosting worse?",
        options={
            "small": "Because its trees are small",
            "right": "Each new tree learns what is left of the training residuals; once the pattern is learned, that is the quirks of single training penguins, which new penguins do not share",
            "rate": "Because a learning rate of 0.1 is too small",
        },
        correct="right",
        why={
            "small": "The small trees are what makes it work: one is too simple, many in a row are not. What is left for tree 500 to learn? See the picture.",
            "rate": "A smaller step only takes more trees to get there, and a bigger one fewer. What is left for the trees after that? See the text after the picture.",
            "right": "In a forest each tree learns the labels themselves, from its own bootstrap sample, and the vote outvotes their quirks. In boosting each tree builds on all the trees before it, so their quirks add up instead. That is why you choose the number of trees on the validation set.",
        },
    )
    choice_too_many
    return (choice_too_many,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Each tree is the tree of [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en), with the mean of its examples in a leaf; it learns the residuals of [How wrong is one prediction?](/course/open/ml-residuals?lang=en), and the learning rate is the same idea as in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en). As in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), the number of trees is chosen on the validation set; in *Early stopping* you do the same for a neural network, epoch by epoch. In the next lesson, *Too good to be true: data leakage*, you see how a score on the validation set can look far better than the model is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_too_many, result_predict, result_round):
    L.wrap_up(result_round, result_predict, choice_too_many)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
