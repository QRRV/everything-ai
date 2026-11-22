# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-permutation-importance. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-permutation-importance", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One short question about [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en) and one about [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="A method shuffles one column and measures how much worse a model gets, over all people at once. It needs the model's answers and the true labels, not the model's insides. What kind of explanation is that?",
        options={
            "local": "Local: one prediction, of one kind of model",
            "right": "Global, and for any model",
            "inside": "Read from the model's own weights or splits",
        },
        correct="right",
        why={
            "right": "Right. It needs only what the model answers and the true labels, so it works for any model: model-agnostic. And it describes the model over all people: global. The forest's own importance, which this lesson also measures, is model-specific: it reads the splits of the trees.",
            "local": "It looks at all people at once, so it is global, and it needs only the model's answers and the labels, so it works for any model. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
            "inside": "Shuffling needs only what the model answers, not its insides: it is model-agnostic, and over all people, global. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def quick_check_forest():
    quick_forest = L.choice(
        "quick-forest",
        question="A random forest grows each of its trees on its own bootstrap sample. What else makes its trees differ?",
        options={
            "labels": "Each tree learns from labels made up at random",
            "depth": "Each tree gets a different maximum depth",
            "right": "At each split, a few features drawn at random",
        },
        correct="right",
        why={
            "right": "Right. At every split, a tree of the forest may choose only from a few features drawn at random. In scikit-learn that is `max_features`, by default the square root of the number of features: 2 of the 4 columns in this lesson.",
            "labels": "Every tree learns from the real labels. Besides its own bootstrap sample, each tree may choose at every split only from a few features drawn at random. See [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
            "depth": "scikit-learn grows every tree of a forest as deep as it goes, unless you set a limit. What differs is the bootstrap sample, and at every split a few features drawn at random. See [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
        },
    )
    quick_forest
    return (quick_forest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A bank lets a random forest decide on loans; as in [Reasons for a decision](/course/open/rai-reason-codes?lang=en), we read a prediction of earning more than 50,000 dollars as the bank's yes. A supervisor asks which features the forest relies on. scikit-learn's forest has an answer ready: `feature_importances_`. In the worked example of this lesson, a forest learns from age, education and hours a week, and from a fourth column of random numbers that has nothing to do with income. Its ready answer gives the random numbers the largest share. A number like that can send a bank, a supervisor or a court looking in the wrong place. This lesson measures it another way: shuffle one column, and see how much worse the model gets on people it has not seen.

    **An everyday example.** How much does a band need its drummer? You can swap the drum part for one from another song and hear how much worse it sounds: that is shuffling. Or you can let the band rehearse without a drummer and hear how well it copes: that is leaving out and fitting again. A band with two drummers barely misses either one, and still needs drums.
    """)
    return


@app.cell(hide_code=True)
def predict_noise():
    guess_noise = L.guess(
        "noise",
        question="A forest with scikit-learn's defaults learns from age, education_num, hours_per_week and a column of random numbers. Its own importance, `feature_importances_`, gives each column a share. Which column gets the largest share?",
        options={
            "education": "education_num",
            "age": "age",
            "noise": "The random numbers",
        },
        reveal="**The random numbers: 48.2 percent**, almost half, twice the share of age (24.3). Yet shuffling them on the test half costs the forest nothing. This lesson shows where the two numbers part ways.",
    )
    guess_noise
    return (guess_noise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Shuffle a column.** Leo Breiman wrote in 2001, in the paper on random forests: "A forest of trees is impenetrable as far as simple interpretations of its mechanism go." Breiman's way in: "After each tree is constructed, the values of the mth variable in the out-of-bag examples are randomly permuted and the out-of-bag data is run down the corresponding tree." The out-of-bag examples are the rows that a tree's bootstrap sample left out, and the tree's mistakes on them are counted again. Shuffling keeps the values of the column, but each now sits in another person's row, so the column no longer tells the model anything about the person. Aaron Fisher, Cynthia Rudin and Francesca Dominici (2019) made it a method for any model and called it model reliance: the loss with the column scrambled divided by the loss without. "An MR(f) value of 1 signifies no reliance"; a value of 2 means the loss doubles. scikit-learn's `permutation_importance` takes the difference: the score before minus the score after shuffling, averaged over a few shuffles.

        **The forest's own number.** Every fitted forest of scikit-learn has `feature_importances_`, the mean decrease in impurity, MDI. For each split, it takes how much purer the split makes the parts, weighed by the people it splits, and adds that up per feature; per tree that becomes a share, and the forest averages the shares over its trees, so they add up to 1. It costs nothing extra: it comes from the training data while the trees grow. The measure goes back to the CART book of Breiman, Friedman, Olshen and Stone; Jerome Friedman (2001) wrote that its motivation "is based purely on heuristic arguments". scikit-learn warns that impurity importances "suffer from two flaws": "First they are computed on statistics derived from the training dataset and therefore do not necessarily inform us on which features are most important to make good predictions on held-out dataset. Secondly, they favor high cardinality features, that is features with many unique values."

        **Why many values win.** Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis and Torsten Hothorn (2007) compared three importance measures of the randomForest function in R, permutation importance among them, and found in their simulations that "for the randomForest function all three variable importance measures are unreliable, and the Gini importance is most strongly biased". They give the cause: "variables with more potential cutpoints are more likely to produce a good criterion value by chance, as in a multiple testing situation." On the training half of this lesson, the column of random numbers offers 999 places to split, against 72 for age, 93 for hours and 15 for education_num. A tree that grows until every leaf is pure keeps finding a split on the random numbers that happens to help on the people it learns from.

        **Measure on people the model has not seen.** scikit-learn shows the same with random columns on data about the Titanic, and adds that permutation importance on the training data of a forest that learned it by heart misleads too: there, the random columns "get a significantly higher importance ranking than when computed on the test set." Christoph Molnar advises: "Estimate PFI on data not used for model training to avoid overly optimistic results, especially with overfitting models." PFI is permutation feature importance.

        **Leaving out asks another question.** Breiman saw it with two identical variables, x1 and x2: "noising each separately will result in the same increase in error rate. But once x1 is entered as a predictive variable, using x2 in addition will not produce any decrease in error rate." Fitting the model again without a feature, which Jing Lei and colleagues call leave one covariate out (LOCO) and Molnar leave one feature out (LOFO), asks, in Molnar's words: "Given the other features, how much will removing a feature worsen the model's predictive performance?" Shuffling asks how much this one fitted model leans on the column. Columns that carry the same information share their importance; scikit-learn: shuffling one of them "results in a lower reported importance value for both features, though they might actually be important." Giles Hooker, Lucas Mentch and Siyu Zhou (2021) warn of the other side: shuffling makes rows that cannot exist, such as "pregnant men", and with flexible models such methods "can vastly over-emphasize correlated features".

        **What it does not tell you.** scikit-learn: "Permutation importance does not reflect the intrinsic predictive value of a feature by itself but how important this feature is for a particular model." Molnar: "Feature importance doesn't tell you how the feature influences the prediction, only how much it affects the loss." And shuffling is random; Molnar again: "When the permutation is repeated, the results might vary greatly." So shuffle more than once, and look at the spread.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **feature importance** | a number per feature for how much a model relies on it; there are several ways to measure it |
        | **permutation importance** | how much worse a model gets on held-out data when one column is shuffled; Fisher and colleagues call it model reliance |
        | **impurity importance (MDI)** | how much a tree's splits on a feature lower the Gini impurity on the training data, weighed by the people they split, as a share; a forest averages it over its trees |
        | **leave one feature out (LOFO)** | fit the model again without the feature, and measure how much worse it gets |
        | `n_repeats` | how many times scikit-learn's `permutation_importance` shuffles each column; it reports the mean and the spread |
        """),
        L.watch_out("Permutation importance describes one fitted model, not the world. scikit-learn warns that features \"deemed of low importance for a bad model (low cross-validation score) could be very important for a good model\". And equally good models can lean on different features, as in [Many good models](/course/open/rai-rashomon?lang=en): Fisher, Rudin and Dominici therefore also measure the range of a feature's importance over all models of one kind that predict well."),
    ])
    return


@app.cell(hide_code=True)
def explore_leaf_cell():
    explore_leaf = L.explore("leaf", "people per leaf, at least", lambda v: C.leaf_picture(v, "en"), start=1,
                             values=C.LEAVES, role="hyper",
                             task="Move the slider from 1 to 50. What happens to the random numbers in the left chart, and in the right one? And to how often the forest is right on the test half?")
    mo.vstack([
        mo.md("**Two importances, as the leaves get bigger**"),
        explore_leaf,
        mo.md(r"""
        A forest of 100 trees on the training half of UCI Adult, with age, education, hours and a column of random numbers. The slider sets `min_samples_leaf`, the fewest people a leaf may hold; 1 is scikit-learn's default, trees that grow until every leaf is pure. Left, the forest's own importance; right, permutation importance on the test half.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure which features a model relies on:

    1. Split the data into a part to train on and a part to test on, and fit the model on the training part.
    2. Measure how often the model is right on the test part.
    3. Shuffle one column of the test part, so that each person gets the value of someone else, and measure again.
    4. The importance of that column is how much worse the model got. Shuffle a few times and take the mean.
    5. Do the same for every column. Where two columns carry the same information, also shuffle them together, or leave one out and fit again: each answers its own question.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The three columns of [Reasons for a decision](/course/open/rai-reason-codes?lang=en), and a fourth: `integers(0, 1000, size=n)` draws n whole numbers from 0 to 999. `.to_numpy()` turns columns of a table into a NumPy array, and `np.column_stack` puts columns side by side in one array. The halves are those of the lessons before: `//` divides and keeps the whole part. `RandomForestClassifier(random_state=0)` has scikit-learn's defaults: 100 trees, each grown until its leaves are pure. `.score` gives how often a model is right, and `feature_importances_` is the forest's own importance. `permutation_importance(model, X, y, n_repeats=5, random_state=0)` shuffles each column five times and gives `importances_mean`, the mean drop in accuracy, and `importances_std`, its spread. Then education_num twice: `_X[:, 1]` is that column, `.copy()` copies an array so that changing the copy leaves the original as it is, `_both[:, [1, 4]] = ...` replaces two columns at once, and `[:, [0, 2, 3, 4]]` keeps only those columns. The index `np.random.default_rng(0).permutation(len(_test))` puts the people of the test half in one shuffled order for both copies, so each person gets both from the same other person. This forest has `min_samples_leaf=20`: each leaf holds at least 20 people, so it cannot learn each person of the training half by heart. Run the cell. It takes about half a minute.
    """)
    return


@app.cell
def example_forest():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    # 1. three columns of UCI Adult and a column of random numbers, in two halves; a forest with its defaults
    _people = pd.read_csv("data/adult.csv.gz")
    _noise = np.random.default_rng(2).integers(0, 1000, size=len(_people))
    _X = np.column_stack([_people[["age", "education_num", "hours_per_week"]].to_numpy(), _noise])
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(random_state=0).fit(_X[_train], _y[_train])

    # 2. how often it is right
    print("right, training half:", round(_forest.score(_X[_train], _y[_train]), 4))
    print("right, test half:", round(_forest.score(_X[_test], _y[_test]), 4))

    # 3 and 4. its own importance; then each column shuffled five times, on the test half and on the training half
    print("own importance:", _forest.feature_importances_.round(3))
    _shuffled = permutation_importance(_forest, _X[_test], _y[_test], n_repeats=5, random_state=0)
    print("shuffled, test half:", _shuffled.importances_mean.round(4))
    print("spread:", _shuffled.importances_std.round(4))
    _learned = permutation_importance(_forest, _X[_train], _y[_train], n_repeats=5, random_state=0)
    print("shuffled, training half:", _learned.importances_mean.round(4))

    # 5. education_num twice: shuffle each copy, shuffle both together, leave one or both out and fit again
    _X5 = np.column_stack([_X, _X[:, 1]])
    _leafy = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X5[_train], _y[_train])
    _right = _leafy.score(_X5[_test], _y[_test])
    print("with a copy, right:", round(_right, 4))
    _each = permutation_importance(_leafy, _X5[_test], _y[_test], n_repeats=5, random_state=0)
    print("each shuffled:", _each.importances_mean.round(4))
    _both = _X5[_test].copy()
    _both[:, [1, 4]] = _both[np.random.default_rng(0).permutation(len(_test))][:, [1, 4]]
    print("both shuffled together:", round(_right - _leafy.score(_both, _y[_test]), 4))
    _one = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X5[_train][:, [0, 2, 3, 4]], _y[_train])
    print("one left out, fit again:", round(_right - _one.score(_X5[_test][:, [0, 2, 3, 4]], _y[_test]), 4))
    _none = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X5[_train][:, [0, 2, 3]], _y[_train])
    print("both left out, fit again:", round(_right - _none.score(_X5[_test][:, [0, 2, 3]], _y[_test]), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With its defaults, the forest is right for 99.6 percent of the training half, which it has learned by heart, and for 76.0 percent of the test half: hardly better than always saying no, 75.9 percent. Its own importance gives the random numbers 0.482, almost half. On the test half, shuffling them costs nothing: −0.0002, that is −0.02 points, well within the spread of 0.0017; shuffling education_num costs 4.4 points, age 2.9 and hours 1.3. On the training half every column looks important, the random numbers too: 20.7 points, because the forest learned them by heart there. Then education_num twice. Shuffled on its own, each copy costs only 1.9 and 1.4 points, since the forest still has the other; shuffled together, 6.1 points, about what education_num costs a forest without a copy at the same leaf size, 6.2 on the slider. Left out and fitted again, one copy costs 0.1 point and both together 3.3: less than shuffling both, because a forest fitted without education_num makes the most of the columns it has left.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_importance():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Shuffle a column by hand

        The box holds `model`, the tree of depth 2 from [Many good models](/course/open/rai-rashomon?lang=en): yes for education above 12.5 and age above 33.5. Write `importance(rows, labels, j, order)`: how much less often `model` is right when column `j` is shuffled. `rows` holds one list per person, `labels` whether each person earns more than 50,000 dollars, and `order` the people in a shuffled order: person `i` gets column `j` of person `order[i]`. `list(rows[i])` makes a copy of a row, so you can change the copy and leave `rows` as it is. The check first calls `importance` with six people, `j = 1` and `order = [1, 0, 3, 4, 5, 2]`, then with other people, columns and orders. The picture runs your function on the 24,421 people of the test half.
        """),
        L.task("Write `importance(rows, labels, j, order)`: right before, minus right with column `j` of person `order[i]`, divided by the number of people."),
    ])
    return


@app.cell(hide_code=True)
def exercise_importance():
    box_importance = L.code_box("importance", lines=19)
    box_importance
    return (box_importance,)


@app.cell(hide_code=True)
def check_importance(box_importance):
    result_importance = L.check("importance", box_importance.value)
    result_importance
    return (result_importance,)


@app.cell(hide_code=True)
def hints_importance():
    L.hints(
        "importance",
        [
            "Count two things in one loop over the people: how often `model` is right for the row as it is, and how often for a copy of the row in which column `j` comes from person `order[i]`. The difference, divided by the number of people, is the importance.",
            "Loop with `for i in range(len(rows)):`. Add 1 to `right` when `model(rows[i]) == labels[i]`. Make `row = list(rows[i])`, put the value that person `order[i]` has in column `j` into it, and add 1 to `shuffled` when `model(row) == labels[i]`.",
            "`row[j] = rows[___][j]`",
        ],
        why="`right = 0` and `shuffled = 0`, then `for i in range(len(rows)):` with `if model(rows[i]) == labels[i]:` adding 1 to `right`, `row = list(rows[i])`, `row[j] = rows[order[i]][j]`, and `if model(row) == labels[i]:` adding 1 to `shuffled`. After the loop, `return (right - shuffled) / len(rows)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_impurity():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The forest's own number, for one tree

        Where does `feature_importances_` come from? The box holds a tree of depth 3 grown on the training half, as scikit-learn keeps it in `tree_`. For each node, `feature` is the column it splits on, `left` and `right` the nodes below it, `impurity` its Gini impurity, and `people` how many people of the training half reach it. A leaf has `-1` in `left` and `right`, and `-2` in `feature`. What a split does is the node's impurity times its people, minus the same for the two nodes below it: how much purer the split makes the parts, weighed by the people. Fill in the loop, so that `gain` holds that drop added up for each column. The last lines turn `gain` into shares; the picture shows yours.
        """),
        L.task("Replace the `...`: for each node with `left[n] != -1`, add its drop to `gain[feature[n]]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_impurity():
    box_impurity = L.code_box("impurity", lines=21)
    box_impurity
    return (box_impurity,)


@app.cell(hide_code=True)
def check_impurity(box_impurity):
    result_impurity = L.check("impurity", box_impurity.value)
    result_impurity
    return (result_impurity,)


@app.cell(hide_code=True)
def hints_impurity():
    L.hints(
        "impurity",
        [
            "Only the nodes that split count: their `left[n]` is not `-1`. For such a node, work out how much the split lowers the impurity, weighed by the people, and add that to the column the node splits on.",
            "Inside `if left[n] != -1:`, call the two nodes below it `a = left[n]` and `b = right[n]`. Work out `drop` from `people` and `impurity` of `n`, `a` and `b`, and add it to `gain[feature[n]]`.",
            "`drop = people[n] * impurity[n] - people[a] * impurity[a] - people[b] * ___`",
        ],
        why="`if left[n] != -1:` with `a = left[n]`, `b = right[n]`, `drop = people[n] * impurity[n] - people[a] * impurity[a] - people[b] * impurity[b]` and `gain[feature[n]] = gain[feature[n]] + drop`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.md(r"""
    ### Exercise 3. Which half?
    """)
    return


@app.cell(hide_code=True)
def choice_train_cell():
    choice_train = L.choice(
        "train",
        question="In the worked example, shuffling the random numbers costs the forest 20.7 points on the training half and nothing on the test half. Which tells you whether the forest uses them for new people?",
        options={
            "right": "The test half: people it has not seen",
            "train": "The training half: it learned there",
            "both": "Both alike: shuffling is the same on each",
        },
        correct="right",
        why={
            "right": "Right. On the training half the forest has learned each person by heart, the random numbers too, so shuffling them there hurts. For new people they tell it nothing, and shuffling them costs nothing. Molnar: estimate it \"on data not used for model training\".",
            "train": "There the forest learned the random numbers by heart: it is right for 99.6 percent of the training half. For new people they tell it nothing, as the test half shows. See *Measure on people the model has not seen* in *The idea*.",
            "both": "The shuffling is the same, but the people are not: the forest learned the training half by heart, the random numbers too, and the test half it has never seen. See *Measure on people the model has not seen* in *The idea*.",
        },
    )
    choice_train
    return (choice_train,)


@app.cell(hide_code=True)
def text_exercise_copy():
    mo.md(r"""
    ### Exercise 4. Leaving out
    """)
    return


@app.cell(hide_code=True)
def choice_copy_cell():
    choice_copy = L.choice(
        "copy",
        question="With education_num in the table twice, leaving one copy out and fitting the forest again costs it 0.1 point. What does that tell you?",
        options={
            "useless": "Education does not help to predict income",
            "right": "The other copy holds it and takes over",
            "unused": "The forest never used that copy",
        },
        correct="right",
        why={
            "right": "Right. Fitting again asks what a feature adds given the others, and the other copy holds the same information. Leaving both out costs 3.3 points, and shuffling both together 6.1: education matters, and the two copies share it.",
            "useless": "Leave both copies out and the forest gets 3.3 points worse; shuffle both together and it gets 6.1 points worse. Education helps; the other copy just takes over. See *Leaving out asks another question* in *The idea*.",
            "unused": "The forest used both: shuffling education_num costs it 1.9 points, the copy 1.4. Fitted again without one, the forest gets it all from the other copy. See *Leaving out asks another question* in *The idea*.",
        },
    )
    choice_copy
    return (choice_copy,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en) you sorted explanations: permutation importance is global and model-agnostic, while the forest's own importance is specific to trees. The forest is that of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en), and its bootstrap samples gave Breiman the first version of this measure: each tree was tested on the rows its sample left out. The next lesson, [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en), asks what this one leaves open: not how much a feature matters, but which way the prediction moves when it changes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_copy, choice_train, result_importance, result_impurity):
    L.wrap_up(result_importance, result_impurity, choice_train, choice_copy)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
