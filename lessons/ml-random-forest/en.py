# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-random-forest. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-random-forest", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pure():
    quick_pure = L.choice(
        "quick-pure",
        question="A decision tree without a maximum depth gets all 206 training penguins right. What does that tell you about new penguins?",
        options={
            "all": "It gets them all right too",
            "right": "Not much: it went on until every part was pure, and may have learned the quirks of single penguins",
            "under": "It underfits",
        },
        correct="right",
        why={
            "all": "On the validation penguins that tree got 9 wrong, more than a tree of depth 2. See the slider in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).",
            "under": "A tree that gets every training example right is not too simple. See the slider in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).",
            "right": "Being perfect on its own training penguins says nothing about others: only the validation set does.",
        },
    )
    quick_pure
    return (quick_pure,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Two lists, each value as likely: 3, 3, 4, 3 and 1, 8, 2, 5. Which has the larger variance?",
        options={
            "first": "3, 3, 4, 3",
            "right": "1, 8, 2, 5",
            "same": "Both the same: they have four numbers each",
        },
        correct="right",
        why={
            "first": "The variance is about how far the values lie from their mean, not how large they are. See *The idea* in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "same": "The variance looks at the squared distance of each value to the mean. See *The idea* in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "Its values lie far from their mean, 4; those of the first lie within 1 of theirs.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en), a tree that grew until every part was pure followed single penguins, so other training penguins give quite another tree. Here the trees use bill length and bill depth, on which the species lie further apart than on bill length and body mass. Draw 206 of the 206 training penguins at random, some more than once, grow a tree until every part is pure, and do that 100 times: of the 68 validation penguins, one such tree gets 1 wrong, another 8, and on average 5. Which tree do you trust?

    **An everyday example.** In 1906 Francis Galton looked at 787 guesses of the weight of an ox at a country fair. Single guesses were often far off, but the middle guess, the median, was 1207 pounds: 9 pounds above the true 1198. That works because the guessers were wrong in different ways. Had they all copied the loudest one, the crowd would have guessed no better than one person.
    """)
    return


@app.cell(hide_code=True)
def predict_vote():
    guess_vote = L.guess(
        "hundred-vote",
        question="100 such trees vote: each validation penguin gets the species most of them say. How many do you think the vote gets wrong?",
        options={
            "average": "About 5, like a tree on average",
            "right": "3",
            "worst": "More than 5: the bad trees spoil it",
        },
        reveal="**3**, and not by luck: ten such votes of 100 trees, each with its own random draws, all get exactly 3 wrong. A few single trees did better, with 1 or 2 wrong, but which ones is luck; the vote gives 3 every time.",
    )
    guess_vote
    return (guess_vote,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **random forest** grows many deep trees and lets them vote, with majority voting, as the neighbors of k-NN do. Each tree learns from its own examples: a **bootstrap** sample, n examples drawn at random from the n of the training set **with replacement**. Each drawn example goes back, so some come twice or more and others not at all: on average a bootstrap sample holds about 63% of the examples. Letting trees on bootstrap samples vote is called **bagging**. A random forest adds one thing: at every split a tree may look at only a few features, drawn at random, so that even the first questions of the trees differ. The trees of a forest form an **ensemble**: models whose predictions are combined.

        Why does the vote vary less than one tree? Different trees get different penguins wrong, and a mistake only a few trees make is outvoted. Only the penguins most trees get wrong stay wrong, and which those are hardly changes from forest to forest; which ones a single tree gets wrong is luck.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **random forest** | many deep decision trees on bootstrap samples, with a few random features per split, that vote | `RandomForestClassifier()` |
        | **bootstrap**, bootstrap sample | drawing n examples from your n with replacement; what you get is a bootstrap sample | `rng.integers(0, n, size=n)` |
        | **with replacement** | each drawn example goes back and can be drawn again | |
        | **bagging** | models on bootstrap samples that vote or are averaged: bootstrap aggregating | |
        | **ensemble** | models whose predictions are combined, by voting or averaging | |
        | `max_features=1` | how many features, drawn at random, a tree may look at in each split | |
        | `n_estimators=100` | how many trees a random forest of scikit-learn grows | |
        """),
        L.watch_out("*Variance* here is the variance of [Expected value and variance](/course/open/prob-expectation?lang=en): how far the mistakes of a forest lie from their average over many random draws."),
        mo.md(r"""
        The picture shows a forest on the 206 training penguins: a circle is an Adelie, a triangle a Chinstrap and a square a Gentoo; here color and shape mean the species. Each tree looks at one random measurement per split, and the <span class="c-pred">blue line</span> is the decision boundary of the vote of the first n trees. The other chart has, for every number of trees, the <span class="c-loss">mistakes</span> on the 68 validation penguins of ten such forests, each with its own random draws: a dot for each, and a line from the fewest to the most; the filled dots are the forest on the left. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_trees_cell():
    def _frame(n):
        n = int(n)
        m = C.MEASURED[n]
        trees = f"{n} {'tree' if n == 1 else 'trees'}"
        alt = (f"The first chart: the 206 training penguins by bill length and bill depth, a color and shape per species. Adelies have short, deep bills, Gentoos long, shallow ones, Chinstraps long, deep ones. A blue line separates the areas where the vote of {trees} predicts each species"
               + ("; with one tree it has sharp corners around single penguins." if n == 1 else "; with only a few trees it still has sharp corners." if n < 5 else "; with more trees it gets smoother.")
               + f" The second chart: for 1 to 100 trees, the validation mistakes of ten forests; with 1 tree they run from 2 to 8, with 100 trees all ten make 3. With {trees}: "
               + (f"from {min(m)} to {max(m)}." if min(m) < max(m) else f"all ten {min(m)}."))
        said = (f"""{trees}: the forest on the left gets <span class="c-loss ea-num">{C.mistakes(0, n)}</span> of the 68 validation penguins wrong; """
                + (f"""the ten forests from <span class="c-loss ea-num">{min(m)}</span> to <span class="c-loss ea-num">{max(m)}</span>""" if min(m) < max(m)
                   else f"""all ten forests <span class="c-loss ea-num">{min(m)}</span>""")
                + f""", with a variance of {L.number(C.variance(m)).rstrip("0").rstrip(".,")}.""")
        return (C.picture(n, {"title": f"Random forest, {trees}", "x": "bill length (mm)", "y": "bill depth (mm)",
                              "every": "Ten forests", "trees": "trees", "mistakes": "mistakes (of 68)", "alt": alt}), said)

    explore_trees = L.explore("trees", "trees", _frame, start=1, values=C.SIZES, role="hyper",
                              task="Move the number of trees from 1 to 100. How far apart are the ten forests with 1 tree, and with 100?")
    explore_trees
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With 1 tree the ten forests make 2 to 8 mistakes, a variance of 4.2, and the boundary has sharp corners around single penguins. With 5 trees they make 2 to 4, and with 100 all ten make exactly 3, a variance of 0: the same three penguins, which most single trees get wrong too. The boundary gets smoother, because the corners only a few trees have are outvoted.

    ### Recipe

    To build a random forest of T trees:

    1. Draw a bootstrap sample: n positions from 0 to n − 1, with replacement, `rng.integers(0, n, size=n)`.
    2. Grow a tree on those examples until every part is pure, with a few random features per split: `DecisionTreeClassifier(max_features=1)`.
    3. Do steps 1 and 2 T times, each time with new random draws.
    4. To predict, let every tree predict, and give each example the class most trees say.

    scikit-learn does it all: `RandomForestClassifier(n_estimators=T, max_features=1)`, with `fit` and `predict`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the project, shuffled with the same seed: 206 to train on and 68 to choose on. Five trees, steps 1 to 4, then scikit-learn's forest of 100 trees. `_votes[:, 54]` is column 54 of the table of votes: what the five trees say about validation penguin 54. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_forest():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    _features = ["bill_length_mm", "bill_depth_mm"]
    _X_train, _y_train = np.array(_train[_features]), np.array(_train["species"])
    _X_val, _y_val = np.array(_val[_features]), np.array(_val["species"])
    _rng = np.random.default_rng(0)
    _votes = []
    # 3. five trees, each with new random draws
    for _i in range(5):
        # 1. a bootstrap sample: 206 positions, drawn with replacement
        _rows = _rng.integers(0, 206, size=206)
        # 2. a tree on those penguins, with one random measurement per split
        _tree = DecisionTreeClassifier(max_features=1, random_state=_i)
        _tree.fit(_X_train[_rows], _y_train[_rows])
        _votes.append(_tree.predict(_X_val))
        print(len(np.unique(_rows)), np.sum(_votes[_i] != _y_val))
    # 4. for each validation penguin, the species most of the five trees say
    _votes = np.array(_votes)
    _said = []
    for _j in range(len(_y_val)):
        _names, _counts = np.unique(_votes[:, _j], return_counts=True)
        _said.append(_names[np.argmax(_counts)])
    print(_votes[:, 54], _said[54], _y_val[54])
    print(np.sum(np.array(_said) != _y_val))
    # scikit-learn: steps 1 to 4 for 100 trees
    _forest = RandomForestClassifier(n_estimators=100, max_features=1, random_state=0)
    _forest.fit(_X_train, _y_train)
    print(np.sum(_forest.predict(_X_val) != _y_val))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The five bootstrap samples hold 125 to 134 different penguins of the 206, and the five trees get 3 to 8 of the 68 validation penguins wrong. About penguin 54, a Chinstrap, two trees say Gentoo and three say Chinstrap: the vote is right. Together the five trees get 3 wrong, as few as the best of them. scikit-learn's forest of 100 trees also gets 3 wrong. It does not count votes but averages what share of each leaf is each species; with one species in every leaf, as in a tree grown until every part is pure, that is the same vote.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bootstrap():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A bootstrap sample

        Step 1 of the recipe. `rng` is a random number generator with seed 1, and `n` is the number of training penguins, 206. Replace the `...`: put in `rows` the positions of a bootstrap sample, `n` whole numbers from 0 to `n - 1`, drawn with replacement. The check looks at what makes a bootstrap sample, not at the numbers themselves, and also tries another `n`.
        """),
        L.task("Put in `rows` the positions of a bootstrap sample of `n` penguins."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bootstrap():
    box_bootstrap = L.code_box("bootstrap", lines=6)
    box_bootstrap
    return (box_bootstrap,)


@app.cell(hide_code=True)
def check_bootstrap(box_bootstrap):
    result_bootstrap = L.check("bootstrap", box_bootstrap.value)
    result_bootstrap
    return (result_bootstrap,)


@app.cell(hide_code=True)
def hints_bootstrap():
    L.hints(
        "bootstrap",
        [
            "Step 1 of the recipe, as in the worked example, with `n` instead of 206.",
            "`rng.integers(low, high, size=...)` gives whole numbers from `low` up to, but not including, `high`. Which `low` and `high` give the positions from 0 to `n - 1`, and how many do you need?",
            "`rows = rng.integers(0, ___, size=___)`.",
        ],
        why="`rng.integers(0, n, size=n)` draws `n` positions, each from 0 to `n - 1`, and every draw can give any position again: with replacement. `rng.permutation(n)` would give every position exactly once: the same penguins in another order.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The vote

        Step 4 of the recipe in a function. `vote(votes)` gets a table of species, one row per tree and one column per penguin, and gives back for each penguin the species most trees say, as an array. Replace the two `...`. The check calls `vote(votes)` with five trees and four made-up penguins: that should give `["Adelie", "Gentoo", "Chinstrap", "Adelie"]`. It also tries other tables.
        """),
        L.task("Make `vote` give back, for each penguin, the species most trees say."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("vote", lines=9)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "vote",
        [
            "Step 4 of the worked example, with `votes` and `j`.",
            "The votes for one penguin are one column of the table: how do you take column `j`? And after `np.unique`, which position in `counts` holds the most votes?",
            "`np.unique(votes[:, ___], return_counts=True)` and `said.append(names[___])`.",
        ],
        why="For the first penguin the five trees say Adelie, Chinstrap, Adelie, Adelie and Gentoo: `names` is `['Adelie' 'Chinstrap' 'Gentoo']`, `counts` is `[3 1 1]`, `np.argmax(counts)` is 0, and `names[0]` is Adelie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.md(r"""
    ### Exercise 3. Why random features?
    """)
    return


@app.cell(hide_code=True)
def choice_features_cell():
    choice_features = L.choice(
        "random-features",
        question="100 trees on bootstrap samples of all four measurements: if every split may look at all four, 77 trees ask their first question about flipper length; with 2 random ones per split, 41 do. Why does a random forest want that?",
        options={
            "better": "So that each tree is right more often",
            "right": "So that the trees differ more: the vote can only outvote mistakes that most trees do not share",
            "faster": "So that the trees learn faster",
        },
        correct="right",
        why={
            "better": "A tree that may not look at the best measurement is, on its own, a little worse. What does the vote gain from trees that differ? See *The idea*.",
            "faster": "They do learn a little faster, but that is not the reason. Think of the guessers of the ox who all copy the loudest one. See *Why this matters*.",
            "right": "Trees that all ask the same first question look alike and are often wrong about the same penguins, and a mistake most trees make stays in the vote. Random features make the trees differ, at a price: each tree on its own is a little worse, so `max_features` is a hyperparameter to choose.",
        },
    )
    choice_features
    return (choice_features,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Each tree is the tree of [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en); the vote is the majority voting of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), and the variance of [Expected value and variance](/course/open/prob-expectation?lang=en) says how much a model changes with its random draws. The bootstrap comes back at the intermediate level in *How sure is a number? Confidence intervals and the bootstrap*. In the next lesson, *A neuron*, you take the first step towards neural networks.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_features, result_bootstrap, result_vote):
    L.wrap_up(result_bootstrap, result_vote, choice_features)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
