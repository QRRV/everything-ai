# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-decision-tree. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-decision-tree", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en) and [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="The computer learned the rule \"coat when it is colder than T °C\" from twelve mornings. How did it choose T?",
        options={
            "middle": "It took the middle between the coldest and the warmest morning",
            "right": "It tried every T and kept the one with the fewest mistakes on the mornings",
            "understand": "It worked out from what cold is which T fits",
        },
        correct="right",
        why={
            "middle": "That would not look at the coats at all. See the recipe in [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en).",
            "understand": "The computer knows nothing about cold; it only sees numbers and answers. See the recipe in [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en).",
            "right": "Try every rule of the kind, count its mistakes, keep the fewest. This lesson does that again and again.",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="A model gets every training example right and many test examples wrong. What does it do?",
        options={
            "under": "It underfits",
            "fine": "It does well",
            "right": "It overfits",
        },
        correct="right",
        why={
            "under": "Then it would get many training examples wrong too. See *The idea* in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "fine": "Then it would do about as well on new examples. See *The idea* in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "right": "Perfect on the examples it learned from, bad on new ones: it follows the noise.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en) the computer learned one rule, coat when it is colder than T °C: one question, two answers. The penguins of [Project: predict something real, end to end](/course/open/ml-project-tabular?lang=en) come in three species, with four measurements each. One threshold can put the Gentoos apart, but leaves the Adelies and the Chinstraps together. So ask a second question in each part, and a third where that is not enough yet.

    **An everyday example.** A field guide finds the name of a bird with questions. Is the bill hooked? Yes: is the bird bigger than a crow? Each answer leads to the next question, until one species is left. Biologists call such a list of questions a key.
    """)
    return


@app.cell(hide_code=True)
def predict_pure():
    guess_pure = L.guess(
        "pure-or-two",
        question="Two trees learn from bill length and body mass of the 206 training penguins. One asks at most two questions and gets 15 of them wrong; the other keeps asking until it gets all 206 right. Which one do you think gets fewer of the 68 validation penguins wrong?",
        options={
            "pure": "The one that gets every training penguin right",
            "two": "The one with at most two questions",
            "same": "They get the same number wrong",
        },
        reveal="**The one with at most two questions:** 7 of the 68 wrong, against 9 for the tree that gets every training penguin right. That tree asks up to seven questions and ends in 23 leaves, some for a single penguin: it also learned their quirks.",
    )
    guess_pure
    return (guess_pure,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **decision tree** predicts with threshold questions, one after another, like a flowchart. The first question is, say, "flipper length at most 207.5 mm?"; yes leads one way and no the other, and on each side a new question can follow. Each question is a **split**: one feature and one threshold that split the examples in two parts. Where no question follows, a **leaf** gives the prediction: the class most of its training examples have.

        A tree learns with the rule of [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en), over and over, in every part: the recipe below has the steps. It stops where a part is pure, with one class only, or where it reaches its **maximum depth**, a hyperparameter: the most questions you allow on any way down. Without a limit the tree goes on until every part is pure, and then it gets every training penguin right.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **decision tree** | a model that predicts with threshold questions, one after another, like a flowchart, until a leaf gives the answer | `DecisionTreeClassifier()` |
        | **split** | one question of a tree: one feature and one threshold, which split the examples in two parts | `x <= t` |
        | **leaf** | the end of a way down the tree, where no question follows; it says the most common class of its training examples | |
        | **depth**, **maximum depth** | the most questions on any way from the top to a leaf; the maximum depth is the limit you set | `max_depth=2` |
        | **Gini impurity** | one number for how mixed the classes in a part are: 0 when they are all the same | `criterion="gini"` |
        | `export_text(tree, feature_names=names)` | the questions of a tree that has learned, as text: one line per question, with the leaves in it | |
        """),
        L.watch_out("A split here is one question of the tree, not the split of the data into a training, a validation and a test set."),
        mo.md(r"""
        The picture shows a tree on two of the measurements, bill length and body mass, of the 206 training penguins of the project: a circle is an Adelie, a triangle a Chinstrap and a square a Gentoo; here color and shape mean the species. The <span class="c-pred">blue line</span> is the decision boundary of the tree: on either side of it, the tree predicts a different species. It has only horizontal and vertical pieces, because each question looks at one measurement. The other chart counts the <span class="c-loss">mistakes</span> for every maximum depth: on the 206 training penguins, the solid line, and on the 68 validation penguins, the dashed line. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_depth_cell():
    def _frame(depth):
        depth = int(depth)
        train, val = (m[C.DEPTHS.index(depth)] for m in C.curve())
        alt = (f"The first chart: the 206 training penguins by bill length and body mass, a color and shape per species. Adelies have short bills, Gentoos are heavy, Chinstraps have long bills and are light. With a maximum depth of {depth}, a blue line of horizontal and vertical pieces separates the areas where the tree predicts each species"
               + ("; there is no area for the Chinstraps." if C.species_in_picture(depth) == 2 else
                  "; the larger the depth, the smaller the boxes it cuts around single penguins.")
               + f" The second chart: the training mistakes fall from 45 at depth 1 to 0 at depth 7; the validation mistakes fall from 16 to 7 at depth 2, 3 and 4, and are 9 at depth 7. At depth {depth}: {train} training and {val} validation penguins wrong.")
        said = (f"""Depth {depth}, {C.leaves(depth)} leaves: <span class="c-loss ea-num">{train}</span> of the 206 training penguins wrong, """
                f"""<span class="c-loss ea-num">{val}</span> of the 68 validation penguins.""")
        if C.species_in_picture(depth) == 2:
            said += " The tree never says Chinstrap."
        return (C.picture(depth, {"title": f"Decision tree, depth {depth}", "x": "bill length (mm)", "y": "body mass (g)",
                                  "every": "Mistakes for every depth", "depth": "maximum depth", "mistakes": "mistakes",
                                  "train": "training, of 206", "val": "validation, of 68", "alt": alt}), said)

    explore_depth = L.explore("depth", "maximum depth", _frame, start=1, values=C.DEPTHS, role="hyper",
                              task="Move the depth from 1 to 7. At which depth are the fewest validation penguins wrong, and what happens to the parts after it?")
    explore_depth
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    At depth 1 the tree asks one question, bill length at most 43.15 mm, and never says Chinstrap. At depth 2 each part gets its own question, and 7 validation penguins are wrong, as at depths 3 and 4. Deeper, the tree cuts ever smaller boxes around single penguins: at depth 7 it gets every training penguin right, but 9 validation penguins wrong. As with the degree of a curve in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en), too few questions underfit and too many overfit. On the validation set you choose depth 2, the first with the fewest.

    ### Recipe

    To grow a decision tree:

    1. For every feature and every threshold, split the examples in two: `x <= t` and `x > t`. Each part says its most common class; count the mistakes of the two parts together.
    2. Keep the split with the fewest mistakes.
    3. Do steps 1 and 2 again in each part, until a part is pure or the tree has reached its maximum depth. Each leaf says the most common class of its examples.
    4. To predict, follow the questions from the top down to a leaf.
    5. Choose the maximum depth on the validation set.

    scikit-learn does steps 1 to 4: `DecisionTreeClassifier(max_depth=...)`, with `fit` and `predict`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the project, shuffled with the same seed: 206 to train on and 68 to choose on. First steps 1 and 2 by hand, for the first question, on all four measurements; `np.unique(x)[:-1]` gives every measured value once, from small to large, but the largest, which would leave the second part empty. Then scikit-learn grows the tree to depth 2. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_tree():
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _y = np.array(_train["species"])
    # 1. one split, bill length at most 45 mm: each part says its most common species
    _x = np.array(_train["bill_length_mm"])
    for _part in [_y[_x <= 45], _y[_x > 45]]:
        _names, _counts = np.unique(_part, return_counts=True)
        print(_names, _counts, len(_part) - max(_counts))
    # 1. and 2. every feature and every threshold: the split with the fewest mistakes
    for _feature in _features:
        _x = np.array(_train[_feature])
        _best, _fewest = None, len(_y)
        for _t in np.unique(_x)[:-1]:
            _wrong = 0
            for _part in [_y[_x <= _t], _y[_x > _t]]:
                _names, _counts = np.unique(_part, return_counts=True)
                _wrong = _wrong + len(_part) - max(_counts)
            if _wrong < _fewest:
                _best, _fewest = _t, _wrong
        print(_feature, _best, _fewest)
    # 3. scikit-learn does the same in each part, to depth 2
    _tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    _tree.fit(_train[_features], _train["species"])
    print(export_text(_tree, feature_names=_features, show_weights=True))
    # 4. predict the validation penguins, and measure
    print(np.mean(_tree.predict(_val[_features]) == _val["species"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bill length at most 45 mm makes 17 + 35 = 52 mistakes: 4 Chinstraps and 13 Gentoos among the Adelies, 2 Adelies and 33 Chinstraps among the Gentoos. Flipper length at most 202 mm is the best split, with 40: it puts the Gentoos apart. Done again in each part, the recipe asks bill length on the left and bill depth on the right: 9 mistakes in all.

    scikit-learn's tree asks about the same three measurements, with other thresholds: it puts each threshold halfway between two measured values, and instead of the fewest mistakes it looks for the split that lowers the Gini impurity of the parts most: a finer score, which breaks many ties between splits with equally many mistakes. A flipper length of at most 202, 205, 206 or 207 mm makes 40 mistakes each; the Gini impurity picks 207, so the threshold is 207.5 mm. The tree makes the same 9 mistakes on the training penguins.

    Read the tree from the top: each `|` further to the right is one question deeper. With `show_weights=True`, `weights` in a leaf counts its training penguins, Adelie, Chinstrap and Gentoo; `class` is what it says. It gets 3 of the 68 validation penguins wrong: 0.96 right. `random_state=0` decides between questions that split equally well, so the tree comes out the same every time.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The mistakes of one split

        Put step 1 of the recipe in a function. `split_mistakes(x, y, t)` gets one measurement `x` of a number of penguins and their species `y`, two NumPy arrays of the same length, and a threshold `t`, and gives back the mistakes of the split `x <= t` when each part says its most common species. Replace the three `...`, as in the worked example. The check calls `split_mistakes(x, y, 200)` with twelve made-up penguins, `x` their flipper lengths: that should give 4. It also tries other thresholds, always between the smallest and the largest `x`, and other penguins.
        """),
        L.task("Make `split_mistakes` give back how many penguins the split at `t` gets wrong."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split-mistakes", lines=9)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split-mistakes", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split-mistakes",
        [
            "Step 1 of the recipe, as in the first loop of the worked example, with `t` instead of 45.",
            "Which mask keeps the species of the penguins whose `x` is at most `t`? And how many penguins of a part are not of its most common species?",
            "`for part in [y[x <= t], y[___]]:` and `wrong = wrong + len(part) - ___`.",
        ],
        why="At 200 mm the first part has 4 Adelies and 2 Chinstraps, the second 4 Gentoos and 2 Chinstraps: `len(part) - max(counts)` is 2 for each, 4 in all.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_depth():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Choose the depth

        The box reads the penguins and divides them as in the worked example, and makes a tree for every maximum depth from 1 to 8. Replace the first `...` with lines that let the tree learn from `train`, on the four measurements, and put in `wrong` how many penguins of `val` it gets wrong. Then put the depth with the fewest mistakes in `best`. The last line is `best`, so the check sees it. The check also tries other depths.
        """),
        L.task("Put the maximum depth with the fewest mistakes on the validation set in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_depth():
    box_depth = L.code_box("choose-depth", lines=19)
    box_depth
    return (box_depth,)


@app.cell(hide_code=True)
def check_depth(box_depth):
    result_depth = L.check("choose-depth", box_depth.value)
    result_depth
    return (result_depth,)


@app.cell(hide_code=True)
def hints_depth():
    L.hints(
        "choose-depth",
        [
            "Steps 3 and 4 of the worked example, with `depth` instead of 2; then step 5 of the recipe, with `np.argmin` as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "`tree.fit` gets the measurements and the species of `train`. Which comparison of `tree.predict(val[features])` with `val[\"species\"]` is `True` for a mistake, and how do you count the `True`s?",
            "`tree.fit(train[features], ___)`, `wrong = np.sum(tree.predict(val[features]) != ___)`, and `best = depths[___]`.",
        ],
        why="The mistakes for the eight depths are 13, 3, 3, 2, 4, 4, 4 and 4. The fewest are at position 3, and `depths[3]` is 4. From depth 6 on, every part of the tree is pure, so it grows no further.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.md(r"""
    ### Exercise 3. Read the tree
    """)
    return


@app.cell(hide_code=True)
def choice_read_cell():
    choice_read = L.choice(
        "read-tree",
        question="The tree of depth 2 from the worked example gets a penguin with a flipper length of 215 mm and a bill depth of 18.5 mm. What does it say?",
        options={
            "gentoo": "Gentoo: its flipper is longer than 207.5 mm",
            "right": "Adelie, from a leaf with only 4 training penguins",
            "chinstrap": "Chinstrap: its bill is deep",
        },
        correct="right",
        why={
            "gentoo": "That is only the first question. A penguin that goes that way gets a second one: which? Look at the lines one `|` further to the right, under `flipper_length_mm >  207.50`.",
            "chinstrap": "Follow the questions from the top: first the flipper length, then, on that side, the bill depth. In which leaf do you end, and what is its `class`?",
            "right": "215 mm is more than 207.5 and 18.5 mm more than 17.65, so it ends in the last leaf: weights 2, 2 and 0, two Adelies and two Chinstraps, and a tie goes to the first species. A leaf of four penguins says little: with other training penguins it could just as well say Chinstrap.",
        },
    )
    choice_read
    return (choice_read,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The rule of [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en) is a tree of depth 1; a tree finds such a rule again in every part. Its maximum depth is a hyperparameter you choose on the validation set, as the degree in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) and k in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en); unlike k-NN, a tree needs no standardizing, because each question looks at one feature. scikit-learn grows it with the same `fit` and `predict` as in [Project: predict something real, end to end](/course/open/ml-project-tabular?lang=en). A deep tree follows single penguins, so with other training penguins it can come out quite different. In the next lesson, *Many trees vote: the random forest*, you let many such trees vote, and the vote varies far less than one tree.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_read, result_depth, result_split):
    L.wrap_up(result_split, result_depth, choice_read)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
