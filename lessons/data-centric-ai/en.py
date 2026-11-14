# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-centric-ai. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-centric-ai", "en")
    C = L.checks   # the 274 penguins with 41 labels the course made wrong, their out-of-fold probabilities, the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en) and one about [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_folds():
    quick_folds = L.choice(
        "quick-folds",
        question="Cross-validation cuts the 274 penguins into 5 folds. How often is each penguin in the validation set?",
        options={
            "never": "Never: all 274 are in the training set",
            "right": "Once: each fold takes one turn",
            "five": "5 times, once for every fold",
        },
        correct="right",
        why={
            "never": "Each fold takes one turn as the validation set, so every penguin is in it once. See *The idea* in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "five": "In the other four turns the penguin is in the training set. See *The idea* in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "right": "Right: each penguin is validated once, by a model that trained on the other four folds. You use that today.",
        },
    )
    quick_folds
    return (quick_folds,)


@app.cell(hide_code=True)
def quick_check_surprise():
    quick_surprise = L.choice(
        "quick-surprise",
        question="A model gave what really happened a probability of 0.1: its surprise is $-\\log(0.1) \\approx 2.3$. What is the surprise of an outcome it gave 0.9?",
        options={
            "same": "Also about 2.3",
            "nine": "About 21, nine times as much",
            "right": "About 0.1: a likely outcome surprises little",
        },
        correct="right",
        why={
            "same": "The surprise depends on the probability: the smaller it is, the larger the surprise. See *The idea* in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "nine": "A higher probability gives a smaller surprise, not a larger one. See *The idea* in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "right": "Right: $-\\log(0.9) \\approx 0.11$. Today a low probability for a label is a large surprise, and a reason to look again.",
        },
    )
    quick_surprise
    return (quick_surprise,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model learns from its labels, and labels have mistakes: typed wrong, judged wrong, copied wrong. In ten well-known test sets, Northcutt and colleagues (2021) estimated that at least 3.3% of the labels are wrong on average, and at least 6% in the validation set of ImageNet. You cannot check every label by hand. Which ones do you check first?

    **An everyday example.** A bookkeeper does not check all 300 receipts again. They look first at the ones that do not fit: a coffee of 450 euros is more likely a typo than a coffee of 4.50.
    """)
    return


@app.cell(hide_code=True)
def predict_found_cell():
    guess_found = L.guess(
        "found-guess",
        question="The course made 41 of the 274 penguin labels wrong. Checking 30 labels at random finds about 4.5 of them. How many do you find if you check the 30 labels that k-NN, trained without them, finds least likely?",
        options={
            "same": "About 5: k-NN cannot know which labels are wrong",
            "some": "About 12",
            "right": "About 25",
        },
        reveal="**About 25.** The 30 most suspect labels hold 25 of the 41 wrong ones here; with other labels made wrong at random, 22 to 27. A wrong label disagrees with the penguins around it, and k-NN notices.",
    )
    guess_found
    return (guess_found,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Wrong labels.** Wrong labels in a dataset are called **label noise**. A model trained on them learns some of the mistakes, and a test set with them measures against the mistakes.

        Here the course makes 41 of the 274 penguin labels outside the test set wrong itself, 15%, at random: each becomes one of the two other species. So it knows which labels are wrong, and can count how many you find. In real data nobody knows that; checking is the only way to find out. The 68 test penguins keep their right labels: you need an honest number.

        **The model as a detective.** A wrong label disagrees with the penguins around it: a penguin labeled Gentoo among Adelies. k-NN with k = 9, the k cross-validation chose, says how much: the probability it gives a species is the share of the 9 nearest penguins that have it. Look at the probability of the label each penguin has. A low probability is a large surprise, $-\log(p)$: that label is suspect. Check the suspects first, from the lowest probability to the highest.

        **Trained without it.** The k-NN must not have stored the penguin itself. Otherwise the penguin is one of its own neighbors, at distance 0, and votes for its own label, right or wrong. So use the folds of cross-validation: fold 1 gets its probabilities from a k-NN that stored only folds 2 to 5, fold 2 from folds 1, 3, 4 and 5, and so on. Every penguin gets a probability from a model that never saw it: an **out-of-fold prediction**.

        **Fix the data.** Check the suspects by hand, in the field notes, on a photo, or with an expert, and fix the labels that are wrong. Then train again and measure on the test set. Keeping the model the same and improving the data is called **data-centric AI**; changing the model, such as choosing another k, is the model-centric way.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **label noise** | wrong labels in a dataset: typed, judged or copied wrong | |
        | **out-of-fold prediction** | a prediction for an example by a model trained on the other folds, without that example | `_p` |
        | **data-centric AI** | improving a system by improving its data, such as fixing labels, while the model stays the same | |
        | `np.mean(a == b)` | the share of places where `a` and `b` are the same: `a == b` gives `True` and `False`, and the mean counts `True` as 1 and `False` as 0 | `np.mean(_given[_train][_nearest] == _given[_i])` |
        | `np.sum(a != b)` | the number of places where `a` and `b` differ | |
        """),
        L.watch_out("A low probability says: look at this one. It does not say that the label is wrong: some penguins look unlike their own species. Change only labels you checked."),
        mo.md(r"""
        The slider says how many labels you check, the most suspect first. Left, how many of the 41 wrong labels you find, against checking at random, the dashed line. Right, how many of the 68 test penguins k-NN gets wrong after you fix what you found: with k = 1, which listens to one neighbor, and with k = 9. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_checked_cell():
    def _frame(c):
        said = C.said(c, L.lang)
        alt = ("Left: how many of the 41 wrong labels you find against how many labels you check, the most suspect first "
               "and at random. Right: the mistakes of k-NN with k = 1 and k = 9 on the 68 test penguins after fixing what "
               "you found. " + said)
        return (C.picture(c, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_checked = L.explore("checked", "labels checked", _frame, start=30, values=C.CHECKS, role="hyper",
                                task="Move the slider from 0 to 100. Where does checking stop finding many wrong labels, and what does fixing do for k = 1 and for k = 9?")
    explore_checked
    return


@app.cell(hide_code=True)
def text_after_checked():
    mo.md(r"""
    The first 30 checks find 25 wrong labels, the next 20 another 13; after 50 checks only 3 are left: one turns up at check 58, and the other two look right to k-NN, with 7 and 8 of their 9 neighbors sharing their label. k = 1 suffers most from wrong labels: 14 test mistakes with them, 6 with all labels right, and 7 after fixing the 25 of the first 30 checks. k = 9 outvotes most wrong labels by itself: 3 mistakes, and 2 after fixing.

    And the model-centric way? Choosing k again with cross-validation, on the labels with the mistakes, picks k = 15, and that also gets 2 test penguins wrong: a larger vote hides random mistakes too. But the labels stay wrong. The next model you train on them suffers again, and cross-validation itself compares with them: a right prediction for a penguin with a wrong label counts as a mistake. Fixed labels stay fixed: every model you train on them later, and every comparison cross-validation makes with them, starts from the right labels.

    ### Recipe

    To find the wrong labels in your data:

    1. Cut the examples into 5 folds, as in cross-validation.
    2. For each fold, train on the other four, and let the model give each example of this fold a probability for its own label. For k-NN: the share of its 9 nearest neighbors with that label.
    3. Sort from the lowest probability to the highest.
    4. Check the first ones by hand, and fix the labels that are wrong.
    5. Train again on the fixed data, and measure on the test set, which played no part.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes the same 41 labels wrong as the slider, gives each of the 274 penguins the probability of its label out of fold, with k = 9, and prints the 8 most suspect: the penguin's number among the 274, its label, the probability, and the species it really is. Then it counts the wrong labels among the 30 most suspect. Run the cell.
    """)
    return


@app.cell
def example_suspects():
    import numpy as np
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    _rest = _order[:274]

    # the course makes 41 of the 274 labels wrong: each becomes one of the two other species
    _rng = np.random.default_rng(15)
    _given = np.array(_y)                                  # a new array with the same labels
    _wrong = _rng.permutation(_rest)[:41]
    for _i in _wrong:
        _others = []
        for _s in ["Adelie", "Chinstrap", "Gentoo"]:
            if _s != _y[_i]:
                _others.append(_s)
        _given[_i] = _others[_rng.integers(0, 2)]

    _p = []
    for _f in range(5):
        # 1. fold _f gets its probabilities from the other four folds
        _val = _rest[55 * _f:55 * (_f + 1)]
        _train = np.concatenate([_rest[:55 * _f], _rest[55 * (_f + 1):]])
        _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
        _X_train = (_X[_train] - _mean) / _std
        for _i in _val:
            # 2. the 9 nearest penguins among the other folds
            _dists = np.linalg.norm(_X_train - (_X[_i] - _mean) / _std, axis=1)
            _nearest = np.argsort(_dists, kind="stable")[:9]
            # 3. the probability of its own label: the share of the 9 with that label
            _p.append(np.mean(_given[_train][_nearest] == _given[_i]))
    # 4. the most suspect first
    _suspects = np.argsort(_p, kind="stable")[:30]
    for _j in _suspects[:8]:
        print(_j, _given[_rest[_j]], round(_p[_j], 2), _y[_rest[_j]])
    print(np.sum(_given[_rest[_suspects]] != _y[_rest[_suspects]]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The 8 most suspect all have probability 0: none of their 9 neighbors has their label. The first six are wrong. Penguin 3, an Adelie of 40.6 mm and 3550 g, has the label Gentoo, and Gentoos weigh around 5000 g. Penguins 107 and 109 are right, though: Adelies with a bill of about 46 mm, almost as long as a Chinstrap's. The model suspects them; you look, and leave them as they are. Of the 30 most suspect labels, 25 are wrong.

    `_given[_train][_nearest]` first takes the labels of the training folds, and then those of the 9 nearest among them: `_nearest` holds positions in `_train`, not in all penguins.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_p_given():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The probability of a penguin's own label

        `p_given(X_train, y_train, x, label, k)` gets standardized training penguins and their labels, one penguin `x` with its label `label`, and `k`. It should give the probability k-NN gives that label: the share of the `k` nearest training penguins whose label is `label`. Replace the two `...`. The check calls `p_given(X_train, y_train, np.array([0.2, -0.9]), "Chinstrap", 3)` with the ten made-up penguins of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), which should give one third, and seven more.
        """),
        L.task("Let `p_given` give the share of the `k` nearest with the penguin's own label."),
    ])
    return


@app.cell(hide_code=True)
def exercise_p_given():
    box_p_given = L.code_box("p-given", lines=8)
    box_p_given
    return (box_p_given,)


@app.cell(hide_code=True)
def check_p_given(box_p_given):
    result_p_given = L.check("p-given", box_p_given.value)
    result_p_given
    return (result_p_given,)


@app.cell(hide_code=True)
def hints_p_given():
    L.hints(
        "p-given",
        [
            "Two steps: which training penguins are the `k` nearest, and what share of them has `label`?",
            "As in k-NN, `np.argsort(dists, kind=\"stable\")` puts the nearest first. Comparing an array with `==` gives `True` and `False`: what does `np.mean` make of those?",
            "`nearest = np.argsort(dists, kind=\"stable\")[___]` and `return np.mean(y_train[___] == label)`.",
        ],
        why="For the example, the 3 nearest are a Chinstrap and two Adelies: `y_train[nearest] == \"Chinstrap\"` gives `True`, `False` and `False`, and their mean is one third.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_suspects():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The most suspect first

        `suspects(p, n)` gets the probabilities `p` of the labels, and should give the positions of the `n` most suspect, the lowest probability first. Replace the two `...`. The check calls `suspects(np.array([0.89, 0.11, 1.0, 0.0, 0.56, 0.22]), 3)`, which should give `[3, 1, 5]`, and five more.
        """),
        L.task("Let `suspects` give the positions of the `n` lowest probabilities, the lowest first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_suspects():
    box_suspects = L.code_box("suspects", lines=6)
    box_suspects
    return (box_suspects,)


@app.cell(hide_code=True)
def check_suspects(box_suspects):
    result_suspects = L.check("suspects", box_suspects.value)
    result_suspects
    return (result_suspects,)


@app.cell(hide_code=True)
def hints_suspects():
    L.hints(
        "suspects",
        [
            "Which function gives positions in the order of their values, from small to large?",
            "`np.argsort` gives the positions from the lowest value to the highest, equal ones in their order with `kind=\"stable\"`. You need only the start of it.",
            "`order = np.argsort(p, kind=\"stable\")` and `return order[___]`.",
        ],
        why="For the example, `np.argsort` gives 3, 1, 5, 4, 0 and 2: probability 0 first, then 0.11 and 0.22. The first 3 are the suspects.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_self():
    mo.md(r"""
    ### Exercise 3. Why without the penguin itself
    """)
    return


@app.cell(hide_code=True)
def choice_self_cell():
    choice_self = L.choice(
        "self",
        question="You give every penguin the probability of its label with k = 1, from a k-NN that stored all 274 penguins, the penguin itself too. What do you get?",
        options={
            "found": "0 for every wrong label and 1 for every right one: it finds them all",
            "right": "1 for every penguin: its nearest neighbor is itself",
            "same": "The same as out of fold: one penguin among 274 hardly matters",
        },
        correct="right",
        why={
            "found": "That would need a model that knows the right species. A k-NN that stored the penguin itself finds the penguin first, at distance 0, with the label it has, wrong or not.",
            "same": "With k = 1 the nearest neighbor decides alone, and that is the penguin itself, at distance 0. Out of fold, it is another penguin.",
            "right": "Right: every penguin agrees with itself, so every probability is 1 and says nothing. The order is then just the table's: checking the first 30 finds 6 of the 41 wrong labels, about what random checking finds (4.5 on average). Out of fold, k = 1 finds 14 and k = 9 finds 25.",
        },
    )
    choice_self
    return (choice_self,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Wrong labels slip through even with the good guidelines of [Labeling well: guidelines and agreement](/course/open/data-annotation?lang=en). The folds of [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en) give every penguin a model that never saw it, and a low probability is a large surprise, as in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en). Fixing the data is a way to make a model better, besides choosing the model and its hyperparameters. The next lesson, *Data you make yourself*, makes new examples when there are too few.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_self, result_p_given, result_suspects):
    L.wrap_up(result_p_given, result_suspects, choice_self)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
