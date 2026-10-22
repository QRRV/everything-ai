# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-cross-validation. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-cross-validation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vote():
    quick_vote = L.choice(
        "quick-vote",
        question="With k = 5, the nearest training penguins of a new penguin are an Adelie, a Chinstrap, a Chinstrap, an Adelie and a Chinstrap, from near to far. What does k-NN predict?",
        options={
            "nearest": "Adelie: the nearest one is an Adelie",
            "right": "Chinstrap",
            "none": "Nothing: it is a tie",
        },
        correct="right",
        why={
            "nearest": "That is what k = 1 would say. With k = 5, all five vote. See *The idea* in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "none": "Count the votes of all five. See the worked example in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "Three Chinstraps against two Adelies: majority voting gives Chinstrap. That was the second validation penguin of the worked example.",
        },
    )
    quick_vote
    return (quick_vote,)


@app.cell(hide_code=True)
def quick_check_choose():
    quick_choose = L.choice(
        "quick-choose",
        question="How did you choose k?",
        options={
            "train": "The k with the fewest mistakes on the training penguins",
            "right": "The k with the fewest mistakes on the validation penguins",
            "test": "The k with the fewest mistakes on the test penguins",
        },
        correct="right",
        why={
            "train": "On the training penguins k = 1 makes no mistakes at all, and says nothing about new penguins. See exercise 3 in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "test": "The test set stays out of it until everything is chosen. See step 5 of the recipe in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "k-NN does not keep those penguins, so they show which k works on new penguins.",
        },
    )
    quick_choose
    return (quick_choose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the lesson before, k = 15 won on the validation set with 2 mistakes, against 4 for k = 9. But that is 68 penguins: a few other penguins in the validation set, and another k wins. You cannot simply take a bigger validation set, as every penguin you move there is one less to train on. What you can do is let every penguin take a turn: split them into parts, and let each part be the validation set once while the others train. Then the mistakes are counted on all of them.

    **An everyday example.** You do not pick the better of two penalty takers after five shots each: one lucky shot can decide. Let them take fifty, and count them all.
    """)
    return


@app.cell(hide_code=True)
def predict_how_many():
    guess_how_many = L.guess(
        "how-many",
        question="Shuffle the 274 penguins outside the test set eight times, and choose k each time on one validation set of 68, with the other 206 to train on. How many different values of k do you think the eight shuffles choose?",
        options={
            "one": "1: always the same k",
            "few": "2 or 3",
            "many": "4 or more",
        },
        reveal="**5 different values**: 15, 25, 1, 25, 3, 9, 1 and 1. With 68 penguins, one mistake more or less decides, and which penguins land in the validation set is luck. Cross-validation, below, chooses k = 9 in five of the eight shuffles, and 5 or 15 in the others.",
    )
    guess_how_many
    return (guess_how_many,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Cross-validation** splits the examples you may use to choose, here the 274 penguins outside the test set, into parts of almost the same size, the **folds**: often 5, as here, or 10. Each fold takes one turn as the validation set: train on the other folds, and count the mistakes on this one. After 5 turns every penguin has been in the validation set exactly once, so you add up the mistakes of the 5 folds: a count over all 274 penguins instead of 68. Do that for every value of the hyperparameter, and keep the value with the fewest. Then train once more, on all 274 with that value, and measure once on the test set.

        It costs more: 5 folds are 5 trainings for every value. For k-NN, training is only standardizing and keeping the penguins, so that is cheap here.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **cross-validation** | | estimating how well a model does on new examples by letting every fold take one turn as the validation set, and adding up the mistakes or the loss over all folds; here, to choose a hyperparameter | |
        | **fold** | | one of the parts, of almost the same size, that the examples are split into for cross-validation | `_val` |
        | `np.concatenate([a, b])` | concatenate | the arrays `a` and `b` one after the other, as one array: `[1, 2]` and `[3]` give `[1, 2, 3]` | `_train` |
        """),
        L.watch_out("This is often called k-fold cross-validation, with k the number of folds: another k than that of k-nearest neighbors. This lesson says 5 folds, and keeps k for the neighbors."),
        mo.md(r"""
        Eight shuffles of the 274 penguins; shuffle 1 is the order of the lesson before. For each, one chart shows, for every k, the share of the 68 penguins of one validation set that k-NN gets <span class="c-loss">wrong</span>; the other the share of all 274 over the 5 folds of cross-validation. k is on a log scale. The thin lines are the other seven shuffles, and the open dot the k each way chooses. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_shuffle_cell():
    def _frame(s):
        s = int(s)
        one, cv = C.chosen(C.one_split(s)), C.chosen(C.cross_validation(s))
        lo1, hi1 = C.spread("one")
        lo2, hi2 = C.spread("cv")
        alt = (f"Two charts of the share of penguins k-NN gets wrong for every k from 1 to 121, eight lines each, one of them bold for shuffle {s}. "
               f"For one validation set of 68, the lines lie far apart, between {L.number(lo1)} and {L.number(hi1)} for k from 1 to 41, and shuffle {s} chooses k = {one}. "
               f"For cross-validation over all 274, the lines lie close together, between {L.number(lo2)} and {L.number(hi2)} for k from 1 to 41, and shuffle {s} chooses k = {cv}. "
               "Almost every line rises steeply from k = 71 to 121.")
        return (C.picture(s, {"one": "One validation set: 68 penguins", "cv": "Cross-validation: all 274", "share": "share wrong", "alt": alt}),
                f"""Shuffle {s}. One validation set chooses <span class="c-hyper">k = {one}</span>; cross-validation chooses <span class="c-hyper">k = {cv}</span>.""")

    explore_shuffle = L.explore("shuffle", "shuffle", _frame, start=1, low=1, high=8, step=1, role="",
                                task="Go through the eight shuffles. Which k does one validation set choose each time, and which does cross-validation choose?")
    explore_shuffle
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    One validation set chooses five different values of k in eight shuffles: 15, 25, 1, 25, 3, 9, 1 and 1. Its lines jump around, as one penguin is 0.015 of 68: a single lucky or unlucky penguin moves them. The lines of cross-validation lie almost on top of each other. It chooses k = 9 in five shuffles, and 5 or 15 in the others; twice k = 5 ties with k = 9, and `np.argmin` takes the first. And in every shuffle it counts more mistakes for k = 1 than for k = 9, while one validation set chose k = 1 three times, twice in a tie with five other values of k.

    ### Recipe

    To choose a hyperparameter with 5-fold cross-validation:

    1. Keep the test set apart, and shuffle the other examples.
    2. Cut the shuffled positions into 5 folds: fold `f` is `order[55 * f:55 * (f + 1)]`, and the training set is the rest, `np.concatenate([order[:55 * f], order[55 * (f + 1):]])`.
    3. For each value of the hyperparameter and each fold: train on the other folds, and count the mistakes on this fold.
    4. Add up the mistakes of the 5 folds, and keep the value with the fewest.
    5. Train on all of them with that value, and measure once on the test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the lesson before, with bill length and body mass: the 68 of the test set apart, and the other 274 in the order of shuffle 1. `_mistakes` is the k-NN of the lesson before: it standardizes with the penguins it trains on, lets the k nearest vote, and counts what it gets wrong. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_cross_validation():
    import numpy as np
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    # 1. the test set apart; the other 274 are shuffled already
    _rest, _test = _order[:274], _order[274:]


    def _mistakes(train, val, k):
        mean, std = _X[train].mean(axis=0), _X[train].std(axis=0)
        X_train, X_val = (_X[train] - mean) / std, (_X[val] - mean) / std
        wrong = 0
        for i in range(len(val)):
            nearest = np.argsort(np.linalg.norm(X_train - X_val[i], axis=1), kind="stable")[:k]
            names, counts = np.unique(_y[train][nearest], return_counts=True)
            if names[np.argmax(counts)] != _y[val][i]:
                wrong = wrong + 1
        return wrong


    _ks = [1, 3, 5, 9, 15, 25, 41, 71, 121]
    _totals = []
    for _k in _ks:
        _per_fold = []
        _total = 0
        for _f in range(5):
            # 2. fold _f validates, the other four folds train
            _val = _rest[55 * _f:55 * (_f + 1)]
            _train = np.concatenate([_rest[:55 * _f], _rest[55 * (_f + 1):]])
            # 3. the mistakes on this fold
            _wrong = _mistakes(_train, _val, _k)
            _per_fold.append(_wrong)
            _total = _total + _wrong
        # 4. added up over the 5 folds: every penguin counted once
        print(_k, _per_fold, _total)
        _totals.append(_total)
    _best = _ks[np.argmin(_totals)]
    print(_best)
    # 5. all 274 to vote, and once on the test set
    print(_mistakes(_rest, _test, _best))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each line gives k, the mistakes on each of the 5 folds, and their sum. For k = 9 the folds count 4, 5, 2, 5 and 3: on about 55 penguins each, anything from 2 to 5, the luck of a single validation set again. Added up, cross-validation counts 19 mistakes of 274 for k = 9, the fewest, and 20 for k = 15: almost as good. In the lesson before, one validation set had 2 mistakes for k = 15 and 4 for k = 9; that difference was mostly which 68 penguins happened to be in it. With k = 9 and all 274 penguins to vote, k-NN gets 2 of the 68 test penguins wrong: that is the number you report. It is a smaller share than 19 of 274: with 68 test penguins, luck plays a part here too. Honest means that the test set played no part in choosing, not that its number is exact.

    `np.concatenate` puts the part before fold `_f` and the part after it one after the other: for the first fold the part before is empty, for the last one the part after. The last fold has 54 penguins, as 274 is not a multiple of 5; `_rest[220:275]` simply stops at the end.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why more reliable?
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-reliable",
        question="Why does cross-validation choose k more reliably than one validation set of 68 penguins?",
        options={
            "train": "In each fold, k-NN keeps more penguins: 219 or 220 instead of 206",
            "right": "It counts the mistakes on all 274 penguins, so a few lucky or unlucky penguins weigh less",
            "test": "It also uses the test set",
        },
        correct="right",
        why={
            "train": "That is true, but a few more penguins to vote change little. What matters most is how many penguins the mistakes are counted on. See *The idea*.",
            "test": "The test set stays apart, in every fold. See step 1 of the recipe.",
            "right": "Every penguin takes one turn in the validation set, so the count is over 274 penguins instead of 68, and luck averages out more.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Folds for a curve

        Cross-validation works for any hyperparameter, also the degree of a curve. `x` and `y` are the 18 temperatures of exercise 2 in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), its 12 to train and 6 to validate together, and `order` shuffles their positions. The box uses 3 folds of 6. Replace the two `...`: put the positions of fold `f` in `val`, and the positions of the other two folds in `train`. The box then fits a curve of each degree on `train`, adds up its squared residuals on `val`, and divides by 18: the MSE over all 18 temperatures, each measured once. The last line is `cv_mses`, so the check sees it. The check also tries other degrees.
        """),
        L.task("Put the positions of fold `f` in `val`, and those of the other folds in `train`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("folds", lines=15)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "folds",
        [
            "Step 2 of the recipe, with folds of 6 instead of 55.",
            "Fold `f` starts at position `6 * f` of `order` and stops before `6 * (f + 1)`. `train` is what comes before that slice, followed by what comes after it.",
            "`val = order[6 * f:___]` and `train = np.concatenate([order[:6 * f], order[___:]])`.",
        ],
        why="With 3 folds of 6, the cross-validation MSEs of degrees 1 to 6 are about 7.26, 1.00, 1.28, 1.65, 1.25 and 2.35: degree 2 has the lowest, as the validation set chose in that lesson.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Exercise 3. And then?
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after-cv",
        question="Cross-validation chose k = 9, with 19 mistakes of 274. What do you do now?",
        options={
            "fold": "Keep the k-NN of the fold with the fewest mistakes",
            "right": "Let k-NN with k = 9 use all 274 penguins, and measure it once on the test set",
            "report": "Report 19 of 274 as how k-NN will do on new penguins",
        },
        correct="right",
        why={
            "fold": "Each fold's k-NN kept only four fifths of the penguins. With the value chosen, use them all. See step 5 of the recipe.",
            "report": "Those 19 helped choose k, so the number is a little too optimistic, as with one validation set. See step 5 of the recipe.",
            "right": "The test set played no part in choosing, so its count is an honest estimate for new penguins: here 2 of 68.",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) one validation set chose the degree; cross-validation lets every example take a turn, and chose degree 2 as well. In [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) k = 15 won on 68 penguins; over all 274, k = 9 to 41 are about as good, and k = 1 is worse in every shuffle. In the next lesson, *Regularization and weight decay*, you keep a flexible model in check another way: by making large weights cost something. How much they cost is one more hyperparameter, chosen the same way.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why, result_folds, choice_after):
    L.wrap_up(choice_why, result_folds, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
