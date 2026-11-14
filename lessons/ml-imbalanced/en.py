# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-imbalanced. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-imbalanced", "en")
    C = L.checks   # the machine runs of AI4I 2020, the models with and without weights, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) and one about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_precision():
    quick_f1 = L.choice(
        "quick-f1",
        question="A model has precision 0.8 and recall 0.2. What is its F1 score?",
        options={
            "mean": "0.5",
            "right": "0.32",
            "product": "0.16",
        },
        correct="right",
        why={
            "mean": "0.5 is the plain mean. The F1 score is $2PR / (P + R) = 2 \\cdot 0.8 \\cdot 0.2 / 1.0 = 0.32$: it stays close to the smaller of the two. See *The idea* in [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "product": "0.16 is the product. The F1 score is twice the product divided by the sum: $2 \\cdot 0.8 \\cdot 0.2 / (0.8 + 0.2) = 0.32$. See *The idea* in [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "right": "$2 \\cdot 0.8 \\cdot 0.2 / (0.8 + 0.2) = 0.32$, close to the smaller of the two. The first model of today also has a high precision and a low recall, and you see how to trade one against the other when failures are rare.",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def quick_check_nearest():
    quick_nearest = L.choice(
        "quick-nearest",
        question="`distances` holds the distance from a new example to every training example. Which gives the positions of its 5 nearest training examples?",
        options={
            "values": "`np.sort(distances)[:5]`",
            "far": "`np.argsort(distances)[-5:]`",
            "right": "`np.argsort(distances)[:5]`",
        },
        correct="right",
        why={
            "values": "`np.sort` gives the 5 smallest distances themselves, not where they are. `np.argsort` gives the positions, from near to far. See the worked example of [Ask the neighbors](/course/open/ml-knn?lang=en).",
            "far": "`np.argsort` goes from near to far, so the last 5 positions are the 5 farthest. See the worked example of [Ask the neighbors](/course/open/ml-knn?lang=en).",
            "right": "The new example is not among the training examples. Today it will be: then the nearest of all is the example itself, at distance 0.",
        },
    )
    quick_nearest
    return (quick_nearest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A factory wants to know which runs of a machine will end in a failure, so that it can check them in time. Failures are rare: in the AI4I 2020 dataset, 339 of 10,000 runs, 3.4%. The data is synthetic: Stephan Matzka made it in 2020 to look like real maintenance data, which companies rarely publish. A model that learns to be right as often as possible learns to say "no failure", and the failures, the examples that matter, get lost. The same happens with fraud, rare diseases and a faulty part on a production line: the class you care about is the rare one.

    **An everyday example.** A smoke alarm that never goes off is right on almost every day of the year. It is wrong on exactly the day that matters.
    """)
    return


@app.cell(hide_code=True)
def predict_always_cell():
    guess_always = L.guess(
        "always-guess",
        question="Of 10,000 runs, 339 end in a failure. A model says \"no failure\" every time. What share of the runs does it get right?",
        options={
            "half": "About 50%",
            "right": "About 97%",
            "few": "About 3%",
        },
        reveal="**96.6%.** It is right on every one of the 9,661 runs without a failure, and finds none of the 339 failures: recall 0%. A model that never warns is right almost always, and useless.",
    )
    guess_always
    return (guess_always,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Right most of the time, finding little.** The course trained logistic regression on 7,000 of the runs, with the five numbers of every run: the air and the process temperature, the rotational speed, the torque and the tool wear. On the other 3,000 runs, with 110 failures, it is right 97.2% of the time, hardly more than "no failure" every time, 96.3%. It finds 31 of the 110 failures: recall 28.2%. Its loss is a mean over all 7,000 runs, and the 229 failures are 3.3% of them, so it gives most runs a small chance of failure, and few chances reach 0.5.

        **Class weights.** Give every example a weight, and let its log loss count that many times. A **class weight** is the weight of every example of one class. With balanced weights, each class weighs the same in total: an example of class $c$ gets

        $$w_c = \frac{n}{2\,n_c},$$

        with $n$ examples, $n_c$ of them of class $c$. Of the 7,000 runs, 229 are failures: every failure weighs $7{,}000 / (2 \cdot 229) = 15.28$, every normal run $7{,}000 / (2 \cdot 6{,}771) = 0.517$, and both classes weigh 3,500 in total. In scikit-learn it is `class_weight="balanced"`. You write the weights in exercise 1.

        **Changing the counts.** Instead of weighting, you can change the training set. **Undersampling** keeps every failure and throws normal runs away at random, until 229 of each are left. **Oversampling** repeats failures at random, until there are 6,771 of each.

        **SMOTE.** Repeating a failure gives copies. **SMOTE** (Chawla and others, 2002) makes new failures: take a failure $a$, one of its $k$ nearest other failures $b$, found as in [Ask the neighbors](/course/open/ml-knn?lang=en), and a random number $u$ between 0 and 1. The new failure is

        $$a + u\,(b - a),$$

        a point on the line from $a$ to $b$: with $u = 0$ it is $a$, with $u = 1$ it is $b$. You write it in exercise 2.

        **What they really change.** The course trained the same logistic regression in four more ways, tried the model as it is with a lower threshold as well, and counted on the 3,000 test runs:

        | Trained | Failures found (of 110) | Alarms | Recall | Precision | AUC |
        |---|---|---|---|---|---|
        | as it is | 31 | 35 | 28.2% | 88.6% | 0.888 |
        | with balanced class weights | 89 | 630 | 80.9% | 14.1% | 0.899 |
        | after undersampling | 91 | 684 | 82.7% | 13.3% | 0.896 |
        | after oversampling | 89 | 629 | 80.9% | 14.1% | 0.900 |
        | after SMOTE, k = 5 | 89 | 622 | 80.9% | 14.3% | 0.901 |
        | as it is, threshold 0.031 | 87 | 630 | 79.1% | 13.8% | 0.888 |

        All four fixes make the model say "failure" far more often: it finds 89 to 91 of the 110 failures, with 622 to 684 alarms. The AUC of [Every threshold at once](/course/open/ml-roc-auc?lang=en), how well the model puts failures above normal runs, hardly moves: 0.888 without, 0.896 to 0.901 with. And the model as it is does nearly the same with a lower threshold: at 0.031 instead of 0.5 it raises as many alarms as with balanced weights, 630, and finds 87 failures. Here the weights and the resampling mostly move the threshold; they teach the model little that is new. What is left to choose is how readily it may say "failure": what costs more, a missed failure or a needless check?

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **class weight** | the weight of every example of one class in the loss | `class_weight="balanced"` |
        | **undersampling** | balancing the training set by throwing away examples of the common class, at random | |
        | **oversampling** | balancing the training set by repeating examples of the rare class, at random | |
        | **SMOTE** | oversampling with new examples, each between an example of the rare class and one of its nearest neighbors of that class | `a + u * (b - a)` |
        | $w_c$ | the balanced weight of an example of class $c$: $n / (2\,n_c)$ | |
        """),
        L.watch_out("Resample only the training part, after the split. The validation and the test set must look like the runs the model will meet, with about 3.4% failures. Oversampling before the split puts copies of the same failure on both sides, and the test set then rewards remembering them."),
        mo.md(r"""
        The slider sets the class weight of a failure; a normal run keeps weight 1. The gray line shows the model without weights at every threshold, as in the precision-recall curve of [Every threshold at once](/course/open/ml-roc-auc?lang=en): its recall to the right, its precision up. The large dot is the model trained with that weight, at threshold 0.5, and the open dots are the other weights. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_weight_cell():
    def _frame(w):
        said = C.weight_said(w, L.lang)
        alt = ("The precision of the model without weights against its recall, at every threshold, falling from the "
               "upper left to the lower right; dots for the models trained with a class weight of 1, 2, 5, 10, 30 and "
               "100 lie close to that line. " + said)
        return (C.weight_picture(w, {"title": "Weight of a failure", "x": "recall (%)", "y": "precision (%)",
                                     "weight": "weight {w}", "alt": alt}), said)

    explore_weight = L.explore("failure-weight", "weight of a failure", _frame, start=1, values=C.WEIGHTS, role="hyper",
                               task="Go from weight 1 to 100. What happens to the recall, what to the precision, and where do the dots lie?")
    explore_weight
    return


@app.cell(hide_code=True)
def text_after_weight():
    mo.md(r"""
    From weight 1 to 100, the model finds 31, 43, 63, 72, 89 and 102 of the 110 failures, with 35, 63, 142, 274, 636 and 1,282 alarms. Every dot lies close to the gray line: a larger weight moves the model along it, as a lower threshold does. Weight 30 is about the balanced weight: 6,771 / 229 = 29.6.

    ### Recipe

    When one class is rare:

    1. Split first. The validation and the test set keep the real share of the rare class.
    2. Judge by the precision and recall of the rare class, or their F1 score, never by accuracy alone.
    3. Try class weights first: they change nothing in the data. Or keep the model and lower its threshold.
    4. Choose the weight or the threshold on the validation set, by what a missed rare example costs against a needless alarm.
    5. Resample, if you do, the training part only: undersampling when there is plenty of data, oversampling or SMOTE when there is little.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads the 10,000 runs, keeps the five numbers and the label `Machine failure`, 1 for a failure, splits the runs at random into 3,000 to test on and 7,000 to train on, and scales the numbers with the mean and the standard deviation of the training part. `LogisticRegression` is the logistic regression of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), with `fit` and `predict` as in [The same model with scikit-learn](/course/open/ml-sklearn?lang=en). The cell trains it twice, without class weights (`None`) and with `class_weight="balanced"`, and prints for both how many of the test failures it finds, how many alarms it raises, its precision and its recall. Run the cell.
    """)
    return


@app.cell
def example_machines():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _numbers = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    _X = np.array(_df[_numbers])
    _y = np.array(_df["Machine failure"])
    print(len(_y), _y.sum())

    _order = np.random.default_rng(0).permutation(len(_y))
    _test, _train = _order[:3000], _order[3000:]           # 3,000 runs to test on, 7,000 to train on
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
    _X = (_X - _mean) / _std                               # scaled with the numbers of the training part

    for _weights in [None, "balanced"]:
        _model = LogisticRegression(class_weight=_weights)
        _model.fit(_X[_train], _y[_train])
        _alarm = _model.predict(_X[_test]) == 1            # True where the model says: failure
        _found = _y[_test][_alarm].sum()                   # the real failures among the alarms
        print(_weights, _found, _alarm.sum(), round(_found / _alarm.sum(), 3), round(_found / _y[_test].sum(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    10,000 runs, 339 failures. Without weights the model raises 35 alarms, and 31 of them are failures: precision 88.6%, but it finds only 31 of the 110, recall 28.2%. With balanced weights it raises 630 alarms and finds 89: recall 80.9%, precision 14.1%. Which is better depends on what a missed failure costs, and what a needless check.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Balanced weights

        `balanced_weights(y)` gets the labels of the training part, 1 for a failure and 0 for a normal run, and should give the weight of every example: $n / (2\,n_c)$, with $n_c$ the number of failures for a failure and the number of normal runs for a normal run. `weights[y == 1] = 2.0` would put 2.0 at every place where `y` is 1. Replace the two `...`. The check calls `balanced_weights(y_train)` with the 7,000 labels of the training part, `_y[_train]` in the worked example, and tries three short arrays.
        """),
        L.task("Let `balanced_weights` give every failure and every normal run its balanced weight."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weights():
    box_weights = L.code_box("balanced-weights", lines=9)
    box_weights
    return (box_weights,)


@app.cell(hide_code=True)
def check_weights(box_weights):
    result_weights = L.check("balanced-weights", box_weights.value)
    result_weights
    return (result_weights,)


@app.cell(hide_code=True)
def hints_weights():
    L.hints(
        "balanced-weights",
        [
            "Both classes should weigh the same in total, n / 2 each. Each example of a class gets its share of that.",
            "The more examples a class has, the less each of them weighs: divide by how many there are. `failures` of the n are failures; how many are normal runs?",
            "`weights[y == 1] = n / (2 * ___)` and `weights[y == 0] = n / (2 * (___))`.",
        ],
        why="Each of the 229 failures weighs 15.28 and each of the 6,771 normal runs 0.517: both classes weigh 3,500. Trained with these weights, logistic regression finds 89 of the 110 test failures, with 630 alarms.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_new():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A new failure with SMOTE

        `new_failure(X_rare, i, j, u)` gets the scaled numbers of the failures, a row per failure, and should make one new failure: on the line from failure `i` to its nearest other failure number `j`, counted from 0 for the nearest, at share `u` of the way. Replace the two `...`. The check calls `new_failure(failures, 0, 2, 0.4)` with the 229 failures of the training part, and tries five points in a plane, where you can see which is nearest.
        """),
        L.task("Let `new_failure` place a new failure on the line between failure `i` and one of its nearest other failures."),
    ])
    return


@app.cell(hide_code=True)
def exercise_new():
    box_new = L.code_box("new-failure", lines=8)
    box_new
    return (box_new,)


@app.cell(hide_code=True)
def check_new(box_new):
    result_new = L.check("new-failure", box_new.value)
    result_new
    return (result_new,)


@app.cell(hide_code=True)
def hints_new():
    L.hints(
        "new-failure",
        [
            "Two steps: the positions of the other failures from near to far, then the point at share `u` of the way from `X_rare[i]` to `b`.",
            "`np.argsort` gives the positions from near to far, as in k-NN, but the nearest of all is failure `i` itself, at distance 0: leave it out. For the point, start at `X_rare[i]` and take a share `u` of the step to `b`.",
            "`nearest = np.argsort(distances)[___:]` and `return X_rare[i] + u * (___)`.",
        ],
        why="The new failure lies on the line from failure i to one of its nearest other failures, so it stays where failures are. With i, j and u drawn at random 6,542 times, these are the SMOTE failures of the table: logistic regression trained with them finds 89 of the 110 test failures, with 622 alarms.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Exercise 3. Every failure found
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "resample-when",
        question="A colleague oversamples the failures of all 10,000 runs, then splits the runs into a part to train on and a part to test on, and trains k-NN with k = 5. It finds every failure of the test set. What explains that?",
        options={
            "balanced": "k-NN works well once the classes are balanced",
            "right": "Copies of every test failure are among the training runs",
            "more": "The test set has many more failures now, so they are easier to find",
        },
        correct="right",
        why={
            "balanced": "Oversampled after the split, k-NN finds 83 of the 110 test failures, 75.5%: good, not perfect. The 100% comes from the order: copies of the same failure landed on both sides, and the nearest training run of every test failure is its own copy, at distance 0.",
            "more": "More failures make precision easier, not recall: recall still asks how many of the failures were found. All 2,913 failures of that test set were found because every one of them had a copy among the training runs, at distance 0.",
            "right": "Every one of the 2,913 failures of that test set had a copy among the training runs, and k-NN found it at distance 0. Oversampled after the split, in the training part only, k-NN finds 83 of the 110 test failures, 75.5%.",
        },    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Precision and recall come from [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), and [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) already showed how accuracy can rank a model that finds nothing first. The threshold and the precision-recall curve come from [Every threshold at once](/course/open/ml-roc-auc?lang=en): class weights mostly move a model along that curve. SMOTE uses the nearest neighbors of [Ask the neighbors](/course/open/ml-knn?lang=en). Exercise 3 is a first look at a test set that knows too much, which a later lesson covers in full. In the next lesson, *Searching settings: grid search and random search*, you choose the settings of a model on these runs better than by trying one at a time.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_new, result_weights):
    L.wrap_up(result_weights, result_new, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
