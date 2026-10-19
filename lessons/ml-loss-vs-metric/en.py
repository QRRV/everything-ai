# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-loss-vs-metric. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-loss-vs-metric", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) and [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_report():
    quick_report = L.choice(
        "quick-report",
        question="You chose a degree on the validation set, where it got an MSE of 0.80. Which number do you report as the estimate for new data?",
        options={
            "val": "0.80, its validation MSE",
            "train": "Its training MSE",
            "right": "Its MSE on the test set",
        },
        correct="right",
        why={
            "val": "The validation set helped choose, so its score flatters the winner. See the watch-out in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "train": "The model learned from those examples. See *The idea* in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "right": "The test set played no part in training or choosing, so its MSE is an honest estimate.",
        },
    )
    quick_report
    return (quick_report,)


@app.cell(hide_code=True)
def quick_check_cross_entropy():
    quick_cross_entropy = L.choice(
        "quick-cross-entropy",
        question="A model gives a photo the probabilities 0.1, 0.6 and 0.3, and its right class is 1. About how big is its cross-entropy?",
        options={
            "right": "0.51",
            "first": "2.30",
            "last": "1.20",
        },
        correct="right",
        why={
            "first": "That is $-\\log(0.1)$, the probability of class 0. Where do you start counting? See *The idea* in [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "last": "That is $-\\log(0.3)$, the probability of class 2. See *The idea* in [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "right": "Class 1 got 0.6: $-\\log(0.6) \\approx 0.51$.",
        },
    )
    quick_cross_entropy
    return (quick_cross_entropy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Gradient descent needs a loss with useful slopes, such as cross-entropy. But the person who uses a spam filter does not care about its cross-entropy: they want to know how often it is right. That is a different number, and it can move differently. So you train on one number and report another, and you need to know which is which.

    **An everyday example.** In driving lessons, the instructor tells you after every turn how far off you were: a little too wide, a bit too fast. The driving test only says pass or fail. You learn from the first; others look at the second.
    """)
    return


@app.cell(hide_code=True)
def predict_better():
    guess_better = L.guess(
        "better",
        question="On five photos, model B has a far lower cross-entropy than model A: on most photos, it gives the right class a much higher probability. Do you think B also gets more of the five photos right?",
        options={
            "more": "Yes, clearly more",
            "same": "Not necessarily: it can be the same number",
            "fewer": "No, fewer",
        },
        reveal="**Not necessarily.** In the worked example below, both get four of the five photos right, although the cross-entropy is about 0.77 for A and 0.31 for B. Whether a photo counts as right only depends on which class is the most likely, not on how sure the model is.",
    )
    guess_better
    return (guess_better,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **metric** is a number that says how good a model is in terms people care about. The best known for classification is **accuracy**: the share of examples whose predicted class is the label. The predicted class is the most likely one, `probs.argmax(axis=1)`; comparing it with `labels` gives `True` or `False` per example, and since `True` counts as 1 and `False` as 0, the mean of those is the share that is right.

        You train on the loss and report the metric, and they are different numbers for a reason. A tiny change in a weight changes the loss a tiny bit, so the loss has slopes for gradient descent. The accuracy only changes when an example crosses to another class; for almost every tiny change it stays exactly the same, so its slope is 0 almost everywhere, and gradient descent has nothing to go on. The two usually move together, but not always: a model can become surer of its right answers, and better on the loss, without a single prediction changing.

        Choose on the validation set by the metric that matters, and report it on the test set, as with the loss.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **metric** | | a number that says how good a model is in terms people care about; you report it, you do not train on it | |
        | **accuracy** | | the share of examples a classifier gets right | `np.mean(probs.argmax(axis=1) == labels)` |
        """),
        L.watch_out("Accuracy can mislead when one class is much more common. If 95 of 100 mails are not spam, a filter that never says spam has an accuracy of 0.95. A later lesson shows numbers that catch this."),
        mo.md(r"""
        Ten students, their hours of study $x$, and whether they passed. A model gives the probability of passing, with a fixed weight of 1:
        """),
        L.formula(r"[[pred:\hat{y}]] = \sigma(x + [[param:b]])"),
        mo.md(r"""
        Move the slider, or drag across the chart of the log loss: it sets the <span class="c-param">bias term</span> $b$. One chart shows the <span class="c-loss">log loss</span>, the other the accuracy: a probability of 0.5 or more counts as passed.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        b = round(b, 1)
        right = C.right(b)
        return (C.picture(b, {"loss_title": "Log loss", "loss": "log loss", "acc_title": "Accuracy", "acc": "accuracy",
                              "alt": f"The log loss for b from −8 to 0, a smooth valley with its lowest point near b = −4.2; at b = {L.number(b, 1)} it is {L.number(C.log_loss(b), 2)}. The accuracy for the same b, a staircase of flat steps that go up and down between 0.5 and 0.8, highest from b = −4.5 to −4.1; at b = {L.number(b, 1)} it is {L.number(right / 10, 1)}."}),
                f"""With <i>b</i> = {L.number(b, 1)}: a log loss of <span class="c-loss ea-num">{L.number(C.log_loss(b), 3)}</span>, and an accuracy of <span class="ea-num">{L.number(right / 10, 1)}</span>: the model is right about {right} of the 10 students.""")

    explore_b = L.explore("bias", "b", _frame, start=-5.5, low=-8, high=0, step=0.1, role="param", task="Find the $b$ with the lowest log loss, and the $b$ with the highest accuracy.")
    explore_b
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The log loss has one lowest point, near $b = -4.2$, and at every $b$ its slope shows the way there. The accuracy is highest, 0.8, on a whole flat step from $b = -4.5$ to $-4.1$, and it only jumps where a student crosses 0.5: a staircase, with no slope to follow.

    ### Recipe

    To judge a classifier by its accuracy:

    1. Take the predicted class of each example: the most likely one, `probs.argmax(axis=1)`.
    2. Compare with the labels: `== labels` gives `True` where the prediction is right.
    3. Take the mean: the share of right predictions.
    4. Train on the loss; choose on the validation set and report on the test set with the metric that matters.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Five photos, three classes, and two models, A and B, that give each photo its probabilities. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_metric():
    import numpy as np

    _labels = np.array([0, 1, 2, 1, 0])
    _a = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5], [0.45, 0.35, 0.2], [0.6, 0.2, 0.2]])
    _b = np.array([[0.9, 0.05, 0.05], [0.45, 0.35, 0.2], [0.05, 0.05, 0.9], [0.1, 0.85, 0.05], [0.9, 0.05, 0.05]])

    for _probs in [_a, _b]:
        # 1. the most likely class of each photo
        _predicted = _probs.argmax(axis=1)
        print(_predicted)
        # 2. and 3. right or not, and the share that is right
        print(_predicted == _labels, np.mean(_predicted == _labels))
        # the cross-entropy, for comparison
        print(-(np.eye(3)[_labels] * np.log(_probs)).sum(axis=1).mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Model A predicts the classes 0, 1, 2, 0 and 0, and gets four right: an accuracy of 0.8. Model B predicts 0, 0, 2, 1 and 0: it gets another photo wrong, but also four right, so also 0.8. The cross-entropy tells them apart: about 0.77 for A, and 0.31 for B, which gives the right class 0.85 or 0.9 on four photos, where A never gives it more than 0.6.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_slope():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. No slope
    """)
    return


@app.cell(hide_code=True)
def choice_slope_cell():
    choice_slope = L.choice(
        "no-slope",
        question="Why does gradient descent not train on accuracy directly?",
        options={
            "number": "Accuracy is not a number you can work out",
            "right": "A tiny change in a weight almost never changes a prediction, so the slope of accuracy is 0 almost everywhere",
            "big": "Accuracy can become larger than 1",
        },
        correct="right",
        why={
            "number": "It is a number: a share between 0 and 1. What does gradient descent need from it? See *The idea*.",
            "big": "A share of examples is never more than 1. Look at the chart of the accuracy: what does a staircase not have?",
            "right": "Flat steps give gradient descent no direction; the loss, which changes with every tiny step, does.",
        },
    )
    choice_slope
    return (choice_slope,)


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Accuracy

        Six photos and three classes: the probabilities of each photo are a row of `probs`, and the labels are in `labels`. Put the accuracy in `accuracy`. Replace the `...`. The last line is `accuracy`, so the check sees it. The check also tries other labels.
        """),
        L.task("Put the share of photos whose most likely class is the label in `accuracy`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("accuracy", lines=7)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Steps 1 to 3 of the recipe, in one line.",
            "Which method gives the place of the largest number, and along which axis are the classes of one photo? What does the mean of `True` and `False` give?",
            "`accuracy = np.mean(probs.argmax(axis=___) == ___)`.",
        ],
        why="The most likely classes are 0, 2, 1, 0, 2 and 1; compared with the labels 0, 2, 1, 1, 0 and 1, four are right. Four of six is about 0.67.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 3. Which one?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-model",
        question="Two spam filters on the validation set. Filter A: cross-entropy 0.40, accuracy 0.85. Filter B: cross-entropy 0.45, accuracy 0.88. What counts is how often the filter is right. Which do you choose?",
        options={
            "a": "Filter A: its loss is lower",
            "right": "Filter B: it is right more often",
            "test": "Neither yet: compare them on the test set first",
        },
        correct="right",
        why={
            "a": "The loss is what each filter trained on. What did the people who use it ask for? See *The idea*.",
            "test": "The test set stays sealed while you choose. See the recipe.",
            "right": "Choose on the validation set by the metric that matters; then report the accuracy of B on the test set.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) you met the loss a classifier trains on; accuracy is the metric you report next to it, and it only looks at the most likely class. In [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) you chose on one set and reported on another; the same holds with a metric. In the next lesson, *Beyond accuracy: precision and recall*, you see which classes a model mixes up, and why accuracy alone can mislead.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_slope, result_accuracy, choice_which):
    L.wrap_up(choice_slope, result_accuracy, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
