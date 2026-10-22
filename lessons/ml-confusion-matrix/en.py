# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-confusion-matrix. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-confusion-matrix", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_accuracy():
    quick_accuracy = L.choice(
        "quick-accuracy",
        question="A model predicts the classes 0, 1, 1 and 2 for four photos with the labels 0, 1, 2 and 2. What is its accuracy?",
        options={
            "right": "0.75",
            "count": "3",
            "half": "0.5",
        },
        correct="right",
        why={
            "count": "That is how many are right. Accuracy is a share. See *The idea* in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "half": "Compare the predictions with the labels one by one: how many match? See the recipe in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "right": "Three of the four match: 0.75.",
        },
    )
    quick_accuracy
    return (quick_accuracy,)


@app.cell(hide_code=True)
def quick_check_metric():
    quick_metric = L.choice(
        "quick-metric",
        question="Why do you report a metric such as accuracy, and not the cross-entropy the model trained on?",
        options={
            "slope": "Because the metric has better slopes for gradient descent",
            "lower": "Because the metric is always lower",
            "right": "Because the metric says how good the model is in terms people care about",
        },
        correct="right",
        why={
            "slope": "It is the other way round: the loss has the slopes. See *The idea* in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "lower": "A share and a loss are not on the same scale at all. See *The idea* in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "right": "You train on the loss, which has slopes, and report what the people who use the model want to know.",
        },
    )
    quick_metric
    return (quick_metric,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Accuracy lumps all mistakes together. But a spam filter can go wrong in two ways: it can let spam through, or it can throw away a mail you needed. Those mistakes cost different things, and when one class is much rarer than the other, accuracy can look good while the filter does nothing useful. To see what a model really does, count which classes it mixes up.

    **An everyday example.** A metal detector at an airport should beep for every knife. But if it also beeps for every belt buckle, the staff stop paying attention to it. Finding all knives and beeping only for knives are two different goals.
    """)
    return


@app.cell(hide_code=True)
def predict_raise():
    guess_raise = L.guess(
        "raise",
        question="A spam filter gives each mail a probability of spam, and calls it spam from 0.5 on. What do you think happens if it only calls a mail spam from 0.8 on?",
        options={
            "both": "It lets less spam through, and throws away fewer mails you need",
            "same": "Nothing changes: the probabilities stay the same",
            "right": "It lets more spam through, but throws away fewer mails you need",
        },
        reveal="**It lets more spam through, but throws away fewer mails you need.** In the worked example below, with 0.8 instead of 0.5 the filter catches 4 instead of 7 of the 10 spam mails, and wrongly calls 0 instead of 3 other mails spam. Its accuracy is 0.85 both times, so accuracy does not show the difference. Which mistake costs more decides where you draw the line.",
    )
    guess_raise
    return (guess_raise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        With two classes, call the class you are looking for **positive**: here, spam. Every mail then lands in one of four cells: a **true positive** is spam the filter calls spam, a **false positive** is a mail that is not spam but is called spam, a **false negative** is spam the filter misses, and a **true negative** is a mail that is not spam and is not called spam. Put the counts in a table, with a row for each true class and a column for each predicted class, and you have a **confusion matrix**. With $K$ classes it is $K$ by $K$, and the right predictions lie on its diagonal.

        From the cells come two metrics:

        - **Precision**: of the mails the filter calls spam, the share that really is spam: $\dfrac{TP}{TP + FP}$. High precision means few false alarms.
        - **Recall**: of the mails that really are spam, the share the filter finds: $\dfrac{TP}{TP + FN}$. High recall means little spam slips through.

        A filter that calls no mail spam has no precision at all: $TP + FP$ is 0, and you cannot divide by 0.

        A filter gives each mail a probability of spam, and calls it spam once that probability reaches a **decision threshold**, often 0.5. Raise the threshold, and the filter calls fewer mails spam: recall can only go down, and precision usually goes up, though it dips each time a spam mail falls below the threshold. Which matters more depends on which mistake costs more; you choose the threshold on the validation set.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **true positive**, **false positive**, **false negative**, **true negative**; $TP$, $FP$, $FN$, $TN$ | | the four outcomes of a prediction with two classes: positive or negative predicted, and right (true) or wrong (false) | `tp`, `fp`, `fn`, `tn` |
        | **confusion matrix** | | a table that counts, for each true class (row), how often each class was predicted (column) | `cm[label, pred]` |
        | **precision** | | of the examples predicted positive, the share that is positive | `tp / (tp + fp)` |
        | **recall** | | of the examples that are positive, the share predicted positive | `tp / (tp + fn)` |
        | **decision threshold** | | the lowest probability at which a model predicts the positive class | `p >= 0.5` |
        """),
        L.watch_out("When this course says precision on its own, it means this metric. The precision of numbers in a computer, how many digits it keeps, is something else: numerical precision."),
        mo.md(r"""
        Forty mails, ten of them spam, each with the probability of spam a filter gave it. Move the slider, or drag across the chart of precision and recall, to set the decision threshold. Below the slider is the confusion matrix, with not spam as class 0 and spam as class 1, so the true positives are at the bottom right, in `cm[1, 1]`. Under that is the chart: precision (solid line) and recall (dashed line) for every threshold.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        t = round(t, 2)
        c = C.counts(t)
        s = C.scores(t)
        precision = "—" if s["precision"] is None else L.number(s["precision"], 2)
        return (draw.array_grids([("counts", [[str(c["tn"]), str(c["fp"])], [str(c["fn"]), str(c["tp"])]],
                                   ["cm[label, pred]", "rows: the label, 0 not spam, 1 spam", "columns: the prediction, 0 not spam, 1 spam"])],
                                 indices=True, title="Confusion matrix").text
                + C.picture(t, {"title": "Precision and recall", "threshold": "decision threshold", "share": "share",
                                "precision": "precision", "recall": "recall",
                                "alt": f"Precision rises, with small dips, from about 0.33 to 1 as the threshold goes up, and recall falls from 1 to 0.1. At the threshold {L.number(t, 2)}: precision {precision}, recall {L.number(s['recall'], 2)}."}),
                f"""With the threshold {L.number(t, 2)}: true positives {c["tp"]}, false positives {c["fp"]}, false negatives {c["fn"]}, true negatives {c["tn"]}. Precision <span class="ea-num">{precision}</span>, recall <span class="ea-num">{L.number(s["recall"], 2)}</span>, accuracy <span class="ea-num">{L.number(s["accuracy"], 3)}</span>.""")

    explore_threshold = L.explore("threshold", "threshold", _frame, start=0.5, low=0.05, high=0.95, step=0.05, role="hyper",
                                  task="Find the highest threshold that still catches at least 8 of the 10 spam mails. What is the precision there?")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    To catch at least 8 of the 10 spam mails, the threshold can go up to 0.45. There precision is 0.73: of the 11 mails the filter calls spam, 3 are not. A higher threshold lets more spam through and usually raises precision, but not always: at 0.5 the spam mail with 0.45 falls below the threshold, and precision dips to 0.70.

    ### Recipe

    To see what a classifier with two classes really does:

    1. Choose the positive class, the one you are looking for, and a decision threshold.
    2. Count the four cells: true positives, false positives, false negatives and true negatives.
    3. Work out precision, $TP / (TP + FP)$, and recall, $TP / (TP + FN)$.
    4. Move the threshold to trade one for the other, depending on which mistake costs more, and choose it on the validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forty mails of the picture: the probabilities the filter gave the ten spam mails, and the thirty other mails. Then the threshold 0.5, and 0.8. `np.sum(a)` is the same as `a.sum()`: it adds up all numbers of `a`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_confusion():
    import numpy as np

    _spam = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2])
    _ham = np.array([0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08,
                     0.08, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01])

    # 1. the positive class is spam; two thresholds
    for _t in [0.5, 0.8]:
        # 2. the four cells: the sum of True and False counts the True ones
        _tp = np.sum(_spam >= _t)
        _fn = np.sum(_spam < _t)
        _fp = np.sum(_ham >= _t)
        _tn = np.sum(_ham < _t)
        print(_t, _tp, _fn, _fp, _tn)
        # 3. precision and recall, and the accuracy to compare
        print(_tp / (_tp + _fp), _tp / (_tp + _fn), (_tp + _tn) / 40)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With the threshold 0.5, the filter finds 7 of the 10 spam mails and wrongly calls 3 other mails spam: precision 0.7, recall 0.7. With 0.8, it finds only 4, but every mail it calls spam is spam: precision 1.0, recall 0.4. The accuracy is 0.85 both times, and says nothing about this difference.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_never():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A filter that never says spam
    """)
    return


@app.cell(hide_code=True)
def choice_never_cell():
    choice_never = L.choice(
        "never-spam",
        question="Of 100 mails, 5 are spam, and a filter never says spam. Which number shows straight away that it is useless?",
        options={
            "accuracy": "Its accuracy, 0.95",
            "right": "Its recall, 0",
            "tn": "Its 95 true negatives",
        },
        correct="right",
        why={
            "accuracy": "0.95 looks good. Which number counts how much of the spam it finds? See *The idea*.",
            "tn": "Those are the mails it gets right without doing anything. See *The idea*.",
            "right": "Of the 5 spam mails it finds none: recall 0. Its precision cannot even be worked out, because it calls no mail spam.",
        },
    )
    choice_never
    return (choice_never,)


@app.cell(hide_code=True)
def text_exercise_precision():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Precision and recall

        Twelve mails: `pred` says which ones a filter calls spam (1) or not (0), and `y` which ones are spam. Because both hold only 0 and 1, `pred * y` is 1 exactly where both are 1, so `tp = np.sum(pred * y)` counts the true positives. Count the false positives in `fp` and the false negatives in `fn` the same way, and then put precision in `precision` and recall in `recall`. Replace the four `...`. The last line shows both, so the check sees them. The check also tries other predictions.
        """),
        L.task("Put the precision and the recall of the filter in `precision` and `recall`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision():
    box_precision = L.code_box("precision-recall", lines=10)
    box_precision
    return (box_precision,)


@app.cell(hide_code=True)
def check_precision(box_precision):
    result_precision = L.check("precision-recall", box_precision.value)
    result_precision
    return (result_precision,)


@app.cell(hide_code=True)
def hints_precision():
    L.hints(
        "precision-recall",
        [
            "Steps 2 and 3 of the recipe: first the cells, then the two shares.",
            "A false positive has `pred` 1 and `y` 0, so `1 - y` is 1 there. Which product is 1 exactly for a false negative? And which cells go below the line for precision, and for recall?",
            "`fp = np.sum(pred * (1 - y))`, `fn = np.sum(___)`, `precision = tp / (tp + ___)` and `recall = tp / (tp + ___)`.",
        ],
        why="The filter calls 5 mails spam, and 4 of them are spam: 4 true positives and 1 false positive, so precision is 4 / 5 = 0.8. Of the 6 spam mails it misses 2: recall is 4 / 6, about 0.67.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_matrix():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A confusion matrix for three classes

        Ten photos of cats (0), dogs (1) and birds (2): their labels in `labels`, and what a model predicted in `preds`. The matrix `cm` starts with all zeros. In the loop, add 1 to the cell in the row of the true class and the column of the predicted class. Replace the `...`. The last line is `cm`, so the check sees it. The check also tries other predictions.
        """),
        L.task("Count every photo in its cell of the confusion matrix `cm`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matrix():
    box_matrix = L.code_box("matrix", lines=9)
    box_matrix
    return (box_matrix,)


@app.cell(hide_code=True)
def check_matrix(box_matrix):
    result_matrix = L.check("matrix", box_matrix.value)
    result_matrix
    return (result_matrix,)


@app.cell(hide_code=True)
def hints_matrix():
    L.hints(
        "matrix",
        [
            "One line in the loop: one photo, one cell, plus 1.",
            "Photo `i` belongs in row `labels[i]` and column `preds[i]`. How did you select one element of a matrix by its row and column?",
            "`cm[labels[i], preds[i]] = ___ + 1`.",
        ],
        why="The cats are all right: 3 in the top left. The dogs: 2 right and 1 taken for a bird. The birds: 2 right, 1 taken for a cat and 1 for a dog.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en) accuracy was the metric; the confusion matrix shows what lies behind it, and precision and recall show what it hides when one class is rare. The decision threshold is one more setting you choose on the validation set. In the next lesson, *Precision and recall in one number: the F1 score*, you put the two in one number that stays honest when one class is rare. And in the project with handwritten digits, a confusion matrix of ten classes shows which digits your model mixes up.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_never, result_precision, result_matrix):
    L.wrap_up(choice_never, result_precision, result_matrix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
