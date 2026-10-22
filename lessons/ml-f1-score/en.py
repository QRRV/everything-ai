# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-f1-score. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-f1-score", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="A filter calls 8 mails spam, and 6 of them are spam. It misses 4 other spam mails. What are its precision and recall?",
        options={
            "swapped": "Precision 0.6, recall 0.75",
            "right": "Precision 0.75, recall 0.6",
            "all": "Precision 0.5, recall 0.6",
        },
        correct="right",
        why={
            "swapped": "The other way round. Precision divides by the mails the filter calls spam, recall by the mails that are spam. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "all": "Precision only looks at the 8 mails the filter calls spam. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "right": "6 of the 8 mails called spam are spam: precision 0.75. The filter finds 6 of the 10 spam mails: recall 0.6.",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="A filter calls a mail spam from a probability of 0.5 on. You raise its decision threshold to 0.7. What happens to its recall?",
        options={
            "up": "It goes up",
            "same": "It always stays the same",
            "right": "It goes down, or stays the same",
        },
        correct="right",
        why={
            "up": "Fewer mails reach a higher threshold, so the filter cannot find more spam than before. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "same": "It stays the same only if no spam mail has a probability from 0.5 to 0.7. See the picture in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "right": "Fewer mails are called spam, so no spam mail can be found that was missed before.",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the lesson before, a filter that never says spam had an accuracy of 0.95 on 100 mails with 5 spam. When one class is rare, accuracy can even rank a model that finds nothing above one that finds almost everything. Precision and recall show what a model does, but they are two numbers, and the threshold trades one for the other. To compare two models, or two thresholds, you want one number that is only high when both are.

    **An everyday example.** A smoke alarm that never goes off is right on almost every day of the year, and useless on the one day there is a fire. One that goes off whenever you make toast catches every fire, but soon nobody listens to it. A good alarm does both: it catches the fire, and stays quiet otherwise.
    """)
    return


@app.cell(hide_code=True)
def predict_everything():
    guess_everything = L.guess(
        "everything",
        question="1,000 handwritten digits, 80 of them a 5. A detector calls every digit a 5: its recall is 1, and its precision 0.08. The plain mean of the two is 0.54. What do you think one fair number for both should say about this detector?",
        options={
            "mean": "About 0.54: half of it is perfect",
            "right": "Something close to 0.08: its weaker number holds it down",
            "one": "1: it finds every 5",
        },
        reveal="**Something close to its weaker number.** Its F1 score is 0.15. One number for both must be low as soon as one of them is low; otherwise calling every digit a 5 would earn more than half of the top score.",
    )
    guess_everything
    return (guess_everything,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **F1 score** puts precision $P$ and recall $R$ in one number:
        """),
        L.formula(r"F_1 = \frac{2 \cdot P \cdot R}{P + R}"),
        mo.md(r"""
        That is their **harmonic mean**: for two numbers, twice their product divided by their sum. Unlike the plain mean, it stays close to the smaller of the two: it always lies between the smaller number and twice the smaller. It is 1 only when both are 1, and 0 as soon as one of them is 0.

        Written with the cells of the confusion matrix, it is $F_1 = 2\,TP / (2\,TP + FP + FN)$. The true negatives are not in it. That is why F1 does not mislead when one class is much rarer than the others, which is called **class imbalance**, as long as the rare class is the positive one: the many examples of the common class that a model rightly leaves alone make accuracy high, but they do not count for F1. A model that never predicts the rare class has 0 true positives, so its F1 score is 0, although its precision cannot be worked out.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **F1 score**, $F_1$ | F one | the harmonic mean of precision and recall; high only when both are high | `2 * tp / (2 * tp + fp + fn)` |
        | **harmonic mean** | | for two numbers $a$ and $b$: $2ab / (a + b)$ | |
        | **class imbalance** | | one class is much rarer than the others, such as the 5s among all digits | |

        Drag the dot over the plane, or move it with the arrow keys: precision from left to right, recall from bottom to top, and the bars beside it follow.
        """),
    ])
    return


@app.cell(hide_code=True)
def plane_f1_cell():
    def _frame(_p, _r):
        _mean, _f1 = (_p + _r) / 2, C.f1(_p, _r)
        _words = {"title": "Precision, recall and two means", "precision": f"precision {L.number(_p)}", "recall": f"recall {L.number(_r)}",
                  "mean": f"plain mean {L.number(_mean)}", "f1": f"F1 score {L.number(_f1)}",
                  "alt": f"Four bars from 0 to 1: precision {L.number(_p)}, recall {L.number(_r)}, their plain mean {L.number(_mean)} and their F1 score {L.number(_f1)}."}
        return (C.bars(_p, _r, _words),
                f"""Precision {L.number(_p)} and recall {L.number(_r)}: the plain mean is <span class="ea-num">{L.number(_mean)}</span>, the F1 score <span class="ea-num">{L.number(_f1)}</span>.""")

    _plane = L.plane("precision-recall", "precision", "recall", _frame, C.STOPS, C.STOPS, start=(0.9, 0.3), role="",
                     task="Find a point where the plain mean of precision and recall is 0.5 or more, but the F1 score is below 0.2.")
    _plane
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Such points lie near two corners: one number 0.9 or more, the other 0.1 or less. F1 stays close to the smaller number, and is high only in the top right corner, where both are high. No real detector lies on the left or bottom edge, except in their corner: a 0 for either one means 0 true positives, and then the other is 0 too, or cannot be worked out.

    ### Recipe

    To judge a classifier by its F1 score:

    1. Count the true positives, false positives and false negatives.
    2. Work out precision, $TP / (TP + FP)$, and recall, $TP / (TP + FN)$.
    3. Work out $F_1 = 2PR / (P + R)$, or straight from the counts, $2\,TP / (2\,TP + FP + FN)$, which also works when precision cannot be worked out.
    4. When one class is rare, make it the positive class, and compare models, or thresholds on the validation set, by F1 instead of accuracy.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four detectors of 5s, on 1,000 handwritten digits of a validation set from QMNIST, 80 of them a 5: one that never says 5; a logistic regression that the course trained on 5,000 other digits, once with the decision threshold 0.05 and once with 0.5; and one that calls every digit a 5. For each, its counts $TP$, $FP$, $FN$ and $TN$. `for _name, _tp, _fp, _fn, _tn in _detectors:` gives each value of each tuple its own name, as `x, y = point` does. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_f1():
    # 1. the counts of each detector
    _detectors = [
        ("never 5", 0, 0, 80, 920),
        ("threshold 0.05", 77, 120, 3, 800),
        ("threshold 0.5", 52, 3, 28, 917),
        ("always 5", 80, 920, 0, 0),
    ]
    for _name, _tp, _fp, _fn, _tn in _detectors:
        # 2. precision, if the detector says 5 at all, and recall
        if _tp + _fp == 0:
            _precision = None
        else:
            _precision = _tp / (_tp + _fp)
        _recall = _tp / (_tp + _fn)
        # 3. F1 from the counts, and the accuracy to compare
        _f1 = 2 * _tp / (2 * _tp + _fp + _fn)
        print(f"{_name}:", _precision, _recall, _f1, (_tp + _tn) / 1000)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each line gives the detector, its precision, recall, F1 score and accuracy. Accuracy ranks the detector that never says 5, with 0.92, above the one with the threshold 0.05, with 0.877, although that one finds 77 of the 80 fives. F1 puts them the right way round: 0 against 0.56. With the threshold 0.5, the detector finds fewer fives, 52, but raises only 3 false alarms: the highest F1 score of the four, 0.77. That only works with the 5s as the positive class: with the other digits as positive, the detector that never says 5 would get an F1 score of 0.96, and the one with the threshold 0.05 only 0.93.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lopsided():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Close to the smaller one
    """)
    return


@app.cell(hide_code=True)
def choice_lopsided_cell():
    choice_lopsided = L.choice(
        "lopsided",
        question="A detector has a precision of 0.2 and a recall of 0.8. What is its F1 score?",
        options={
            "mean": "0.5",
            "right": "0.32",
            "product": "0.16",
        },
        correct="right",
        why={
            "mean": "That is the plain mean: one high number pulls it up, however low the other is. F1 stays close to the smaller number. See *The idea*.",
            "product": "That is the product of the two. F1 is twice the product, divided by the sum. See *The idea*.",
            "right": "2 · 0.2 · 0.8 / (0.2 + 0.8) = 0.32: much closer to 0.2 than to 0.8.",
        },
    )
    choice_lopsided
    return (choice_lopsided,)


@app.cell(hide_code=True)
def text_exercise_f1():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The F1 score of a detector

        Twenty digits: `pred` says which ones a detector calls a 5 (1) or not (0), and `y` which ones are a 5. As in the lesson before, `tp = np.sum(pred * y)` counts the true positives. Count the false positives in `fp` and the false negatives in `fn`, and put the F1 score in `f1`. Replace the three `...`. The last line is `f1`, so the check sees it. The check also tries other digits.
        """),
        L.task("Put the F1 score of the detector in `f1`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("f1", lines=9)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "f1",
        [
            "Steps 1 and 3 of the recipe: first the counts, then F1.",
            "A false positive has `pred` 1 and `y` 0, a false negative `pred` 0 and `y` 1. F1 from the counts needs no precision and recall first.",
            "`fp = np.sum(pred * (1 - y))`, `fn = np.sum(___)` and `f1 = 2 * tp / (___)`.",
        ],
        why="The detector calls 8 digits a 5, and 3 of them are: 3 true positives and 5 false positives. It misses 1 of the 4 fives. F1 = 2 · 3 / (2 · 3 + 5 + 1) = 0.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rare():
    mo.md(r"""
    ### Exercise 3. A rare class
    """)
    return


@app.cell(hide_code=True)
def choice_rare_cell():
    choice_rare = L.choice(
        "rare-pick",
        question="Of 1,000 card payments, 10 are fraud. Model A never says fraud. Model B catches 8 of the 10, and wrongly flags 40 honest payments. Which model is better at finding fraud, and which number shows it?",
        options={
            "accuracy": "A: its accuracy is 0.99, against 0.958 for B",
            "right": "B: its F1 score is about 0.28, against 0 for A",
            "neither": "Neither: both get payments wrong",
        },
        correct="right",
        why={
            "accuracy": "Accuracy counts the 990 honest payments that A leaves alone without doing anything. Which number leaves the true negatives out? See *The idea*.",
            "neither": "Every model makes mistakes; the question is which one does its job better. Work out F1 for both: step 3 of the recipe.",
            "right": "B has 8 true positives, 40 false positives and 2 false negatives: F1 = 16 / 58, about 0.28. A finds no fraud at all: F1 0.",
        },
    )
    choice_rare
    return (choice_rare,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) precision and recall showed what accuracy hides; F1 puts them in one number that leaves the true negatives out. It weighs them the same: when one mistake costs much more than the other, look at the two themselves, or at the F-beta score, which gives recall more weight than precision, or less, and choose the threshold by what each mistake costs. And F1 still belongs to one decision threshold. In the next lesson, *Every threshold at once: the ROC curve and AUC*, you look at all thresholds together.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lopsided, result_f1, choice_rare):
    L.wrap_up(choice_lopsided, result_f1, choice_rare)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
