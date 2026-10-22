# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-roc-auc. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-roc-auc", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="A detector has a precision of 0.5 and a recall of 0.5. What is its F1 score?",
        options={
            "product": "0.25",
            "right": "0.5",
            "sum": "1",
        },
        correct="right",
        why={
            "product": "That is the product. F1 is twice the product divided by the sum. See *The idea* in [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "sum": "That is the sum. F1 lies between precision and recall, close to the smaller. See *The idea* in [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "right": "2 · 0.5 · 0.5 / (0.5 + 0.5) = 0.5: when the two are the same, F1 is that number too.",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def quick_check_cells():
    quick_cells = L.choice(
        "quick-cells",
        question="Which cells of the confusion matrix does the F1 score leave out?",
        options={
            "fp": "The false positives",
            "right": "The true negatives",
            "none": "None: it uses all four",
        },
        correct="right",
        why={
            "fp": "The false positives are in it: they pull precision down. See *The idea* in [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "none": "Look at 2 · TP / (2 · TP + FP + FN): one cell is missing. See *The idea* in [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "right": "F1 is 2 · TP / (2 · TP + FP + FN): the many examples a model rightly leaves alone do not count.",
        },
    )
    quick_cells
    return (quick_cells,)




@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Precision, recall and F1 all belong to one decision threshold: move it, and they change. While you build a model, you often have not chosen the threshold yet, and two filters can swap places as it moves. You want one number for how well the probabilities put the spam above the other mails, whatever the threshold.

    **An everyday example.** A teacher can put thirty essays in order from best to worst, well or badly, before anyone has set the pass mark: the order is the teacher's work, the pass mark a separate choice.
    """)
    return


@app.cell(hide_code=True)
def predict_random():
    guess_random = L.guess(
        "random",
        question="A filter gives every mail a random probability. Pair each spam mail with each mail that is not spam. In what share of the pairs do you think the spam mail gets the higher probability?",
        options={
            "none": "Almost none",
            "right": "About half",
            "threshold": "That depends on the threshold",
        },
        reveal="**About half.** By chance, a random probability puts the spam mail first in half of the pairs. That share is the AUC: 0.5 for guessing, 1 when every spam mail is above every other mail, and 0.935 for the filter of the forty mails. It needs no threshold.",
    )
    guess_random
    return (guess_random,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Each of the forty mails of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) has the probability of spam the filter gave it; ten of them are spam. For every threshold, work out two shares:

        - the **true positive rate**: of the spam mails, the share called spam, $TP / (TP + FN)$: recall, under another name;
        - the **false positive rate**: of the other mails, the share wrongly called spam, $FP / (FP + TN)$.

        Each threshold is one point, with the false positive rate across and the true positive rate up. Join the points, and you have the **ROC curve**: from (0, 0), where nothing is called spam, to (1, 1), where everything is. As the threshold goes down, the curve steps up by 1/10 for every spam mail the threshold passes, and right by 1/30 for every other mail; a spam mail and another mail with the same probability make one slanted step. A filter that puts all spam above the other mails goes straight up to the top left corner first; guessing stays near the diagonal.

        The area under the curve is the **ROC AUC**, or AUC: a number from 0 to 1. Each of the 30 other mails is a step of 1/30 to the right, at the height of the share of spam mails above it. So the area is the share of the 10 × 30 pairs of a spam mail and another mail in which the spam mail has the higher probability, a tie counting half. It only depends on the order of the probabilities, not on a threshold.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **true positive rate**, TPR | | of the positive examples, the share predicted positive: another name for recall | `tpr` |
        | **false positive rate**, FPR | | of the negative examples, the share wrongly predicted positive | `fpr` |
        | **ROC curve** | R O C | the true positive rate against the false positive rate, for every decision threshold | |
        | **ROC AUC**, AUC | A U C | the area under the ROC curve: the share of pairs of a positive and a negative example in the right order, a tie counting half | `auc` |
        | **precision-recall curve** | | precision against recall, for every decision threshold; with a rare class it shows what the ROC curve hides | |
        """),
        L.watch_out("ROC stands for receiver operating characteristic, from radar in the Second World War. In medicine, recall is called sensitivity and 1 minus the false positive rate specificity, so an ROC curve there shows sensitivity against 1 − specificity."),
        mo.md(r"""
        Move the slider, or drag across the chart of the mails, to set the threshold: every mail right of the line is called spam. The other chart shows the ROC curve, the dot of that threshold, the dashed diagonal of guessing and, in grey, the area under the curve.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_roc_cell():
    def _frame(t):
        t = round(t, 2)
        r = C.rates(t)
        return (C.picture(t, {"mails": "The 40 mails", "probability": "probability of spam", "spam": "spam", "other": "not spam",
                              "roc": f"ROC curve, AUC {L.number(C.auc(), 3)}", "fpr": "false positive rate", "tpr": "true positive rate", "guess": "guessing",
                              "alt": f"First chart: the ten spam mails as filled dots in a row at the top and the thirty other mails as open dots below, by probability, with the threshold {L.number(t, 2)} as a line and {r['tp']} spam mails and {r['fp']} other mails right of it. Second chart: the ROC curve, a staircase far above the dashed diagonal of guessing, the area under it in grey, and a dot at the false positive rate {L.number(r['fpr'], 2)} and the true positive rate {L.number(r['tpr'], 2)}."}),
                f"""With the threshold {L.number(t, 2)}, the filter calls {r["tp"]} of the 10 spam mails spam, a true positive rate of <span class="ea-num">{L.number(r["tpr"], 2)}</span>, and {r["fp"]} of the 30 other mails, a false positive rate of <span class="ea-num">{L.number(r["fpr"], 2)}</span>.""")

    explore_roc = L.explore("threshold", "threshold", _frame, start=0.5, low=0, high=1, step=0.05, role="hyper",
                            task="Find the lowest threshold at which the filter calls at most 3 of the 30 other mails spam. What share of the spam does it catch there?")
    explore_roc
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The lowest is 0.45: there the filter calls 3 of the 30 other mails spam, a false positive rate of 0.1, and catches 8 of the 10 spam mails, a true positive rate of 0.8; at 0.4 a fourth other mail crosses the line. The curve answers that question for every threshold at once. The area, the AUC of 0.935, does not move with the slider.

    ### Recipe

    The score of a classifier is the number it gives each example, here the probability of spam. To draw its ROC curve and work out its AUC:

    1. Take every threshold, from above the highest score down to below the lowest.
    2. At each one, work out the true positive rate and the false positive rate.
    3. Plot the true positive rate up against the false positive rate across, and join the points.
    4. The AUC is the area under it: the share of pairs of a positive and a negative example in which the positive one has the higher score, a tie counting half.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forty mails of the picture: six thresholds from high to low, each one point of the curve, and then the AUC, from the spam mails above each of the 30 other mails and the ties. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_roc():
    import numpy as np

    _spam = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2])
    _ham = np.array([0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08,
                     0.08, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01])

    # 1. thresholds from high to low
    for _t in [1.0, 0.8, 0.6, 0.45, 0.2, 0.0]:
        # 2. the true positive rate and the false positive rate
        _tpr = np.sum(_spam >= _t) / len(_spam)
        _fpr = np.sum(_ham >= _t) / len(_ham)
        # 3. one point of the curve: across, then up
        print(_t, _fpr, _tpr)
    # 4. the AUC: for each other mail, the spam mails above it and the ties; a tie counts half
    _higher = 0
    _ties = 0
    for _h in _ham:
        _higher = _higher + np.sum(_spam > _h)
        _ties = _ties + np.sum(_spam == _h)
    _above = _higher + 0.5 * _ties
    print(_higher, _ties, _above, _above / (len(_spam) * len(_ham)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first six lines are points of the curve: threshold, false positive rate, true positive rate. The last line: in 279 of the 300 pairs the spam mail has the higher probability, and 3 are a tie, at 0.35 and at 0.2. With the ties counting half, that makes 280.5: an AUC of 0.935.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Two rates at one threshold

        Twelve mails, split by their label: `pos` holds the probabilities of spam of the 5 spam mails, and `neg` those of the 7 others. Put the true positive rate at the threshold `t` in `tpr`, and the false positive rate in `fpr`, as in the worked example. Replace the two `...`. The last line shows both, so the check sees them. The check also tries another threshold.
        """),
        L.task("Put the true positive rate in `tpr` and the false positive rate in `fpr`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("rates", lines=8)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "rates",
        [
            "Step 2 of the recipe, as in the worked example: a count divided by the number of mails of one class.",
            "`np.sum(pos >= t)` counts the spam mails called spam. Which mails are wrongly called spam, and how many mails does each class have?",
            "`tpr = np.sum(pos >= t) / len(___)` and `fpr = np.sum(___ >= t) / len(neg)`.",
        ],
        why="At 0.5 the filter calls 4 mails spam: 3 of the 5 spam mails, a true positive rate of 0.6, and 1 of the 7 others, a false positive rate of about 0.14.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_auc():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The AUC by counting pairs

        The same twelve mails; one spam mail and one other mail both have 0.4. In the loop, add to `above`, for each other mail `n`, the spam mails with a higher probability and half of those with the same one. Then put the AUC in `auc`. Replace the two `...`. The last line is `auc`, so the check sees it. The check also tries other spam mails.
        """),
        L.task("Put the AUC of the twelve mails in `auc`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_auc():
    box_auc = L.code_box("auc", lines=9)
    box_auc
    return (box_auc,)


@app.cell(hide_code=True)
def check_auc(box_auc):
    result_auc = L.check("auc", box_auc.value)
    result_auc
    return (result_auc,)


@app.cell(hide_code=True)
def hints_auc():
    L.hints(
        "auc",
        [
            "Step 4 of the recipe, as in the worked example.",
            "`np.sum(pos > n)` counts the spam mails above `n`, and `np.sum(pos == n)` the ties. How many pairs of one spam mail and one other mail are there?",
            "`above = above + np.sum(pos > n) + 0.5 * ___` and `auc = above / (len(pos) * ___)`.",
        ],
        why="Of the 35 pairs, the spam mail has the higher probability in 26, and one pair is a tie: 26.5 / 35, about 0.757.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ### Exercise 3. Only the order
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="A second filter gives each of the forty mails half the probability the first filter gives it. What is its AUC?",
        options={
            "half": "Half as much: about 0.47",
            "right": "The same: 0.935",
            "threshold": "That depends on the threshold",
        },
        correct="right",
        why={
            "half": "Halving changes the numbers, but does a spam mail that was above another mail stay above it? See *The idea*.",
            "threshold": "The AUC belongs to no threshold: it counts pairs. See step 4 of the recipe.",
            "right": "Halving keeps the order of every pair, and the AUC only looks at that order. At the threshold 0.5, though, the second filter calls no mail spam: no probability reaches 0.5 any more.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The ROC curve shows every threshold of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) at once. The F1 score of [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) belongs to one threshold; the AUC belongs to none, so it compares models before you choose a threshold. With a rare class, the false positive rate divides by the many negatives, so it stays small even when the false alarms outnumber the hits: the detector of 5s with the threshold 0.05 in that lesson had a false positive rate of 0.13, while 120 of its 197 alarms were false. Then look at the precision-recall curve as well. In the next lesson, *Always beat a simple baseline*, you compare a model with the simplest rule you can think of.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rates, result_auc, choice_order):
    L.wrap_up(result_rates, result_auc, choice_order)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
