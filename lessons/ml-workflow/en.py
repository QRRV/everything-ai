# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-workflow. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-workflow", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Always beat a simple baseline](/course/open/ml-baselines?lang=en), [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) and [Where does this data come from?](/course/open/data-provenance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="A model predicts house prices with an MSE of 900 on the test set. Always predicting the mean price of the training set gets an MSE of 850 on the same houses. What does that tell you?",
        options={
            "small": "The model is good: an MSE of 900 is small for prices of a few hundred thousand euros",
            "right": "The model has not shown that it learned anything from the features",
            "cheat": "The baseline cheats: it was worked out on the test set",
        },
        correct="right",
        why={
            "small": "An MSE has no fixed scale: whether 900 is small, only a comparison tells you. Which number is lower? See [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "cheat": "The mean comes from the training set, as the model does; only the measuring happens on the test set. See [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "right": "For an MSE lower is better, and the mean alone gets 850: the features did not help the model on this metric.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="A filter moves 8 mails to the spam folder, and 6 of them are spam. There were 10 spam mails in all. What are its precision and recall?",
        options={
            "right": "Precision 0.75, recall 0.6",
            "swapped": "Precision 0.6, recall 0.75",
            "same": "Both 0.6",
        },
        correct="right",
        why={
            "swapped": "Those are the other way round. Which of the two looks only at the mails the filter moved? See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "same": "That is the recall. What do you divide by for the precision? See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "right": "Of the 8 mails it moved, 6 are spam: precision 0.75. Of the 10 spam mails, it found 6: recall 0.6.",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="You find a set of photos on a website, with no license anywhere. What may you do with it?",
        options={
            "free": "Use it freely: it is on the internet",
            "credit": "Use it and share it, as long as you name the website it came from",
            "right": "Assume you may not copy or share it, and ask the maker first",
        },
        correct="right",
        why={
            "free": "Being on the internet does not make something free to use. What does no license mean? See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "credit": "Naming the source is what some licenses ask; here there is none. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "No license means no permission: ask the maker before you share it or build on it.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You know the pieces of a project by now: a metric, a training, validation and test set, a baseline, a model, and a confusion matrix that shows where it goes wrong. A real project is more than its pieces. It starts before the first line of code, with a question, and it does not end with the first model. The order matters too: a goal you set after seeing the results bends towards them, as a test score does when you choose with the test set.

    **An everyday example.** A dish you cook for the first time is rarely right at once. You taste it, change one thing, and cook it again.
    """)
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first",
        question="A webshop asks you for a model that predicts which customers will leave. What do you think you should do first?",
        options={
            "data": "Collect as much data about the customers as you can, from every system the shop has",
            "model": "Train a first model, to see how far you get",
            "goal": "Agree on what counts as leaving, and how you will judge a model",
            "method": "Choose the best kind of model for the job",
        },
        reveal="**Agree on what counts as leaving, and how you will judge a model.** Is a customer who orders nothing for six months gone? And which mistake costs more: taking a loyal customer for one who is leaving, or missing one who is? Until that is agreed, any result can be made to look good afterwards. It may even turn out that a simple rule does the job, and then you need no model at all.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **machine learning workflow** is the order of the steps of a project, and the way back:

        1. **Question.** What decision should the model help with? First check whether a fixed rule already does it: then you do not need machine learning. If not, write it as a task: what are the examples, the features and the label?
        2. **Metric.** How will you judge a model, and how good is good enough? That follows from what each kind of mistake costs. Agree on it before you train.
        3. **Data.** Where does it come from, and may you use it? Split it into a training, validation and test set before you look closely.
        4. **Baseline.** What does the simplest prediction get on the validation set: the most common class, the mean, or a simple rule?
        5. **Model.** Train on the training set, and choose on the validation set, with the metric of step 2.
        6. **Round again.** Look at the examples it gets wrong, and go back to the step that holds it back: more or better data, other features, another model, and now and then a better question.

        Only when everything has been chosen do you measure the model, and the baseline, once on the test set, and report the two side by side.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **machine learning workflow** | | the steps of a project in their order: question, metric, data, baseline, model, and round again until it is good enough | |
        """),
        L.watch_out("Agree on the metric and the goal before you see results. Chosen afterwards, they bend towards the model you happen to have, as a test set does when you choose with it."),
        mo.Html(C.picture([("1. Question", "what should it decide?"), ("2. Metric", "how good is good enough?"), ("3. Data", "where from? split it"),
                           ("4. Baseline", "the simplest prediction"), ("5. Model", "train, choose on validation"), ("6. Round again", "where does it go wrong?")],
                          "back to the step that holds it back", "The machine learning workflow",
                          "Six boxes in a loop: 1 question, 2 metric and 3 data down the left, then across to 4 baseline, and 5 model and 6 round again up the right. A dashed arrow runs from 6 back to 3, where the round starts again: back to the step that holds it back.")),
        mo.md(r"""
        Steps 1 and 2 come first, and the test set is put aside once. Then steps 3 to 6 go round: from step 6 back to the step that holds the model back, most often the data or the model, until the model is good enough, or until it is clear that it will not be.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To run a machine learning project:

    1. Write down the question, and check that no fixed rule answers it.
    2. Agree on the metric and the goal, from what each mistake costs.
    3. Check where the data comes from and whether you may use it, and split it.
    4. Measure a baseline on the validation set.
    5. Train a model, and choose on the validation set by the metric.
    6. Look at what it gets wrong, go back to the step that holds it back, and repeat. At the end, measure once on the test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A spam filter, from step 1 to step 6. There is already a filter that gives every mail a probability of spam, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en); what is left to choose in step 5 is its threshold. Here are 20 mails from the validation set, with their probability and whether they are spam. `(_probs >= _t) * 1` turns `True` and `False` into 1 and 0: the predictions of the filter. And `_probs[_y == 0]` picks from `_probs` with a mask made from the labels: the probabilities of the real mails. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_workflow():
    import numpy as np

    # 1. the question: move spam to a folder of its own; a real mail may hardly ever end up there
    # 2. the metric: a precision of at least 0.9, and then the highest recall
    # 3. the data: mails that may be used for this, split before anyone looked closely;
    #    here 20 mails of the validation set, their probability of spam, and 1 for spam
    _probs = np.array([0.97, 0.12, 0.83, 0.66, 0.04, 0.35, 0.92, 0.28, 0.58, 0.09,
                       0.45, 0.62, 0.02, 0.81, 0.22, 0.88, 0.06, 0.41, 0.74, 0.18])
    _y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    # 4. the baseline: call every mail spam; its precision is the share of spam
    print(np.mean(_y))
    # 5. the model: a threshold, chosen by the metric; each line: threshold, precision, recall
    for _t in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9]:
        _pred = (_probs >= _t) * 1
        _tp = np.sum(_pred * _y)
        print(_t, _tp / np.sum(_pred), _tp / np.sum(_y))
    # 6. round again: the real mails with a high probability, and the spam below 0.85
    _ham = _probs[_y == 0]
    _spam = _probs[_y == 1]
    print(_ham[_ham >= 0.5])
    print(_spam[_spam < 0.85])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    8 of the 20 mails are spam, so calling every mail spam gives a precision of 0.4: far below the goal of 0.9, and calling nothing spam finds no spam at all. With the filter, the precision mostly rises with the threshold: at 0.8 it dips, because a spam mail falls below it. Only from 0.85 does it reach 0.9. There the recall is 0.375: the filter moves 3 of the 8 spam mails. That meets the goal, but it barely helps, and that is what step 6 is for. Two real mails get 0.83 and 0.62, and five spam mails get between 0.45 and 0.81. Reading those seven mails shows where the filter goes wrong. Perhaps it needs other features, more examples of this kind of spam, or another model. Then you go round again. The test set waits until the end.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_need():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Is machine learning needed?
    """)
    return


@app.cell(hide_code=True)
def choice_need_cell():
    choice_need = L.choice(
        "need-ml",
        question="A library wants to send a reminder for every book that is more than 21 days late. What does that need?",
        options={
            "model": "A classifier, trained on old loans with their return dates and reminders",
            "right": "No machine learning: the fixed rule, more than 21 days late, does it",
            "baseline": "A baseline first, and then a model that beats it",
        },
        correct="right",
        why={
            "model": "What would a model learn here that the library does not already know exactly? See step 1 in *The idea*.",
            "baseline": "A baseline is for a task a model has to learn. Does anything have to be learned here? See step 1 in *The idea*.",
            "right": "The rule is known and exact: you write it as code. Machine learning is for decisions whose rule you cannot write down, such as spam or handwriting.",
        },
    )
    choice_need
    return (choice_need,)


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ### Exercise 2. Which step?

    A team builds a spam filter. Put everything the team does in the step of the workflow it belongs to.
    """)
    return


@app.cell(hide_code=True)
def sort_steps_cell():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps",
        [
            "Ask for each one: does it happen before there are examples, to the examples, or to a model?",
            "Two of them are about the examples. Which one comes after there is a model, and looks at where it goes wrong?",
            "Question: what it is for. Metric: \\_\\_\\_. Data: permission, and \\_\\_\\_. Baseline: never spam. Model: training. Round again: \\_\\_\\_.",
        ],
        why="What it is for is the question; at most 1 real mail in 20 in the spam folder is the metric; permission and putting the test set aside are the data; never spam is the baseline; training and choosing the threshold is the model; reading the false alarms starts the next round.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Let the metric choose

        Five spam filters were measured on the validation set: `names`, `precision` and `recall` hold their names and scores, in the same order. The goal is the same as in the worked example: a precision of at least 0.9, and then the highest recall. Replace the `...` with the choice: when a filter reaches a precision of at least 0.9, and its recall is higher than `best_recall`, it becomes `best`, and its recall the new `best_recall`. The last line is `best`, so the check sees it. The check also tries other filters.
        """),
        L.task("Put in `best` the name of the filter with the highest recall among those with a precision of at least 0.9."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick-model", lines=12)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick-model", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick-model",
        [
            "Two questions for each filter, in this order: does it reach the goal for precision, and does it find more spam than the best so far?",
            "An `if` inside an `if`: the outer one asks about `precision[i]`, the inner one about `recall[i]`. What has to change when both are true?",
            "`if precision[i] >= 0.9:`, inside it `if recall[i] > best_recall:`, and inside that `best = ___` and `best_recall = ___`.",
        ],
        why="Four filters reach a precision of at least 0.9: B, C, D and E, with recalls of 0.40, 0.75, 0.60 and 0.30. Of those, C finds the most spam. A has the highest recall of all, 0.90, but its precision of 0.85 misses the goal.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Always beat a simple baseline](/course/open/ml-baselines?lang=en) gave you step 4, [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) the metrics of step 2 and the look at mistakes of step 6, and [Where does this data come from?](/course/open/data-provenance?lang=en) the questions of step 3. The workflow puts them in order, and turns them into a loop. In the next lesson, *Project: predict something real, end to end*, you run the whole workflow on a real table, the Palmer penguins, with scikit-learn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_need, result_pick, result_steps):
    L.wrap_up(choice_need, result_steps, result_pick)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
