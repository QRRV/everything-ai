# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-code-tests. Route: course/route/lessen/systemen.yaml. Checks, the versions with a bug, the course's runs on 16 hours, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-code-tests", "en")
    C = L.checks   # the versions with a bug, the course's runs on 16 hours, the checks of your tests, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en) and [Reading error messages and finding bugs](/course/open/py-errors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_days():
    quick_days = L.choice(
        "quick-days",
        question="In your project ops, the validation set is every fifth day of 2011, each with all its hours. Why whole days, and not 1,733 hours picked at random?",
        options={
            "faster": "Whole days make the validation set smaller, so measuring after every epoch takes less time",
            "right": "The hours of one day share weather and busy moments, so a validation day must be new to the model",
            "pandas": "A mask on the day is the only kind of mask that pandas can make on a table of hours",
            "night": "Whole days keep the night hours out of the training set, since almost nobody rents a bike in the middle of the night",
        },
        correct="right",
        why={
            "right": "With random hours, the model has learned from the other hours of the same days, so the validation hours are easier than a new day: the RMSE came out about 7 bikes lower.",
            "faster": "73 whole days have 1,733 hours, as many as 1,733 random hours would: what do the hours of one day share? See [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).",
            "pandas": "A mask can pick rows by any column, such as the hour: what do the hours of one day share that makes a split by day better? See [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).",
            "night": "Every validation day keeps all its hours, the night hours too, and so does every training day: what do the hours of one day share? See [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).",
        },
    )
    quick_days
    return (quick_days,)


@app.cell(hide_code=True)
def quick_check_errors():
    quick_errors = L.choice(
        "quick-errors",
        question="A function you wrote runs without an error message. What does that tell you?",
        options={
            "answer": "That its answers are right, since Python stops with an error at every mistake in a function",
            "checked": "That it has been tested, since Python tries every new function on a few inputs of its own before it runs it",
            "right": "Only that Python could run it: whether its answers are right, inputs whose answer you know will tell",
            "types": "That every value in it has the right type, so only its numbers can still be a little off",
        },
        correct="right",
        why={
            "right": "A bug without an error gives a wrong answer quietly, such as 0 as the largest number of `[-3, -7]`, where −3 was meant. Inputs whose answer you know make it show.",
            "answer": "Python stops only where it cannot go on, such as at a name it does not know: what about a function that runs and gives a wrong number? See [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
            "checked": "Python runs what you wrote and tries nothing by itself: who tries the function on inputs whose answer is known? See [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
            "types": "Python does not check types before it runs a function, and a wrong answer can have the right type: how do you find a bug without an error? See [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
        },
    )
    quick_errors
    return (quick_errors,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en), a version of your model trained in a few seconds. A real model trains for hours or days, on machines that cost money for every hour. Most bugs in training code do not stop the run with an error: the code runs, the loss falls a little, and the model comes out worse than it could be. You find out at the end, if you find out at all.

    Tests find such bugs in seconds, before the long run. Engineers at Google listed 28 tests and checks for models in production, from the data to the model to how it runs, in *The ML Test Score* (Breck and others, 2017). This lesson takes three tests in that spirit, for the code of your project *ops*: a test of one function on one row, a test that a network can learn a few hours by heart, and a check that runs on every prediction once the model is in use.

    **An everyday example.** A baker tries a new oven with one small bun before baking a hundred loaves. The bun takes ten minutes, and if it comes out raw or burnt, the baker knows something is wrong with the oven, without losing a day of flour.
    """)
    return


@app.cell(hide_code=True)
def predict_column():
    guess_column = L.guess(
        "column",
        question="In this lesson a fresh network of 577 numbers trains for 400 steps on only 16 hours of 2011. Suppose a bug in the training code turns `y` into a row of 16 numbers, while the network gives a column of 16. What does the loss on these 16 hours do, do you think?",
        options={
            "zero": "It still falls to almost 0: the network learns the 16 hours by heart all the same",
            "right": "It falls for a while, then stops far above 0 and stays there",
            "grow": "It grows with every step, until it is far above where it started",
            "same": "It stays where it started, since the network cannot learn at all",
        },
        reveal="**It stops far above 0.** The loss falls from 36,930 to about 21,091 within the first 100 steps and stays there, while the right code gets it to 0.99. With `y` as a row, `model(x) - y` gives 16 × 16 differences: every prediction against every one of the 16 hours. The one number closest to all 16 hours at once is their mean, so the network learns to say 125.9 bikes for every hour, and 21,091 is the variance of the 16 hours: the mean squared distance to their mean. The run gives no error, and the loss even falls by more than a third.",
    )
    guess_column
    return (guess_column,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A test is a function that asserts.** `assert` followed by a condition does nothing when the condition is True, and stops with an `AssertionError` when it is False. A test is a function whose name starts with `test`, that calls your code on an example and asserts what must come out. pytest, the usual tool for tests in Python, finds every such function in the files whose names start with `test_`, and runs them all at once; you run it after every change, and in this lesson the course runs your tests for you. A test passes when it ends without an error, and fails when an assert, or any other error, stops it.

        **A unit test: one function, one example you can work out by hand.** As in [Reading error messages and finding bugs](/course/open/py-errors?lang=en), you test a small piece with an input whose answer you know. For `inputs` from your project, one row is enough: hour 8 must give a 1 in column 8 of the 24 hour columns, and a feature exactly at its mean must give 0. You choose the row so that you know every one of its 34 numbers.

        **A good test fails when the code is wrong.** A test that passes for the right code is only half a test: it must also fail for code with a bug. The checks of this lesson try your tests on the course's version of the code and on versions with bugs that happen in real projects, and show for each version whether your test passes or fails.

        **Shapes and ranges.** Many bugs in training code show in the shape of a tensor or in a value that cannot be: a row where a column was meant, a NaN, a negative number of bikes. A line such as `assert pred.shape == y.shape` before the loss costs nothing and catches the bug of the guess above, as the worked example shows.

        **Overfit one batch.** A network of 577 numbers can learn 16 hours by heart: trained on those hours only, with nothing else to learn, its loss must come close to 0. If it does not, the network cannot learn even what is easiest, and the reason is almost always a bug: a wrong shape, a step that never happens, gradients that pile up. The test takes a second, while a long run on all the data would hide the bug behind numbers that look plausible. Andrej Karpathy made this test widely known in *A Recipe for Training Neural Networks* (2019).
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The picture shows the guess: with the right code the loss falls to about 1, and with `y` as a row it stops at 21,091. On a scale where every step up is ten times more, both lines fit in one chart.

        **A regression test keeps a score.** Your packed model has a validation RMSE of 53.04 bikes per hour. A regression test asserts after every change that the RMSE stays below a fixed bound: when a change makes the model worse, the test fails. A regression here is something that worked and broke again, not the task of predicting a number.

        **A model assertion runs while the model is in use.** Tests run before the model goes out; a model assertion looks at every prediction once it is in use, and flags the ones that cannot be right. The course's model of your project expects −10.8 bikes at 4 in the morning of 14 July 2011, and fewer than 0 bikes in 235 of the 8,645 hours of 2011, all from midnight to 5 in the morning. Rounding those to 0 would hide them; a model assertion counts them, so that someone looks, and the flagged hours are good examples to learn from later. Daniel Kang and others gave the idea this name in 2020.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **unit test** | | a small test of one function, on an example whose answer you know | `def test_one_row():` |
        | `assert` | | does nothing when the condition after it is True; stops with an `AssertionError` when it is False | `assert x.shape == (1, 34)` |
        | **overfit one batch** | | train a fresh network on a few examples only: its loss must come close to 0, or the code has a bug | `train_epoch(model, opt, x, y, 16)` |
        | **regression test** | | a test that something which worked before still works after a change, such as a score that may not get worse | `assert rmse <= 55` |
        | **model assertion** | | a check on every prediction while the model is in use, that flags predictions that cannot be right | `impossible(pred)` |
        | **NaN** | nan | not a number: what `torch.tensor(0.0) / 0` gives, and how a table marks a missing value; it is equal to nothing, not even to itself | `float("nan")` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Tests for ML code:

    1. For each function, pick an example whose answer you can work out by hand, such as one row, and assert what must come out: a unit test.
    2. Make sure each test fails for code with a bug: try it on a version with a bug you know, and see it fail.
    3. Before a long run, assert shapes and ranges: predictions and targets of the same shape, no NaN, no value that cannot be.
    4. Overfit one small batch: a fresh network trained on a few examples only must get its loss close to 0.
    5. Keep the score in a regression test: a fixed bound a little above the score you have.
    6. Once the model is in use, flag every prediction that cannot be right with a model assertion, and count them.
    7. Run all tests after every change.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four hours, as in the worked example of [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en): the network says 30, 40, 220 and 110 bikes, as a column of shape (4, 1), and there were 10, 40, 250 and 100. With `y` as a column too, `pred - y` has the shape (4, 1): the differences 20, 0, −30 and 10, their squares 400, 0, 900 and 100, and the loss is their mean, 350.

    With `y` as a row of shape (4,), broadcasting makes `pred - y` a table of shape (4, 4): each of the 4 predictions minus each of the 4 numbers of bikes, 16 differences, and the loss is the mean of their 16 squares: 14,300. Now the loss rewards the wrong thing. A network that says exactly 10, 40, 250 and 100 gets 17,100, and one that says 100, the mean, for every hour gets 8,550, the lowest any network can get: the variance of the four hours. So training pulls every prediction to the mean, as in the guess.

    The line `assert pred.shape == y.shape` sees (4, 1) and (4,), and fails before a single step.
    """)
    return


@app.cell(hide_code=True)
def text_few():
    mo.md(r"""
    ### The 16 hours

    The test of exercise 3 trains on 16 hours of 2011, which `.sample(16, random_state=0)` picks from all 8,645: the same 16 every time. The cell reads the hours with `read_hours` from your project, and prints the date, the hour and the bikes of each of the 16 with `.to_string(index=False)`, then the fewest bikes with `.min()`, the most with `.max()` and their mean with `.mean()`. Run the cell.
    """)
    return


@app.cell
def example_few():
    from ops.demand import read_hours

    hours = read_hours()
    few = hours.sample(16, random_state=0)   # the same 16 hours every time
    print(few[["dteday", "hr", "cnt"]].to_string(index=False))
    print("from", few["cnt"].min(), "to", few["cnt"].max(), "bikes, mean", few["cnt"].mean())
    return


@app.cell(hide_code=True)
def text_exercise_one_row():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A unit test of one row

        `test_one_row()` tests `inputs` from your project, which comes from `ops.demand` together with `FEATURES`, the names of the 10 features. The lines of the test are given, and it imports `pandas` as `pd`. `scale` holds a made-up mean and standard deviation for each feature, in the order of `FEATURES`. `zip` pairs each name with its mean, and `dict` makes a row of them: every feature exactly at its mean. Then `row["temp"]` becomes 0.75, one standard deviation of 0.25 above its mean of 0.5, and the hour `row["hr"]` becomes 8. `pd.DataFrame([row])` makes a table of that one row, and `inputs` gives `x`, a tensor of shape (1, 34). The last line asserts `x.shape`; it passes for every version of `inputs` the check tries.

        Add lines with `assert` under it, indented as the others, that say what must be in two columns of `x`. `x[0, 8]` is the number in column 8 of the first row. The first 24 columns are the hours 0 to 23, one-hot; then come the 10 features in the order of `FEATURES`: weekday, workingday, holiday, weathersit, temp, atemp, hum, windspeed, season and mnth, each standardized: minus the mean, divided by the standard deviation.

        The check runs `test_one_row()` on the right `inputs`, where it must pass, and on four versions with a bug, where it must fail: hour 8 in column 9, the features before the hours, the mean not subtracted, and the mean of the given hours subtracted instead of the mean in `scale`. Under the check you see how your test does on each. When it is right, the check saves your test as `work/projects/ops/test_inputs.py`.
        """),
        L.task("Add lines with `assert` that catch all four versions with a bug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one_row():
    box_one_row = L.code_box("one-row", lines=14)
    box_one_row
    return (box_one_row,)


@app.cell(hide_code=True)
def check_one_row(box_one_row):
    result_one_row = L.check("one-row", box_one_row.value)
    result_one_row
    return (result_one_row,)


@app.cell(hide_code=True)
def hints_one_row():
    L.hints(
        "one-row",
        [
            "Hour 8 is one-hot: one of the 24 hour columns of `x` holds 1, the other 23 hold 0. The features come after the 24 hour columns, in the order of `FEATURES`. A line with `assert` and `==` asserts one number of `x`.",
            "temp is the fifth feature in `FEATURES`, at index 4. For this row, every feature but temp is exactly at its mean: (value − mean) / standard deviation = 0. temp gives (0.75 − 0.5) / 0.25. The feature columns are 24 to 33.",
            "`    assert x[0, ___] == 1`",
        ],
        why="`assert x[0, 8] == 1` and `assert x[0, 28] == 1`, both indented in `test_one_row`: hour 8 in column 8, and temp, (0.75 − 0.5) / 0.25 = 1, in column 24 + 4 = 28.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_impossible():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A model assertion

        `impossible(pred)` gets a tensor of predictions, of any shape, and gives back a mask of the same shape: True for every prediction that cannot be right, False for the others. Two kinds cannot be right: fewer than 0 bikes, and NaN, which an hour gives when one of its inputs is missing, such as its humidity. 0 bikes can happen, late at night. A comparison such as `pred < 0` gives a mask. `torch.isnan(t)` is True where the tensor `t` holds NaN; `==` cannot find NaN, since NaN is equal to nothing. `|` combines two masks: True where at least one of the two is True. Python does `|` before `<`, so give each comparison its own parentheses. Replace the `...`.

        The check calls `impossible(PRED)`, with `PRED` the 24 predictions of the course's model for Thursday 14 July 2011; it should be True at the places 3 and 4, the hours for which the model expects −6.3 and −10.8 bikes. Then it tries three tensors of its own: one with a NaN, a column, and one with zeros. When it is right, the check saves your function as `work/projects/ops/assertions.py`.
        """),
        L.task("Make `impossible` True for every prediction below 0 and for every NaN."),
    ])
    return


@app.cell(hide_code=True)
def exercise_impossible():
    box_impossible = L.code_box("impossible", lines=5)
    box_impossible
    return (box_impossible,)


@app.cell(hide_code=True)
def check_impossible(box_impossible):
    result_impossible = L.check("impossible", box_impossible.value)
    result_impossible
    return (result_impossible,)


@app.cell(hide_code=True)
def hints_impossible():
    L.hints(
        "impossible",
        [
            "Two masks, each from one comparison on `pred`: one for fewer than 0 bikes, and one from `torch.isnan`. `|` puts them together: True where either one is True. Each comparison goes between its own parentheses.",
            "NaN is not even equal to itself: `float(\"nan\") == float(\"nan\")` is False, so `==` never finds it. In `PRED` only the hours 3 and 4 are below 0, and `PRED` holds no NaN; one of the tensors the check tries does.",
            "`    return (pred < 0) | ___`",
        ],
        why="`return (pred < 0) | torch.isnan(pred)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_overfit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Overfit one batch

        `test_overfit_one_batch()` tests `train_epoch` from your project, which it imports from `ops.demand` with the other parts it needs. The lines of the test are given. `read_hours` reads the hours of 2011 into `hours`, and `.sample(16, random_state=0)` puts the 16 hours of the cell above in `few`. `inputs` and `targets` make `x` and `y` of them, with `scaling` on all the hours. `torch.manual_seed(0)` makes the start the same every time, `DemandNet(16)` is a fresh network of 577 numbers, and `torch.optim.Adam` with `lr=0.03`, a learning rate of 0.03, is its optimizer, which gets the numbers of the network from `model.parameters()`, as in [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en). `first` is the loss before training: the mean squared error, with `.mean()` over the squared differences, and `.item()` turns it into a Python number. `for _ in range(400)` runs `train_epoch` 400 times; with a batch of 16, each time is one step on all 16 hours. `last` is the loss after them.

        Replace the `...` after `assert` with a condition on `first` and `last`. The right `train_epoch` gets the loss from 36,930 to below 1. The check runs your test on it, where it must pass, and on three versions with a bug, where it must fail: one without `opt.step()`, one that compares `model(x)` with `y` as a row, and one without `opt.zero_grad()`. Each run takes a fraction of a second. When it is right, the check saves your test as `work/projects/ops/test_training.py`.
        """),
        L.task("Make the assert fail unless the loss comes close to 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overfit():
    box_overfit = L.code_box("overfit", lines=16)
    box_overfit
    return (box_overfit,)


@app.cell(hide_code=True)
def check_overfit(box_overfit):
    result_overfit = L.check("overfit", box_overfit.value)
    result_overfit
    return (result_overfit,)


@app.cell(hide_code=True)
def hints_overfit():
    L.hints(
        "overfit",
        [
            "`last` and `first` are numbers: the loss after and before training. A condition such as `last < first` asks only that the loss falls, and a network that can learn 16 hours by heart should get much further. Multiply `first` by a small number to make a bound.",
            "With `y` as a row the loss stops at 21,091, 57% of where it started, and without `opt.zero_grad()` it ends at 28,134. The right version ends at 0.99, less than a thousandth of 36,930.",
            "`    assert last < ___ * first`",
        ],
        why="`assert last < 0.01 * first`: below 1% of where it started. Any bound from about 0.0001 to 0.5 times `first` passes for the right version here and fails for all three bugs.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Three versions of train_epoch, and your test on each

    The button trains a fresh network on the 16 hours three times, 400 steps each, with three versions of `train_epoch`: the right one, the one with `y` as a row, and the one without `opt.zero_grad()`. The chart shows the loss of the version that trains now, with the versions done before it as thin dashed lines. Beside it you see what the network says for each of the 16 hours, from the quietest to the busiest, against the bikes that were really rented. Then your test from exercise 3 runs on each version. When the run is done, you can go back through the steps.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Train three versions on 16 hours")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_overfit, run_button):
    result_overfit   # run again with your test once the check saves it
    if not run_button.value:
        _out = mo.md("*Press the button to train the three versions.*")
    else:
        _now = {}

        def _show(_name, _step, _loss, _preds, _done):
            if _now.get("name") != _name:   # a new version: a chart of its own, with the versions done before it
                _chart = L.live(C.live_labels(_name, L.lang), C.STEPS, role="loss", low=C.LOW, high=C.HIGH, window=1)
                for _other, _points in _done:
                    _chart.reference(_points, C.version_name(_other, L.lang), role="loss")
                _chart.watch(lambda: C.now_picture(_now, L.lang))
                _now["chart"], _now["name"], _now["preds"] = _chart, _name, []
            _now["step"] = _step
            if _preds is not None:
                _now["preds"] = _preds
            _now["chart"].add(_step, _loss, C.progress_line(_name, _step, _loss, L.lang))

        _r = C.demo(C.saved_test(), show=_show)
        if isinstance(_r, str):
            _note = C.slowed(L.lang)
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _out = mo.vstack([
                L.explore("rewind", "steps", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.STEPS,
                          low=0, high=C.STEPS, step=10, role="",
                          task="Go to step 50, then to 100 and to 400. Where are the dots of each version at step 400, and between which steps do the right version and the version without `opt.zero_grad()` part ways?"),
                mo.md(C.verdicts(_r, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        The right version gets the loss from 36,930 to 0.99: after 400 steps almost every dot sits on its bar, a network that knows the 16 hours by heart. With `y` as a row, all the dots move to one height, 125.9 bikes, the mean of the 16 hours, and the loss stays at 21,091. Without `opt.zero_grad()`, the first 50 steps look like the right ones, 11,725 against 11,481; then the gradients of all earlier steps pile up, every step keeps pushing the way earlier steps pushed, and the loss jumps between 5,439 and 28,134. A test that looked at step 50 only would have missed that bug.

        None of the three versions stopped with an error. The version without `opt.step()`, which the check of exercise 3 also tries, would draw a flat line: its loss stays at 36,930.
        """),
        L.watch_out("Overfitting one batch shows that the code can learn; it does not show that the model is good. A network that knows 16 hours by heart says nothing about a new day: that is what the validation set is for."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_regression():
    mo.md(r"""
    ### Exercise 4. A regression test for the score

    Your packed model has a validation RMSE of 53.04 bikes per hour. After every change to the code, a regression test trains it again and asserts something about the new `rmse`.
    """)
    return


@app.cell(hide_code=True)
def choice_regression_cell():
    choice_regression = L.choice(
        "regression",
        question="Which assert keeps the score from getting worse, without failing when nothing is wrong?",
        options={
            "exact": "`assert rmse == 53.04`: the score must stay exactly what it is now",
            "baseline": "`assert rmse <= 132.8`: no worse than a model that always says the mean of the training hours",
            "right": "`assert rmse <= 55`: a fixed bound a little above the score you have now",
            "better": "`assert rmse < last_rmse`: every new run must beat the run before it",
        },
        correct="right",
        why={
            "right": "53.04 moves a little without anything being wrong: with the seeds 1 and 2 the same version ends at 52.8 and 53.5. A bound of 55 lets that pass, and fails when a change makes the model clearly worse, such as a change that brings it to 79.1, no better than the mean of the same hour on the same kind of day.",
            "exact": "Another seed, or a harmless change to the code, moves 53.04 by a few tenths: the test would fail without anything being wrong. What must a test of a score allow?",
            "baseline": "A model 100 bikes off would still pass: the test catches only a model worse than saying that mean for every hour. How far may the score rise before something is wrong?",
            "better": "Two runs where nothing is wrong differ a little either way, so this test fails on about half of them. Which fixed number should the score stay under?",
        },
    )
    choice_regression
    return (choice_regression,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Reading error messages and finding bugs](/course/open/py-errors?lang=en) you tested small pieces with inputs whose answer you knew; here those tests became functions that pytest can run, and each one must also fail for code with a bug. Your project *ops* from [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en) now has three of them: `test_inputs.py`, `test_training.py`, and the model assertion in `assertions.py`. Tests of code check what the code does; the next lesson, *Does the model behave? Behavioral tests*, checks what the model does, such as more bikes in good weather. In *Project: tests and a model card* your demand model gets both, and in *Project: your model as a service, watched for drift* it answers requests.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_regression, result_impossible, result_one_row, result_overfit):
    L.wrap_up(result_one_row, result_impossible, result_overfit, choice_regression)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
