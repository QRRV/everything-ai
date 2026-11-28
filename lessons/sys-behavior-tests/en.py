# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-behavior-tests. Route: course/route/lessen/systemen.yaml. Checks, the five versions of the model, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-behavior-tests", "en")
    C = L.checks   # the course's model and four versions with a bug, the checks of your tests, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Tests for ML code](/course/open/sys-code-tests?lang=en) and [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tests():
    quick_tests = L.choice(
        "quick-tests",
        question="A test you wrote passes for the right version of your code. What else must be true before you trust it?",
        options={
            "fast": "It must finish within a second, since pytest stops every test that takes longer than that",
            "right": "It must fail for a version of the code with a bug you know, or it may pass for any code",
            "many": "It must hold three or more lines with `assert`, one per kind of bug a function has",
            "random": "It must also pass on random inputs, since one example whose answer you know shows nothing",
        },
        correct="right",
        why={
            "right": "A test that passes for every version catches nothing. In Tests for ML code each test had to fail for every version with a bug, such as an `inputs` that put hour 8 in column 9.",
            "fast": "pytest stops no test for taking its time; a slow test is only a nuisance: what shows that a test can catch a bug? See [Tests for ML code](/course/open/sys-code-tests?lang=en).",
            "many": "One line with `assert` can catch several bugs, and three lines can catch none: what shows that a test catches a bug? See [Tests for ML code](/course/open/sys-code-tests?lang=en).",
            "random": "One example whose answer you know is what a unit test is made of: what shows that the test can fail? See [Tests for ML code](/course/open/sys-code-tests?lang=en).",
        },
    )
    quick_tests
    return (quick_tests,)


@app.cell(hide_code=True)
def quick_check_rmse():
    quick_rmse = L.choice(
        "quick-rmse",
        question="Your packed model has a validation RMSE of 53.04 bikes per hour. What does that one number tell you?",
        options={
            "every": "That no hour of a validation day is more than 53.04 bikes off, at night or in busy hours",
            "new": "That the model will be about 53.04 bikes off per hour in 2012 too, since the validation days are new",
            "half": "That about half of the hours of the validation days are more than 53.04 bikes off, and half are less",
            "right": "How far off the model is over all hours of the validation days, in one number, not where or when",
        },
        correct="right",
        why={
            "right": "The RMSE squares every error, takes the mean over the 1,733 validation hours and then the square root: one number for all hours together. Whether the model gets the night right, or rain, it does not say, and that is what this lesson tests.",
            "every": "Some hours are much further off and many are closer: the RMSE is the square root of the mean squared error. What does one number over 1,733 hours leave out? See [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).",
            "new": "The validation days are new to the model, but they are days of 2011, with the weather and the riders of that year: does a score on 2011 say how 2012 goes? See [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).",
            "half": "That would be the median of the errors, not the RMSE, the square root of the mean squared error: what does one number over all the hours tell, and what not? See [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en).",
        },
    )
    quick_rmse
    return (quick_rmse,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your packed model has a validation RMSE of 53.04 bikes per hour: one number over 1,733 hours. It does not tell you whether the model gets the simple things right, such as a quiet night, whether rain makes it expect fewer bikes, or whether its answer changes when something changes that should not matter. The model assertion of [Tests for ML code](/course/open/sys-code-tests?lang=en) found 235 hours of 2011 for which the model expects fewer than 0 bikes, and the RMSE gave no sign of them.

    In 2020, Marco Tulio Ribeiro and others tested models that read text, from large companies and from research, which scored high on their test sets. With small tests of one ability each, they found failures in every one of them. Their method, CheckList, won the best paper award of the conference ACL that year. Testing a model the way engineers test software, one ability at a time, is called behavioral testing. This lesson writes such tests for your demand model, of the three kinds CheckList uses.

    **An everyday example.** A driving test is not a score averaged over a thousand kilometers. The examiner checks a few things every driver must get right, such as stopping at a red light; things that must not matter, such as the color of the car; and things that must change the driving the right way, such as slowing down in fog.
    """)
    return


@app.cell(hide_code=True)
def predict_later():
    guess_later = L.guess(
        "later",
        question="A version of the model also got the column instant: the row number of each hour in the table, 1 for the first hour of 2011 and 8,645 for the last. Its validation RMSE is 49.98 bikes per hour, lower than the 53.04 of the course's model. Give it the same hours of 2011, with only the row numbers moved on a year, as the hours of 2012 have them. What does it do, do you think?",
        options={
            "same": "It expects the same bikes as before, since nothing about the hours themselves has changed",
            "right": "Its answers move, by up to tens of bikes in an hour, though only the row number changed",
            "error": "It stops with an error, since it never saw a row number above 8,645 while it learned",
            "more": "It expects more bikes in every hour, since in 2011 the later rows had more bikes",
        },
        reveal="**Its answers move.** For the same hours, a year later in the table, the version expects 12.4 fewer bikes on average, and up to 66.7 more or fewer in a single hour: 8,454 of the 8,645 hours move by more than 1 bike. On the real hours of 2012 it does worse than the course's model, with an RMSE of 145.0 against 130.7. Both do far worse on 2012 than on 2011, since 2012 had many more riders; *Project: your model as a service, watched for drift* comes back to that. The validation days could not show the problem: every fifth day of the year is one, among the training days, so there the row number looked like a fine feature. An invariance test shows it in a second, as exercise 2 does.",
    )
    guess_later
    return (guess_later,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Behavioral testing: the model from the outside.** The tests of [Tests for ML code](/course/open/sys-code-tests?lang=en) look inside: one function, one row, one batch. A behavioral test looks at the model from the outside, as it is used: a table of hours goes in, and bikes per hour come out. It tests one ability at a time, with cases you choose. It needs no real number of bikes: you say what must come out, or how one answer must relate to another. So it can run on any hour, also on hours that never happened, such as a summer day with rain in every hour.

        **Three kinds of tests.** CheckList names three:

        - A **minimum functionality test** (MFT) asks for simple cases with a known answer, which every useful model must get right. From 1 to 4 at night on a working day, almost nobody rents a bike: in 2011, fewer than 40 bikes in all but two such hours, which had 50 and 114. A useful model expects few bikes there.
        - An **invariance test** (INV) changes something that must not matter, and asks for the same answer. The column instant is the row number of each hour in the table; moving it on changes nothing about the weather, the hour or the day, so it must not change what the model expects.
        - A **directional expectation test** (DIR) changes something that matters, and asks the answer to move the known way. With rain instead of clear weather, fewer people take a bike: the model must expect fewer bikes, not more.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The picture shows two of them on one day. The dashed line, rain in every hour, lies below the solid line all day: the direction holds. From 1 to 4 at night, the model stays below the short line at 50 bikes: the minimum holds.

        **Change a copy.** An invariance test and a directional expectation test change a column of a table, and compare. `day.copy()` gives a new table with the same rows and columns; changing it leaves `day` as it was. `rainy = day` makes no new table, only a second name for the same one: a change through `rainy` changes `day` too, and a test that compares the two compares a table with itself.

        **The fixture `expect`.** Your tests get the model as a function, `expect`: a table of hours goes in, and a tensor with one number per hour comes out, what the model expects in bikes. pytest gives a test function the value of the fixture whose name matches its argument, so a test written as `def test_rain(expect):` gets the fixture `expect`. This lesson puts the file `conftest.py` in your project, which pytest reads first; `@pytest.fixture` above its function `expect` makes that function a fixture, and it loads your package from `ops/model` with `load` from `ops/serving.py`. The checks of this lesson give your tests a version of the model in its place.

        **Bugs around the model.** A behavioral test runs on the model as it is used, so it also sees bugs in the code around it, which the RMSE of training never meets. The checks try your tests on the course's model and on four versions with a bug. The hour in UTC, the clock of the whole world, 5 hours later than in Washington in winter. The two temperatures in degrees, where the data divides them by 41 and 50. The weather the wrong way round, 1 for heavy rain and 4 for clear weather. And the version with the row number from the guess. A good test passes for the course's model and fails for the bug it is meant to catch.

        **Count the failures.** Some tests are rules the model must pass. Others are questions: CheckList counts for each test how many of its cases fail, and a high count shows where the model is weak, which is worth writing down for whoever uses the model.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **behavioral testing** | | testing a model from the outside, one ability at a time, with cases you choose, instead of one score over a test set | `expect(hours)` |
        | **minimum functionality test** | MFT | a test of simple cases with a known answer, which every useful model must get right | `expect(night)` |
        | **invariance test** | INV | a test that changes something that must not matter, and asks for the same answer | `moved["instant"]` |
        | **directional expectation test** | DIR | a test that changes something that matters, and asks the answer to move the known way | `rain["weathersit"] = 3` |
        | `.copy()` | copy | a new table with the same rows and columns; changing it leaves the table it came from as it was | `day.copy()` |
        | `.between(1, 4)` | between | True for each value from 1 to 4, both ends included | `work["hr"].between(1, 4)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Behavioral tests of a model:

    1. Write down in words what the model must do: simple cases it must get right, changes that must not matter, and changes that must move the answer one way.
    2. For each, choose the hours: real ones from the data, or hours you change on purpose.
    3. For a minimum functionality test, assert a bound that every useful model meets, with room to spare.
    4. For an invariance test, change a copy and assert the same answer.
    5. For a directional expectation test, change a copy and assert that the answer moves the known way, on average or for each case.
    6. Make sure each test fails for a version with a bug you know.
    7. Count per test how many cases fail, and write down where the model is weak.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Thursday 14 July 2011 was clear all day. At 3 at night the course's model expects −6.3 bikes: below 50, so the quiet night holds for this hour, though fewer than 0 is a flaw of its own. The version with the hour in UTC adds 5 hours, as in winter, so the model gets hour 8 instead, and then it expects 425.6: far above 50, so a test of the quiet night fails for that version.

    At 8 in the morning the model expects 431.2 bikes. With rain in that hour, weathersit 3 instead of 1, it expects 373.5: 57.7 fewer, the known way. The version with the row number expects 440.4 at 8 that day, and 417.5 for the same hour a year later in the table: 22.9 fewer, though nothing about the hour changed. For the course's model both give 431.2, since it never gets the row number.
    """)
    return


@app.cell(hide_code=True)
def text_day():
    mo.md(r"""
    ### 14 July, as it was and with rain

    The cell reads the hours of 2011 with `read_hours` from your project and takes Thursday 14 July, day 194 counted from 0 for 1 January. It makes a copy with `.copy()`, and puts light rain, weathersit 3, in every hour of the copy. `C.expect` is the course's packed model of your project, the one the checks call the course's model: it gives what the model expects for each hour, and `.round()` rounds to whole bikes. `print` shows both. Run the cell.
    """)
    return


@app.cell
def example_day():
    from ops.demand import read_hours

    hours = read_hours(2011)
    day = hours[hours["day"] == 194]   # Thursday 14 July 2011
    rainy = day.copy()
    rainy["weathersit"] = 3            # light rain in every hour of the copy
    print("as it was:", C.expect(day).round())
    print("with rain:", C.expect(rainy).round())
    return


@app.cell(hide_code=True)
def text_exercise_quiet_night():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A quiet night

        `test_quiet_night(expect)` is a minimum functionality test. pytest gives it `expect`, the fixture: a function that takes a table of hours and gives what the model expects in each, as a tensor with one number per hour. The test reads every hour of 2011 with `read_hours` from `ops.demand` into `HOURS`. `work` keeps the hours of the working days, and `night` the hours from 1 to 4 at night of those days, with `.between(1, 4)`: 939 hours. `pred` holds what the model expects for each of them. In 2011, fewer than 40 bikes were rented in all but two of these hours, which had 50 and 114.

        Add a line with `assert` under it, indented as the others, that fails when the model expects many bikes in any of these hours. A comparison of a tensor with a number, such as `pred > 10`, gives a tensor with a True or False for every hour; `.all()` gives True only when every one of them is True.

        The check runs your test on the course's model, where it must pass, and on two versions with a bug, where it must fail: the hour in UTC and the temperatures in degrees. Under the check you see how your test does on each. When it is right, the check saves your test as `work/projects/ops/test_night.py`.
        """),
        L.task("Add a line with `assert` that fails when the model expects many bikes in any of these night hours."),
    ])
    return


@app.cell(hide_code=True)
def exercise_quiet_night():
    box_quiet_night = L.code_box("quiet-night", lines=10)
    box_quiet_night
    return (box_quiet_night,)


@app.cell(hide_code=True)
def check_quiet_night(box_quiet_night):
    result_quiet_night = L.check("quiet-night", box_quiet_night.value)
    result_quiet_night
    return (result_quiet_night,)


@app.cell(hide_code=True)
def hints_quiet_night():
    L.hints(
        "quiet-night",
        [
            "Compare `pred` with a bound: the comparison gives a True or False for each of the 939 hours, and `.all()` turns them into the one True or False that `assert` needs.",
            "Choose a bound that a good model stays under at night, with room to spare: in 2011 all but two such hours had fewer than 40 bikes, and the course's model expects up to 47.8. A version that sees the morning in the night expects hundreds.",
            "`    assert (___).all()`",
        ],
        why="`assert (pred < 50).all()`, indented in `test_quiet_night`: every night hour below 50 bikes. Any bound above 47.8 and up to a few hundred passes for the course's model and fails for both bugs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_row_number():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A test that always passes

        `test_row_number(expect)` is an invariance test: moving the row number of every hour on by a year, 8,645 rows, must not change what the model expects. It reads `HOURS` with `read_hours` from `ops.demand`, as in exercise 1. `moved = HOURS` is meant to give the table to change, the next line adds 8,645 to its column instant, and the `assert` compares what the model expects for `moved` with what it expects for `HOURS`: `==` for each hour, and `.all()` over all of them.

        The test has a bug. It passes for both versions the check tries: the course's model, and the version with the row number, which the test was written to catch. Find the bug, and change one line. When it is right, the check saves your test as `work/projects/ops/test_row_number.py`.
        """),
        L.task("Change one line, so that the test fails for the version with the row number and still passes for the course's model."),
    ])
    return


@app.cell(hide_code=True)
def exercise_row_number():
    box_row_number = L.code_box("row-number", lines=9)
    box_row_number
    return (box_row_number,)


@app.cell(hide_code=True)
def check_row_number(box_row_number):
    result_row_number = L.check("row-number", box_row_number.value)
    result_row_number
    return (result_row_number,)


@app.cell(hide_code=True)
def hints_row_number():
    L.hints(
        "row-number",
        [
            "Run the test in your head. After the line with `moved`, how many tables are there? Which table does the next line change, and what does the `assert` then compare?",
            "`moved = HOURS` makes no new table: it gives the one table a second name. Adding 8,645 through `moved` changes `HOURS` too, so the model gets the same rows twice. The test needs a table of its own to change.",
            "`    moved = ___`",
        ],
        why="`moved = HOURS.copy()`: a new table with the same rows, so that the next line changes the row numbers of `moved` only. The version with the row number then expects other bikes for `moved` than for `HOURS`, and the test fails for it, as it should.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rain():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Rain

        `test_rain(expect)` is a directional expectation test. It reads `HOURS` with `read_hours` from `ops.demand`, as in exercise 1. `clear` holds the 5,645 hours of 2011 with clear weather, weathersit 1, and `rain` is a copy of `clear`, made with `.copy()`, with weathersit 3, light rain, in every hour.

        Add a line with `assert` under it that fails unless the model expects fewer bikes with rain than in the same hours with clear weather. `expect(rain)` and `expect(clear)` each give a tensor of 5,645 numbers, and `.mean()` gives the mean of a tensor.

        The check runs your test on the course's model and on a year with 50% more riders, where it must pass, and on the weather the wrong way round, where it must fail. When it is right, the check saves your test as `work/projects/ops/test_rain.py`.
        """),
        L.task("Add a line with `assert` that fails unless rain makes the model expect fewer bikes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rain():
    box_rain = L.code_box("rain", lines=10)
    box_rain
    return (box_rain,)


@app.cell(hide_code=True)
def check_rain(box_rain):
    result_rain = L.check("rain", box_rain.value)
    result_rain
    return (result_rain,)


@app.cell(hide_code=True)
def hints_rain():
    L.hints(
        "rain",
        [
            "Compare one number of `expect(rain)` with one number of the other tensor: what the model expects on average, with `.mean()`.",
            "Comparing the two tensors hour by hour would ask every one of the 5,645 hours to drop, and at night the model expects about the same with rain or without. The mean of a tensor is one number, and `assert` needs one True or False.",
            "`    assert expect(rain).mean() < ___`",
        ],
        why="`assert expect(rain).mean() < expect(clear).mean()`: for the course's model 121.6 against 157.4 bikes per hour, 22.8% fewer; for the weather the wrong way round 137.7 against 109.2.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your tests on five versions

    Your three tests, as the checks saved them in your project, now run on all five versions: the course's model and the four versions with a bug. Where a test of yours is not in your project yet, the course's own test runs in its place. Go through the versions with the slider: the picture shows what each one expects on 14 July, as it was, with rain in every hour, and a year later in the table, and under it how your tests did.
    """)
    return


@app.cell(hide_code=True)
def demo_versions(result_quiet_night, result_rain, result_row_number):
    (result_quiet_night, result_row_number, result_rain)   # run again once a check saves a test
    _r = C.demo()
    _out = mo.vstack([
        L.explore("versions", "version", lambda _v: C.version_frame(_r, _v, L.lang), start=1, low=1, high=5, step=1,
                  role="",
                  task="Go through the five versions. Which one moves the busy morning into the night, which one expects thousands of bikes in every hour, and for which one does the purple line part from the solid line?"),
        mo.Html(C.demo_key(L.lang)),
        mo.Html(C.matrix_picture(_r, L.lang)),
        mo.md(C.verdicts(_r, L.lang)),
    ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Each version with a bug fails one of the tests. The hour in UTC moves the busy morning into the night: at 3 at night on 14 July it expects 425.6 bikes. The temperatures in degrees put almost every hour in the thousands, and the short line of the quiet night lies at the bottom of the picture. The weather the wrong way round puts the dashed line above the solid one. The version with the row number is the only one whose purple line parts from the solid line, and only the test of the row number catches it. No single test catches all four bugs; a test of each kind does.

        ### Where the course's model fails

        The checks need tests that pass for the course's model. CheckList also asks another question: for how many cases does a test fail? The picture runs five tests on the course's model, case by case.
        """),
        mo.Html(C.report_picture(L.lang)),
        mo.md(r"""
        The quiet night and the row number hold in every case. Fewer than 0 bikes in 235 of the 8,645 hours is the flaw the model assertion found. Rain, tested hour by hour instead of on average, fails in 272 of the 5,645 clear hours, 261 of them before 7 in the morning, when the model expects almost nothing either way. And one invariance test fails in 333 of the 2,495 weekend hours: whether a weekend hour is a Saturday or a Sunday changes what the model expects by more than 20 bikes there, up to 38.0, while in 2011 the two days had about as many bikes per hour, 142.2 and 143.8. Exercise 4 asks why.
        """),
        L.watch_out("Passing all your behavioral tests does not make a model right: the tests check only what you thought of. CheckList's advice is to go through the abilities one by one, such as the time of day, the weather and the day of the week, and to try each kind of test on each of them."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_weekend():
    mo.md(r"""
    ### Exercise 4. Saturday or Sunday

    The invariance test of the weekend asks that, for each of the 2,495 weekend hours of 2011, the course's model expects almost the same with Saturday as with Sunday, within 20 bikes. In 333 hours the two answers are further apart, up to 38.0, most of them between 8 in the morning and 4 in the afternoon, and late in the evening. The inputs of the model are those of [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en): the hour as 24 one-hot columns, and every feature as one standardized number, the day of the week too.
    """)
    return


@app.cell(hide_code=True)
def choice_weekend_cell():
    choice_weekend = L.choice(
        "weekend",
        question="Why does the course's model treat a Saturday and a Sunday this differently?",
        options={
            "floats": "Two predictions are never exactly equal, since floats differ in their last digits, so a test with a margin fails too",
            "right": "The day of the week is one number, from 0 for Sunday to 6 for Saturday, so for the model the two lie far apart",
            "data": "In 2011 the Saturdays had clearly more bikes than the Sundays, and the model learned that difference well",
            "small": "A network of 577 numbers is too small to learn anything about weekends, so its answers there are guesses",
        },
        correct="right",
        why={
            "right": "The column weekday goes into the network as one standardized number: −1.50 for Sunday and 1.49 for Saturday, the two ends of the week. Every neuron sees the day through one weight on that number, so for the model Sunday lies next to Monday and furthest from Saturday. With the day as 7 one-hot columns, as the hour has 24, the model could learn each day on its own. The validation RMSE of 53.04 does not show this; a behavioral test does, and the next step is to write it down for whoever uses the model.",
            "floats": "Floats differ in their last digits, by millionths of a bike, and the test allows 20 bikes: what in the inputs makes Saturday and Sunday so different for the model?",
            "data": "In 2011 the two days had about as many bikes per hour, 142.2 on Saturdays and 143.8 on Sundays: so where does the difference the model makes come from?",
            "small": "The same network gets the busy mornings of the working days and the quiet nights right: what in its inputs puts Saturday and Sunday so far apart?",
        },
    )
    choice_weekend
    return (choice_weekend,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Tests for ML code](/course/open/sys-code-tests?lang=en) your tests looked at the code; here they looked at the model, from the outside, as CheckList does: a minimum it must get right, an answer that must not change, and an answer that must move the right way. Your project *ops* now has three more tests, `test_night.py`, `test_row_number.py` and `test_rain.py`, and `conftest.py` gives them your packed model. The failures that remain, such as fewer than 0 bikes at night and Saturday against Sunday, are what a model card writes down: *A model card for your model* shows how, and in *Project: tests and a model card* your demand model gets both. In *Project: your model as a service, watched for drift* it answers requests, and the hours of 2012, with many more riders than 2011, come back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weekend, result_quiet_night, result_rain, result_row_number):
    L.wrap_up(result_quiet_night, result_row_number, result_rain, choice_weekend)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
