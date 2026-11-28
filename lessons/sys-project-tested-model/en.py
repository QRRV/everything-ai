# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-project-tested-model. Route: course/route/lessen/systemen.yaml. Checks, the course's model and a version without weekday, the demo that runs pytest, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("sys-project-tested-model", "en")
    C = L.checks   # the course's model, the checks of your parts, the run of every test, the card and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en) and [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Which of these is a directional expectation test of your demand model?",
        options={
            "right": "With rain instead of clear weather in the same hours, the model must then expect fewer bikes",
            "night": "From 1 to 4 at night on a working day, the model must expect fewer than 50 bikes in every hour",
            "row": "Moving the row number of every hour on by a year must not change what the model expects",
            "unit": "`inputs` must put hour 8 of a row in column 8 of its one-hot part, and in no other column",
        },
        correct="right",
        why={
            "right": "A directional expectation test changes something that matters, the weather, and asks the answer to move the known way: with rain, fewer bikes. Your `test_rain` does that on the mean over 5,645 clear hours.",
            "night": "Few bikes at night is a simple case with a known answer, which every useful model must get right: which kind of test asks for an answer to move the known way? See [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en).",
            "row": "A change that must not matter, with the same answer asked for, is an invariance test: which kind changes something that matters, and asks for the answer to move? See [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en).",
            "unit": "That tests one function from the inside, with one row whose answer you know: a unit test of the code, not a test of the model from the outside. Which kind asks the answer of the model to move the known way? See [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="The digit reader gets 94.5% of 2,000 test digits right. What should its model card give beside that one number?",
        options={
            "decimals": "The same 94.5% with two more decimals, so that two versions can be told apart",
            "history": "The 94.5% of every earlier version, so that a reader sees how the model got better",
            "right": "The share right per group and per condition, each with the number of digits behind it",
            "train": "The share right on the training digits as well, so that a reader sees it did not overfit",
        },
        correct="right",
        why={
            "right": "A mean over 2,000 digits can hide a group: the students' digits went wrong almost twice as often as the employees'. With the number of digits behind each share, a reader also sees which numbers are too small to say much.",
            "decimals": "More decimals make the one number more exact, but it still hides who the model gets wrong: what does a card add so that a reader sees that? See [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en).",
            "history": "Earlier versions say how the model got here, not for whom it works now: what does a card add to one number over all digits? See [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en).",
            "train": "The training digits say how well the model learned them, not how it does for each group of new digits: what does a card add to one number over all test digits? See [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en).",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your demand model is packed, and it has five tests from two lessons: two of its code and three of its behavior. A team does not run its tests one at a time. It runs all of them with one command, before every new version, and wants one answer: green when every test did what it should, red with the name of the test that did not. Some failures are known and cannot be fixed today, such as Saturday against Sunday from [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en), and they must not turn every run red.

        Whoever uses the model needs what the numbers and the tests found, in words: what the model is for and what not, how far off it is overall and for which hours, and its known limits. That is a model card, as in [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en). Written by hand, a card is out of date as soon as the next version is packed. So in this step of your project, code writes the card from the package and its numbers, and one command runs every test.

        **An everyday example.** A car gets its yearly inspection: one visit checks the brakes, the lights and the tires, all on one list. A known flaw that may stay for now, such as a dent, goes on the report as a note, not as a failure. And the report goes with the car to its next owner.
        """),
        draw.project_steps(["A demand model, tracked and packed", "Tests and a model card", "Your model as a service, watched for drift"], 1,
                           here_label="← today", title="Where your project stands: ops"),
    ])
    return


@app.cell(hide_code=True)
def predict_days():
    guess_days = L.guess(
        "days",
        question="Over all 1,733 hours of the validation days, your model is 53.04 bikes per hour off. 1,119 of those hours are on working days, and 614 on weekends and holidays, with about as many bikes per hour. Split the RMSE into those two groups. What comes out, do you think?",
        options={
            "same": "About 53 for both groups, since the model gets the kind of day as one of its inputs",
            "work": "Clearly higher on working days, since their rush hours have the most bikes of all",
            "right": "Clearly higher on weekends and holidays, by more than 10 bikes per hour",
            "count": "Higher on working days, since they have almost twice as many hours to get wrong",
        },
        reveal="**Higher on weekends and holidays.** 60.9 bikes per hour on their 614 hours, against 48.2 on the 1,119 hours of the working days, though both have about as many bikes per hour, 144.9 and 142.0. Furthest off are the 5 holidays among the validation days: 78.7 on their 119 hours. The training days held only 5 holidays too, so the model learned holidays from 5 days. The one RMSE of 53.04 hides all of this. Exercise 1 measures it per group.",
    )
    guess_days
    return (guess_days,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Every test, one command.** pytest looks in a folder for files whose names begin with `test_`, and in them for functions whose names begin with `test`. It reads `conftest.py` first, for fixtures such as `expect`, runs every test it found, and ends with one line that counts them, such as `5 passed in 1.35s`. In your project, pytest on the folder `ops` finds five tests in five files: `test_one_row` and `test_overfit_one_batch` from [Tests for ML code](/course/open/sys-code-tests?lang=en), and `test_quiet_night`, `test_row_number` and `test_rain` from [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en). All the tests of a project together are its suite. The suite is green when every test does what it should, and red as soon as one does not.

        **A known failure.** The invariance test of the weekend fails for your model, in 333 of the 2,495 weekend hours, since the day of the week goes in as one number. Fixing that takes a new version with other inputs. Until then, what do you do with the test? Delete it, and nothing tells you when a version gets it right, or wrong in a new way. Leave it failing, and every run is red, so a new failure looks just like the old one. pytest has a mark for this: `@pytest.mark.xfail(reason="...")` on the line above `def`, as `@pytest.fixture` stands above `expect` in `conftest.py`. It says that the test is expected to fail, and why. pytest still runs the test. When it fails, pytest reports **XFAIL**, an `x` instead of a dot, and the suite stays green: `5 passed, 1 xfailed`.

        **Strict.** One day a test marked xfail may pass: a new version no longer has the limit. pytest then reports **XPASS**, and on its own that keeps the suite green, so nobody notices, and the card keeps a limit that is gone. With `strict=True` in the mark, pytest counts XPASS as a failure, `[XPASS(strict)]`, and the suite turns red: time to remove the mark, and the line on the card.

        **Numbers per group.** In [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en), the card gave the accuracy of the digit reader per group and per condition, with the number of digits behind each share. For your model the metric is the RMSE, and the groups are hours: per kind of day, working day or not, and per weather. The picture splits the validation RMSE of the course's model that way.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The dashed line is the RMSE of all 1,733 hours together. Weekends and holidays, clear weather, and light rain lie above it; working days and mist lie below it. The hours under each bar say how much each number can be trusted: 182 hours of light rain say less than 1,119 hours of working days.

        **A card made by code.** A card written by hand is right for one version. Your project packs a new version whenever the model is trained again, with other numbers, and a card that nobody rewrites then gives the numbers of an older model. So code writes the card, from `config.json` of the package, the RMSE per group and the known limits, each time the tests run. It goes beside the package as `README.md`: on the Hugging Face Hub, the file `README.md` of a model is its model card. The card is in Markdown, plain text with a few signs: `#` before a heading, `|` between the cells of a table, and `-` before each item of a list. And it is in English, in both languages of the course, as most model cards are: whoever uses the model may read no Dutch.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **suite** | sweet | all the tests of a project together, run with one command | `pytest ops` |
        | **xfail** | ex-fail | a mark that says a test is expected to fail: pytest still runs it, and reports XFAIL instead of a failure | `@pytest.mark.xfail(reason="...")` |
        | **XPASS** | ex-pass | what pytest reports when a test marked xfail passes | |
        | `strict=True` | strict | in an xfail mark: a test that passes counts as a failure, so the suite turns red when a known limit is gone | `@pytest.mark.xfail(..., strict=True)` |
        | **README.md** | read me | the text file that says what a folder holds; on the Hugging Face Hub, the model card of a model | `ops/model/README.md` |
        | **Markdown** | | plain text with a few signs for headings, tables and lists, which a site shows as a page with layout | `## Known limits` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Tests and a card for a model in a project:

    1. Keep every test of the model in one folder, in files whose names begin with `test_`, and run them all with one command.
    2. Measure the metric per group and per condition, with the number of cases behind each number.
    3. For a test the model fails, and that you will not fix now, add an xfail mark with a reason and `strict=True`, and write the same limit on the card.
    4. Let code write the card, from the package and its numbers, so that it changes with each version.
    5. When the strict mark turns into a failure, the limit is gone: remove the mark, and the line on the card.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four hours: two of a working day and two of a Saturday. The model expects 30, 40, 220 and 110 bikes; 10, 40, 250 and 100 were rented. The errors are 20, 0, −30 and 10.

    For the working day, the squares are 400 and 0, with a mean of 200, and the root of 200 is 14.1 bikes per hour. For the Saturday, the squares are 900 and 100, with a mean of 500, and the RMSE is 22.4. For all four hours together, the mean of the four squares is 1,400 / 4 = 350, and the RMSE is 18.7: not 18.25, the mean of 14.1 and 22.4, since the root comes last: the mean of the four squares is the mean of 200 and 500, and the root of that mean is not the mean of the two roots. In the card, the working day becomes the row `| working day | 2 | 14.1 |`: the group, its hours, and its RMSE to one decimal.
    """)
    return


@app.cell(hide_code=True)
def text_val():
    mo.md(r"""
    ### The validation days

    The cell reads the hours of 2011 with `read_hours` from your project and keeps the validation days, every fifth day, as in [Project: a demand model, tracked and packed](/course/open/sys-project-tracked-model?lang=en). `C.expect` is the course's packed model of your project: it gives what the model expects for each hour. `targets` gives the bikes really rented, as a column, and `.reshape(-1)` makes them one row of numbers, like `pred`. `sorted(set(...))` gives each value of a column once, in order. Run the cell.
    """)
    return


@app.cell
def example_val():
    from ops.demand import read_hours, targets

    hours = read_hours(2011)
    val = hours[hours["day"] % 5 == 4]   # every fifth day: the 73 validation days
    pred = C.expect(val)                 # what the course's model expects in each hour
    y = targets(val).reshape(-1)         # the bikes really rented, one row like pred
    print("hours:", len(y))
    print("kinds of day:", sorted(set(val["workingday"])))
    print("weather:", sorted(set(val["weathersit"])))
    print("first 6 hours, model:", pred[:6].round())
    print("first 6 hours, real: ", y[:6])
    return


@app.cell(hide_code=True)
def text_exercise_per_group():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The RMSE per group

        `rmse_per(hours, pred, column)` measures how far off the model is for each value of one column, such as `workingday`. `hours` is a table of hours, `pred` what the model expects for each of them, and `column` the name of the column. `y` holds the bikes really rented, from `targets`, as one row with `.reshape(-1)`. `table` starts empty. The loop goes over each value of the column once, in order, with `sorted(set(hours[column]))`. `mask` is True for the hours with that value: `.to_numpy()` takes the column out of pandas, and `torch.tensor` makes it a tensor. `error` holds the errors of those hours, what the model expects minus the bikes rented. Each value gets an entry in `table`, under `int(value)`: a dict with the number of hours, `len(error)`, under the key `"hours"`, and the RMSE under the key `"rmse"`, the names the card reads. The function ends with `return table`. `import torch` and the import of `targets` from `ops.demand` are there for the function.

        Replace the `...` with that dict, with the RMSE of `error`, the root of the mean of the squared errors, as a Python number.

        The check calls `rmse_per(val, pred, "workingday")`, with the validation days of the cell above and the course's model, then other columns and a small table of its own. Under the check, your function splits the validation days by kind of day and by weather. When it is right, the check saves your function as `work/projects/ops/report.py`.
        """),
        L.task("Replace the `...` with the dict of the group: its hours, and the RMSE of `error` as a Python number."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_group():
    box_per_group = L.code_box("per-group", lines=13)
    box_per_group
    return (box_per_group,)


@app.cell(hide_code=True)
def check_per_group(box_per_group):
    result_per_group = L.check("per-group", box_per_group.value)
    result_per_group
    return (result_per_group,)


@app.cell(hide_code=True)
def hints_per_group():
    L.hints(
        "per-group",
        [
            "After `table[int(value)] =` comes a dict with two keys: `\"hours\"` with `len(error)`, and `\"rmse\"` with the RMSE of the group. That RMSE needs only `error`, the errors of the group, and it has to come out as a plain Python number, not as a tensor.",
            "The RMSE in three steps, each an operator or a method of a tensor: `** 2` squares every error, `.mean()` takes the mean of the squares, and `.sqrt()` the square root of that mean. What comes out is still a tensor with one number in it; `.item()` gives that number as a plain Python number.",
            "`        table[int(value)] = {\"hours\": len(error), \"rmse\": ___}`",
        ],
        why="`\"rmse\": (error ** 2).mean().sqrt().item()`: the root of the mean squared error of the hours under `mask`, as a Python number. For the validation days: 48.2 bikes per hour on working days and 60.9 on the other days; 54.1 when clear, 48.0 with mist and 60.1 in light rain.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_weekend():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A known limit, marked

        `test_weekend(expect)` is the invariance test of the weekend from [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en). It reads `HOURS` with `read_hours` from `ops.demand`. `days_off` keeps the hours of the days that are not working days, and `weekend` those of them that are not holidays: 2,495 hours. `saturday` is a copy of `weekend`, made with `.copy()`, with weekday 6 in every hour, and `sunday` a copy with weekday 0. The `assert` asks that `expect` gives answers within 20 bikes of each other for the two copies, in every hour: `.abs()` makes each difference positive, and `.all()` asks it of every hour. `import pytest` is there for the mark.

        The course's model fails this test, in 333 hours. Add one line, right above `def test_weekend(expect):`, that marks the test as a known limit: expected to fail, with a reason, and strict.

        The check reads the mark on your test as pytest reads it, and runs your test with two models: the course's model, `h16-lr0.01`, and a version trained the same way without the column weekday, for which Saturday and Sunday are the same. Under the check you see what pytest would report for each. When it is right, the check saves your test as `work/projects/ops/test_weekend.py`.
        """),
        L.task("Add the line that marks `test_weekend` as expected to fail, with a reason, and strict."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weekend():
    box_weekend = L.code_box("weekend", lines=15)
    box_weekend
    return (box_weekend,)


@app.cell(hide_code=True)
def check_weekend(box_weekend):
    result_weekend = L.check("weekend", box_weekend.value)
    result_weekend
    return (result_weekend,)


@app.cell(hide_code=True)
def hints_weekend():
    L.hints(
        "weekend",
        [
            "The mark goes on the line right above `def test_weekend(expect):`, beginning with `@`, as `@pytest.fixture` does in `conftest.py`. `@pytest.mark.xfail` says the test is expected to fail; `reason` says why, and `strict=True` makes a pass count as a failure.",
            "The reason is a short text in quotes: what in the inputs of the model puts Saturday and Sunday far apart? Give it by name, `reason=`, since pytest reads a first value without a name as something else.",
            "`@pytest.mark.xfail(reason=___, strict=True)`",
        ],
        why="`@pytest.mark.xfail(reason=\"weekday is one number, so Saturday and Sunday lie far apart\", strict=True)`, right above `def test_weekend(expect):`. For the course's model pytest reports XFAIL and the suite stays green; for the version without weekday it reports `[XPASS(strict)]`, a failure that says the limit is gone.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_card():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The card

        `card(config, groups, limits)` writes the card as one text in Markdown. `config` is the dict of `config.json` from the package. `groups` is a dict with, for each group, its hours and its RMSE, such as `{"working day": {"hours": 1119, "rmse": 48.18}}`. `limits` is a list of sentences. `lines` begins with the heading, with the version from `config['version']`, then two sentences on what the model is for and what not, with the run from `config['run']`, a heading with the validation RMSE from `config['val_rmse']`, rounded to two decimals with `round`, and the first two lines of a table: the header `| Group | Hours | RMSE |`, and `|---|---|---|`, which Markdown needs under a header. The loop goes over `groups.items()`: each `name` with its `row`. After the loop come a heading `## Known limits` and a line with `- ` before each `limit`. `"\n".join(lines)` joins the lines into one text, with `\n`, the sign for a new line, between them.

        Replace the `...` with a line that adds one row of the table to `lines`: the name of the group, its hours, and its RMSE rounded to one decimal, each between `|` signs.

        The check calls your `card` with the course's package, its RMSE per group from exercise 1 and its known limits, and once more with other groups and no limits. Under the check you see your card as a site shows Markdown. When it is right, the check saves your function as `work/projects/ops/card.py`.
        """),
        L.task("Replace the `...` with a line that adds one row of the table to `lines`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_card():
    box_card = L.code_box("card", lines=13)
    box_card
    return (box_card,)


@app.cell(hide_code=True)
def check_card(box_card):
    result_card = L.check("card", box_card.value)
    result_card
    return (result_card,)


@app.cell(hide_code=True)
def hints_card():
    L.hints(
        "card",
        [
            "`lines.append` adds one line to the list. The row is an f-string with `name` and `row['hours']` in its cells, like the header `| Group | Hours | RMSE |`, and the RMSE from `row` in the last cell.",
            "The RMSE needs rounding to one decimal: `round` with two values, the number and how many decimals. In an f-string, the whole call goes between `{` and `}`.",
            "`        lines.append(f\"| {name} | {row['hours']} | ___ |\")`",
        ],
        why="`lines.append(f\"| {name} | {row['hours']} | {round(row['rmse'], 1)} |\")`, indented in the loop: one row per group, such as `| working day | 1119 | 48.2 |`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_heavy_rain():
    mo.md(r"""
    ### Exercise 4. Heavy rain

    The table of your card has a row for each weather of the validation days: clear, mist, and light rain or snow. Heavy rain, weathersit 4, has no row. In all of 2011, one hour had heavy rain: 4 in the afternoon on Wednesday 26 January, a training day, with 36 bikes.
    """)
    return


@app.cell(hide_code=True)
def choice_heavy_rain_cell():
    choice_heavy_rain = L.choice(
        "heavy-rain",
        question="What should your card say about heavy rain?",
        options={
            "nothing": "Nothing at all: a weather without hours is not a limit of the model, so the card has nothing to report",
            "train": "The RMSE of the one hour with heavy rain on a training day, since one number is better than none",
            "right": "That heavy rain was not tested, since no validation hour shows how far off the model is then",
            "light": "The RMSE of light rain for heavy rain too, since heavy rain is the same kind of weather, only more of it",
        },
        correct="right",
        why={
            "right": "No validation hour had heavy rain, so no number says how far off the model is then, and it learned heavy rain from one hour. A card that leaves it out looks as if nothing goes wrong: a group nobody measured does not show up at all, as [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en) warned. So the card says what was not tested, such as a line among the known limits, and the run below adds one.",
            "nothing": "Without hours, the card cannot give a number, but a reader who sees no row for heavy rain may think the model does fine then: what does a card say about a group nobody measured?",
            "train": "The model learned from that hour, so its error there says how well it learned that one hour, and one hour is far too few for a number: what is left for the card to say?",
            "light": "Heavy rain keeps more riders at home than light rain, and the model saw it once: does the number of light rain say anything about heavy rain, or is it a guess?",
        },
    )
    choice_heavy_rain
    return (choice_heavy_rain,)


@app.cell(hide_code=True)
def text_demo(result_card, result_per_group, result_weekend):
    result_per_group, result_weekend, result_card  # made again after the checks, which save your parts
    mo.md(r"""
    ### Every test, and the card

    The button runs every test of your project at once, with pytest. It first copies your project to `work/check/sys-project-tested-model/run`, and puts the course's own version in the copy of any file your project does not have yet, so the run also works when you skipped a lesson; where your project has no package yet, the copy gets the course's. pytest runs in a Python process of its own, as in a terminal, and that takes a few seconds. Then your `rmse_per` and your `card` write the card of your package, and the course saves it beside the package as `work/projects/ops/model/README.md`. In `limits`, your `card` gets the known limits of your model, and a line for each group that no validation hour falls in, such as the heavy rain of Exercise 4.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Run every test and write the card")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_card, result_per_group, result_weekend, run_button):
    result_per_group, result_weekend, result_card   # run again with your files once a check saves them
    if not run_button.value:
        _out = mo.md("*Press the button to run every test of your project.*")
    else:
        mo.output.replace(mo.md("**pytest is running** …"))
        _r = C.demo()
        if _r.get("error"):
            _out = mo.md(C.summary(_r, L.lang))
        else:
            _out = mo.vstack([
                mo.Html(C.suite_picture(_r, L.lang)),
                mo.Html(C.card_view(_r, L.lang)),
                mo.md(C.summary(_r, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        With the course's files, pytest runs six tests: five pass, and `test_weekend` fails as its mark expects, so the suite is green, `5 passed, 1 xfailed`. The tests themselves take about a second, most of it `test_overfit_one_batch`, which trains a network; starting Python and PyTorch takes the rest. The card has the version of the package, `cb4f177c9137` for the course's, its validation RMSE of 53.04, five rows from `rmse_per`, and four known limits: fewer than 0 bikes at night, Saturday against Sunday, 2012, and heavy rain, not tested.

        The third limit comes from hours the model never saw: the 8,734 hours of 2012. With many more riders than in 2011, the model is 130.7 bikes per hour off there, more than twice its 53.04. The validation days could not show that: they are days of 2011, among the training days. *Project: your model as a service, watched for drift* comes back to 2012.

        In a terminal, in the folder of the course, `uv run pytest work/projects/ops` runs the same tests on your project itself, and with `-rx` added, pytest prints the reason beside each XFAIL.
        """),
        L.watch_out("An xfail mark counts any error as the expected failure, also one that has nothing to do with the limit, such as a typo in the name of a column in the test. Then the suite stays green while the test tests nothing. `raises=AssertionError` in the mark narrows it: only an `assert` that fails counts as expected, and any other error fails the test."),
    ])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Tests for ML code](/course/open/sys-code-tests?lang=en) and [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en) your tests looked at the code and at the model, one at a time; now one command runs all of them, and a known limit is marked, not hidden. The card of [For whom does it work? Model and system cards](/course/open/sys-model-card?lang=en) is now written by code, from the package and its numbers, and lies beside the package. Your project *ops* has three more files: `report.py`, `test_weekend.py` and `card.py`. In *Project: your model as a service, watched for drift*, your model answers requests, and the hours of 2012 come back, as data that has moved away from what the model learned.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_heavy_rain, result_card, result_per_group, result_weekend):
    L.wrap_up(result_per_group, result_weekend, result_card, choice_heavy_rain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
