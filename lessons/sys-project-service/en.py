# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-project-service. Route: course/route/lessen/systemen.yaml. Checks, the course's service, the log of 2012, the demo that runs your service, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("sys-project-service", "en")
    C = L.checks   # the course's service, the tables of the checks, the server of the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Before and after the model: the serving path](/course/open/sys-serving-path?lang=en) and [When the world changes: drift](/course/open/sys-drift?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_skew():
    quick_skew = L.choice(
        "quick-skew",
        question="A service scales the inputs of each request with the mean and the standard deviation of that day, while the model learned with those of its training hours. What is that called?",
        options={
            "drift": "Data drift: the world sends the service other inputs than the ones the model learned from",
            "right": "Training-serving skew: the inputs are made one way in training and another way in serving",
            "rule": "A business rule: a choice of the company that sits after the model, such as when to call the crew",
            "overfit": "Overfitting: the model learned its training hours by heart, and goes wrong on new ones",
        },
        correct="right",
        why={
            "right": "The model learned from inputs scaled with the mean and standard deviation of training. A service that scales with other numbers gives it other inputs for the same hour, so its answers go off: with the scale of each day's batch, 73 bikes per hour off on average instead of 34.",
            "drift": "Drift is a change in the world: the hours would come in with other weather or other riders. Here the world stays the same, and the service makes the inputs in another way: what is that called? See [Before and after the model: the serving path](/course/open/sys-serving-path?lang=en).",
            "rule": "A business rule sits after the model and turns a number into a decision, such as calling the crew from 400 bikes on. This happens before the model: what does the course call inputs made one way in training and another way in serving? See [Before and after the model: the serving path](/course/open/sys-serving-path?lang=en).",
            "overfit": "Overfitting is about what the model learned. Here the model stays the same, and its inputs are made in another way than in training: what is that called? See [Before and after the model: the serving path](/course/open/sys-serving-path?lang=en).",
        },
    )
    quick_skew
    return (quick_skew,)


@app.cell(hide_code=True)
def quick_check_psi():
    quick_psi = L.choice(
        "quick-psi",
        question="The PSI of the temperature in March 2012, against March in the training hours of 2011, is 1.15. What does that say?",
        options={
            "error": "The model is 115% further off in March 2012 than it was on its validation days of 2011",
            "share": "1.15% of the hours of March 2012 had a temperature that never came up in training",
            "right": "The temperatures of March 2012 lie in other bins than in training, far beyond 0.25",
            "year": "Nothing yet: a PSI only says something once it is measured over a whole year of data",
        },
        correct="right",
        why={
            "right": "Above 0.25, a distribution has moved a lot. March 2012 was far warmer than March 2011: half of its hours lay in the top bin of March in training, where about a tenth belongs.",
            "error": "The PSI compares the shares of the values in the bins of training: it holds no answer of the model and no rental. What does it say about the temperatures themselves? See [When the world changes: drift](/course/open/sys-drift?lang=en).",
            "share": "The PSI is not a share of the hours: it adds up, over ten bins, how far the share now lies from the share in training. What does 1.15, far above 0.25, say about the temperatures? See [When the world changes: drift](/course/open/sys-drift?lang=en).",
            "year": "The PSI works on any two sets of values, such as one month against the same month of training, as the lesson did with March. What does 1.15 say about the temperatures? See [When the world changes: drift](/course/open/sys-drift?lang=en).",
        },
    )
    quick_psi
    return (quick_psi,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your demand model is packed and tested. Now it has to work: a service answers every request, at any hour, for months. Two things can then go wrong without anyone noticing. The service can get slow, and only a record of every request shows when, and how slow. And the world can move away from the hours the model learned from: 2012 had many more riders than 2011. A team wants to know when the model needs training again, and it wants to know from numbers, not from a complaint.

        So in this step of your project, your service writes one line per request, with how long the answer took, and two checks read those lines: a drift check of what came in, month by month, and, once the real rentals are in, how far off the model was. A rule on those numbers says when to train again.

        **An everyday example.** A bakery bakes each morning what its forecast says, and writes down what it baked and what it sold. Whether the weather changed, the baker sees in the morning. Whether the forecast was any good shows only in the evening, when the sales of the day are in.
        """),
        draw.project_steps(["A demand model, tracked and packed", "Tests and a model card", "Your model as a service, watched for drift"], 2,
                           here_label="← today", title="Where your project stands: ops"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_months():
    guess_months = L.guess(
        "months",
        question="Your drift check compares each month of 2012 with the same month of 2011. The temperatures of January 2012 lie far from those of January 2011, a PSI of 0.87; those of July 2012 lie close to July 2011, a PSI of 0.06. In which of the two months is your model further off, do you think?",
        options={
            "january": "January, by far, since its weather lies furthest from what the model learned",
            "same": "About the same in both, since the model gets the weather as one of its inputs",
            "right": "July, clearly, although its weather lies close to that of July 2011",
            "neither": "Neither: both stay close to the 53.04 of its validation days",
        },
        reveal="**July.** An RMSE of 119.4 bikes per hour in July 2012, against 77.6 in January: 2.25 times the 53.04 of the validation days, against 1.46. The weather of July 2012 looked like that of July 2011, but 274 bikes were rented per hour instead of 190, and the model, which sees the hour and the weather, kept answering as in 2011. A drift in the inputs says that the weather moved, not how far off the model is. Exercise 3 measures that, with the real rentals.",
    )
    guess_months
    return (guess_months,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One line per request.** `time.perf_counter()` gives the time on a clock that counts in seconds, from a moment of its own: one reading alone says nothing, but the difference between two readings is the time that went by, in seconds. Read it right before the answer and right after it, and you have the latency of the answer; a log gives it in milliseconds, thousandths of a second. The service writes it in one line of JSON per request, with the answer and the request itself, as the structured logging of [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en) does: `{"ms": 0.4, "bikes": 431, "request": {...}}`. A file with one JSON object on each line is called JSON Lines, often with the extension `.jsonl`, as `runs.jsonl` of your project already is. Each new line goes at the end of the file, so nothing written is lost, and a program can read the file line by line, even while the service writes on.

        **Drift, month by month.** The log holds every request, so it holds the inputs the model really got. Your drift check reads them per month and compares each month with the same month of the training hours, with the PSI of [When the world changes: drift](/course/open/sys-drift?lang=en): a month against a whole year would mostly measure the season. The course has put that PSI in your project, as `ops/psi.py`: `psi_of(then, now)` makes ten bins from the values `then` and gives the PSI of the values `now`. The check runs on three inputs, the temperature, the humidity and the wind, and on the answers of the model, the column `bikes`, as soon as a month is in the log.

        **The real rentals come later.** How far off an answer was shows only when the hour is over and its bikes are counted, and a team often gets those counts in a batch, a day or a month later. Once the rentals of a month are in, your `rmse_per` from [Project: tests and a model card](/course/open/sys-project-tested-model?lang=en) measures the RMSE of that month. Against what? The package has its own number: `val_rmse` in `config.json`, its RMSE on the validation days. On new hours, a model is always a little further off, so a rule leaves some room: train again when a month is more than `LIMIT` times as far off, here 1.5 times.

        **A warning, and a decision.** The drift check is there at once, but it says that the data moved, not that the model is wrong. The error is what counts, but it comes in late. A service watches both: drift as an early warning, and the error of each month as the decision to train again.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The top row is one request: `serve` times the model and gives the answer back. Every request leaves one line in the log. From the log, the drift check compares each month with the same month of training at once; the error of a month waits for its rentals.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **latency** | | how long a service takes for one answer | `ms` |
        | **millisecond** (ms) | | a thousandth of a second | |
        | `time.perf_counter()` | perf counter | a clock in seconds for measuring how long something takes: only the difference between two readings means something | `start = time.perf_counter()` |
        | **JSON Lines** | jason lines | a file with one JSON object on each line, often with the extension `.jsonl` | `requests.jsonl` |
        | `LIMIT` | | in your project: how many times as far off as on its validation days a month may be before the model trains again | `LIMIT = 1.5` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A model as a service, watched for drift:

    1. Read the clock right before and right after every answer, and write one line of JSON per request, with the latency, the answer and the request.
    2. Read the log per month, and compare each month with the same month of the training data: the PSI of each input and of the answers.
    3. Once the real outcomes of a month are in, measure its RMSE, and hold it against the RMSE of the package on its validation days.
    4. Train again when a month is more than `LIMIT` times as far off; read drift of the inputs as a warning, not as the decision.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A request comes in. The clock reads 5,031.2140 seconds right before the answer and 5,031.2146 seconds right after it: 0.0006 seconds went by, which is 0.6 milliseconds. The log line says `"ms": 0.6`.

    A package with a validation RMSE of 40.0 bikes per hour, and a `LIMIT` of 1.5: the limit is 1.5 × 40.0 = 60.0. Its first month is 52.3 bikes per hour off, its second 72.4. The first stays under 60.0: no training yet, though it is 1.31 times as far off as on the validation days. The second is above it, at 1.81 times: the rule names the second month.
    """)
    return


@app.cell(hide_code=True)
def text_request():
    mo.md(r"""
    ### A request, and the clock

    `C.REQUEST` is the body of a request for 8:00 on Tuesday 10 April 2012: the hour and the 10 features, as in [Your model behind an API](/course/open/sys-serving-api?lang=en). That hour, 692 bikes were rented. `C.answer` is the course's service: it gives the answer of the course's packed model, in whole bikes, never fewer than 0. Around the answer, the cell reads `time.perf_counter()` twice, as `first` and `second`. Run the cell, and look at the answer, next to the 692 bikes of that hour, and at the two readings.
    """)
    return


@app.cell
def example_request():
    import time

    request = C.REQUEST                  # Tuesday 10 April 2012 at 8:00, as the body of a request
    print(request)
    first = time.perf_counter()          # a reading of the clock, in seconds
    body = C.answer(request)             # what the course's model answers
    second = time.perf_counter()         # a second reading, right after the answer
    print(body)
    print(first)
    print(second)
    return


@app.cell(hide_code=True)
def text_exercise_serve():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One line per request

        `serve(answer, request, log)` answers one request and writes one line to the log. `answer` is the model of the service: it takes a request and gives back a dict such as `{"bikes": 431}`. `request` is the body of the request, a dict with `hr` and the 10 features. `log` is the open log file: `log.write` adds text at its end. `start` reads the clock with `time.perf_counter()` right before the answer, `body` holds the answer, and `ms` how long it took. `line` is the line for the log: a dict with `"ms"`, rounded to three decimals with `round`, the bikes of the answer, and the request. `json.dumps` writes it as one line of text, and `"\n"` ends the line. The function ends with `return body`, the answer the server sends back. `import json` and `import time` are there for the function.

        Replace the `...` with how long the answer took, in milliseconds.

        The check calls `serve(C.answer, C.REQUEST, log)`, with a clock of its own that moves only while the model answers: `C.answer` takes 0.4 ms. In other cases, `C.slow_answer` takes 1.5 seconds, and `C.EVENING` is the same Tuesday at 17:00. Under the check, your `serve` answers the 24 hours of 14 July 2011, with the real clock of your computer. When it is right, the check saves your function as `work/projects/ops/service.py`.
        """),
        L.task("Replace the `...` with how long the answer took, in milliseconds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_serve():
    box_serve = L.code_box("serve", lines=12)
    box_serve
    return (box_serve,)


@app.cell(hide_code=True)
def check_serve(box_serve):
    result_serve = L.check("serve", box_serve.value)
    result_serve
    return (result_serve,)


@app.cell(hide_code=True)
def hints_serve():
    L.hints(
        "serve",
        [
            "`ms` is how long `answer` took, from two readings of the clock: `start`, read right before `answer`, and a new reading of `time.perf_counter()`, right after it. The difference between the two is in seconds.",
            "The later reading minus the earlier one gives a time above 0. There are 1,000 milliseconds in a second: what does that make of a time in seconds?",
            "`    ms = (___ - start) * 1000`",
        ],
        why="`ms = (time.perf_counter() - start) * 1000`: the reading after the answer minus the reading before it, in seconds, times 1,000 for milliseconds. For the course's model, a few tenths of a millisecond per answer.",
    )
    return


@app.cell(hide_code=True)
def text_tables():
    mo.md(r"""
    ### The log as a table

    Your drift check reads the log as a table, a row per request: its fields, and the answer as the column `bikes`. `C.served()` gives the log of the course's service for every request of 2012, read that way. `C.trained()` gives the training hours of 2011 as the same table, with the course's answers on them as `bikes`, and the bikes rented as `cnt`. `temp` is the temperature divided by 41, `hum` the humidity from 0 to 1, and `windspeed` the wind divided by 67, as the data has them. Run the cell.
    """)
    return


@app.cell
def example_tables():
    train = C.trained()     # the training hours of 2011, with the course's answers as the column bikes
    served = C.served()     # every request of 2012 in the log of the course's service, as a table
    print(len(train), "training hours,", len(served), "requests")
    print(served[["mnth", "hr", "temp", "hum", "windspeed", "bikes"]].head(3))
    return


@app.cell(hide_code=True)
def text_exercise_drift():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Drift, month by month

        `drift_by_month(train, served, column)` gives the PSI of each month of the log, for one column, against the same month of the training hours. `train` is the table of the training hours, `served` the table of the log, and `column` the name of a column, such as `"temp"`. `table` starts empty. The loop goes over each month in the log once, in order, with `sorted(set(served["mnth"]))`. `now` holds the values of `column` in that month of the log, as a NumPy array, with `.to_numpy()`. `then` should hold the same for the training hours. `psi_of(then, now)` gives the PSI, and `table` keeps it under `int(month)`. The function ends with `return table`. The import of `psi_of` from `ops.psi` is there for the function.

        Replace the `...` with the values of `column` in the same month of the training hours.

        The check calls `drift_by_month(train, served, "temp")`, with the tables of the cell above; then for the answers, the humidity, and the wind of a log of January to March only. Under the check, your function gives the PSI of each month for the temperature and for the answers. When it is right, the check saves your function as `work/projects/ops/drift.py`.
        """),
        L.task("Replace the `...` with the values of `column` in the same month of the training hours."),
    ])
    return


@app.cell(hide_code=True)
def exercise_drift():
    box_drift = L.code_box("drift", lines=11)
    box_drift
    return (box_drift,)


@app.cell(hide_code=True)
def check_drift(box_drift):
    result_drift = L.check("drift", box_drift.value)
    result_drift
    return (result_drift,)


@app.cell(hide_code=True)
def hints_drift():
    L.hints(
        "drift",
        [
            "`now` holds the values of `column` in the month, from the log, `served`. `then` needs the same from the table the model learned from: `train`, which has the column `\"mnth\"` too, so `month` picks its rows, and `.to_numpy()` makes the values an array.",
            "A mask picks rows: a comparison of the column `\"mnth\"` of `train` with `month` is True for the training hours of that month. Put it between the square brackets of `train`, then take the column the way `now` does.",
            "`        then = train[train[\"mnth\"] == ___][column].to_numpy()`",
        ],
        why="`then = train[train[\"mnth\"] == month][column].to_numpy()`: the values of `column` in the training hours of the same month. Against the same month, the temperature of 2012 lies above 0.25 in January, February, March, May and November; against all of training, every month would.",
    )
    return


@app.cell(hide_code=True)
def text_rentals():
    mo.md(r"""
    ### The rentals come in

    Once the rentals of 2012 are in, `read_hours(2012)` from your project gives every hour of 2012 with its rentals as `cnt`. `C.answers()` gives what the course's service answered for each of those hours, as a tensor, and `C.config()` the `config.json` of the course's package, with `val_rmse`, its RMSE on the validation days. Run the cell.
    """)
    return


@app.cell
def example_rentals():
    from ops.demand import read_hours

    hours = read_hours(2012)     # every hour of 2012, with the bikes rented as cnt
    pred = C.answers()           # what the course's service answered for each of those hours
    config = C.config()          # config.json of the course's package, as a dict
    print(len(hours), "hours,", len(pred), "answers")
    print("version:", config["version"], "  val_rmse:", config["val_rmse"])
    return


@app.cell(hide_code=True)
def text_exercise_retrain():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. When to train again

        `train_again(hours, pred, config)` names the months in which the model needs training again. `hours` holds the hours whose rentals are in, `pred` what the service answered for each of them, and `config` the dict of `config.json` from the package. `LIMIT` says how many times as far off as on its validation days a month may be: 1.5. `table` comes from `rmse_per` of your project, per month: a dict with `"hours"` and `"rmse"` for each month. `months` starts empty, and the loop goes over the months of `table`, in order. The test after `if` decides whether `months.append(month)` adds the month. The function ends with `return months`. The import of `rmse_per` from `ops.report` is there for the function.

        Replace the `...` with the test: is the RMSE of the month more than `LIMIT` times the RMSE of the package on its validation days?

        The check calls `train_again(hours, pred, config)`, with the cell above; then with another config, with January to March only, and with a few hours of its own. Under the check, you see the RMSE of each month and the months your function names. When it is right, the check saves your function as `work/projects/ops/retrain.py`.
        """),
        L.task("Replace the `...` with the test of a month against `LIMIT` times the validation RMSE of the package."),
    ])
    return


@app.cell(hide_code=True)
def exercise_retrain():
    box_retrain = L.code_box("retrain", lines=13)
    box_retrain
    return (box_retrain,)


@app.cell(hide_code=True)
def check_retrain(box_retrain):
    result_retrain = L.check("retrain", box_retrain.value)
    result_retrain
    return (result_retrain,)


@app.cell(hide_code=True)
def hints_retrain():
    L.hints(
        "retrain",
        [
            "`table[month]` is a dict with two keys: which of the two holds how far off the model was in that month? The RMSE of the package on its validation days is in `config`, under the key `\"val_rmse\"`.",
            "The test is True for a month that is too far off: its RMSE above the limit, which is `LIMIT` times the validation RMSE. Write the limit with names, so that it changes with every new package, and let a month exactly at the limit not count.",
            "`        if table[month][\"rmse\"] > ___:`",
        ],
        why="`if table[month][\"rmse\"] > LIMIT * config[\"val_rmse\"]:`: a month counts when it is more than 1.5 times as far off as on the validation days, 79.56 bikes per hour for the course's package. February 2012, at 89.5, is the first.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Exercise 4. The next step

    In February 2012 your model has an RMSE of 89.5 bikes per hour, 1.69 times its 53.04, and on 1 March, when the rentals of February are in, your `train_again` names February.
    """)
    return


@app.cell(hide_code=True)
def choice_next_cell():
    choice_next = L.choice(
        "next",
        question="What should your project do next?",
        options={
            "limit": "Raise `LIMIT` from 1.5 to 3, so that the rule names only the months in which the model is really far off",
            "february": "Swap the model at once for one trained on the hours of February 2012 alone, since those are the newest data",
            "right": "Train a new version on 2011 and on the months of 2012 so far, test it, and serve it if it beats the old one",
            "add": "Keep the model as it is, and double every answer, since January and February 2012 had 2.2 times the rentals of 2011",
        },
        correct="right",
        why={
            "right": "The rule says the model no longer fits, and your project already has every step to fix that: train a version on data that hold the change, run its tests and write its card, and serve it only when it is less far off on the latest hours than the version that runs now. A loop that does this by itself, again and again, is what a later lesson calls MLOps.",
            "limit": "Over all of 2012, a limit of 3 would name only September and October, while from March to November the model is more than twice as far off as on its validation days. A limit raised until the rule is quiet hides the problem: what does the rule say about the model?",
            "february": "One month of winter holds no summer hours and only 692 hours in all: a model trained on it alone knows nothing of what 2011 taught the old one. Which data hold both what the model knew and what changed?",
            "add": "Looking back over 2012, July had 1.4 times the rentals per hour of July 2011, not 2.2: one factor for every month is far off by summer. And a model that doubles its answers is a new version that nobody tested. What does your project do with a new version before it serves?",
        },
    )
    choice_next
    return (choice_next,)


@app.cell(hide_code=True)
def text_demo(result_drift, result_retrain, result_serve):
    result_serve, result_drift, result_retrain   # made again after the checks, which save your parts
    mo.md(r"""
    ### Your service, a year of requests

    The button starts your service on 127.0.0.1, with the server of [Your model behind an API](/course/open/sys-serving-api?lang=en), your package from `work/projects/ops/model` and your `serve`, and sends it every hour of 2012 as a request, month by month: 8,734 requests. Each leaves a line in `work/projects/ops/requests.jsonl`. After each month, your `drift_by_month` compares the month in the log with the same month of training, the rentals of the month come in, and your `train_again` says whether the model needs training again. The chart shows the RMSE of each month; dashed, the RMSE of your package on its validation days, and `LIMIT` times it. Beside it, what the service saw that month. Where a part of yours is missing, or does not do yet what its check asks, the course's takes its place. Each month takes a second or two.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Start your service and send it 2012")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_drift, result_retrain, result_serve, run_button):
    result_serve, result_drift, result_retrain   # run again with your files once a check saves them
    if not run_button.value:
        _out = mo.md("*Press the button to start your service and send it the requests of 2012.*")
    else:
        _w = C.live_words(L.lang)
        _now = {}

        def _month(_m, _run):
            if not _now:   # from the first month on: the dashed lines, and what the service saw beside the chart
                for _points, _label in C.references(_run, L.lang):
                    _chart.reference(_points, _label, role="ink-2")
                _chart.watch(lambda: C.month_picture(_now["run"], _now["m"], L.lang))
            _now["run"], _now["m"] = _run, _m
            _chart.add(_m, _run["rmse"][_m], C.progress_line(_run, _m, L.lang))

        _chart = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, 12,
                        role="loss", low=0, window=1, xticks=C.MONTH_TICKS)
        _chart.line(_w["start"])
        _run = C.demo(show=_month)
        if not _run["months"]:
            _out = _chart.stopped(C.summary(_run, L.lang))
        else:
            _out = mo.vstack([
                L.explore("rewind", "month", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=_run["months"][-1], values=_run["months"], role="",
                          task="Go back to January, then forward month by month: in which months does the drift check go above 0.25, and in which does your train_again name the month?"),
                mo.md(C.summary(_run, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        With the course's files and package, your `train_again` names February to December: in February the model has an RMSE of 89.5 bikes per hour, and from March to November more than twice as far off as on its validation days, up to 165.6 in September. The drift check of the inputs went above 0.25 in seven months, January to May, September and November, and it did not follow the error. January, with the largest drift of the temperature after March, had the smallest error. June, July, August, October and December, with no input above 0.25, were 1.84 to 3.10 times as far off. The answers of the model stayed under 0.1 in every month: they depend most on the hour of the day, so even a warmer month moved them only a little. What changed most was the number of riders, and only the rentals show that, once they are in.

        On the course's computer, `serve` measured about 0.3 ms per answer, and the client waited about 1.5 ms per request: the rest of the wait is the connection, the JSON and the thread of the server, outside `serve`. A latency in a log is always the latency of the part between its two readings of the clock.
        """),
        L.watch_out("Measure with `time.perf_counter()`, not with `time.time()`. The clock of `time.time()` follows the time of day, and a computer may set it forward or back while a request runs, such as when it syncs with a time server: one latency in the log then comes out below 0, or far too large."),
    ])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The server of [Your model behind an API](/course/open/sys-serving-api?lang=en) now answers with your own package, and every request leaves a structured log line, as in [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en). The PSI of [When the world changes: drift](/course/open/sys-drift?lang=en) runs month by month on what your service received, and your `rmse_per` from [Project: tests and a model card](/course/open/sys-project-tested-model?lang=en) measures each month once its rentals are in. Your project *ops* has three more files, `service.py`, `drift.py` and `retrain.py`, and a log, `requests.jsonl`. In the next lesson, *Someone else's computer: the cloud*, your service could run on a computer you rent instead of your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_next, result_drift, result_retrain, result_serve):
    L.wrap_up(result_serve, result_drift, result_retrain, choice_next)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
