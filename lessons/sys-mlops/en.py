# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-mlops. Route: course/route/lessen/systemen.yaml. Checks, the teams, the drift of two years and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-mlops", "en")
    C = L.checks   # the teams, the drift of two years, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: your model as a service, watched for drift](/course/open/sys-project-service?lang=en) and [A box that runs anywhere: containers](/course/open/sys-containers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_retrain():
    quick_retrain = L.choice(
        "quick-retrain",
        question="In the project, the function train_again named the months in which the bike model should be trained again. When did a month count?",
        options={
            "right": "When its RMSE was more than 1.5 times the RMSE on the validation set",
            "psi": "When the PSI of the model's answers rose above 0.1",
            "every": "In every month, since the world changes all the time",
            "slow": "When the 99th percentile of the answers took more than a second, since a slow model has to be replaced",
        },
        correct="right",
        why={
            "right": "train_again compared the RMSE of each month of 2012 with 1.5 × 53.04 = 79.56, and named February to December. In the project, a person still had to read that list and start the training. This lesson puts that step in a loop that runs by itself.",
            "psi": "The PSI of the answers stayed under 0.1 all year, while the error grew. Which number did train_again compare? See [Project: your model as a service, watched for drift](/course/open/sys-project-service?lang=en).",
            "every": "Training every month costs time and compute, even in months in which the model is still good. Which number decided? See [Project: your model as a service, watched for drift](/course/open/sys-project-service?lang=en).",
            "slow": "A slow answer is a matter for the service, not for the training. Which number of the model decided? See [Project: your model as a service, watched for drift](/course/open/sys-project-service?lang=en).",
        },
    )
    quick_retrain
    return (quick_retrain,)


@app.cell(hide_code=True)
def quick_check_layers():
    quick_layers = L.choice(
        "quick-layers",
        question="In the quick Dockerfile, only service.py changes. Which steps are built again?",
        options={
            "all": "All steps, since a new image starts from nothing",
            "right": "Only the last step, which copies service.py",
            "pip": "The step with pip and every step after it, since packages may change at any time",
            "none": "None: the old image still runs",
        },
        correct="right",
        why={
            "right": "Each step of a Dockerfile is a layer, and a layer is built again only when its own step or a layer below it changes. With service.py copied last, a change to it rebuilds one small layer in a tenth of a second. In this lesson, building the image is one step of a pipeline, after the tests.",
            "all": "Docker keeps every layer whose step and files did not change. Which layers did change? See [A box that runs anywhere: containers](/course/open/sys-containers?lang=en).",
            "pip": "The packages are listed in requirements.txt, which did not change, so the layer with pip stays. Which layer holds service.py? See [A box that runs anywhere: containers](/course/open/sys-containers?lang=en).",
            "none": "The old image holds the old service.py. What does a build do with the one step whose file changed? See [A box that runs anywhere: containers](/course/open/sys-containers?lang=en).",
        },
    )
    quick_layers
    return (quick_layers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In this module, you built every step around a model by hand: tracked runs, versions, tests, a service with a log, a measurement of drift, a container. In many teams, each of those steps is the work of a person: someone trains in a notebook, reads the numbers, copies a file to a server, and looks at the monitoring now and then. That works for one model, once. It breaks when the world changes every month and the model has to keep up: then every step by hand is a wait, and while the team waits, a worse model answers. MLOps puts the steps together in one loop that runs by itself, and a maturity model tells a team how far along it is and which step to automate next.

    **An everyday example.** A home cook bakes bread by hand: weighing, kneading, baking, tasting. A bakery that sells bread every day puts the same steps in machines, and checks each batch before it goes to the shop: a batch that fails never reaches the shelf. When the flour changes, the bakery sees it in its own measurements and adjusts the recipe before customers complain. And the way from kitchen to bakery goes one machine at a time.
    """)
    return


@app.cell(hide_code=True)
def predict_wait():
    guess_wait = L.guess(
        "wait",
        question="The error of a model grows a little each month, as the world changes. When the error is more than 1.5 times the error right after training, an alert goes off. Team A then trains by hand: two more months pass before the new model is in production. The pipeline of team B trains, tests and releases the new model by itself, by the next month. Over two years, team B serves a model above the limit in 4 months. And team A, do you think?",
        options={
            "same": "4 months too, since the same alert starts both teams",
            "two": "6 months: two months more, but only at the first alert",
            "right": "9 months: about three months above the limit at each alert",
            "all": "All 24 months, since a team that works by hand never catches up again",
        },
        reveal="**9 months.** Team A gets three alerts, and each one costs three months above the limit: the month of the alert and the two months of waiting, in which the error grows on. Team B gets four alerts, one month each. The months of waiting count far more than the number of alerts: with three months of waiting, half of the two years would be above the limit. Exercise 2 counts this, and the demo shows it month by month.",
    )
    guess_wait
    return (guess_wait,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One loop.** **MLOps**, short for machine learning operations, is the way of working that keeps a model good in production, with the tools for it. At its heart is one loop of the steps of this module. Training runs tracked and reproducible, as in [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en). Tests check the code and what the model does, as in [Tests for ML code](/course/open/sys-code-tests?lang=en) and [Does the model behave? Behavioral tests](/course/open/sys-behavior-tests?lang=en). A model that passes goes out as a version in a registry and an image, as in [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en). It answers behind an API, and monitoring watches the answers and the inputs, as in [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en) and [When the world changes: drift](/course/open/sys-drift?lang=en). When the monitoring sees the model get worse, the loop starts again at training, with new data.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **CI/CD.** The words come from software. With continuous integration, every change to the code is built and tested by itself; with continuous delivery or deployment, a change that passes the tests goes to production by itself, or with one click. **CI/CD** is the pair. In ML, the tests check more than code: the data and the model too. And what gets delivered is not only a service, but the pipeline that trains the model.

        **Continuous training.** With **continuous training**, the pipeline trains the model again by itself. Google's page lists when: on demand, on a schedule, when new data arrive, when the model gets worse, and when the data shift. The new model passes the same tests as the first one, and replaces the model in production only when it is better: the pipeline must be able to say no.

        **How far along is a team?** A **maturity model** describes the way from everything by hand to a loop that runs by itself, in levels. The page of Google Cloud has three: level 0, a manual process; level 1, ML pipeline automation, with continuous training; level 2, CI/CD pipeline automation, in which changes to the pipeline itself are tested and delivered by themselves. The page of Microsoft, last updated on 3 June 2026, has five:

        | Level | Name | What runs by itself |
        |---|---|---|
        | 0 | No MLOps | nothing: training, release and the application all by hand |
        | 1 | DevOps but no MLOps | the code of the application is built, tested and released by itself; the model is trained by hand and handed over |
        | 2 | Automated training | training runs as a pipeline, with tracked runs; the release of a model is still done by hand |
        | 3 | Automated model deployment | a model that passes the tests goes to production through CI/CD |
        | 4 | Full MLOps automated operations | numbers from production start new training by themselves |

        Both pages were read on 28 November 2026. Microsoft notes that a team often shows traits of more than one level at once. This lesson uses a simple reading: a team is at the highest level whose step, and every step below it, runs by itself. What a maturity model gives is not the number, but the next step: the lowest step that a person still does.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **MLOps** | | the way of working, with its tools, that keeps a model good in production: tracked training, tests, releases and monitoring in one loop | |
        | **CI/CD** | | every change to the code built and tested by itself (continuous integration), and released when the tests pass (continuous delivery or deployment) | |
        | **continuous training** | | a pipeline that trains the model again by itself, on a schedule, on new data or on an alert, and releases it only after the same tests | |
        | **maturity model** | | a series of levels from everything by hand to everything by itself, to see where a team is and what comes next | |
        """),
        L.watch_out("Automating a step you have never done by hand makes the same mistakes, only faster. Do each step by hand first, as in this module, and automate it once you know what right looks like."),
        L.watch_out("Continuous training without the tests of the release step can put a worse model into production all by itself, for example when a batch of new data is broken. Every release that runs by itself needs a test that can stop it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    How far along is a team, and what is its next step:

    1. Write down the steps of the loop: train, test, release, serve, watch, and back to train.
    2. For each step, ask: does it run by itself, or does a person do it, and how long does that take?
    3. Count from the bottom: the level is the number of steps in a row that run by themselves.
    4. Automate the first step that a person still does, with a test that can stop the pipeline.
    5. Measure what it gives: the months from an alert to a new model, and the months above the limit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The release step of a pipeline, for the bike model: a new model goes into production only when every test passes and its RMSE on the validation set is lower than that of the model in production. In the cell, `tests` holds the result of each test, `new_rmse` the RMSE of the new model, and `old_rmse` the 53.04 of the model in production. The loop sets `passed` to `False` at a test that fails. Run the cell.
    """)
    return


@app.cell
def example_gate():
    tests = {"shapes": True, "loss falls": True, "warmer, more bikes": True}
    new_rmse = 51.2      # the new model, on the validation set
    old_rmse = 53.04     # the model in production
    passed = True
    for _name in tests:
        if not tests[_name]:
            passed = False
    release = passed and new_rmse < old_rmse
    print("all tests pass:", passed)
    print("release the new model:", release)
    return new_rmse, old_rmse, passed, release, tests


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    All three tests pass, and 51.2 is lower than 53.04, so the new model goes out. Set one test to `False`, and `passed` becomes `False`: the model stays out, however good its RMSE. Set `new_rmse` to 53.04, and the old model stays: a new model has to be better, not just as good, since every release carries a risk. This gate is what makes continuous training safe. Exercise 1 first places a team on the maturity model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_level():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How far along is a team?

        `level(team, steps)` gets a team and the steps of the levels, in order. `C.STEPS` is `["code", "training", "release", "retraining"]`: the code of the application built and tested by itself (level 1), training as a pipeline (level 2), the release by the pipeline (level 3), and retraining that monitoring starts (level 4). A team is a dict with `True` for a step that runs by itself and `False` for a step a person does. The level is the number of steps in a row from the start that run by themselves. The `for` loop walks the steps, `n` counts them, and `break` stops the loop. Replace the `...` with the question whether a person still does this `step`.

        `C.TEAMS` holds four teams. The check calls `level(C.TEAMS["shop"], C.STEPS)`: a shop whose code and training run by themselves, but whose model is copied to the server by hand. Then a bike rental that releases by itself but trains by hand, a bank, a lab, a team with every step by itself, and two teams with other steps.
        """),
        L.task("Count the steps that run by themselves, up to the first one a person does."),
    ])
    return


@app.cell(hide_code=True)
def exercise_level():
    box_level = L.code_box("level", lines=8)
    box_level
    return (box_level,)


@app.cell(hide_code=True)
def check_level(box_level):
    result_level = L.check("level", box_level.value)
    result_level
    return (result_level,)


@app.cell(hide_code=True)
def hints_level():
    L.hints(
        "level",
        [
            "A step a person does has `False` in `team`. The loop has to stop at that step: the question after `if` is true when the value of `step` in `team` is `False`. How do you look up a value in a dict, and how do you turn `False` into a question that is true?",
            "In the worked example, `if not tests[_name]:` was true for a test that failed. For the bike rental, the loop meets `True` for the code, then `False` for the training: it must stop there, with `n` at 1.",
            "`        if not ___:`",
        ],
        why="`if not team[step]:` is true at the first step a person does, and `break` stops the counting there. The shop has `True`, `True`, `False`: level 2. The bike rental has `True`, `False`, `True`: level 1, since a release by itself does not count above a training by hand.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bad_months():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Months above the limit

        `bad_months(drift, limit, delay)` follows a model in production month by month. `drift` holds for each month how much the error grows: `1.06` is 6% more. The `for` loop takes the growth of each month as `grow`. `error` starts at `1.0`, the error right after training. In a month in which `error` is above `limit`, the model is wrong too often: `bad` counts that month, and the alert goes off. The team then needs `delay` more months before the new model is in production: `waited` counts the months since the alert, and when the new model is there, `error` goes back to `1.0` and `waited` to 0. Replace the `...` with the question whether the team has waited long enough.

        `C.DRIFT` holds the 24 months of two years. The check calls `bad_months(C.DRIFT, 1.5, 2)`: a team that trains by hand and needs 2 more months after an alert. Then 0, 1 and 3 months of waiting, a lower limit, a world that does not change, a world that changes fast, and a single month.
        """),
        L.task("Let the new model come after the months of waiting the team needs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bad_months():
    box_bad_months = L.code_box("bad_months", lines=15)
    box_bad_months
    return (box_bad_months,)


@app.cell(hide_code=True)
def check_bad_months(box_bad_months):
    result_bad_months = L.check("bad_months", box_bad_months.value)
    result_bad_months
    return (result_bad_months,)


@app.cell(hide_code=True)
def hints_bad_months():
    L.hints(
        "bad_months",
        [
            "At each month above the limit, the question is: is the new model there now? `waited` counts the months of waiting so far, and `delay` is how many the team needs. When are the two the same?",
            "With `delay` at 0, the new model comes at once: at the first month above the limit, `waited` is 0, and `error` goes back to `1.0`. With `delay` at 2, the month of the alert and two more months go by first.",
            "`            if waited ___:`",
        ],
        why="`if waited == delay:` lets the new model come after exactly `delay` months of waiting. With 2 months, each of the three alerts of `C.DRIFT` costs three months above the limit: 9 in all. With 0 months, there are four alerts of one month each: 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.md(r"""
    ### Exercise 3. Which step of the loop?

    Eight practices from this module. Put each one under the step of the loop it belongs to: train, test, release or watch.
    """)
    return


@app.cell(hide_code=True)
def sort_loop_cell():
    sort_loop = L.sort("loop")
    sort_loop
    return (sort_loop,)


@app.cell(hide_code=True)
def check_loop(sort_loop):
    result_loop = L.check("loop", sort_loop)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "loop",
        [
            "The picture of the loop, under *One loop*. Ask for each practice: does it happen while a model is made, before it goes out, while it goes out, or while it answers?",
            "Two practices go in each column.",
            "Train: the seeds and \\_\\_\\_. Test: the loss on one batch and \\_\\_\\_. Release: the registry and \\_\\_\\_. Watch: the 99th percentile and \\_\\_\\_.",
        ],
        why="**Train:** seeds and pinned packages, and a line per run. **Test:** the loss on one batch, and warmer weather with more bikes. **Release:** a version in the registry, and a container image. **Watch:** the 99th percentile, and the PSI per month. A pipeline runs these steps one after another, and the watch step starts the training again.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Exercise 4. The next step

    A team builds and tests the code of its service by itself. Its training runs every night as a pipeline, with each run tracked. Once a month, a data scientist reads the numbers and copies the best model to the server by hand.
    """)
    return


@app.cell(hide_code=True)
def choice_next_cell():
    choice_next = L.choice(
        "next",
        question="What is its next step?",
        options={
            "retrain": "Retraining that starts by itself when monitoring sees drift",
            "code": "Automatic builds and tests for the code of the service",
            "hourly": "Training every hour instead of every night, so the model is always fresh and never behind",
            "right": "A release by the pipeline: a model that passes the tests goes out by itself",
        },
        correct="right",
        why={
            "right": "The code and the training run by themselves: level 2. The first step a person still does is the release, so that is the next step, to level 3. Only then does retraining by itself pay off, since every new model would otherwise wait for the copy by hand.",
            "retrain": "Retraining by itself is level 4, but each new model would still wait for the copy by hand. Which step does a person still do?",
            "code": "The code is built and tested by itself already. Which step does a person still do?",
            "hourly": "Training more often brings the model to the server no sooner. Which step does a person still do?",
        },
    )
    choice_next
    return (choice_next,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### How long may a team wait?

    The model of exercise 2 in production for two years, with the drift of `C.DRIFT` and the limit at 1.5. Choose how many months pass after an alert before the new model is there: 0 for a pipeline that trains, tests and releases by itself, more for a team that works by hand.
    """)
    return


@app.cell(hide_code=True)
def demo_delay():
    L.explore("delay", "months of waiting after an alert", lambda _v: C.frame(_v, L.lang), start=2,
              values=C.DELAYS, role="", task="Go through the four values: what does each month of waiting add, and why does a team that waits get fewer alerts?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Without waiting, 4 of the 24 months are above the limit; with 3 months of waiting, 12: half of the two years. A team that waits gets fewer alerts, three instead of four, since its new models come later, but each alert lasts longer. And a fixed plan that trains every 6 months, alert or not, trains 4 times as well, as often as the pipeline, yet has 10 months above the limit: it trains when the calendar says so, not when the error does. Continuous training trains at the moments the error asks for, and quickly.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Project: your model as a service, watched for drift](/course/open/sys-project-service?lang=en), train_again named the months for new training, and a person had to act on that list; in [A box that runs anywhere: containers](/course/open/sys-containers?lang=en), the service became an image that any server can start. MLOps puts those steps, and the others of this module, in one loop that runs by itself, and a maturity model shows which step to automate next. In the next lesson, *Is the data good? Four ways to look*, the loop gets a step at its start: checking the data before it goes into training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_next, result_bad_months, result_level, result_loop):
    L.wrap_up(result_level, result_bad_months, result_loop, choice_next)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
