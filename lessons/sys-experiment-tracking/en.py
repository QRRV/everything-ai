# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-experiment-tracking. Route: course/route/lessen/systemen.yaml. Checks, the course's log, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-experiment-tracking", "en")
    C = L.checks   # the course's log of twelve runs, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Data as text: JSON](/course/open/py-json?lang=en) and [Reading loss curves](/course/open/torch-loss-curves?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question="A line of a log holds the text `{\"lr\": 0.5, \"done\": true}`. What does `json.loads` turn that text into?",
        options={
            "text": "The same text, as a string",
            "right": "A dict whose value for `\"done\"` is `True`",
            "strings": "A dict whose values are the strings `\"0.5\"` and `\"true\"`",
            "error": "An error, since `true` is not Python",
        },
        correct="right",
        why={
            "right": "`json.loads` turns JSON text into Python values: the object becomes a dict, `0.5` a float and `true` the `True` of Python. That is how a log of runs comes back: each line of text becomes a dict again.",
            "text": "`json.loads` reads the text and gives back the Python values it describes. What does a JSON object become? See [Data as text: JSON](/course/open/py-json?lang=en).",
            "strings": "In JSON, a number and `true` are not strings: strings have double quotes. What do they become in Python? See [Data as text: JSON](/course/open/py-json?lang=en).",
            "error": "`true` is JSON, not Python, and `json.loads` knows it. What does it become? See [Data as text: JSON](/course/open/py-json?lang=en).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def quick_check_epoch():
    quick_epoch = L.choice(
        "quick-epoch",
        question="After every epoch you measure the training loss and the validation loss. The training loss keeps falling; the validation loss is lowest after epoch 8 and rises after it. Which epoch's model do you keep?",
        options={
            "first": "Epoch 1: it has not overfit yet",
            "last": "The last epoch, with the lowest training loss",
            "right": "Epoch 8, where the validation loss is lowest",
            "test": "Whichever epoch reads the most test digits right",
        },
        correct="right",
        why={
            "right": "With overfitting, the best model is the one of the epoch with the lowest validation loss. So a log needs the validation loss of every epoch, not only of the last one.",
            "first": "After epoch 1 the model has barely learned: both losses are still high. Which epoch did the best on digits it never trained on? See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "last": "A falling training loss with a rising validation loss is overfitting: the model learns its training digits by heart. Which epoch did the best on the validation set? See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "test": "Choosing by the test digits uses them up: they no longer measure how well the model does on digits the choice never saw. Which set is there for choosing? See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
        },
    )
    quick_epoch
    return (quick_epoch,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In March 2018, the engineer Pete Warden wrote a blog post called *The Machine Learning Reproducibility Crisis*. Models were hard to train again from what their makers had written down: "even the original author sometimes couldn't train the same model and get similar results!" The post's diagnosis: "we're still back in the dark ages when it comes to tracking changes and rebuilding models from scratch." In June 2018, the company Databricks released MLflow, an open-source tool for exactly that problem. Its announcement put it plainly: "It's hard to track experiments."

    You have met the problem in this course already. A training run in a cell shows its losses, and when you run the cell again with another learning rate, the old numbers are gone. After a dozen runs, which learning rate gave the lowest validation loss, and after which epoch? This lesson keeps every run in a log, so you can still answer that next week.

    **An everyday example.** Someone baking bread changes the flour one week and the oven temperature the next. Without notes, a good loaf is luck you cannot repeat. With a notebook that holds, for each bake, what went in, the temperature and how it came out, you can look up the best one and bake it again.
    """)
    return


@app.cell(hide_code=True)
def predict_last():
    guess_last = L.guess(
        "last-epoch",
        question="The course trained the digits MLP 12 times on 1,000 training digits: three learning rates, two sizes of the hidden layer and two seeds, 20 epochs each, with the validation loss after every epoch. In how many of the 12 runs is the validation loss lowest after the last epoch?",
        options={
            "all": "All 12: more training only helps",
            "most": "About 9 of the 12",
            "right": "2 of the 12",
            "none": "None of the 12",
        },
        reveal="**2 of the 12.** Only the two runs with learning rate 0.05 and seed 1 reach their lowest validation loss in epoch 20. The two runs with learning rate 0.05 and seed 0 are lowest after epoch 19, and at most 0.001 higher after epoch 20: they are still learning too. In the 8 runs with learning rates 0.2 and 0.5, the lowest point lies after epoch 4 to 10, and the validation loss rises after it. A log with only the last epoch would hide the best model of those 8 runs. The demo below goes through all 12.",
    )
    guess_last
    return (guess_last,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A run and its log.** A run is one round of training with its own settings: a learning rate, a size, a seed. **Experiment tracking** means that every run writes down, by itself, what went in and what came out: its settings before it starts, its measurements while it trains and its result at the end. Later you can compare all runs, find the best one and repeat it, because its settings are in the log, including the seed from [Same run, same result: seeds and determinism](/course/open/sys-reproducibility?lang=en).

        **Params and metrics.** Tracking tools use two words. A **param** is a setting of the run, logged once, before training: the learning rate, the number of neurons, the seed, the version of PyTorch. A **metric** is a measurement, logged many times: the training loss and the validation loss after every epoch, each with its **step**, the number of the epoch. MLflow's API reference says it in two sentences: "A param can be logged only once for a run." and "A metric can be logged multiple times." With the steps, the log holds the whole curve from [Reading loss curves](/course/open/torch-loss-curves?lang=en), so you can still find the epoch with the lowest validation loss.

        **One run per line.** The log is a text file in the format **JSON Lines**: each line is one JSON value, here one run, as `json.dumps` from [Data as text: JSON](/course/open/py-json?lang=en) writes it. A new run adds one line at the end of the file and leaves every earlier line as it was. To read the log, you go over its lines and turn each one back into a dict with `json.loads`. The format's own site, jsonlines.org, calls it "a great format for log files", and suggests the ending `.jsonl`.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **MLflow.** You will write a small tracker yourself, in a few lines of Python. MLflow has the same calls, and much more: `mlflow.log_param("lr", 0.5)` logs a setting, `mlflow.log_metric("val_loss", 0.48, step=2)` a measurement, and `mlflow.search_runs()` gives all runs as a pandas DataFrame, with a column per param and metric, such as `params.lr` and `metrics.val_loss`. That column holds the last value logged, so sorting by it sorts by the last epoch. Since version 3.7, MLflow keeps its runs in a database file, `mlflow.db`, and `mlflow server` shows them as a table in your browser, at `http://localhost:5000`. MLflow is open source, under the Apache 2.0 license; TensorBoard and Weights & Biases do the same job in their own way. This course does not install any of them: your tracker shows the idea, and the tools add a server, a table to sort and charts to compare runs.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **experiment tracking** | | every training run records its settings, its metrics and its result, so you can compare and repeat runs later | |
        | **MLflow** | em-el-flow | an open-source tool for experiment tracking, with the calls `log_param` and `log_metric` | `mlflow.log_metric("val_loss", 0.48, step=2)` |
        | **JSON Lines** | jason lines | a text format with one JSON value per line, usually in a file ending in `.jsonl` | `json.dumps(run) + "\n"` |
        | **`open(path, "a")`** | | opens a file to add text at its end, and makes the file when it is missing | `with open(path, "a") as f:` |
        | **`f.write(text)`** | | writes the text into the open file as it is, without a newline at the end | `f.write(json.dumps(run) + "\n")` |
        """),
        L.watch_out("`open(path, \"w\")` empties a file before it writes. Open your log with `\"w\"` instead of `\"a\"`, and every earlier run is gone. In the research domain, the lesson *Keeping an experiment log* also writes the version of the code into every run."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A run that keeps track of itself:

    1. Give every run a name, and a dict with `"params"` and `"metrics"`.
    2. Before training, use `log_param` to log every setting that can change the result: the learning rate, the size, the seed, the version of PyTorch.
    3. After every epoch, log every measurement with `log_metric`, with the epoch as its step.
    4. After training, add the run as one JSON line to the end of the log, with `open(path, "a")`.
    5. To compare runs, read the log back line by line with `json.loads`, and look for the lowest validation loss of each run, in any epoch.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1, 2 and 4 of the recipe, with two epochs of the validation loss typed in by hand, then step 5: the log read back. `C.log_path("example.jsonl")` gives a file in your work folder, `work/runs/example.jsonl`. Run the cell.
    """)
    return


@app.cell
def example_tracker():
    import json
    import torch

    # 1. a run: its name, its settings (params) and its measurements (metrics)
    def start_run(name):
        return {"name": name, "params": {}, "metrics": {}}

    def log_param(run, key, value):
        run["params"][key] = value

    # 2. the settings, once, before training
    run = start_run("first-try")
    log_param(run, "lr", 0.5)
    log_param(run, "seed", 0)
    log_param(run, "torch", torch.__version__)
    run["metrics"]["val_loss"] = [[1, 0.61], [2, 0.48]]   # two epochs, by hand for now

    # 4. the run as one line of JSON, added to the end of the log
    path = C.log_path("example.jsonl")
    with open(path, "a") as _f:
        _f.write(json.dumps(run) + "\n")

    # 5. the log read back: one run per line
    with open(path) as _f:
        runs = [json.loads(line) for line in _f]
    print(len(runs))
    print(runs[-1])
    return json, log_param, path, run, runs, start_run, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first number counts the runs in the log. Run the cell again, and it goes up by one: each run adds a line, and the earlier lines stay. The last line comes back as a dict, with the settings under `"params"` and the two measurements under `"metrics"`, each a list of two: the step, then the value. The version of PyTorch is in it too: `torch.__version__` gives it as text, so the log says which PyTorch made the run. What is still missing is a function that adds a measurement for you, after every epoch: that is your first exercise.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_metric():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own log_metric

        `log_metric(run, key, value, step)` adds one measurement to a run, as `log_param` adds a setting. Under `run["metrics"]`, each metric has a list, such as `[[1, 0.61], [2, 0.48]]` for the validation loss after epochs 1 and 2: one item per call, each a list of two, with the `step` first and the `value` second. The `if` line and the line under it make the list when `key` is not in `run["metrics"]` yet. Replace the `...` with one line that adds the item to the end of the list of `key`.

        The check runs your function with a few lines after it, such as the run of the worked example with its two epochs, and compares what the log holds. The picture under it fills a log for one of the course's runs.
        """),
        L.task("Add each measurement to the end of the list of its metric, with its step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_metric():
    box_metric = L.code_box("log-metric", lines=5)
    box_metric
    return (box_metric,)


@app.cell(hide_code=True)
def check_metric(box_metric):
    result_metric = L.check("log-metric", box_metric.value)
    result_metric
    return (result_metric,)


@app.cell(hide_code=True)
def hints_metric():
    L.hints(
        "log-metric",
        [
            "The list of `key` is `run[\"metrics\"][key]`: the two lines above it make it when it is missing. `append` adds one item to the end of a list.",
            "The item is itself a list of two: the step first, then the value, as in `[2, 0.48]`. Which two names of the function hold them?",
            "`    run[\"metrics\"][key].append(___)`",
        ],
        why="`run[\"metrics\"][key].append([step, value])` adds the measurement to the end of the list of `key` and keeps what was there: after 20 epochs, the list holds 20 items, one per step.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The course's log

    The course trained the twelve runs from the question above, logged each one with `log_param` and `log_metric` as in this lesson, and added each run as one line to a log. `C.RUNS` holds the twelve runs, read back with `json.loads`. The slider picks a run. At the top, its training loss (solid) and validation loss (dashed) after every epoch, on a log scale, with the lowest validation loss marked; below it, the lowest validation loss of all twelve runs, with the chosen run drawn larger; at the bottom, the start of its line in the log.
    """)
    return


@app.cell(hide_code=True)
def demo_log():
    L.explore("run", "run", lambda _v: C.frame(_v, L.lang), start=0, low=0, high=11, step=1,
              role="", task="Find the run with the lowest validation loss, and a run whose validation loss is still falling.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Sorted by the lowest validation loss, `lr0.5-h128-s0` leads: 0.3965, after epoch 8. Sorted by the validation loss after the last epoch, another run would lead: `lr0.2-h128-s1`, with 0.4474. Which run is best depends on what you compare, and the log holds every epoch, so you can choose. The same settings with seed 1, `lr0.5-h128-s1`, reach 0.4183, after epoch 4: as [Same run, same result: seeds and determinism](/course/open/sys-reproducibility?lang=en) showed, one run per setting says little. The four runs with learning rate 0.05 are still learning after 20 epochs; a next run could train them longer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The best run in the log

        `best_run(runs, key)` gets a list of runs, as read back from a log, and the name of a metric, and gives back the name of the run that reaches the lowest value of that metric, in any step. `lowest(run, key)` is ready: with `min` and a list comprehension over the `step` and `value` of each item, it gives the lowest value of `key` in one run. In `best_run`, `best` starts as the first of the `runs`, and the loop `for run in runs` goes over all of them; `best = run` keeps a run that does better than `best` so far. Replace the `...` with that comparison.

        The check calls `best_run(C.RUNS, "val_loss")`, with the twelve runs of the demo, then other runs and other metrics.
        """),
        L.task("Keep the run whose lowest value of the metric is lower than that of the best run so far."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-run", lines=10)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-run", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-run",
        [
            "`lowest(run, key)` gives the lowest value of `key` in `run`, over all its steps. For a loss, lower is better. Which run does `run` have to beat to take the place of the best one?",
            "`run` has to beat the best run so far, which `best` holds. Which call gives the lowest value of `key` in that run?",
            "`        if lowest(run, key) < ___:`",
        ],
        why="`lowest(run, key) < lowest(best, key)` is `True` when `run` goes lower than every run before it; then `best = run` keeps it. For the course's log, that is `lr0.5-h128-s0`, at 0.3965 after epoch 8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lines():
    mo.md(r"""
    ### Exercise 3. One line per run

    Your log adds each run as one line at the end of `runs.jsonl`. A colleague keeps all runs in one JSON list in `runs.json` instead: after every run, the program reads the list, adds the new run to it, and writes the whole list again.
    """)
    return


@app.cell(hide_code=True)
def choice_lines_cell():
    choice_lines = L.choice(
        "lines",
        question="What does one line per run avoid?",
        options={
            "size": "A file that takes more space on the disk",
            "right": "Losing every run when a write stops halfway",
            "read": "A log that Python cannot read back",
            "order": "Runs that end up in another order than they ran",
        },
        correct="right",
        why={
            "right": "Writing the whole list again rewrites every earlier run too. If the program stops halfway, say the computer loses power, the file can end in the middle of the list, and `json.load` cannot read any of it. A new line only adds text at the end: a broken last line costs only that run, and since the earlier lines are still whole, your code can skip the broken one.",
            "size": "Both files hold the same runs as the same JSON text, give or take a comma and two brackets. What happens to the one list when a write stops halfway?",
            "read": "`json.load` reads a JSON list as easily as `json.loads` reads a line. What happens to the list when a write stops halfway?",
            "order": "A list keeps its items in the order they were added, as a file keeps its lines. What else is different when the program stops while it writes?",
        },
    )
    choice_lines
    return (choice_lines,)


@app.cell(hide_code=True)
def text_exercise_repeat():
    mo.md(r"""
    ### Exercise 4. Enough to train it again?

    Your log of the run `run-7` holds `lr`, `hidden` and `epochs` as params, and the training loss and the validation loss of every epoch as metrics. A colleague wants to train the same run again, on a computer of the same kind, and get the same model.
    """)
    return


@app.cell(hide_code=True)
def choice_repeat_cell():
    choice_repeat = L.choice(
        "repeat",
        question="What should the log hold too?",
        options={
            "time": "How long each epoch took",
            "loss": "The test loss of every epoch, next to the other two",
            "name": "Who trained it, and on which day",
            "right": "The seed and the versions of the packages",
        },
        correct="right",
        why={
            "right": "With the same seed, the same versions and the same hardware, a run on a CPU gives the same model again, as in [Same run, same result: seeds and determinism](/course/open/sys-reproducibility?lang=en). The losses are what the run gave, not what it needs. So a run logs every setting that can change its result, as params.",
            "time": "The time of an epoch says something about the computer, not about what the run needs to come out the same. Which random numbers does the run depend on?",
            "loss": "A test loss is something the run gives, like the other two losses. What does the colleague need to set before training, to get the same model?",
            "name": "A name and a date help you find a run, but training does not depend on them. What does the model depend on, besides `lr`, `hidden` and `epochs`?",
        },
    )
    choice_repeat
    return (choice_repeat,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Same run, same result: seeds and determinism](/course/open/sys-reproducibility?lang=en) showed what a run needs to come out the same; this lesson writes it down with every run. [Data as text: JSON](/course/open/py-json?lang=en) gave `json.dumps` and `json.loads`, here one line per run, and [Reading loss curves](/course/open/torch-loss-curves?lang=en) the reason to keep every epoch: the best model may lie in the middle. In the next lesson, *Which model is this? Versions and a model registry*, the model of the best run gets a version of its own, so you always know which model you are using.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lines, choice_repeat, result_best, result_metric):
    L.wrap_up(result_metric, result_best, choice_lines, choice_repeat)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
