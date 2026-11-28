# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-project-tracked-model. Route: course/route/lessen/systemen.yaml. Checks, the hours of 2011, the course's runs, the demo and the pictures: checks.py; the files the course puts in your project: project/ops/demand.py and project/ops/safetensors_io.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("sys-project-tracked-model", "en")
    C = L.checks   # the hours of 2011, the course's runs, the checks of your parts, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en) and [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_package():
    quick_package = L.choice(
        "quick-package",
        question="Your network is saved as a safetensors file. What is in that file?",
        options={
            "pickle": "The network as Python objects, which Python builds again when it loads the file, class and all",
            "right": "A JSON header with the name, dtype, shape and place of each tensor, then the bytes of all the numbers",
            "settings": "The weights and every setting of the model, such as its number of neurons and its learning rate, in one header",
            "code": "The code of the network's class, with the weights written into it as Python numbers",
        },
        correct="right",
        why={
            "right": "The header says where the numbers of each tensor are, and the numbers follow as raw bytes: reading the file takes slices and runs nothing. The settings that build the network go in a config.json beside it.",
            "pickle": "That is what `torch.save` writes, with Python's pickle, which can run any code while it loads: what does a safetensors file hold instead? See [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).",
            "settings": "The header of a safetensors file holds the dtype, the shape and the place of each tensor: where do the class and the settings of the network go? See [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).",
            "code": "A safetensors file holds no code at all, which is what makes it safe to load: what does it hold, after its header? See [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).",
        },
    )
    quick_package
    return (quick_package,)


@app.cell(hide_code=True)
def quick_check_val():
    quick_val = L.choice(
        "quick-val",
        question="You train four versions of a model, each with other hyperparameters. On which data do you choose the best version?",
        options={
            "train": "On the training set: the versions learned from it, so it shows best what each of them can do",
            "right": "On the validation set: data that none of the versions learned from, kept apart from the test set",
            "test": "On the test set, the data you keep for the very end, so that the choice rests on the final number",
            "all": "On all the data at once, training and validation together, so that the choice rests on more rows",
        },
        correct="right",
        why={
            "right": "The validation set says how each version does on data it did not learn from. The test set stays untouched by every choice, so that at the end it still says how far off the chosen version is on new data.",
            "train": "A version that learns the training set by heart scores best there and worst on new data: which data did no version learn from? See [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "test": "A test set that helped you choose has seen a choice, so its number comes out too good: which data do you choose on, then? See [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "all": "Part of all the data is the training data, which the versions learned from: on which data does a version show what it does with rows it never saw? See [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
        },
    )
    quick_val
    return (quick_val,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A bike-sharing company has docks with bikes all over a city. In the morning rush, the docks in the suburbs empty and those in the center fill up; vans then drive bikes back, and the company needs to know how many bikes people will rent in the coming hours. A model can say that. In this project, *ops*, you build such a model in three steps, and you run it the way a team runs a model: in this lesson you train versions of it, keep track of each one, and pack the best; in the next steps you test it, and put it behind a service that watches whether the world still looks like the data it learned from.

        The data are real: every hour of 2011 and 2012 at Capital Bikeshare in Washington, D.C., with the weather of that hour, collected by Hadi Fanaee-T and João Gama (2013). In [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en) and [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en), you tracked the course's runs and packed the course's network. Now the parts are yours, and they go in your project: the split, the tracked run and the package.

        **An everyday example.** A bakery tries four recipes for a new bread. The baker writes each recipe and its result in a notebook, judges the loaves on days the recipes were not tuned on, and then writes the winning recipe on a card with everything a colleague needs to bake it: the amounts, the oven, the time, and which batch it came from.
        """),
        draw.project_steps(["A demand model, tracked and packed", "Tests and a model card", "Your model as a service, watched for drift"], 0,
                           here_label="← today", title="Where your project stands: ops"),
    ])
    return


@app.cell(hide_code=True)
def predict_split():
    guess_split = L.guess(
        "split",
        question="In this lesson the validation set is 73 whole days of 2011, every fifth day, with all their 1,733 hours. Suppose you took 1,733 hours at random instead, from all the days of the year, and trained on the rest. With the course's best settings, how does the validation RMSE change, do you think?",
        options={
            "same": "It stays about the same: 1,733 hours either way",
            "right": "It gets lower, by about 7 bikes per hour",
            "higher": "It gets higher, by about 7 bikes per hour",
            "much": "It gets lower, by about 40 bikes per hour",
        },
        reveal="**About 7 lower.** Over three seeds, the RMSE on 1,733 random hours is 45.7 bikes per hour on average, and on the 73 whole days 53.1. With random hours, the model has learned from the other hours of the same days, with the same weather and the same busy moments, so the validation hours are easier. A new day is never like that: on a new day the model is about 53 bikes off, as the whole days say.",
    )
    guess_split
    return (guess_split,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The hours.** The course hosts the table as `data/bike-sharing-hour.csv.gz`. Each row is one hour. `cnt` is the number of bikes rented in that hour: what the model must say. `hr` is the hour of the day, from 0 to 23; `weekday`, `workingday` and `holiday` say what kind of day it is; `weathersit` goes from 1, clear, to 4, heavy rain; `temp`, `atemp` (how warm it feels), `hum` and `windspeed` are scaled to between 0 and 1 by the makers of the table; `season` and `mnth` give the time of year. Your project uses 2011: 8,645 hours of 365 days, since some hours are missing. The hours of 2012 wait for the last step of the project.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The picture shows why the hour matters so much, and why it matters differently on different days. On a working day, people ride to work and back: one peak at 8 in the morning, a higher one at 5 in the afternoon. On a Saturday, there is one wide hill in the afternoon.

        **The inputs and the network.** The course puts `work/projects/ops/demand.py` in your project, with the parts that are not the subject of this lesson. `read_hours()` reads the hours of 2011 and adds a column `day`: 0 for 1 January, up to 364 for 31 December. `inputs(hours, scale)` makes 34 numbers per hour. The hour becomes 24 one-hot columns, as in [One-hot vectors for classes](/course/open/ml-one-hot?lang=en): two peaks are not a straight line, and a column per hour lets the network give every hour its own weight. The other 10 columns are standardized, as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en): minus their mean, divided by their standard deviation. `scaling(train)` works out those means and standard deviations on the training hours. `DemandNet(hidden)` is a network with one hidden layer of `hidden` neurons: a linear layer from 34 numbers to `hidden`, ReLU, and a linear layer to one number, the bikes. `train_epoch` trains it for one epoch in mini-batches, with the mean squared error as the loss.

        **Whole days for validation.** The hours of one day belong together: they share the weather, the kind of day and its busy moments. If some hours of a day are in the training set and others in the validation set, the model has seen that day already, and the validation set says less about a new day: a leak, as in [Too good to be true: data leakage](/course/open/ml-data-leakage?lang=en). So the split goes by day: every fifth day, the days 4, 9, 14 and so on, goes to the validation set with all its hours. That gives 73 days, spread over the whole year: every season, every day of the week, holidays too.

        **A version is a tracked run.** Each set of hyperparameters, such as 16 neurons and a learning rate of 0.01, trains another network: another version of the model, here with the name `h16-lr0.01`. Your `train_version` trains it, and after every epoch it measures the root mean squared error on the validation days, as in [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en): the square root of the mean squared difference, in bikes per hour. The run is a dict with the name, the hyperparameters and the metrics, and goes as one line of JSON into the file `runs.jsonl`, as in [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en).

        **Choose by the last epoch.** In *Keeping track of your runs*, the best run was the one with the lowest validation loss at any epoch. But `train_version` gives back the network after its last epoch: the network of an earlier epoch is gone, unless you saved it then, as in a checkpoint. So you choose the version whose last epoch has the lowest validation RMSE: that is the network you have.

        **Pack it.** Your `pack` writes the weights of the chosen version as `model.safetensors`, with `save_file` from `work/projects/ops/safetensors_io.py`: the format of [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en), which the course put in your project too. The `version` in the package is the first 12 characters of the sha256 of the bytes of that file, as in [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en): the name `h16-lr0.01` says how a version was trained, and these 12 characters say which file it is. Beside it goes `config.json`, with what the weights do not say: the class and the number of neurons, how the inputs are made, the means and standard deviations of the training hours, what the output means, the run it came from with its validation RMSE, and `version`. Whoever loads the package checks first that `version` matches the file.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **demand** | | how much of something people want in a time: here, the bikes rented in an hour | `cnt` |
        | `~` | tilde | turns a mask around: every True becomes False, and every False True | `hours[~val_day]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A model, tracked and packed:

    1. Split the data so that the validation set is like new data: here whole days, spread over the year.
    2. Work out how the inputs are made on the training set only, such as the means and standard deviations.
    3. For every version, train with a fixed seed, measure the validation metric after every epoch, and add the run as one line to the file of runs.
    4. Choose the version with the best metric at its last epoch, since that is the network you have.
    5. Pack it: the weights in a safetensors file, its `version` from the bytes of that file, and a config.json with everything else someone needs to use it.
    6. Load the package back, check `version`, and make sure it gives the same predictions as the network you trained.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A version says, for four hours, 30, 40, 220 and 110 bikes; there were 10, 40, 250 and 100. The differences are 20, 0, −30 and 10, their squares 400, 0, 900 and 100, and the mean of the squares is 350. The validation RMSE is $\sqrt{350} \approx 18.7$ bikes per hour. The hour that is 30 off counts for more than half of it: squares weigh large mistakes more.

    After two epochs, the run of `h16-lr0.01` is one line in `runs.jsonl`:

    ```json
    {"name": "h16-lr0.01", "params": {"hidden": 16, "lr": 0.01, "epochs": 2, "batch": 64, "seed": 0}, "metrics": {"val_rmse": [[1, 146.87], [2, 103.32]]}}
    ```

    Its last pair is `[2, 103.32]`: after epoch 2 the network is 103.32 bikes per hour off on the validation days, so a package of this version gets `"val_rmse": 103.32` in its config.json, and not 146.87.
    """)
    return


@app.cell(hide_code=True)
def text_hours():
    mo.md(r"""
    ### The hours of 2011

    The cell reads the hours with `read_hours` from your project, counts them with `len` and the days with `.nunique()`, shows the first three rows with `.head(3)`, and the busiest hour with `.max()`. Run the cell.
    """)
    return


@app.cell
def example_hours():
    from ops.demand import read_hours

    hours = read_hours()   # every hour of 2011, with a column day: 0 for 1 January
    print(len(hours), "hours of", hours["day"].nunique(), "days")
    print(hours[["dteday", "day", "hr", "weathersit", "temp", "cnt"]].head(3))
    print("busiest hour:", hours["cnt"].max(), "bikes")
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Whole days for validation

        `split_days(hours)` gets a table of hours with a column `day`, and gives back two tables: the training set and the validation set. The `return` line is given: `hours[val_day]` keeps the rows where the mask `val_day` is True, and `~` turns the mask around, so `hours[~val_day]` keeps all the other rows. Replace the `...` with the mask: True for every hour of the days 4, 9, 14 and so on, up to 364. `%` gives the remainder of a division: `14 % 5` is 4.

        The check calls `split_days(HOURS)`, with `HOURS` the 8,645 hours of 2011. It should give a training set of 6,912 hours and a validation set of 1,733 hours, from the days 4, 9, 14 and on. Then it tries three small tables of its own: ten days with one very busy hour, the last ten days of the year, and days of which some are missing. When it is right, the check saves your function in your project, as `work/projects/ops/split.py`.
        """),
        L.task("Make `val_day` True for every hour of every fifth day, the days 4, 9, 14 and so on."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=5)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "The mask compares a column of `hours` with a number for every row at once, as `hours[\"hr\"] == 8` would. Here the column is the day, `%` gives its remainder after dividing by 5, and `val_day` is True where that remainder is 4.",
            "Day 4 is 5 January, and day 9 is 10 January: `4 % 5`, `9 % 5` and `364 % 5` are all 4, and `0 % 5`, `5 % 5` and `10 % 5` are 0. All hours of one day have the same `day`, so a mask on the day keeps the hours of a day together.",
            "`    val_day = ___ == 4`",
        ],
        why="`val_day = hours[\"day\"] % 5 == 4`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train(result_split):
    result_split  # made again after the check of exercise 1, which saves your split
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A version as a tracked run

        `train_version(name, params, data, log_path, show=None)` trains one version and keeps it as a run. The first lines are given: `from ops.demand import DemandNet, train_epoch` takes the network and one epoch of training from your project, and `range` counts the epochs from 1. `data` holds four tensors: `x` and `y`, the inputs and the bikes of the training hours, and `x_val` and `y_val`, those of the validation hours. `torch.manual_seed` makes the run the same every time for the same `params["seed"]`. `DemandNet` makes the network, `model`, with `params["hidden"]` neurons, and `torch.optim.Adam` makes `opt`, an optimizer for `model.parameters()` that takes smarter steps than plain gradient descent, with the learning rate `params["lr"]`; a later lesson explains Adam. `run` is the dict of the run, as in [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en). In every `epoch`, `train_epoch` trains `model` on all training hours, in mini-batches of `params["batch"]`. Then `torch.no_grad()` measures without building a graph, `float` turns `rmse` into a number for the run with `.append`, and `show`, when it is given, gets the epoch, the model and the run: the demo draws with it.

        Replace the two `...`. First `rmse`: the validation RMSE of `model` on `x_val` and `y_val`, as a tensor with one number. `model(x_val)` and `y_val` both have the shape (n, 1). `.mean()` gives the mean of a tensor, and `.sqrt()` its square root. Then, after the loop, one line that adds `run` to the open file `f` as one line of JSON, with `json.dumps` and `f.write`. `open(log_path, "a")` opens the file at `log_path` to add to its end.

        The check calls `train_version("check", {"hidden": 16, "lr": 0.01, "epochs": 3, "batch": 64, "seed": 0}, DATA, LOG)`, with `DATA` the four tensors of the split of 2011, and `LOG` an empty file. The run should have `val_rmse` `[[1, 146.87], [2, 103.32], [3, 87.53]]`, to two decimals, and the file one line. `name` is the name of the run, here `"check"`. Then the check tries two small cases of its own, one with a file that already holds a run. When it is right, the check saves your function as `work/projects/ops/train.py`.
        """),
        L.task("Make `rmse` the validation RMSE of the model, and add the run to the file as one line of JSON."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=24)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_split):
    result_split  # check again when exercise 1 saves your split
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "`rmse` is built in three steps: the differences between `model(x_val)` and `y_val`, squared with `**`; then `.mean()`; then `.sqrt()`. The line for the file is `f.write`, with the text of `json.dumps(run)` and a line break `\"\\n\"` after it, so that the next run starts on a line of its own.",
            "In the worked example the differences are 20, 0, −30 and 10, the squares 400, 0, 900 and 100, their mean 350, and its square root 18.7. Without `.sqrt()`, the first epoch of the check would give about 21,570 instead of 146.87: the mean squared error, in bikes squared.",
            "`            rmse = (___).mean().sqrt()`",
        ],
        why="`rmse = ((model(x_val) - y_val) ** 2).mean().sqrt()`, and after the loop `f.write(json.dumps(run) + \"\\n\")`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pack(result_train):
    result_train  # made again after the check of exercise 2, which saves your function
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Pack the chosen version

        `pack(model, run, scale, folder)` puts one version in a package in `folder`. The first lines are given: `from ops.demand import FEATURES` and `from ops.safetensors_io import save_file` take two parts of your project, `folder.mkdir(parents=True, exist_ok=True)` makes the folder, `weights = folder / "model.safetensors"` is the path of a file in that folder (the `/` joins a folder and a name), `save_file` writes `model.state_dict()` to that file, and `open(weights, "rb")` opens it to read its bytes, as `f`. Under it, `config` holds what someone needs to use the weights: the class, the number of neurons `run["params"]["hidden"]`, how the inputs are made, the `FEATURES` from your project, the means and standard deviations in `scale`, what the output means, and the name of the run. `json.dump` writes it as `config.json`, with `indent=2`, so that a person can read it too. `hashlib` is the module of the hashes.

        Replace the two `...`. `version`: the first 12 characters of the sha256 of the bytes of the file, with `hashlib.sha256`, `.hexdigest()` and a slice, as in [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en). `val_rmse`: the validation RMSE of the last epoch of `run`. `run["metrics"]["val_rmse"]` is the list of pairs `[epoch, value]`.

        The check calls `pack(MODEL, RUN, SCALE, FOLDER)`: `MODEL` is a `DemandNet(16)` as it starts, so that the check needs no training; `RUN` the course's run of `h16-lr0.01`, with 30 epochs; `SCALE` the means and standard deviations of the training hours of 2011; and `FOLDER` an empty folder. The config should have `"val_rmse": 53.04`, and as its `version` the first 12 characters of the sha256 of `model.safetensors`. Then it tries two cases of its own: a run of three epochs, and a run whose lowest RMSE is not at its last epoch. When it is right, the check saves your function as `work/projects/ops/pack.py`.
        """),
        L.task("Make `version` the start of the sha256 of the weights file, and `val_rmse` the RMSE of the last epoch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pack():
    box_pack = L.code_box("pack", lines=27)
    box_pack
    return (box_pack,)


@app.cell(hide_code=True)
def check_pack(box_pack, result_train):
    result_train  # check again when exercise 2 saves your function
    result_pack = L.check("pack", box_pack.value)
    result_pack
    return (result_pack,)


@app.cell(hide_code=True)
def hints_pack():
    L.hints(
        "pack",
        [
            "Reading the open file `f` gives all its bytes. `hashlib.sha256` makes their hash, `.hexdigest()` writes it as 64 characters, and `[:12]` keeps the first 12; the result goes in `version`. For `val_rmse`, `[-1]` gives the last pair of a list, and `[1]` the value in a pair.",
            "The version must change as soon as one byte of `model.safetensors` changes. The path of the file stays the same for every model you pack there, and the way Python prints the weights shows only a few digits of each: neither says which bytes you hand over. `RUN` ends with the pair `[30, 53.04]`.",
            "`        version = hashlib.sha256(___).hexdigest()[:12]`",
        ],
        why="`version = hashlib.sha256(f.read()).hexdigest()[:12]`, and in the config `\"val_rmse\": run[\"metrics\"][\"val_rmse\"][-1][1],`.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_pack, result_split, result_train):
    result_split, result_train, result_pack  # made again after the checks, which save your parts
    mo.vstack([
        mo.md(r"""
        ### Train four versions, and pack the best

        The button trains four versions with your parts: your `split_days` makes the split, `scaling` works out the means and standard deviations on your training hours, and your `train_version` trains `h16-lr0.001`, `h16-lr0.01`, `h64-lr0.001` and `h64-lr0.01`, each for 30 epochs with seed 0, and adds each run to `work/projects/ops/runs.jsonl`. The chart shows the validation RMSE of the version that trains now, with the versions already done as thin dashed lines. Beside it you see what that version expects for each hour of Thursday 14 July 2011, a validation day, against the bikes that were really rented. Then the version with the lowest RMSE at its last epoch goes to your `pack`, which writes it to `work/projects/ops/model/`, and the course loads the package back with its own reader. When the run is done, you can go back through the epochs.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Train four versions and pack the best")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_pack, result_split, result_train, run_button):
    result_split, result_train, result_pack   # train again with your files once a check saves them
    trained = None
    if not run_button.value:
        _out = mo.md("*Press the button to train the four versions.*")
    else:
        _now = {}

        def _show(_name, _epoch, _rmse, _preds, _done):
            if _now.get("name") != _name:   # a new version: a chart of its own, with the versions done before it
                _chart = L.live(C.live_labels(L.lang), C.EPOCHS, role="loss", low=C.LOW, high=C.HIGH, window=1)
                for _other, _points in _done:
                    _chart.reference(_points, _other, role="loss")
                _chart.watch(lambda: C.now_picture(_now, L.lang))
                _now["chart"], _now["name"] = _chart, _name
            _now["epoch"], _now["preds"] = _epoch, _preds
            _now["chart"].add(_epoch, _rmse, C.progress_line(_name, _epoch, _rmse, L.lang))

        mo.output.replace(mo.md("**Splitting the hours** …"))
        _r = C.demo(show=_show)
        if isinstance(_r, str):
            _note = C.broken(_r, L.lang)
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            trained = _r
            _out = mo.vstack([
                L.explore("rewind", "epochs", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPOCHS,
                          low=1, high=C.EPOCHS, step=1, role="",
                          task="Go back to epoch 1, then forward to 10 and to 30. When does the line of what the best version expects for 14 July first show two peaks, and which two versions are still far above the other two at 30?"),
                mo.Html(C.package_picture(_r, L.lang)),
                mo.md(C.summary(_r, L.lang)),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        With the course's parts, the four versions end at 85.6 (`h16-lr0.001`), 53.0 (`h16-lr0.01`), 71.0 (`h64-lr0.001`) and 53.9 (`h64-lr0.01`) bikes per hour. `h16-lr0.01` wins, and its package holds 577 numbers in a file of 2,596 bytes. Is 53 good? A model that always says the mean of the training hours is 132.8 bikes off, and a rule that says the mean of the same hour on the same kind of day, working day or not, 79.1: the network is a third better than that rule. The two versions with learning rate 0.001 are far from done at epoch 30: whether they would catch up with more epochs, this run does not say.

        `h64-lr0.01` was lower once: 51.5 after epoch 29. But the network of epoch 29 is gone, and the one you have is 53.9 off. And the two versions with learning rate 0.01 lie within a bike per hour of each other, about as much as the seed moves one version: with seeds 0, 1 and 2, `h16-lr0.01` ends at 53.0, 52.8 and 53.5, and `h64-lr0.01` at 53.9, 53.5 and 53.5. `h16-lr0.01` is lower with seeds 0 and 1, and with seed 2 they tie.
        """),
        L.watch_out("The validation days chose the version, so its 53.0 is a little too good for new days: of four versions, you took the one that did best on these 73 days. Only data that no choice has seen says how far off the model is on a new day, such as the days of 2012, which wait for the last step of your project."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_scale():
    mo.md(r"""
    ### Exercise 4. The inputs in the service

    In the last step of your project, a service gets one hour at a time, and makes its 34 inputs with `inputs(hours, scale)` from your project before it asks the network.
    """)
    return


@app.cell(hide_code=True)
def choice_scale_cell():
    choice_scale = L.choice(
        "scale",
        question="Which means and standard deviations must the service give `inputs` as `scale`?",
        options={
            "own": "Those of the hour the request is about, so that the inputs fit that hour",
            "both": "Those of all the hours of 2011 and 2012 together: more hours give better means and standard deviations",
            "right": "Those in config.json: the means and standard deviations of the training hours of 2011",
            "none": "None: the service can give the network the raw values, since it learned what each column means",
        },
        correct="right",
        why={
            "right": "The network learned its weights on inputs made with these numbers: a warm Tuesday must become the same 34 numbers in the service as it did in training. So your `pack` writes them in config.json, beside the weights, and the service reads them from there.",
            "own": "One hour has one value in each column, so its standard deviation is 0, and dividing by 0 gives no number; and minus its own mean, every column would be 0 for every hour. With which numbers were the inputs made while the network learned?",
            "both": "With other numbers, the same temperature becomes another input, and the network answers for weather that was not there. With which numbers were the inputs made while the network learned?",
            "none": "The network learned on standardized inputs, most of them between −2 and 2. A raw value such as `mnth` 12 lies far outside that, where the network never learned anything. Which numbers make the inputs the way the network learned them?",
        },
    )
    choice_scale
    return (choice_scale,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en), [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en) and [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en), you learned each part on the course's models. Now your project `ops` has all of them together: a validation set of whole days, every version as a run in `runs.jsonl`, and the best version packed with the hash of its weights and its settings, ready to hand over. The next lesson, *Tests for ML code*, gives your project tests that catch a bug in training code before a long run. In *Project: tests and a model card* you test your model, and in *Project: your model as a service, watched for drift* it answers requests, and meets the hours of 2012.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scale, result_pack, result_split, result_train):
    L.wrap_up(result_split, result_train, result_pack, choice_scale)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
