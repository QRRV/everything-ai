# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-figure. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-figure", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Guided reproduction 3: sanity checks](/course/open/repro-guided-sanity?lang=en) and [Figures that show something](/course/open/res-figures?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_decay():
    quick_decay = L.choice(
        "quick-decay",
        question="Your sanity check learns one batch by heart without weight decay, and the paper's runs use a weight decay of 1.0. Why does the check leave it out?",
        options={
            "fast": "Weight decay makes each step slower, and a sanity check has to be done in a minute or two",
            "right": "The check asks only whether the model can learn, and weight decay pulls the other way",
            "paper": "The paper too leaves weight decay out for the first steps of a run",
        },
        correct="right",
        why={
            "fast": "Weight decay costs hardly any time: it shrinks every weight a little at each step. The check leaves it out because it works against learning a batch by heart. See [Guided reproduction 3](/course/open/repro-guided-sanity?lang=en).",
            "paper": "The paper's runs use weight decay from the first step on. The check leaves it out because it asks another question than the run does. See [Guided reproduction 3](/course/open/repro-guided-sanity?lang=en).",
            "right": "Weight decay keeps the weights small, against what learning one batch by heart needs. In the long run of this lesson it is back, at the paper's 1.0.",
        },
    )
    quick_decay
    return (quick_decay,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="You have three runs of one setting, each with another seed, and each a curve over the steps. What does a figure that shows something draw?",
        options={
            "best": "The run that looks best, so that the message is clear",
            "mean": "The mean of the three runs, as one curve",
            "right": "All three runs, in one style, so that the spread shows",
        },
        correct="right",
        why={
            "best": "The best run alone hides how much the runs differ, and makes the result look surer than it is. See [Figures that show something](/course/open/res-figures?lang=en).",
            "mean": "A mean can hide that one run did something else; with three runs, draw them all. See [Figures that show something](/course/open/res-figures?lang=en).",
            "right": "Every run, or the spread over seeds, belongs in the figure: the reader sees at once how much of the result is the seed.",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        The claim of the paper is a figure: Figure 1 of Power and others (2022), the accuracy on the training and the validation data against the steps on a log axis, with validation reaching 99% long after training. Your plan says how to read one run. A reproduction is a figure of your own, of the same quantities on the same kind of axes, drawn from logs you kept: then anyone can lay it next to the paper's and see whether it shows the same. Three things make that figure: a run long enough for validation to jump, logs at steps that cover a log axis evenly, and more than one seed, because one run can be luck.

        **An everyday example.** A time-lapse of a plant growing from a seed: in the first hours much changes, later little per hour. A camera that takes a photo after 1, 2, 5, 10, 20, 50 and 100 hours catches every stage with a few photos, and three pots side by side show which growth is typical and which is one pot's own.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           3, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_plateau():
    _r = C.read(C.course_runs()[0])
    guess_plateau = L.guess(
        "plateau-guess",
        question=f"In the course's run of seed 0, training accuracy reaches 99% at step {_r['t']}. What do you think validation accuracy does in the thousands of steps before it jumps?",
        options={
            "chance": "It stays at chance: about 1%, one right answer in 97",
            "flat": "It stays flat at about 30%",
            "climb": "It climbs slowly and steadily to 99%",
        },
        reveal="**It stays flat at about 30%,** from about step 300 to about 5,000, and then jumps. In the paper's Figure 1, validation stays low instead; exercise 5 asks why.",
    )
    guess_plateau
    return (guess_plateau,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## The idea

    To **redraw a figure** is to draw the paper's figure again from your own logs: the same quantities on the same kind of axes, so that the two can lie side by side. For Figure 1 that is the accuracy on all training and all validation equations, against the steps on a log axis. Your kit gets two parts for it.

    1. **A logger**, `run_seed`: one run, which after chosen steps measures the accuracy on all training and all validation equations, and keeps the step and the two accuracies in a log. Measuring all 9,409 equations takes as long as several training steps, so a run does not log after every step. On a log axis, the steps from 1 to 10 take as much room as those from 1,000 to 10,000, so the steps it logs are spread evenly on that axis: {C.PER_FACTOR} for every factor of 10, $10^{{k/{C.PER_FACTOR}}}$ rounded, for $k = 0, 1, 2, \dots$
    2. **A multi-seed runner**, `run_seeds`: one run for each seed, each added as one line to an experiment log. The seed fixes the split, the starting weights and the batches, so each run is another draw of all three. A seed that is in the log already is not run again: when your computer stops halfway, the runs that are done stay done.

    Below is the course's run of seed 0, with the paper's settings at {round(100 * C.FRACTION)}% of the table and {C.number(C.STEPS, 0, L.lang)} steps. It shows the whole run. As it plays, the moment the plan looks for appears once a curve reaches 99%: $t$ for training, $v$ for validation; at the end comes the plan's verdict.
    """)
    return


@app.cell(hide_code=True)
def play_replay_cell():
    _log = C.course_runs()[0]
    _r = C.read(_log)
    play_replay = play.Replay(
        steps=_log["step"], train=_log["train"], val=_log["val"], threshold=C.THRESHOLD, t_index=_r["t_index"], v_index=_r["v_index"],
        verdict=f"The plan's verdict: <b>{_r['verdict']}</b>. Validation reaches 99% {round(_r['times'])} times as late as training, and the plan asks for at least {C.DELAY}.",
        decimal="," if L.lang == "nl" else ".",
        labels={"task_kick": L.w["task"], "task": "Play the run, or drag across it. At which step does validation reach 99%, and how many times as late as training is that? What does validation do in between?",
                "title": "Accuracy during the course's run, seed 0", "x_name": "step", "train": "training", "val": "validation",
                "play": "Play", "pause": "Pause", "back": "Back", "fwd": "Forward", "step_name": "step",
                "alt": f"The accuracy on the training and the validation equations during the course's run of seed 0, over {C.number(C.STEPS, 0, L.lang)} steps on a log axis. Training rises from about 1% to 99% at step {_r['t']}; validation stays flat at about 30% and reaches 99% at step {C.number(_r['v'], 0, L.lang)}."})
    mo.ui.anywidget(play_replay)
    return


@app.cell(hide_code=True)
def text_idea_after():
    _r = C.read(C.course_runs()[0])
    _low, _high = C.plateau(C.course_runs()[0])
    mo.vstack([
        mo.md(rf"""
        Training reaches 99% at step {_r['t']}, validation at step {C.number(_r['v'], 0, L.lang)}: {round(_r['times'])} times as late. In between, validation lies flat, between {C.number(100 * _low, 0, L.lang)}% and {C.number(100 * _high, 0, L.lang)}% from step 500 to step 4,000, while training is at 100%; then it jumps to 100% in a few thousand steps. That is the claim of the paper, in a run of your kit's own settings.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **redraw a figure** | draw the paper's figure again from your own logs, with the same quantities on the same kind of axes | `plt.plot(log["step"], log["val"], "--")` |
        | **logger** | the part of a run that measures and keeps, after chosen steps, what the figure needs | `log["val"].append(...)` |
        | log-spaced steps | steps spread evenly on a log axis: as many from 10 to 100 as from 1,000 to 10,000 | `log_steps(12000, 20)` |
        | **multi-seed runner** | runs the same experiment once for each seed, and keeps every run | `run_seeds(settings, [0, 1, 2], steps, path)` |
        | $t$, $v$ | the first logged steps at which training and validation reach 99%: the moments of your plan | `first_step(...)` |
        """),
        L.watch_out(f"A logged moment is a little late. The plan reads $t$ and $v$ at logged steps, and the real moment lies between a logged step and the one before it. With {C.PER_FACTOR} logged steps for every factor of 10, each is $10^{{1/{C.PER_FACTOR}}} = 1.12$ times the one before: $v$ and $t$ are each at most 12% late, so $v / t$ can be off by about that much either way. At {round(_r['times'])} against the plan's {C.DELAY}, that does not change the verdict; close to {C.DELAY}, log more often."),
        mo.md(r"""
        ### Recipe

        To redraw a figure of training curves from your own runs:

        1. Choose the steps to log: spread evenly over the axis the figure uses, with the last step among them.
        2. For each seed, train, and after each of those steps measure what the figure shows, on all the data it is about.
        3. Keep every run in an experiment log, one line per run, so that the runs that are done stay done.
        4. Draw every run on the axes of the figure, with a line style of its own for each quantity, such as training solid and validation dashed, and with its message in the title.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Why not log every 500 steps? The cell below works out what that gives for a run of 12,000 steps, and how much of a log axis it leaves empty: step 1 of the recipe. Run it.
    """)
    return


@app.cell
def example_every():
    import math

    _every = list(range(500, 12_001, 500))   # 1. logging every 500 steps
    print("logged steps:", len(_every), "the first at step", _every[0])
    # the share of a log axis from step 1 to 12,000 that lies before step 500
    print("share before step 500:", round(math.log10(500) / math.log10(12_000), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Logging every 500 steps gives 24 logs, the first at step 500. On a log axis from step 1 to 12,000, two thirds of the width, 0.66, lies before step 500, and there training goes from 1% to 100%: that part of the figure would be empty. Log-spaced steps fill it, with as many logs from 1 to 10 as from 1,000 to 10,000.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.vstack([
        mo.md(rf"""
        ## Your turn

        ### Exercise 1. Steps on a log axis

        Complete `log_steps(budget, per_factor)`: the steps at which a run logs. For $k = 0, 1, 2, \dots$ the step is $10^{{k / \text{{per\_factor}}}}$, rounded to a whole step, as long as that is not above `budget`; at the start several values of $k$ round to the same step, and each step comes in the list once. The last step of the run, `budget`, comes at the end, if it is not there yet. The check calls `log_steps(100, 5)`, which should give `[1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100]`, and other budgets. When it is right, it is saved in your project as `repro/log.py`.
        """),
        L.task("Complete `log_steps`: a `while` loop over `k` that adds each new rounded step, then the budget."),
    ])
    return


@app.cell(hide_code=True)
def exercise_log():
    box_log = L.code_box("log-steps", lines=14)
    box_log
    return (box_log,)


@app.cell(hide_code=True)
def check_log(box_log):
    result_log = L.check("log-steps", box_log.value)
    result_log
    return (result_log,)


@app.cell(hide_code=True)
def hints_log():
    L.hints(
        "log-steps",
        [
            "`10 ** (k / per_factor)` is a float, and `round` makes it a whole step. A `while` loop goes on as long as its condition holds: which condition stops it at the budget?",
            "With `per_factor = 20`, `k = 0` to `3` all round to 1. Before `append`, test whether the step is in `steps` already, with `not in`; and do not forget `k = k + 1`.",
        ],
        why="With 20 logged steps for every factor of 10, a run of 12,000 steps logs 72 times, 43 of them before step 500: logging every 500 steps would log 24 times, none of them there.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The logger

        Complete `accuracy(model, x, y)` and the logging in `run_seed(settings, seed, steps)`. The rest of the run is given: the split, the tokens and the model as in [Guided reproduction 2](/course/open/repro-guided-build?lang=en), AdamW with the paper's settings and its warmup, and batches drawn by a generator of their own. `accuracy` gives the share of the equations whose guess at the last place is the answer, as a number. After each step in `at`, the loop adds to `log` the step, the accuracy on all training equations and that on all validation equations. The check runs it with small settings, addition modulo 7, and compares the logs with the course's. When it is right, it is saved in your project as `repro/run.py`.
        """),
        L.task("Complete `accuracy` and the logging: after each step in `at`, the step and both accuracies."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run-seed", lines=44)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run-seed", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run-seed",
        [
            "The guess is the token with the highest logit: `.argmax(-1)` of the logits at the last place. `guess == y` is a tensor of `True` and `False`; as floats, their mean is the share that is right.",
            "`log[\"step\"]`, `log[\"train\"]` and `log[\"val\"]` are lists: `append` to each, only when `step in at`. The training equations are `xt` and `yt`, the validation equations `xv` and `yv`.",
        ],
        why="Measured on all equations, the accuracy of a step is the same number each time you run this seed; measured on a random batch, it would jump with the batch.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_seeds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The multi-seed runner

        Complete `run_seeds(settings, seeds, steps, path)`: for each seed that is not in `done`, one run of your `run_seed`, with `device` and `progress` passed on, added to the file at `path` as one line of JSON, as in [Keeping an experiment log](/course/open/res-experiment-log?lang=en). Reading the seeds that are done is given: `done` holds them, and is empty when the log is new. The check runs it with a quick stand-in for `run_seed` that does not train, in a folder of its own. When it is right, it is saved in your project as `repro/seeds.py`.
        """),
        L.task("Complete `run_seeds`: run each seed that is not in `done`, and add its log to the file as one line."),
    ])
    return


@app.cell(hide_code=True)
def exercise_seeds():
    box_seeds = L.code_box("run-seeds", lines=24)
    box_seeds
    return (box_seeds,)


@app.cell(hide_code=True)
def check_seeds(box_seeds):
    result_seeds = L.check("run-seeds", box_seeds.value)
    result_seeds
    return (result_seeds,)


@app.cell(hide_code=True)
def hints_seeds():
    L.hints(
        "run-seeds",
        [
            "`if seed not in done:` runs the indented lines only for a seed that is not done yet: the run, and writing its line.",
            "Open the file with `\"a\"`, so that each run comes after the ones before it; one line is `json.dumps(log)` with a newline at its end.",
        ],
        why="A run that stops halfway, because the computer sleeps or the lesson closes, loses only the seed it was on: the next time, `run_seeds` starts with that seed.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_figure():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Redraw Figure 1

        Draw Figure 1 again, from the course's logs of seeds 0, 1 and 2 in `data/course-grokking.jsonl`: for each run, its training accuracy solid and its validation accuracy dashed, against `log["step"]`, on a log axis of steps, with the one message of the figure as its title and a label on both axes. The legend is given: it names the first two lines. The check compares your lines with the course's figure. Once your own runs are in your experiment log, the same code draws them: only the file is another.
        """),
        L.task("Draw both curves of every run against its steps, on a log x-axis, with a title that says what all three show."),
    ])
    return


@app.cell(hide_code=True)
def exercise_figure():
    box_figure = L.code_box("figure", lines=16)
    box_figure
    return (box_figure,)


@app.cell(hide_code=True)
def check_figure(box_figure):
    result_figure = L.check("figure", box_figure.value)
    result_figure
    return (result_figure,)


@app.cell(hide_code=True)
def hints_figure():
    L.hints(
        "figure",
        [
            "`plt.plot(xs, ys, \"--\")` draws a dashed line, and `\"-\"` a solid one. In the loop, `log` is one run: draw two lines for it, training first.",
            "The x-axis of Figure 1 is a log axis: `plt.xscale(\"log\")`. What do all three runs show, in one sentence? That is the title.",
        ],
        why="With every run in the figure, a reader sees at once how little the jump depends on the seed: in all three, validation reaches 99% 45 to 50 times as late as training.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_plateau():
    mo.md(r"""
    ### Exercise 5. Why 30%?

    Your figure shows more than the claim. Two of the runs dip for a moment, training too, and pick up again at the next logged step: a short setback of training, worth a line in your report. And validation lies flat for thousands of steps, at a height Figure 1 does not have.
    """)
    return


@app.cell(hide_code=True)
def choice_plateau_cell():
    choice_plateau = L.choice(
        "plateau",
        question=f"In your figure, validation lies flat at about 30% for thousands of steps while training is at 100%; in the paper's Figure 1, of division, it stays low. In the split of seed 0, {C.number(100 * C.MIRROR['share'], 1, L.lang)}% of the validation equations have their mirror, b + a, in the training data. Why does validation stop at about 30%?",
        options={
            "leak": "About 30% of the validation equations are in the training data too, so it has seen them",
            "chance": "With 97 possible answers, 30% is what guessing gives",
            "part": "The model has learned 30% of the rule, because it trains on 30% of the table",
            "right": "It gets right the equations whose mirror b + a it trained on, and almost no others",
        },
        correct="right",
        why={
            "leak": "Your `dataset` splits the table: each equation is in the training data or in the validation data, never in both. What about 30% of the validation equations do have is their mirror in the training data.",
            "chance": "Guessing among 97 answers gets one in 97 right, about 1%. 30% is far above chance: the model knows something about those equations.",
            "part": f"A model that knew part of the rule would get a part of every kind of equation right. At step {C.number(C.MIRROR['step'], 0, L.lang)} of seed 0, it gets {C.number(100 * C.MIRROR['mirrored'], 1, L.lang)}% of the mirrored validation equations right and {C.number(100 * C.MIRROR['others'], 1, L.lang)}% of the others: it knows the mirror, not the rule.",
            "right": f"At step {C.number(C.MIRROR['step'], 0, L.lang)} of seed 0, the model gets {C.number(100 * C.MIRROR['mirrored'], 1, L.lang)}% of the validation equations with their mirror in the training data right, and {C.number(100 * C.MIRROR['others'], 1, L.lang)}% of the others. Division has no such mirror, so Figure 1 has no such plateau: a difference between your figure and the paper's, with its cause, for your report.",
        },
    )
    choice_plateau
    return (choice_plateau,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### What your project can do now

    The button trains seeds 0, 1 and 2 with your kit, one after another: your `run_seeds` with your `run_seed`, on your ModernGPT with your data and your settings, {C.number(C.STEPS, 0, L.lang)} steps each, into your experiment log `work/projects/repro/runs.jsonl`. You see each run while it trains. On the course's CPU, with one thread, a seed takes about {round(C.STEPS * C.SECONDS_PER_STEP / 60)} minutes, all three about {round(3 * C.STEPS * C.SECONDS_PER_STEP / 60)}; on a GPU, a few minutes. That is where your runner earns its keep: close the lesson halfway, and the seeds that are done stay in your log; the next press goes on with the others. While it trains, the rest of this lesson waits: you can read on, or work in another lesson. Where a part of your kit is missing or broken, the course's version stands in.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train seeds 0, 1 and 2 with your kit")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_run, result_seeds):
    import time as _time

    result_run, result_seeds  # run again when an exercise saves a part of the kit
    _ready = result_run.state == "correct" and result_seeds.state == "correct"
    _labels = {"x": "step", "y": "accuracy", "train": "training", "val": "validation"}
    trained = None
    if not train_button.value:
        _out = mo.md("*Press the button to train your seeds with your kit.*" if _ready else "*Finish exercises 2 and 3 first, then press the button: it trains with your `run_seeds` and your `run_seed`.*")
    elif set(C.SEEDS) <= {_log["seed"] for _log in C.my_runs() or []}:
        _out = mo.md("*Seeds 0, 1 and 2 are in your experiment log already, so your `run_seeds` skips them. To train them again, delete `work/projects/repro/runs.jsonl`.*")
    else:
        _began = {}

        def _progress(_log):
            _step, _now = _log["step"][-1], _time.perf_counter()
            _spent = _now - _began.setdefault(_log["seed"], _now)
            _left = f", about {round(_spent / _step * (C.STEPS - _step) / 60)} to go" if _step >= 1000 else ""   # before step 1,000, the many logs of the start make a guess too high
            mo.output.replace(mo.vstack([
                mo.md(f"**Seed {_log['seed']}, step {C.number(_step, 0, L.lang)} of {C.number(C.STEPS, 0, L.lang)}** · {C.number(_spent / 60, 1, L.lang)} minutes{_left} · training {C.number(100 * _log['train'][-1], 1, L.lang)}%, validation {C.number(100 * _log['val'][-1], 1, L.lang)}%"),
                mo.Html(C.figure_picture([_log], {**_labels, "title": f"Accuracy during your run, seed {_log['seed']}", "alt": f"Your run of seed {_log['seed']} up to step {_step}: training at {C.number(100 * _log['train'][-1], 1, L.lang)}%, validation at {C.number(100 * _log['val'][-1], 1, L.lang)}%."})),
            ]))

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `run_seed` and your `run_seeds` are in your project, and this cell uses them.*")
        elif isinstance(_r, str):
            _out = mo.md({"seeds": "*Your `repro/seeds.py` or `repro/run.py` stops with an error. Did you change it after its check? Delete it, and check its exercise again.*",
                          "model": "*Your ModernGPT, your data or your settings stop with an error. Did you change `llm/modern_gpt.py`, `repro/data.py` or `repro/settings.py` after its check? Delete the file you changed, and check its exercise again.*"}[_r])
        else:
            trained = _r
            _stand = (" The course's version stood in for: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
            _out = mo.md(f"*Done: seeds 0, 1 and 2 are in your experiment log, `work/projects/repro/runs.jsonl`.{_stand}*")
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_mine(trained):
    trained  # read the experiment log again after a run
    _mine = C.my_runs()
    if not _mine:
        _out = mo.md("*Your runs play here once they are in your experiment log.*")
    else:
        _log = _mine[0]
        _r = C.read(_log)
        _seen = [(_l["seed"], C.read(_l)) for _l in _mine]
        _when = lambda _s: C.number(_s, 0, L.lang) if _s is not None else "not reached"
        _rows = "\n".join(f"| {_seed} | {_when(_x['t'])} | {_when(_x['v'])} | {C.number(_x['times'], 0, L.lang) if _x['times'] else '–'} | {_x['verdict']} |" for _seed, _x in _seen)
        _out = mo.vstack([
            mo.ui.anywidget(play.Replay(
                steps=_log["step"], train=_log["train"], val=_log["val"], threshold=C.THRESHOLD, t_index=_r["t_index"], v_index=_r["v_index"],
                verdict=f"The plan's verdict: <b>{_r['verdict']}</b>.", decimal="," if L.lang == "nl" else ".",
                labels={"task_kick": L.w["task"], "task": f"Play your run of seed {_log['seed']}: does it show what the course's run showed?",
                        "title": f"Accuracy during your run, seed {_log['seed']}", "x_name": "step", "train": "training", "val": "validation",
                        "play": "Play", "pause": "Pause", "back": "Back", "fwd": "Forward", "step_name": "step",
                        "alt": f"The accuracy on the training and the validation equations during your run of seed {_log['seed']}, over {C.number(_log['step'][-1], 0, L.lang)} steps on a log axis. The plan's verdict: {_r['verdict']}."})),
            mo.md(f"""
| seed | $t$ | $v$ | $v / t$ | the plan's verdict |
|---|---|---|---|---|
{_rows}

To draw them, copy your figure code of exercise 4 into a cell of your own and let it open `work/projects/repro/runs.jsonl`.
"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Your kit now runs every seed of a setting into one experiment log, and your figure code draws whatever that log holds.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The plan of [Guided reproduction 1](/course/open/repro-guided-claim?lang=en), now read in real runs; every run in one figure, as in [Figures that show something](/course/open/res-figures?lang=en); one line per run, as in [Keeping an experiment log](/course/open/res-experiment-log?lang=en); and the mirror cells of the table of [Guided reproduction 2](/course/open/repro-guided-build?lang=en), now visible as a plateau. Your kit now has `repro/log.py`, `repro/run.py` and `repro/seeds.py`. The next lesson, *Guided reproduction 5: the ablation*, turns weight decay off and on and varies the training fraction, to find which part the jump needs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_log, result_run, result_seeds, result_figure, choice_plateau):
    L.wrap_up(result_log, result_run, result_seeds, result_figure, choice_plateau)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
