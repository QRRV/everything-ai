# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-ablation. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-ablation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en) and [Guided reproduction 4: reproduce the key figure](/course/open/repro-guided-figure?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_two():
    quick_two = L.choice(
        "quick-two",
        question="Your reproduction uses the paper's warmup of 10 steps. You want to know whether the jump needs it. Which runs answer that?",
        options={
            "right": "The run with your kit's settings, and the same run with the warmup taken out",
            "two": "A run without warmup and without weight decay, against the run with your kit's settings",
            "seeds": "Two runs without warmup, one with seed 0 and one with seed 1",
        },
        correct="right",
        why={
            "two": "Two parts are out at once, so a difference could come from either. Take out one part, and compare with the control run. See [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).",
            "seeds": "Both runs lack the warmup, so neither shows what the warmup does; together they show how much the seed moves a run. See [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).",
            "right": "An ablation takes one part out and compares with the control run, the same in every other way: a difference then comes from that part.",
        },
    )
    quick_two
    return (quick_two,)


@app.cell(hide_code=True)
def quick_check_close():
    quick_close = L.choice(
        "quick-close",
        question="A run gives t = 200 and v = 1,995, read at logged steps, 20 for every factor of 10. Your plan asks for a delay of at least 10 times. What do you say?",
        options={
            "sure": "No delay, and a sure one: 9.98 is below the plan's 10, whatever the logs missed",
            "delay": "A delay: validation needs about 1,800 more steps than training did, which is long",
            "right": "No delay, but close: either moment can be up to 12% late",
        },
        correct="right",
        why={
            "sure": "Each logged moment can be up to 12% late, so 1,995 / 200 = 9.98 could really be anything from about 8.9 to 11.2. By the plan it is no delay, but a close one. See [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).",
            "delay": "The plan counts how many times as late, not how many steps more: 1,995 is 9.98 times 200, just below its 10. See [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).",
            "right": "By the plan it is no delay; but with logged moments up to 12% late, a ratio of 9.98 is a close call, worth more logs or more seeds.",
        },
    )
    quick_close
    return (quick_close,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your figure shows the claim in your own runs. But the paper says more than its key figure: Figure 1 (center) shows that the time until validation reaches 99% grows quickly as the fraction of training data shrinks, and Figure 2 (left) that weight decay "improves generalization the most". Which of the two does the jump in your reproduction need? An ablation finds out: change one setting, keep the rest as it was, and compare with the control run. Here two settings, each at a few values, in every combination: so that you also see whether one of them matters only at some values of the other.

        **An everyday example.** Does the bread rise because of the yeast, or because of the warm spot by the heater? Bake with and without yeast, in the warm spot and in a cold one: four loaves. If only the one with yeast in the warm spot rises, it needs both; if both loaves with yeast rise, the yeast is enough.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           4, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_decay():
    _r = C.read(C.run(1.0, C.FRACTION))
    guess_decay = L.guess(
        "no-decay",
        question=f"In the course's run at {round(100 * C.FRACTION)}% of the table, validation reached 99% at step {C.number(_r['v'], 0, L.lang)}. With weight decay off, and everything else the same, what do you think happens within the {C.number(C.STEPS, 0, L.lang)} steps?",
        options={
            "later": "Validation reaches 99% later, but within the 12,000 steps",
            "same": "Validation reaches 99% at about the same step",
            "never": "Validation does not reach 99% within the 12,000 steps",
        },
        reveal=f"**It does not reach 99% within the budget:** after {C.number(C.STEPS, 0, L.lang)} steps, validation is at {C.number(100 * C.run(0.0, C.FRACTION)['val'][-1], 0, L.lang)}%. What \"within the budget\" means, and at which fractions it does reach 99%, *The idea* shows.",
    )
    guess_decay
    return (guess_decay,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## The idea

    An **ablation grid** runs every combination of a few values of each setting you change: weight decay 1 or 0, times the fractions 30%, 50% and 90%, makes 6 runs. One of them is the control run, with your kit's settings: the paper's weight decay of 1, at {round(100 * C.FRACTION)}% of the table. Every line of the experiment log keeps the values its run changes, so that you can read the grid back as a table.

    Each run has a **budget**: here {C.number(C.STEPS, 0, L.lang)} steps. When validation does not reach 99% within it, the plan says "not reached": not within this budget, which is not the same as never. The run of the paper's own Figure 1 (left), division modulo 97 at 50% of the table, was trained without weight decay, with Adam and a budget of $10^6$ steps (Appendix A.1.2), and there validation took close to $10^6$ steps to catch up with training.

    Below are the course's runs, seed 0, with weight decay (solid) and without (dashed), at fractions from 30% to 90%: the validation accuracy over the steps, and a dot where it reaches 99%. Move the slider to set the fraction.
    """)
    return


@app.cell(hide_code=True)
def explore_fraction_cell():
    def _frame(fraction):
        _f = round(fraction / 100, 1)
        _on, _off = C.read(C.run(1.0, _f)), C.read(C.run(0.0, _f))
        _say_on = (f"reaches 99% at step {C.number(_on['v'], 0, L.lang)}, {C.number(_on['times'], 1, L.lang)} times as late as training" if _on["v"] is not None
                   else f"ends at {C.number(100 * C.run(1.0, _f)['val'][-1], 0, L.lang)}%")
        _say_off = (f"reaches 99% at step {C.number(_off['v'], 0, L.lang)}, {C.number(_off['times'], 1, L.lang)} times as late as training" if _off["v"] is not None
                    else f"ends at {C.number(100 * C.run(0.0, _f)['val'][-1], 0, L.lang)}% after {C.number(C.STEPS, 0, L.lang)} steps")
        return (C.val_picture(_f, {"title": f"Validation at {round(100 * _f)}% of the table", "x": "step", "y": "validation accuracy",
                                   "with": "with weight decay", "without": "without weight decay",
                                   "alt": f"Validation accuracy over {C.number(C.STEPS, 0, L.lang)} steps on a log axis, at {round(100 * _f)}% of the table. With weight decay, validation {_say_on}; without, it {_say_off}."}),
                f"At {round(100 * _f)}%, with weight decay, validation {_say_on}; without weight decay, it {_say_off}.")

    explore_fraction = L.explore("fraction", "% of the table", _frame, start=round(100 * C.FRACTION), low=30, high=90, step=10, role="hyper",
                                 task="From which fraction on does validation reach 99% without weight decay, within the budget? And with weight decay, from which fraction on is it less than 10 times as late as training?")
    explore_fraction
    return


@app.cell(hide_code=True)
def text_idea_after():
    _no = C.first_reached(0.0)
    _close = next(_f for _f in C.FRACTIONS if C.read(C.run(1.0, _f))["verdict"] == "no delay")
    mo.vstack([
        mo.md(rf"""
        Without weight decay, validation reaches 99% within the budget only from {round(100 * _no)}% on, and late even there: {C.number(C.read(C.run(0.0, _no))['times'], 1, L.lang)} times as late as training. Below that it ends somewhere between {C.number(100 * C.run(0.0, C.FRACTIONS[0])['val'][-1], 0, L.lang)}% and {C.number(100 * C.run(0.0, round(_no - 0.1, 1))['val'][-1], 0, L.lang)}%. With weight decay it reaches 99% at every fraction, and the smaller the fraction, the further it lags behind training: {C.number(C.read(C.run(1.0, C.FRACTION))['times'], 1, L.lang)} times as late at {round(100 * C.FRACTION)}%, and from {round(100 * _close)}% on less than 10 times. So within this budget, weight decay brings the jump sooner at every fraction, and below {round(100 * _no)}% it makes the difference between a jump and none. At {round(100 * _close)}%, the ratio is {C.number(C.read(C.run(1.0, _close))['times'], 1, L.lang)}: a close call.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **ablation grid** | every combination of a few values of each setting you change, each run against the same control run | `grid(settings, changes)` |
        | **budget** | the most steps a run gets; a result says what happened within it | `steps` |
        | not reached | validation did not reach 99% within the budget, which is not the same as never | `"not reached"` |
        """),
        L.watch_out("One seed per combination is one roll of the dice. A clear difference, such as a jump against none, survives that; a close call does not: at the fraction where the delay comes near 10, another seed can tip the verdict either way. The next lesson repeats the runs over seeds."),
        mo.md(r"""
        ### Recipe

        To find out which settings an effect needs:

        1. Choose the settings to change, and a few values of each, around those of the paper.
        2. Count the runs: the numbers of values multiplied, times the time of one run.
        3. Run every combination with the same budget, seed and code, and keep the values each run changes with its log.
        4. Read each run with your plan, and lay the verdicts out as a table: one setting down, the other across.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    How long does the grid take? The cell below counts the runs of weight decay 1 or 0 times three fractions, and of the course's grid with seven fractions, at the course's seconds per step on one thread: step 2 of the recipe. Run it.
    """)
    return


@app.cell
def example_count():
    _decays = [1.0, 0.0]
    _fractions = [0.3, 0.5, 0.9]
    _runs = len(_decays) * len(_fractions)   # 2. the numbers of values multiplied
    _hours = _runs * 12_000 * 0.113 / 3600   # steps per run, seconds per step
    print("runs:", _runs, "hours:", round(_hours, 1))
    _runs = len(_decays) * 7                 # the course's grid: seven fractions
    print("runs:", _runs, "hours:", round(_runs * 12_000 * 0.113 / 3600, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Six runs take 2.3 hours on the course's CPU, and with seven fractions, 14 runs, 5.3 hours: every value you add to one setting adds a run for every value of the other. On a GPU each run takes a few minutes. Choose the values that answer your question, and no more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_grid():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Every combination

        Complete `grid(settings, changes)`: every combination of the values in `changes`, each a copy of `settings` with those values. `changes` holds a list of values for each key it changes. Start from one copy of `settings`; for each key, make a new list from every run so far, once for each value of that key, and let `runs` become that list. So the first key changes slowest, as in a table read row by row. The check calls `grid({"lr": 0.001, "weight_decay": 1.0, "batch": 512}, {"lr": [0.001, 0.003], "batch": [256, 512]})`, which should give 4 settings, and other grids. When it is right, it is saved in your project as `repro/grid.py`.
        """),
        L.task("Complete `grid`: for each key, a copy of every run so far for every value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grid():
    box_grid = L.code_box("grid", lines=13)
    box_grid
    return (box_grid,)


@app.cell(hide_code=True)
def check_grid(box_grid):
    result_grid = L.check("grid", box_grid.value)
    result_grid
    return (result_grid,)


@app.cell(hide_code=True)
def hints_grid():
    L.hints(
        "grid",
        [
            "Two loops inside the loop over the keys: the outer one over `runs`, the inner one over `changes[key]`. Each round of the inner loop adds one copy to `new`.",
            "`dict(run)` makes a copy, so changing it leaves `run` as it was; `one[key] = value` sets the value in the copy. After both loops, `runs = new`.",
        ],
        why="Copying matters: `one = run` would be the same dictionary, and every run made from it would change along.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run_grid():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The ablation runner

        Complete `run_grid(settings, changes, steps, path)`: for each combination of `grid(settings, changes)` whose values are not in `done`, one run of your `run_seed` with seed 0, `device` and `progress` passed on, with the values of the keys added to its log, and the log added to the file at `path` as one line of JSON. `values` is the list of its values of the keys, in the order of `keys`, as each item of `done` is. The rest is given, as in your `run_seeds`. The check runs it with a quick stand-in for `run_seed` that does not train, in a folder of its own. When it is right, it is saved in your project as `repro/ablation.py`.
        """),
        L.task("Complete `run_grid`: run each combination that is not done, add its values to its log, and add the log to the file."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_grid():
    box_run_grid = L.code_box("run-grid", lines=24)
    box_run_grid
    return (box_run_grid,)


@app.cell(hide_code=True)
def check_run_grid(box_run_grid):
    result_run_grid = L.check("run-grid", box_run_grid.value)
    result_run_grid
    return (result_run_grid,)


@app.cell(hide_code=True)
def hints_run_grid():
    L.hints(
        "run-grid",
        [
            "`values = [one[key] for key in keys]` gives the values of this combination in the order of `done`. Then `if values not in done:` as in your `run_seeds`.",
            "The run trains with `one`, not with `settings`: `run_seed(one, 0, steps, device, progress)`. Before writing, `log[key] = one[key]` for each key, so the line says which combination it is.",
        ],
        why="Without its values in the log, a line cannot say which combination it was, and the table cannot be read back.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_figure():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The moments against the fraction

        Draw the course's runs with weight decay as the paper's Figure 1 (center) draws its runs: for each fraction, `t` and `v`, the first logged steps at which training and validation reach 99%, with `first_step` and `THRESHOLD` of your plan. Draw `t` and `v` against the fractions, each a line with dots, `t` first and solid, `v` dashed, on a log y-axis, with the one message of the figure as its title and a label on both axes. The legend is given. The check compares your lines with the course's figure.
        """),
        L.task("Collect the fraction, t and v of each run with weight decay, and draw t and v against the fraction on a log y-axis."),
    ])
    return


@app.cell(hide_code=True)
def exercise_figure():
    box_figure = L.code_box("figure", lines=20)
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
            "`first_step(log[\"step\"], log[\"train\"], THRESHOLD)` gives `t` of one run; with `log[\"val\"]` it gives `v`. Append them, and `log[\"fraction\"]`, to the three lists.",
            "`plt.plot(fractions, t, \"o-\")` draws a line with dots; `\"o--\"` a dashed one. `plt.yscale(\"log\")` makes the y-axis a log axis.",
        ],
        why="On a log y-axis the distance between the lines is how many times as late validation is: the gap your plan reads.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shows():
    mo.md(r"""
    ### Exercise 4. What does it show?
    """)
    return


@app.cell(hide_code=True)
def choice_shows_cell():
    _no = C.first_reached(0.0)
    choice_shows = L.choice(
        "what-shows",
        question=f"The course's ablation, seed 0, {C.number(C.STEPS, 0, L.lang)} steps per run, fractions 30% to 90%: with weight decay, validation reaches 99% at every fraction; without weight decay, only from {round(100 * _no)}% on, and later than with it. What may your report say?",
        options={
            "right": "Within this budget, weight decay brings the jump sooner, and below " + f"{round(100 * _no)}% decides whether there is one at all",
            "never": "Without weight decay, this model never generalises below " + f"{round(100 * _no)}% of the table",
            "fraction": "Only the fraction of the table matters for the jump; weight decay just makes every run a little faster",
        },
        correct="right",
        why={
            "never": "\"Not reached\" means not within this budget. The run of the paper's Figure 1 (left) had no weight decay either, and validation got there after close to a million steps.",
            "fraction": f"Below {round(100 * _no)}%, the runs without weight decay do not reach 99% within the budget at all, and those with it do: within this budget, weight decay makes the difference between a jump and none.",
            "right": "That is what the grid shows, and no more: one seed per combination, one budget, one task. The next lesson repeats the runs over seeds and compares them with the paper.",
        },
    )
    choice_shows
    return (choice_shows,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### What your project can do now

    The button trains the grid with your kit: your `run_grid` with your `grid` and your `run_seed`, on your ModernGPT with your data and your settings, weight decay 1 and 0 at 30%, 50% and 90%, {C.number(C.STEPS, 0, L.lang)} steps each, into your experiment log `work/projects/repro/ablation.jsonl`. A chart shows each run while it trains: its validation accuracy, with its training accuracy dashed, on a log scale of the steps as in the figure above, and the run's weight decay and fraction in its title. On a GPU the six runs take a few minutes each; on the course's CPU, with one thread, about {C.number(6 * C.STEPS * C.SECONDS_PER_STEP / 3600, 1, L.lang)} hours together. Close the lesson halfway, and the runs that are done stay in your log; the next press goes on with the others. While it trains, the rest of this lesson waits: you can read on, or work in another lesson. Where a part of your kit is missing or broken, the course's version stands in.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train the grid with your kit")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_grid, result_run_grid):
    import time as _time

    result_grid, result_run_grid  # run again when an exercise saves a part of the kit
    _ready = result_grid.state == "correct" and result_run_grid.state == "correct"
    trained = None
    if not train_button.value:
        _out = mo.md("*Press the button to train the grid with your kit.*" if _ready else "*Finish exercises 1 and 2 first, then press the button: it trains with your `run_grid` and your `grid`.*")
    elif len(C.my_runs() or []) >= 6:
        _out = mo.md("*The six runs of the grid are in your experiment log already, so your `run_grid` skips them. To train them again, delete `work/projects/repro/ablation.jsonl`.*")
    else:
        _kept = [(_l["weight_decay"], _l["fraction"]) for _l in C.my_runs() or []]
        _todo = [(_wd, _f) for _wd in C.CHANGES["weight_decay"] for _f in C.CHANGES["fraction"] if (_wd, _f) not in _kept]   # what this press trains, in the order of the grid
        _starts, _lives = [], {}   # when each run of this press started; its live chart

        def _progress(_log):
            _step, _now = _log["step"][-1], _time.perf_counter()
            if _step == 1:   # the first logged step of a new run
                _starts.append(_now)
            _spent, _n = _now - _starts[-1], len(_starts)
            if _n not in _lives:   # its settings; none only when your run_grid trains a run that is in the log already
                _title = f"Weight decay {C.number(_todo[_n - 1][0], 0, L.lang)} at {round(100 * _todo[_n - 1][1])}% of the table" if _n <= len(_todo) else f"Run {_n} of this press"
                _lives[_n] = L.live({"title": _title, "x": "step", "y": "accuracy", "alt": f"{_title}: the validation accuracy so far, on a log scale of the steps; dashed, the training accuracy."},
                                    C.STEPS, role="ink", low=0.0, high=1.15, window=1, xlog=True)   # room above 1 for the name of the training line
            _live, _of = _lives[_n], f" of {len(_todo)}" if _n <= len(_todo) else ""
            _live.reference(list(zip(_log["step"], _log["train"])), "training", role="ink")   # it grows with the run
            _live.extend(list(zip(_log["step"][len(_live.xs):], _log["val"][len(_live.xs):])), f"**Run {_n}{_of} in this press, step {C.number(_step, 0, L.lang)} of {C.number(C.STEPS, 0, L.lang)}** · {C.number(_spent / 60, 1, L.lang)} minutes · training {C.number(100 * _log['train'][-1], 1, L.lang)}%, validation {C.number(100 * _log['val'][-1], 1, L.lang)}%")

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Finish exercises 1 and 2 first: then your `grid` and your `run_grid` are in your project, and this cell uses them.*")
        elif isinstance(_r, str):
            _out = mo.md({"grid": "*Your `repro/ablation.py`, `repro/grid.py` or `repro/run.py` stops with an error. Did you change it after its check? Delete it, and check its exercise again.*",
                          "model": "*Your ModernGPT, your data or your settings stop with an error. Did you change `llm/modern_gpt.py`, `repro/data.py` or `repro/settings.py` after its check? Delete the file you changed, and check its exercise again.*"}[_r])
        else:
            trained = _r
            _stand = (" The course's version stood in for: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
            _out = mo.md(f"*Done: the six runs of the grid are in your experiment log, `work/projects/repro/ablation.jsonl`.{_stand}*")
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_table(trained):
    trained  # read the experiment log again after a run
    _mine = C.my_runs()
    _runs = _mine or [C.run(_wd, _f) for _wd in (1.0, 0.0) for _f in (0.3, 0.5, 0.9)]
    _when = lambda _s: C.number(_s, 0, L.lang) if _s is not None else "–"
    _rows = "\n".join(f"| {C.number(_l['weight_decay'], 0, L.lang)} | {round(100 * _l['fraction'])}% | {_when(_x['t'])} | {_when(_x['v'])} | {C.number(_x['times'], 1, L.lang) if _x['times'] else '–'} | {_x['verdict']} |"
                      for _l, _x in ((_l, C.read(_l)) for _l in _runs))
    _whose = "Your grid" if _mine else "The course's grid, until yours is in your experiment log"
    mo.md(f"""
**{_whose}**, read with the plan:

| weight decay | fraction | $t$ | $v$ | $v / t$ | the plan's verdict |
|---|---|---|---|---|---|
{_rows}
""")
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The ablation of [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en), now with two settings in every combination; the runner and the plan of [Guided reproduction 4](/course/open/repro-guided-figure?lang=en), now over a grid; and the paper's Figure 1 (center) and Figure 2 (left), now tested on your own task. Your kit now has `repro/grid.py` and `repro/ablation.py`. The next lesson, *Guided reproduction 6: does it match the paper?*, repeats the runs over seeds and compares your numbers with the paper's.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_grid, result_run_grid, result_figure, choice_shows):
    L.wrap_up(result_grid, result_run_grid, result_figure, choice_shows)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
