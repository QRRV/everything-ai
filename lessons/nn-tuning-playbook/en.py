# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-tuning-playbook. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-tuning-playbook", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Adam](/course/open/opt-adam?lang=en) and [A reusable training loop](/course/open/nn-trainer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_correction():
    quick_correction = L.choice(
        "quick-correction",
        question="Adam's average $m$ starts at 0, with $\\beta_1 = 0.9$. The first gradient is 2. What are $m$ and its correction $\\hat{m}$ after that step?",
        options={
            "right": "$m = 0.2$ and $\\hat{m} = 2$",
            "both": "$m = 0.2$ and $\\hat{m} = 0.2$",
            "full": "$m = 2$ and $\\hat{m} = 2$",
        },
        correct="right",
        why={
            "both": "$m$ is $0.1 \\times 2 = 0.2$, but the correction divides it by $1 - 0.9^1 = 0.1$. See [Adam](/course/open/opt-adam?lang=en).",
            "full": "$m = 0.9 \\times 0 + 0.1 \\times 2 = 0.2$: the start at 0 pulls it down; $\\hat{m}$ corrects that. See [Adam](/course/open/opt-adam?lang=en).",
            "right": "$0.2 / (1 - 0.9) = 2$. This lesson smooths losses the same way.",
        },
    )
    quick_correction
    return (quick_correction,)


@app.cell(hide_code=True)
def quick_check_validation():
    quick_validation = L.choice(
        "quick-validation",
        question="Your `Trainer` keeps the parameters of the epoch with the lowest validation loss. Why the validation loss, and not the training loss?",
        options={
            "speed": "The validation loss is faster to compute",
            "right": "The training loss keeps falling when the model starts to overfit",
            "same": "They are the same, so either would do",
        },
        correct="right",
        why={
            "speed": "Speed is not the reason: it is about what the loss says. See [A reusable training loop](/course/open/nn-trainer?lang=en).",
            "same": "On data the model trains on, the loss keeps falling; on new data it can rise again. See [A reusable training loop](/course/open/nn-trainer?lang=en).",
            "right": "This lesson chooses settings the same way: by the validation loss.",
        },
    )
    quick_validation
    return (quick_validation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Of all the settings of a training run, the learning rate matters most: too large and the loss blows up, too small and the model hardly learns in the time you have. Trying every value with a full run takes long. An **LR range test** finds a sensible range in one short run, and a small sweep then picks the best value in it. Then you go on to the next setting, one at a time.
    """)
    return


@app.cell(hide_code=True)
def predict_best():
    guess_best = L.guess(
        "best-rate",
        question="An MLP with one hidden layer of 128 neurons learns the handwritten digits for one epoch, with SGD and momentum 0.9. Which of these learning rates ends with the lowest validation loss?",
        options={
            "small": "0.001",
            "right": "0.1",
            "large": "1",
        },
        reveal="**0.1.** With 0.001 the validation loss is still 2.14 after the epoch: hardly lower than at the start, 2.30. With 1 the steps are too big, and it ends at about 1.9. With 0.1 it ends at 0.34, and 0.03 and 0.3 come close, at about 0.45 and 0.47. The three options lie a factor of a thousand apart, and only the middle one learns well in one epoch.",
    )
    guess_best
    return (guess_best,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: one run, growing learning rates.** A range test is one short training run in which the learning rate grows by the same factor at every step, from far too small to far too large. With $n$ steps from $\eta_{\min}$ to $\eta_{\max}$, step $i$ uses
        """),
        L.formula(r"[[hyper:\eta_i]] = [[hyper:\eta_{\min}]] \left( \frac{[[hyper:\eta_{\max}]]}{[[hyper:\eta_{\min}]]} \right)^{i / (n - 1)}"),
        mo.md(r"""
        The loss of every step is noted. At first the steps are too small and the loss hardly moves; then it falls; then the steps get too large and it blows up. The test stops once the loss is 4 times what it was at the start.

        **Step 2: smooth.** The loss of one mini-batch is noisy. A running average, corrected for its start at 0 as in Adam, shows the trend. It lags a few steps behind, so its lowest point lies a little to the right of where the single losses are lowest.

        **Step 3: read it.** The lowest smoothed loss is about where training stops being stable, so a good learning rate lies at or below it: from about a tenth of it up to it.

        **Step 4: sweep.** In that range, train a few short runs with values an equal factor apart, and keep the one with the lowest validation loss. Change only the learning rate, then fix it and go on to the next setting. Changing two at once tells you neither.

        The picture: a range test on the handwritten digits, from 0.0001 to 10 in 100 steps. The circles are the losses of single steps, the line their smoothed loss, and the dot its lowest point; the band runs from a tenth of that point up to it.
        """),
        mo.Html(C.range_picture({"title": "An LR range test", "x": "learning rate", "y": "loss", "lowest": "the dot: lowest at {lr}",
                                 "band": "sweep", "lang": L.lang,
                                 "alt": "Loss against the learning rate on a log scale from 0.0001 to 10. The loss of single steps scatters around a smoothed line that stays near 2.3 up to about 0.003, falls to about 0.9 near a learning rate of 0.3, and then shoots up, where the test stops. A light band runs from a tenth of that lowest point up to it."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **LR range test** | one short run with a learning rate that grows by the same factor every step, to see where training works and where it blows up | |
        | **hyperparameter sweep**, or sweep | short runs that each try one value of a setting, with all else the same | |
        | $\eta_{\min}$, $\eta_{\max}$, $n$ | the first and the last learning rate of the test, and its number of steps | `lr_min`, `lr_max`, `steps` |
        """),
        L.watch_out("A range test belongs to its setup: the model, the batch size, the optimizer. Change one of them and run it again. And the lowest point is not the best learning rate: it is close to where training breaks down, so stay below it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find a learning rate:

    1. Range test: one short run, the learning rate growing by the same factor every step, from far too small to far too large; note every loss.
    2. Smooth the losses with a running average, corrected as in Adam, and find the learning rate at its lowest point.
    3. Sweep a few values an equal factor apart, from a tenth of that learning rate up to it, each in a short run.
    4. Keep the value with the lowest validation loss, fix it, and go on to the next setting.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Learning rates an equal factor apart, from 0.01 to 1 in 5 steps: each is the one before it times the same factor. The factor is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_factor():
    _lr_min, _lr_max, _n = 0.01, 1.0, 5
    _factor = 1.0  # complete: the factor between neighbours, (_lr_max / _lr_min) ** (1 / (_n - 1))
    _lr = _lr_min
    for _i in range(_n):
        print(round(_lr, 4))
        _lr = _lr * _factor
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the factor is 1 and every line prints 0.01. Completed, the factor is $100^{1/4} \approx 3.16$, and the lines are 0.01, 0.0316, 0.1, 0.3162 and 1.0: from $\eta_{\min}$ to $\eta_{\max}$ in 4 equal factors, evenly spread on a log scale. Multiplying by the factor $i$ times is the same as the formula of Step 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_range():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The learning rates of a range test

        Complete `lr_range(lr_min, lr_max, steps)`: the list of `steps` learning rates of Step 1, from `lr_min` up to `lr_max`, each the same factor larger than the one before. The check calls `lr_range(0.001, 1.0, 4)`, which should give `[0.001, 0.01, 0.1, 1.0]`, and tries other ranges and numbers of steps.
        """),
        L.task("Make `lr_range` give `steps` learning rates an equal factor apart, from `lr_min` to `lr_max`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_range():
    box_range = L.code_box("lr-range", lines=6)
    box_range
    return (box_range,)


@app.cell(hide_code=True)
def check_range(box_range):
    result_range = L.check("lr-range", box_range.value)
    result_range
    return (result_range,)


@app.cell(hide_code=True)
def hints_range():
    L.hints(
        "lr-range",
        [
            "The formula of Step 1, for each `i` of the loop, with `lr_min`, `lr_max` and `steps`.",
            "In Python, `a ** b` is `a` to the power `b`, and the n of the formula is `steps`. Check your line: at `i = 0` it must give `lr_min`, at `i = steps - 1` it must give `lr_max`.",
        ],
        why="`lrs.append(lr_min * (lr_max / lr_min) ** (i / (steps - 1)))`: for `lr_range(0.001, 1.0, 4)` the factors are 1, 10, 100 and 1000.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Read the test

        Complete `range_pick(lrs, losses, beta)`: smooth `losses` with a running average `avg` that starts at 0, `avg = beta * avg + (1 - beta) * loss`, correct it for its start as Adam does, and give back the learning rate of `lrs` where the corrected average is lowest. The check calls `range_pick([0.001, 0.01, 0.1, 1.0], [2.3, 2.0, 1.2, 5.0], 0.5)`, which should give 0.1, and tries a lucky dip, another `beta`, and `beta = 0`.
        """),
        L.task("Make `range_pick` give the learning rate at the lowest corrected running average of the losses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("range-pick", lines=9)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("range-pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "range-pick",
        [
            "Three lines in the loop: the running average, its correction, and keeping the best so far, as your `Trainer` keeps the best validation loss.",
            "The correction at step `i` divides by `1 - beta ** (i + 1)`: `i` counts from 0, Adam's steps from 1.",
        ],
        why="`avg = beta * avg + (1 - beta) * losses[i]`, `smooth = avg / (1 - beta ** (i + 1))`, and `if best is None or smooth < best: best, best_lr = smooth, lrs[i]`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your range test, and a sweep

    The cell below runs the range test of the picture with your `lr_range`, reads it with your `range_pick`, and then trains three short runs, one epoch each: at a tenth of the learning rate it found, at a third of it, and at the rate itself. Until your functions are right, the course's stand in.
    """)
    return


@app.cell(hide_code=True)
def demo_sweep(box_pick, box_range, result_pick, result_range):
    _range, _pick, _r_yours, _p_yours = C.functions_of(box_range.value if result_range.state == "correct" else None,
                                                       box_pick.value if result_pick.state == "correct" else None)
    _lrs = list(_range(C.LO, C.HI, C.STEPS))
    _losses = C.range_test(_lrs, 0)
    _found = _pick(_lrs[:len(_losses)], _losses, C.BETA)
    _values = (_found / 10, _found / 3, _found)
    _val = C.sweep(_values)
    _best = min(range(3), key=lambda k: _val[k])
    _who = ("your `lr_range` and your `range_pick`" if _r_yours and _p_yours else
            "your `lr_range` and the course's `range_pick`" if _r_yours else
            "the course's `lr_range` and your `range_pick`" if _p_yours else "the course's functions")
    _rows = [f"| {C.rate(_v, L.lang)} | {C.number(_l, 3, L.lang)}{' (lowest)' if _k == _best else ''} |" for _k, (_v, _l) in enumerate(zip(_values, _val))]
    mo.md("\n".join([
        f"With {_who}, the test stopped after {len(_losses)} steps, and its lowest smoothed loss lies at a learning rate of {C.rate(_found, L.lang)}. The sweep, one epoch each:",
        "",
        "| learning rate | validation loss |",
        "|---|---|",
        *_rows,
        "",
        "The test cut a search over five factors of ten down to one; three short runs then picked the value.",
    ]))
    return


@app.cell(hide_code=True)
def text_exercise_sweep():
    mo.md(r"""
    ### Exercise 3. Which values to sweep
    """)
    return


@app.cell(hide_code=True)
def choice_sweep_cell():
    choice_sweep = L.choice(
        "sweep-values",
        question="Your range test has its lowest smoothed loss at a learning rate of 0.3. Which three values do you try in the sweep?",
        options={
            "close": "0.29, 0.3 and 0.31",
            "above": "0.3, 1 and 3",
            "right": "0.03, 0.1 and 0.3",
        },
        correct="right",
        why={
            "close": "These differ by about 3%: three runs that tell you almost nothing, while the good value may lie a factor of 3 lower.",
            "above": "Above the lowest point, the loss of the test was already rising, and soon blew up: training is unstable there.",
            "right": "From a tenth of the lowest point up to it, an equal factor apart: about 3 each time.",
        },
    )
    choice_sweep
    return (choice_sweep,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The learning rate of [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en), now found in one short run, with the corrected average of [Adam](/course/open/opt-adam?lang=en) and the validation loss that your `Trainer` of [A reusable training loop](/course/open/nn-trainer?lang=en) keeps. When you later train a model of your own, a range test is a cheap first run before the long ones.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sweep, result_pick, result_range):
    L.wrap_up(result_range, result_pick, choice_sweep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
