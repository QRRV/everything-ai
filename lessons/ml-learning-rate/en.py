# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-learning-rate. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-learning-rate", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_update():
    quick_update = L.choice(
        "quick-update",
        question="$w = 2$, the slope of the loss there is 50, and the learning rate is 0.01. What is $w$ after one update step?",
        options={
            "plus": "2.5",
            "right": "1.5",
            "no-lr": "−48",
        },
        correct="right",
        why={
            "plus": "That steps with the slope. Which way does an update step go? See the formula in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).",
            "no-lr": "That subtracts the whole slope. What is it multiplied by first? See the formula in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).",
            "right": "2 − 0.01 · 50 = 2 − 0.5 = 1.5. The slope is positive, so $w$ goes down.",
        },
    )
    quick_update
    return (quick_update,)


@app.cell(hide_code=True)
def quick_check_lr():
    quick_lr = L.choice(
        "quick-lr",
        question="What is the learning rate?",
        options={
            "right": "A small number you choose yourself, that sets how big each step is",
            "learned": "A parameter the model learns, like $w$",
            "slope": "The slope of the loss",
        },
        correct="right",
        why={
            "learned": "Who chooses the learning rate: you, or gradient descent? See the table of new words in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).",
            "slope": "The slope changes at every step; the learning rate multiplies it. See the formula in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).",
            "right": "You choose it before training starts, and every step is the learning rate times the slope.",
        },
    )
    quick_lr
    return (quick_lr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lesson the learning rate was given, and gradient descent reached the bottom in a few steps. But you choose the learning rate yourself, and with a bad choice training goes wrong: too small, and it takes forever; too big, and every step jumps over the bottom, further and further. Training a large model takes days, so you want to see early on which it is. The loss shows it.

    **An everyday example.** A shower whose water takes a moment to follow the tap. Turn it a hair at a time, and you wait forever for warm water. Turn it hard, and it swings from ice-cold to scalding and back.
    """)
    return


@app.cell(hide_code=True)
def predict_double():
    guess_double = L.guess(
        "double",
        question="The learning rate of the previous lesson was 0.00005. What do you think happens with twice as much, 0.0001?",
        options={
            "faster": "$w$ gets to the bottom twice as fast",
            "zigzag": "$w$ jumps over the bottom, back and forth, but still gets there",
            "away": "$w$ runs away from the bottom",
        },
        reveal="**It jumps over the bottom, back and forth, but still gets there.** The first step goes from 2.5 to 3.40, past the bottom at 3.1; the next one back to 2.95, then 3.17, each jump smaller. With twice as much again, 0.0002, the jumps grow instead: 4.30, then 0.70, then 7.91. Then $w$ runs away.",
    )
    guess_double
    return (guess_double,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The <span class="c-hyper">learning rate</span> decides how training goes. Too small: every step is tiny, and the loss goes down slowly. Well chosen: the loss drops fast, and then hardly changes any more: training converges. Too big: the steps jump over the bottom; much too big, and every jump lands higher than the one before, so the loss grows, step after step: training diverges. You see which it is in the loss curve, the loss after each step. A learning rate that is a bit too big can also look slow in the loss curve: each jump lands only a little lower. The steps on the loss show the difference: steps that are too small stay on one side of the bottom, steps that are too big jump from side to side.
        """),
        mo.Html(C.three_curves({"titles": ["η = 0.000005: too small", "η = 0.00005: converges", "η = 0.00015: diverges"], "x": "step", "y": "loss",
                                "alt": "Three loss curves, each over 10 steps from w = 2.5. With a learning rate of 0.000005 the loss goes down slowly, from 2790 to 665. With 0.00005 it drops to about 100 in three steps and stays there. With 0.00015 it grows every step, from 2790 to about 246000."})),
        mo.md(r"""
        Three runs of 10 steps from $w = 2.5$, with $b = 50$. Mind the vertical axes: the run that diverges ends at about 246000.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **loss curve** | | a chart of the loss after each step of training | `plt.plot(losses)` |
        | **convergence** | | training settles: further steps hardly change the loss or the parameters any more | |
        | **divergence** | | training runs away: the loss grows step after step, usually because the learning rate is too big | |
        | `abs(x)` | | the size of a number, without its sign | `abs(-0.05)` is `0.05` |
        """),
        L.watch_out("Converged does not mean the loss is as low as it can get. It only means further steps hardly change it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To choose a learning rate:

    1. Start with a learning rate, and run a few steps.
    2. Look at the loss curve.
    3. If the loss goes down, but slowly, look at the steps: if they stay on one side of the bottom, make the learning rate bigger; if they jump from side to side, or the loss goes up, make it smaller.
    4. Repeat until the loss drops fast and then settles.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Gradient descent for $w$, 10 steps from $w = 2.5$, with three learning rates. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_rates():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w):
        _residuals = _price - (w * _area + 50)
        return (_residuals ** 2).mean()

    _h = 0.001
    # 1. three learning rates, each with 10 steps
    for _lr in [0.000005, 0.00005, 0.00015]:
        _w = 2.5
        _losses = [_loss(_w)]
        for _step in range(10):
            _slope = (_loss(_w + _h) - _loss(_w)) / _h
            _w = _w - _lr * _slope
            _losses.append(_loss(_w))
        # 2. the loss before the first step and after the last one
        print(_lr, _losses[0], _losses[-1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    All three start at a loss of 2790. With 0.000005, it is 665 after 10 steps: still going down, slowly. Step 3 says: make it bigger. With 0.00005, it is 99.92: the bottom. With 0.00015, it is 246154: the loss grew at every step. Step 3 says: make it smaller. (Python writes very small numbers short: `5e-06` is 0.000005, with the 5 in the sixth place after the point, and `5e-05` is 0.00005.)
    """)
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Read the losses
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "too-small",
        question="Three runs of gradient descent from $w = 3.8$, each with its own learning rate. The loss after each step:<br>run A: 3797, 4611, 5577, 6780, 8213<br>run B: 3797, 135, 100, 100, 100<br>run C: 3797, 2769, 2027, 1491, 1104<br>Which run has a learning rate that is too small?",
        options={
            "a": "Run A",
            "b": "Run B",
            "c": "Run C",
        },
        correct="c",
        why={
            "a": "In run A the loss grows at every step. Is that a learning rate that is too small, or too big? See *The idea*.",
            "b": "Run B drops fast and then hardly changes. What does that mean? See *The idea*.",
            "c": "In run C the loss goes down at every step, but slowly: the steps are too small. A bigger learning rate gets there sooner.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Choose a learning rate

        Move the slider to choose the learning rate $\eta$. One chart shows the loss for every $w$, with the <span class="c-param">points</span> where $w$ was after each of 10 steps from $w = 2.5$; the other, the <span class="c-loss">loss curve</span>, where a loss above 3000 runs off the top. A bigger learning rate gets there faster, until the steps start to jump too far.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lr_cell():
    def _frame(lr):
        ws, losses = C.run(lr)
        return (C.picture(lr, {"left": "The steps on the loss", "right": "The loss curve", "x": "step", "loss": "loss",
                               "alt": f"The loss for every w, a valley with its lowest point at w = 3.1, and the points where w was after each of 10 steps with the learning rate {L.number(lr, 6)}; the last one at w = {L.number(ws[-1], 3)}. The loss after each step, ending at {L.number(losses[-1])}." + (" Above 3000 it runs off the top of the loss curve." if max(losses) > 3000 else "")}),
                f'After 10 steps, the <span class="c-loss">loss</span> is <span class="c-loss ea-num">{L.number(losses[-1])}</span>.')

    explore_lr = L.explore("pick-lr", "η", _frame, start=0.000005, low=0.000005, high=0.00015, step=0.000005, role="hyper",
                           task="Find the biggest learning rate with which the loss is still below 100 after 10 steps.")
    explore_lr
    return (explore_lr,)


@app.cell(hide_code=True)
def check_lr(explore_lr):
    result_lr = L.check("pick-lr", explore_lr)
    result_lr
    return (result_lr,)


@app.cell(hide_code=True)
def hints_lr():
    L.hints(
        "pick-lr",
        [
            "Watch both charts while you move the slider: does the loss go down slowly, drop fast, or grow? Do the points stay on one side of the bottom, or jump from side to side?",
            "Start small and move the slider up, one notch at a time. Where does the loss after 10 steps first go above 100?",
            "The edge lies between 0.0001 and 0.00012. Which notch is the last one with a loss below 100?",
        ],
        why="0.000105 is the biggest: its loss after 10 steps is 99.99. Every learning rate from about 0.00003 gets below 100; smaller ones crawl. From about 0.00007 the steps jump over the bottom, from side to side; up to 0.000105 they still settle within 10 steps, from 0.00011 not, and from 0.000135 on the loss grows: training diverges.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_converged():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Has it converged?

        Finish `converged(losses)`. It gets the losses of a run, the one at the start and then one after each step, and gives back `True` when the run has converged: when the last step changed the loss by less than 0.01, up or down. Otherwise it gives back `False`. `abs(x)` gives a number without its sign: `abs(-0.05)` is `0.05`. Replace the `...`. The check calls `converged([2790.2, 267.4, 110.5, 100.6, 99.98, 99.93])`: that should give `False`. It also tries other runs.
        """),
        L.task("Make `converged` say whether the last step hardly changed the loss."),
    ])
    return


@app.cell(hide_code=True)
def exercise_converged():
    box_converged = L.code_box("converged", lines=4)
    box_converged
    return (box_converged,)


@app.cell(hide_code=True)
def check_converged(box_converged):
    result_converged = L.check("converged", box_converged.value)
    result_converged
    return (result_converged,)


@app.cell(hide_code=True)
def hints_converged():
    L.hints(
        "converged",
        [
            "Converged means the last step hardly changed the loss. Which two losses tell you what the last step did?",
            "`losses[-1]` is the last loss. The change can be up or down; which number tells you its size, whatever its sign?",
            "`return abs(___ - ___) < 0.01`.",
        ],
        why="`losses[-1] - losses[-2]` is what the last step did to the loss, and `abs` drops the sign, so a drop counts as a change too. In the example, the last step changed the loss by 0.05: not converged yet.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en) the learning rate was given. Now you choose it yourself, and read from the loss curve whether it is too small, well chosen or too big. You will look at loss curves every time you train a model from here on, up to your own language model. In the next lesson, *Two parameters: the gradient*, you measure a slope for each parameter, $w$ and $b$, at the same time.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_runs, result_converged, result_lr):
    L.wrap_up(choice_runs, result_lr, result_converged)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
