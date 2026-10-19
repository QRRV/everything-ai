# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-gradient-descent-1d. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import play
    from everything_ai.lesson import lesson

    L = lesson("ml-gradient-descent-1d", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The derivative as a slope](/course/open/math-slope?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sign():
    quick_sign = L.choice(
        "quick-sign",
        question="At $w = 3.3$, the slope of the loss is about 3030. Which way lowers the loss?",
        options={
            "bigger": "Making $w$ bigger",
            "smaller": "Making $w$ smaller",
            "either": "Either way: the slope is far from 0",
        },
        correct="smaller",
        why={
            "bigger": "A positive slope: what does the loss do when $w$ goes up? See step 4 of the recipe in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "either": "The sign of the slope says more than its size. See *The idea* in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "smaller": "A positive slope means the loss goes up when $w$ goes up, so a smaller $w$ lowers it.",
        },
    )
    quick_sign
    return (quick_sign,)


@app.cell(hide_code=True)
def quick_check_steep():
    quick_steep = L.choice(
        "quick-steep",
        question="At $w = 2.5$ the slope of the loss is −8990, at $w = 3.0$ it is −1479. Where is the valley steeper?",
        options={
            "far": "At $w = 2.5$",
            "near": "At $w = 3.0$",
            "same": "Equally steep: both slopes are negative",
        },
        correct="far",
        why={
            "near": "Which of the two numbers is further from 0? See *The idea* in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "same": "The sign says which way; what does the size say? See *The idea* in [The derivative as a slope](/course/open/math-slope?lang=en).",
            "far": "−8990 is much further from 0 than −1479: at $w = 2.5$ the loss changes much faster.",
        },
    )
    quick_steep
    return (quick_steep,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The slope says which way the loss goes down. But how far should you step? With a fixed step you crawl while the bottom is far away, and near the bottom you step right over it. Let the slope decide that too. Step after step, you end up at the bottom. This is how most models in this course learn, up to large language models.

    **An everyday example.** A drop of honey runs down the inside of a bowl. Where the side is steep it runs fast, where the bowl flattens it creeps, and at the bottom it stops. Unlike a rolling ball, it never shoots past: how fast it goes depends only on how steep it is right where it is.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="You start at $w = 2.5$, and each step moves $w$ by a small number times the slope. As you get closer to the bottom, what do you think happens to the steps?",
        options={
            "same": "They stay the same size",
            "smaller": "They get smaller",
            "bigger": "They get bigger",
        },
        reveal="**They get smaller.** Near the bottom the valley is flatter, so the slope is smaller, and so is a step that is a small number times the slope. At the bottom itself the slope is 0, and $w$ stops moving. Take the steps below, and watch them shrink.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Gradient descent finds the bottom of the loss step by step. Each step, the update step, moves the parameter against the slope, by a small number times the slope:
        """),
        L.formula(r"[[param:w_\text{new}]] = [[param:w]] - [[hyper:\eta]] \cdot [[grad:\mathcal{L}'(w)]]"),
        mo.md(r"""
        The minus makes the step go against the <span class="c-grad">slope</span>: a positive slope makes $w$ smaller, a negative slope makes it bigger. The <span class="c-hyper">learning rate</span> $\eta$ is a small number that you choose yourself; the model does not learn it. It sets how big the steps are.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **gradient descent** | | finding the bottom of the loss step by step: each step moves the parameter a little against the slope | a loop |
        | **update step** | | one step of gradient descent: the new value is the old value minus the learning rate times the slope | `w = w - lr * slope` |
        | **learning rate**, $\eta$ | eta | a small number that you choose yourself and that sets how big each step is | `lr` |

        The name comes from the gradient: the slope when a model has more than one parameter; you meet it in a later lesson.

        Take the steps yourself: **Step** takes one, **Play** takes them all. The <span class="c-loss">curve</span> is the loss for every $w$, with $b = 50$; the <span class="c-param">points</span> show where $w$ was after each step, from $w = 2.5$, with $\eta$ = 0.00005. Beside it: how far $w$ moves in each step.
        """),
    ])
    return


@app.cell(hide_code=True)
def play_steps_cell():
    play_steps = play.Descent(
        C.loss, C.slope, w_range=(2.0, 4.2), ymax=10000, steps=8, w0=C.START, lr=C.LR, show_lr=False, beside="step",
        decimal="," if L.lang == "nl" else ".", xticks=[2, 2.5, 3, 3.5, 4], yticks=[0, 5000, 10000],
        labels={"task_kick": L.w["task"], "task": "Take steps until <i>w</i> hardly moves any more. Then drag the orange point to the other side of the valley, and take the steps again.",
                "title": "The loss for every w", "title2": "How far w moves in each step", "x_name": "w", "steps_name": "step",
                "step": "Step", "reset": "Back", "play": "Play", "pause": "Pause", "lr_name": "learning rate", "w_name": "starting value of w",
                "step_name": "step", "gone": "w has left the picture ↗",
                "alt": "The loss for every w from 2 to 4.2, a valley with its lowest point at w = 3.1, with the points where w was after each step of gradient descent.",
                "alt2": "How far w moves in each of the eight steps of gradient descent: 0.449, 0.112, 0.028, 0.007 and smaller still."})
    mo.ui.anywidget(play_steps)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The first step is big, 0.45, because the valley is steep at $w = 2.5$. Then the steps shrink: 0.11, 0.03, 0.007. After a few steps, $w$ hardly moves any more: it is at the bottom, at about 3.1. From the other side of the valley the slope is positive, so the steps go to the left; they shrink in the same way and end at the same bottom.

    ### Recipe

    To find the bottom of the loss with gradient descent:

    1. Choose a starting value for the parameter and a learning rate.
    2. Measure the slope of the loss where the parameter is now.
    3. Update: the new value is the old value minus the learning rate times the slope.
    4. Repeat steps 2 and 3, until the parameter hardly changes any more.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Five update steps for $w$, with $b = 50$, from $w = 2.5$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_descent():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w):
        _residuals = _price - (w * _area + 50)
        return (_residuals ** 2).mean()

    # 1. a starting value and a learning rate
    _w = 2.5
    _lr = 0.00005
    _h = 0.001
    # 4. repeat
    for _step in range(5):
        # 2. the slope where w is now
        _slope = (_loss(_w + _h) - _loss(_w)) / _h
        # 3. the update step
        _w = _w - _lr * _slope
        print(_step + 1, _slope, _w, _loss(_w))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each line shows the step, the slope it used, the new $w$ and its loss. After one step, $w$ is about 2.949, and the loss dropped from 2790 to 267. After five steps, $w$ is about 3.097, with a loss of about 99.93: almost the lowest point you found by hand, without trying a single step back. The learning rate is tiny, 0.00005, because the slopes here are in the thousands: 0.00005 times −8983 is −0.45, and minus that is a step of 0.45 to the right.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One update step by hand
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-step",
        question="At $w = 3$ the slope of the loss is −1479, and the learning rate is 0.0001. What is $w$ after one update step?",
        options={
            "right": "3.1479",
            "wrong-way": "2.8521",
            "no-lr": "1482",
        },
        correct="right",
        why={
            "wrong-way": "That steps with the slope, not against it. What does the minus in the update step do with a negative slope? See *The idea*.",
            "no-lr": "That subtracts the whole slope. What is the slope multiplied by first? See the update step in *The idea*.",
            "right": "3 − 0.0001 · (−1479) = 3 + 0.1479 = 3.1479. The slope is negative, so $w$ goes up.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Gradient descent for b

        Now $w$ stays 3, and gradient descent looks for the best $b$, starting at $b = 0$, with the learning rate `lr = 0.2`. The box has the loss as `loss(w, b)` and the tiny step `h` for measuring a slope. The learning rate is much bigger than the 0.00005 for $w$: the slope for $b$ is in the tens, not in the thousands. The loop in the box runs 25 rounds. Replace the `...` with one update step for $b$. The last line is `b`, so the check sees where it ended. The check also tries another learning rate.
        """),
        L.task("Fill the loop with one update step for b."),
    ])
    return


@app.cell(hide_code=True)
def exercise_b():
    box_b = L.code_box("descend-b", lines=18)
    box_b
    return (box_b,)


@app.cell(hide_code=True)
def check_b(box_b):
    result_b = L.check("descend-b", box_b.value)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "descend-b",
        [
            "Each round is steps 2 and 3 of the recipe: first the slope, then the update.",
            "The slope for $b$ is measured as in [The derivative as a slope](/course/open/math-slope?lang=en). Which parameter do you change by `h`? Then the new $b$ comes from the old $b$, `lr` and the slope.",
            "`slope = (loss(w, b + h) - loss(w, b)) / h`, and then `b = b ___ lr * slope`.",
        ],
        why="Each round measures the slope for $b$ and moves $b$ against it: `b = b - lr * slope`. After 25 rounds, $b$ is about 57.75, the bottom of the valley; with the slider in [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en), in steps of 1, you found 58.",
    )
    return


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Exercise 3. Why do the steps shrink?
    """)
    return


@app.cell(hide_code=True)
def explain_cell():
    explain_smaller = L.explain(
        "why-smaller",
        question="The learning rate stays the same the whole time. Why do the steps of gradient descent still get smaller near the bottom? Write one or two sentences.",
        model_answer="A step is the learning rate times the slope. Near the bottom the valley is flatter, so the slope is smaller, and so is the step; at the bottom the slope is 0, and the parameter stops moving.",
        points=[
            "a step is the learning rate times the slope",
            "near the bottom the slope is smaller",
            "at the bottom the slope is 0, so the parameter stops moving",
        ],
    )
    explain_smaller
    return (explain_smaller,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en) you tried steps both ways; in [The derivative as a slope](/course/open/math-slope?lang=en) the slope told you which way. Gradient descent puts it together: step against the slope, by the learning rate times the slope, again and again. In the next lesson, *Too small, too big: the learning rate*, you see what happens when you choose the learning rate too small or too big.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, explain_smaller, result_b):
    L.wrap_up(choice_one, result_b, explain_smaller)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
