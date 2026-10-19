# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-slope. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-slope", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_valley():
    quick_valley = L.choice(
        "quick-valley",
        question="The loss for every $w$ is a valley, with its lowest point at $w = 3.1$. You are at $w = 2.5$. What happens to the loss if you make $w$ a little bigger?",
        options={
            "up": "It goes up",
            "down": "It goes down",
            "same": "It stays the same",
        },
        correct="down",
        why={
            "up": "On which side of the lowest point are you? See the chart with the slider in [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en).",
            "same": "Is the valley flat at $w = 2.5$? See the chart with the slider in [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en).",
            "down": "You are to the left of the lowest point, so a step to the right goes downhill: the loss goes down.",
        },
    )
    quick_valley
    return (quick_valley,)


@app.cell(hide_code=True)
def quick_check_lowest():
    quick_lowest = L.choice(
        "quick-lowest",
        question="What does the lowest point of that valley tell you?",
        options={
            "best": "The $w$ with the lowest loss: the line that fits the houses best",
            "zero": "Where the loss is 0",
            "steep": "Where the line of the model is steepest",
        },
        correct="best",
        why={
            "zero": "Is the loss 0 there? Look at the number on the vertical axis at the bottom of the valley in [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en).",
            "steep": "The valley shows the loss, not the line of the model. See *The idea* in [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en).",
            "best": "At the lowest point, no other $w$ gives a lower loss: that $w$ gives the line that fits the houses best.",
        },
    )
    quick_lowest
    return (quick_lowest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Fitting by hand means trying: a step one way, a step the other way, and comparing the losses. That works for one parameter. A large model has millions, and trying both ways for each of them, again and again, would take forever. What you want is one number that says, right where you are, which way the loss goes down, and how steeply.

    **An everyday example.** You walk down a hill in thick fog. You cannot see the valley, but under your feet you feel which way the ground slopes, and how steeply.
    """)
    return


@app.cell(hide_code=True)
def predict_steep():
    guess_steep = L.guess(
        "steeper",
        question="The loss for every $w$ is a valley, with its lowest point at $w = 3.1$. Where do you think the valley is steeper: at $w = 2.5$ or at $w = 3.0$?",
        options={
            "near": "At $w = 3.0$",
            "far": "At $w = 2.5$",
            "same": "Equally steep",
        },
        reveal="**At $w = 2.5$.** Far from the lowest point, the valley is steep: a small step changes the loss a lot. Close to it, the valley gets flatter, and at the lowest point itself it is flat. Move the slider below and watch the green line.",
    )
    guess_steep
    return (guess_steep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The derivative of the loss at a $w$ says how fast the loss changes when you change $w$ a tiny bit: the slope of the valley right there. You can measure it: change $w$ by a tiny step $h$, and divide the change in the loss by $h$.
        """),
        L.formula(r"[[grad:\mathcal{L}'(w)]] \approx \frac{[[loss:\mathcal{L}(w + h)]] - [[loss:\mathcal{L}(w)]]}{h}"),
        mo.md(r"""
        The sign says which way the loss goes. A positive <span class="c-grad">slope</span>: the loss goes up when $w$ goes up, so a smaller $w$ is better. A negative slope: a bigger $w$ is better. A slope near 0: you are near the lowest point. The size says how steep: the change in the loss divided by the change in $w$. A small step in $w$ changes the loss by about the slope times that step.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **derivative**, $\mathcal{L}'(w)$ | L prime of w | how fast the loss changes when you change $w$ a tiny bit: the slope of the loss at that $w$ | `slope` |
        | **finite difference** | | an estimate of the derivative from two points close together: change $w$ by a tiny $h$, and divide the change in the loss by $h$ | `(loss(w + h) - loss(w)) / h` |
        | $h$ | h | the tiny step by which you change $w$ | `h` |
        | $\approx$ | is about | about equal to | |

        Move the slider, or drag across the chart. The <span class="c-loss">curve</span> is the loss for every $w$, with $b = 50$, as in the previous lesson. The <span class="c-grad">green line</span> has the slope of the loss at the $w$ you chose.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_slope_cell():
    def _frame(w):
        return (C.picture(w, {"title": "The slope of the loss", "loss": "loss",
                              "alt": f"The loss for every w from 2 to 4.2, a valley with its lowest point at w = 3.1. At w = {L.number(w, 1)} there is a point, with a short green line that has the slope of the valley there: {L.number(C.slope(w), 0)}."}),
                f'At <i>w</i> = {L.number(w, 1)}, the <span class="c-grad">slope</span> is <span class="c-grad ea-num">{L.number(C.slope(w), 0)}</span>.')

    explore_slope = L.explore("slope-w", "w", _frame, start=2.5, low=2.0, high=4.2, step=0.1, task="From $w$ = 2 to 4.2: where is the slope negative, where positive, and where about 0?")
    explore_slope
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Left of the lowest point the <span class="c-grad">slope</span> is negative, and right of it positive. At $w = 3.1$, at the bottom, it is 31: next to the thousands around it, that is about 0. The farther from the bottom, the steeper the valley, and the farther the slope from 0.
        """),
        L.note("The finite difference: you may know it from school as the *difference quotient*."),
        mo.md(r"""
        ### Recipe

        To measure the slope of the loss for a parameter:

        1. Work out the loss where you are.
        2. Change the parameter by a tiny step $h$, such as 0.001, and work out the loss again.
        3. Divide the change in the loss by $h$: new loss minus old loss, then divided by $h$.
        4. Read the sign: positive, make the parameter smaller; negative, make it bigger.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The slope of the loss at $w = 2.5$, with $b = 50$, measured with three steps $h$, each ten times smaller than the one before. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_slope():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w):
        _residuals = _price - (w * _area + 50)
        return (_residuals ** 2).mean()

    _w = 2.5
    for _h in [0.1, 0.01, 0.001]:
        # 1. and 2. the loss where you are, and a tiny step further
        _change = _loss(_w + _h) - _loss(_w)
        # 3. the change divided by the step
        print(_h, _change / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With $h$ = 0.1 the slope comes out at about −8239; with 0.01, −8916; with 0.001, −8983. Each time $h$ gets ten times smaller, the answer moves about ten times less: by 677, then by 67. So it settles at about −8990: the derivative at $w = 2.5$. Step 4: the slope is negative, so the loss goes down when $w$ goes up. A bigger $w$ is better, as the valley showed. The size means: a step of 0.01 in $w$ lowers the loss by about 90.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_way():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which way is down?
    """)
    return


@app.cell(hide_code=True)
def choice_way_cell():
    choice_way = L.choice(
        "which-way",
        question="At $w = 3.5$, with $b = 50$, the slope of the loss is about 6040. To lower the loss, which way do you move $w$?",
        options={
            "up": "Make $w$ bigger",
            "stay": "Leave $w$ as it is: you are at the lowest point",
            "down": "Make $w$ smaller",
        },
        correct="down",
        why={
            "up": "A positive slope: what happens to the loss when $w$ goes up? See the sign in *The idea*.",
            "stay": "At the lowest point, what is the slope? See *The idea*: is 6040 close to that?",
            "down": "The slope is positive: the loss goes up when $w$ goes up. So a smaller $w$ lowers the loss.",
        },
    )
    choice_way
    return (choice_way,)


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The slope for b

        In the box, the loss depends on both parameters: `loss(w, b)`. With $w = 3$ and $b = 50$, measure the slope of the loss for $b$: how fast the loss changes when you change $b$ a tiny bit. Put it in `slope_b`. Replace the `...`. The last line is `slope_b`, so the check sees it. The check also tries another $b$.
        """),
        L.task("Put the slope of the loss for b in `slope_b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_b():
    box_b = L.code_box("slope-b", lines=16)
    box_b
    return (box_b,)


@app.cell(hide_code=True)
def check_b(box_b):
    result_b = L.check("slope-b", box_b.value)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "slope-b",
        [
            "Follow the recipe: the loss where you are, the loss a tiny step further, and then divide.",
            "The loss now has two parameters. Which of the two do you change by `h`, and which stays as it is?",
            "`slope_b = (loss(w, ___) - loss(w, b)) / h`.",
        ],
        why="`loss(w, b + h)` is the loss a tiny step further in $b$, and `loss(w, b)` the loss where you are. Their difference divided by `h` is the slope: about −15.5. Negative, so a bigger $b$ lowers the loss.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. One step downhill

        Finish `next_w(w, b, step)`. It does one step of fitting by hand, but with the slope instead of trying both ways. It measures the slope of the loss for $w$, and gives back $w$ moved by `step` in the direction that lowers the loss. Replace the `...`. The check calls `next_w(2.5, 50, 0.1)`: that should give `2.6`. It also tries other places.
        """),
        L.task("Make `next_w` give back w one step downhill."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next():
    box_next = L.code_box("next-w", lines=16)
    box_next
    return (box_next,)


@app.cell(hide_code=True)
def check_next(box_next):
    result_next = L.check("next-w", box_next.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next-w",
        [
            "Two steps: first the slope for $w$, as in the recipe; then a step against it.",
            "Step 4 of the recipe says which way to go for each sign of the slope. With `if`, you choose between the two ways.",
            "`slope = ___`, then `if slope > 0:` with `return w ___ step`, and otherwise the other way.",
        ],
        why="`(loss(w + h, b) - loss(w, b)) / h` measures the slope for $w$. Is it positive, then the loss goes up when $w$ goes up, so `w - step` goes downhill; otherwise `w + step`. At $w = 2.5$ the slope is about −8983, so the answer is 2.6.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Fitting a line by hand](/course/open/ml-fit-by-hand?lang=en) you found the lowest point by trying a step each way. Now one number, the derivative, tells you right where you are which way the loss goes down, and how steeply. Measuring it with a tiny step still takes two losses per parameter. In the next lesson, *Slope formulas you can trust*, you see that for simple functions the slope follows a short formula, which gives it at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_way, result_b, result_next):
    L.wrap_up(choice_way, result_b, result_next)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
