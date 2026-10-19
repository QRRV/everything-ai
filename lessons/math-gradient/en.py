# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-gradient. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-gradient", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grows():
    quick_grows = L.choice(
        "quick-grows",
        question="The losses of a run are 2790, 4314, 6736 and 10506. What do you change?",
        options={
            "bigger": "Make the learning rate bigger",
            "smaller": "Make the learning rate smaller",
            "more": "Take more steps",
        },
        correct="smaller",
        why={
            "bigger": "The loss grows at every step. Is that a learning rate that is too small, or too big? See *The idea* in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "more": "The loss grows at every step. What will more steps do? See *The idea* in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "smaller": "The loss grows at every step: training diverges, because the steps are too big.",
        },
    )
    quick_grows
    return (quick_grows,)


@app.cell(hide_code=True)
def quick_check_converged():
    quick_converged = L.choice(
        "quick-converged",
        question="What does it mean that training has converged?",
        options={
            "zero": "The loss is 0",
            "lowest": "The loss is as low as it can get",
            "settled": "Further steps hardly change the loss any more",
        },
        correct="settled",
        why={
            "zero": "The loss of the houses never gets to 0. See the table of new words in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "lowest": "That is a common mix-up. See the watch-out in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "settled": "Converged means training has settled: further steps hardly change the loss or the parameters.",
        },
    )
    quick_converged
    return (quick_converged,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far, one parameter moved and the other stayed fixed. But a model has both, $w$ and $b$, and a large model has millions. Gradient descent needs to know, for every parameter at once, which way the loss goes down.

    **An everyday example.** You stand on a hillside, with a hiking map. You can check how steep it is if you walk east, and how steep if you walk north. Those two slopes together tell you which way is straight uphill, and so which way is straight down.
    """)
    return


@app.cell(hide_code=True)
def predict_both():
    guess_both = L.guess(
        "both",
        question="At $w = 3$, $b = 50$, the slope of the loss for $w$ is about −1472, and for $b$ about −15.5. Which do you think you should change to lower the loss fastest?",
        options={
            "w": "Only $w$: its slope is much steeper",
            "both": "Both: each slope is about its own parameter",
            "b": "Only $b$",
        },
        reveal="**Both.** Each slope says which way to move its own parameter to lower the loss, and both are negative: make $w$ and $b$ a bit bigger. Changing only $w$ lowers the loss too, but leaves out what $b$ can do. The slope for $w$ is so much larger partly because $w$ is multiplied by the area: a step of 1 in $w$ moves every prediction by 48 to 130, a step of 1 in $b$ by 1.",
    )
    guess_both
    return (guess_both,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        With more than one parameter, you measure a slope for each parameter in turn, while the others stay as they are. That is a partial derivative. Together, in the order of the parameters, they form the gradient:
        """),
        L.formula(r"[[grad:\nabla \mathcal{L}]] = \left( [[grad:\frac{\partial \mathcal{L}}{\partial w}]],\ [[grad:\frac{\partial \mathcal{L}}{\partial b}]] \right)"),
        mo.md(r"""
        The <span class="c-grad">gradient</span> points uphill: in the direction in which the loss rises fastest. Each part says in which direction of its own parameter the loss goes up; in the other direction, it goes down.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **partial derivative**, $\frac{\partial \mathcal{L}}{\partial w}$ | partial L by partial w | the slope of the loss for one parameter, while the others stay as they are | `slope_w` |
        | **gradient**, $\nabla \mathcal{L}$ | nabla L, or the gradient of L | the partial derivatives of all parameters, in a list; it points in the direction in which the loss rises fastest | `[slope_w, slope_b]` |
        | **contour plot** | | a map of the loss over two parameters, like the contour lines on a hiking map: each line joins the places with the same loss; where lines lie close together, the loss changes fast | |
        """),
        mo.Html(C.picture({"title": "The loss over w and b", "lowest": "lowest", "along_w": "w: −1472", "along_b": "b: −15.5",
                           "alt": "A contour plot of the loss of the houses, with w from 2.6 to 3.6 across and b from −20 to 140 up. The innermost line is a long, narrow oval around the lowest point near w = 3.15, b = 46, lying diagonally from top left to bottom right; the other lines cross the picture diagonally. At w = 3, b = 50 there is a point with two green arrows: one to the right, along w, with slope −1472, and one up, along b, with slope −15.5."})),
        mo.md(r"""
        Every <span class="c-loss">line</span> joins the places $(w, b)$ where the loss is the same, in steps of 600; the cross is the lowest point. The loss is a long, narrow valley. At $w = 3$, $b = 50$, both slopes are negative: the <span class="c-grad">arrows</span> show which way to move each parameter to lower the loss, to the right along $w$ and up along $b$. They point against the gradient, which points uphill.
        """),
        L.watch_out(r"The two axes have very different scales: across, the picture is 1 wide in $w$; up, it is 160 high in $b$. The arrows show the two slopes one at a time; their lengths are not the slopes, and together they do not show the direction of the gradient."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure the gradient of the loss:

    1. Work out the loss where you are.
    2. For each parameter in turn: change only that one by a tiny step $h$, work out the loss again, and divide the change by $h$.
    3. Put the slopes in a list, in the order of the parameters: that is the gradient.
    4. Read it: each sign says which way to move its own parameter to lower the loss.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The gradient of the loss at $w = 2.5$, $b = 50$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_gradient():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w, b):
        _residuals = _price - (w * _area + b)
        return (_residuals ** 2).mean()

    _w = 2.5
    _b = 50
    _h = 0.001
    # 1. the loss where you are
    _now = _loss(_w, _b)
    # 2. one slope per parameter: only that one changes
    _slope_w = (_loss(_w + _h, _b) - _now) / _h
    _slope_b = (_loss(_w, _b + _h) - _now) / _h
    # 3. together, in order: the gradient
    print(_slope_w, _slope_b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The gradient is about [−8983, −98], measured with $h$ = 0.001; the exact slope for $w$ is −8990. Step 4: both slopes are negative, so a bigger $w$ and a bigger $b$ both lower the loss. Compare it with $w = 3$, $b = 50$ in the picture, about [−1472, −15.5]: closer to the bottom, both slopes are closer to 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ways():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two ways at once
    """)
    return


@app.cell(hide_code=True)
def choice_ways_cell():
    choice_ways = L.choice(
        "two-ways",
        question="Somewhere, the gradient of the loss is $(-30,\\ 4)$: $\\frac{\\partial \\mathcal{L}}{\\partial w} = -30$ and $\\frac{\\partial \\mathcal{L}}{\\partial b} = 4$. Which change lowers the loss?",
        options={
            "right": "Make $w$ bigger and $b$ smaller",
            "both-up": "Make $w$ and $b$ both bigger",
            "w-down": "Make $w$ smaller and $b$ bigger",
        },
        correct="right",
        why={
            "both-up": "Look at each sign on its own. What does a positive slope for $b$ say? See step 4 of the recipe.",
            "w-down": "That goes uphill for both. Which way does a negative slope ask you to move? See step 4 of the recipe.",
            "right": "The slope for $w$ is negative, so a bigger $w$ lowers the loss; the slope for $b$ is positive, so a smaller $b$ does. Together: against the gradient.",
        },
    )
    choice_ways
    return (choice_ways,)


@app.cell(hide_code=True)
def text_exercise_gradient():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The gradient as a function

        Finish `gradient(w, b)`. It measures the slope of the loss for $w$ and for $b$ at the $w$ and $b$ it gets, with the tiny step `h`, and gives them back as a list: `[slope_w, slope_b]`. Replace the `...`. The check calls `gradient(3, 50)`: that should give about `[-1471.7, -15.5]`. It also tries other places.
        """),
        L.task("Make `gradient` give back the slope for w and the slope for b, in a list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradient():
    box_gradient = L.code_box("gradient", lines=16)
    box_gradient
    return (box_gradient,)


@app.cell(hide_code=True)
def check_gradient(box_gradient):
    result_gradient = L.check("gradient", box_gradient.value)
    result_gradient
    return (result_gradient,)


@app.cell(hide_code=True)
def hints_gradient():
    L.hints(
        "gradient",
        [
            "Step 2 of the recipe, twice: one slope for $w$ and one for $b$.",
            "For each slope, only one parameter changes by `h`. Which one stays as it is in the slope for $w$?",
            "`slope_w = (loss(___, b) - loss(w, b)) / h`, the same for `slope_b`, and `return [___, ___]`.",
        ],
        why="`loss(w + h, b)` changes only $w$, and `loss(w, b + h)` only $b$. Each change divided by `h` is a partial derivative, and the list of the two, $w$ first, is the gradient: about [−1471.7, −15.5] at $w = 3$, $b = 50$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_steep():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Where is it steepest?
        """),
        mo.Html(C.picture({"title": "Three places on the loss", "lowest": "lowest", "along_w": "", "along_b": "",
                           "alt": "The same contour plot of the loss over w and b, with three points: A at w = 3.0, b = 50, close to the lowest point; B at w = 2.8, b = 80, on the floor of the long valley; C at w = 3.3, b = 100, on its steep side, where the lines lie close together."},
                          slopes=False)),
    ])
    return


@app.cell(hide_code=True)
def choice_steep_cell():
    choice_steep = L.choice(
        "steepest",
        question="At which point does the loss change fastest when you move a little?",
        options={
            "a": "At A",
            "b": "At B",
            "c": "At C",
        },
        correct="c",
        why={
            "a": "A lies close to the lowest point. How far apart are the lines around it? See *contour plot* in the table of new words.",
            "b": "B lies on the floor of the long valley. Do you cross many lines if you move a little from B?",
            "c": "Around C the lines lie closest together: a small step across them crosses many, so the loss changes fast. The gradient there is about [11298, 134].",
        },
    )
    choice_steep
    return (choice_steep,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The derivative as a slope](/course/open/math-slope?lang=en) you measured the slope for one parameter. Now you measure one per parameter, while the others stay as they are, and together they form the gradient, which points uphill. In [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en) you stepped against one slope; in the next lesson, *Fitting w and b with gradient descent*, you step against the whole gradient, and fit $w$ and $b$ together.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_steep, choice_ways, result_gradient):
    L.wrap_up(choice_ways, result_gradient, choice_steep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
