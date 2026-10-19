# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-gradient-descent-2d. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-gradient-descent-2d", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two parameters: the gradient](/course/open/math-gradient?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_direction():
    quick_direction = L.choice(
        "quick-direction",
        question="Somewhere, the gradient of the loss is $(12,\\ -3)$. Which change lowers the loss?",
        options={
            "uphill": "Make $w$ bigger and $b$ smaller",
            "right": "Make $w$ smaller and $b$ bigger",
            "both-down": "Make $w$ and $b$ both smaller",
        },
        correct="right",
        why={
            "uphill": "That goes with the gradient, uphill. See step 4 of the recipe in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "both-down": "Look at each sign on its own. What does a negative slope for $b$ say? See step 4 of the recipe in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "right": "The slope for $w$ is positive, so a smaller $w$ lowers the loss; the slope for $b$ is negative, so a bigger $b$ does.",
        },
    )
    quick_direction
    return (quick_direction,)


@app.cell(hide_code=True)
def quick_check_contour():
    quick_contour = L.choice(
        "quick-contour",
        question="On a contour plot, the lines lie far apart around a point. What does that tell you?",
        options={
            "slow": "The loss changes slowly there",
            "fast": "The loss changes fast there",
            "low": "The loss is low there",
        },
        correct="slow",
        why={
            "fast": "Where the lines lie close together, a small step crosses many of them. And far apart? See *contour plot* in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "low": "The spacing of the lines says how fast the loss changes, not how high it is. See *contour plot* in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "slow": "Far-apart lines mean that a small step crosses few of them: the loss changes slowly there.",
        },
    )
    quick_contour
    return (quick_contour,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en), $w$ stayed 3 while gradient descent fitted $b$, and $b$ ended at about 57.75. But the lowest point of the loss lies at about $w = 3.15$, $b = 46$: as long as $w$ stays put, $b$ can never get there. Both have to move. In the previous lesson you measured the gradient, a slope for each parameter; now you step against it, and $w$ and $b$ move together, downhill. A large model does exactly this, for all of its parameters at once.

    **An everyday example.** A hiker in thick fog wants to get down a mountain. They cannot see the valley, so at every step they feel which way the ground drops most steeply, and take a small step that way.
    """)
    return


@app.cell(hide_code=True)
def predict_path():
    guess_path = L.guess(
        "path",
        question="Gradient descent starts at $w = 0$, $b = 0$, far from the bottom of a long, narrow valley. What path do you think it takes?",
        options={
            "straight": "Straight to the lowest point",
            "valley": "First down into the valley, then along its floor to the lowest point",
            "circles": "Around the lowest point, in circles",
        },
        reveal="**First down into the valley, then along its floor.** The steepest way down points into the valley, not to the lowest point. Once on the floor, the slopes are small, and the steps along the floor are small too. See the picture below.",
    )
    guess_path
    return (guess_path,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        With two parameters, gradient descent does an update step for each, with the same learning rate, using the gradient measured where you are:
        """),
        L.formula(r"[[param:w_\text{new}]] = [[param:w]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial w}]] \qquad [[param:b_\text{new}]] = [[param:b]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial b}]]"),
        mo.md(r"""
        Both slopes are measured first, at the same $w$ and $b$; only then do $w$ and $b$ move. Together the step goes against the <span class="c-grad">gradient</span>: the way the loss goes down fastest.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **loss landscape** | | the loss drawn over the parameters, so that training looks like walking downhill; the contour plot is a map of it | |
        """),
        L.watch_out("The loss landscape is a picture of a function, and walking downhill is a way of telling the story, not what the computer does. With more than two parameters you cannot draw it, but the update steps work just the same."),
        L.note("In this lesson the area is in hundreds of square meters: 62 m² is 0.62. So $w$ is now the price per 100 m²: the lowest point at $w = 314.5$ is 3.145 thousand euros per m², the same line as in the previous lesson. Why the change? With square meters, the valley is so narrow that gradient descent needs about 100,000 steps; a later lesson, *Why gradient descent likes scaled features*, shows why."),
        mo.Html(C.picture({"title": "Gradient descent across the loss landscape", "step": "{k}",
                           "alt": "A contour plot of the loss of the houses over w and b, a long, narrow valley lying diagonally, with its lowest point near w = 314.5, b = 45.8. A path of 200 steps of gradient descent starts at w = 0, b = 0, jumps in one step to the valley floor near w = 164, b = 183, and then walks along the floor to the lowest point. The places after 0, 1, 10, 50 and 200 steps are marked with their number."})),
        mo.md(r"""
        Each <span class="c-loss">line</span> joins the places $(w, b)$ with the same loss; the cross is the lowest point. The <span class="c-param">points</span> show where $w$ and $b$ were after each step, with $\eta$ = 0.3; the numbers count the steps. The first step lands on the valley floor; from there, about 200 steps walk along it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fit $w$ and $b$ with gradient descent:

    1. Choose starting values for $w$ and $b$, and a learning rate.
    2. Measure both slopes, at the same $w$ and $b$.
    3. Update both: each moves against its own slope, by the learning rate times that slope.
    4. Repeat steps 2 and 3 until $w$ and $b$ hardly change any more.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One update step for $w$ and $b$, from $w = 0$, $b = 0$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_update():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58]) / 100   # area in hundreds of m²
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w, b):
        _residuals = _price - (w * _area + b)
        return (_residuals ** 2).mean()

    # 1. starting values and a learning rate
    _w = 0
    _b = 0
    _lr = 0.3
    _h = 0.001
    print(_loss(_w, _b))
    # 2. both slopes, at the same w and b
    _slope_w = (_loss(_w + _h, _b) - _loss(_w, _b)) / _h
    _slope_b = (_loss(_w, _b + _h) - _loss(_w, _b)) / _h
    print(_slope_w, _slope_b)
    # 3. update both
    _w = _w - _lr * _slope_w
    _b = _b - _lr * _slope_b
    print(_w, _b, _loss(_w, _b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At $w = 0$, $b = 0$ the model predicts 0 for every house, and the loss is about 100275. Both slopes are negative, about −548 and −611, so both parameters go up: $w$ to about 164.4 and $b$ to about 183.4. The loss drops to about 1825: one step from the far side of the landscape to the valley floor. Step 4 repeats this; the picture shows 200 rounds.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One step for two parameters
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-update",
        question="$w = 2$ and $b = 1$. The gradient there is $(4,\\ -10)$, and the learning rate is 0.1. Where are $w$ and $b$ after one update step?",
        options={
            "wrong-way": "$w = 2.4$, $b = 0$",
            "mixed": "$w = 1.6$, $b = 0$",
            "right": "$w = 1.6$, $b = 2$",
        },
        correct="right",
        why={
            "wrong-way": "Both moved with their slope, uphill. What does the minus in the update step do? See the formula in *The idea*.",
            "mixed": "$w$ is right. What does the minus do with the negative slope for $b$? See the formula in *The idea*.",
            "right": "$w$: 2 − 0.1 · 4 = 1.6. $b$: 1 − 0.1 · (−10) = 1 + 1 = 2. Each moves against its own slope.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_descend():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Gradient descent for w and b

        The box has the houses, with the area in hundreds of m², the loss as `loss(w, b)`, starting values $w = 0$ and $b = 0$, the learning rate `lr = 0.3`, the tiny step `h`, and a loop of 50 rounds. Replace the `...` with one update step for both parameters. The last line is `[w, b]`, so the check sees where they ended. The check also tries another learning rate.
        """),
        L.task("Fill the loop with one update step for w and b together."),
    ])
    return


@app.cell(hide_code=True)
def exercise_descend():
    box_descend = L.code_box("descend-wb", lines=20)
    box_descend
    return (box_descend,)


@app.cell(hide_code=True)
def check_descend(box_descend):
    result_descend = L.check("descend-wb", box_descend.value)
    result_descend
    return (result_descend,)


@app.cell(hide_code=True)
def hints_descend():
    L.hints(
        "descend-wb",
        [
            "Each round is steps 2 and 3 of the recipe: first both slopes, then both updates.",
            "Measure the slope for $b$ before $w$ moves: otherwise you measure it somewhere else. How do you make sure of that?",
            "`slope_w = ...` and `slope_b = ...` as in the worked example, then `w = w - lr * ___` and `b = b - lr * ___`.",
        ],
        why="Each round measures both slopes at the same $w$ and $b$, and then moves each against its own slope. After 50 rounds, $w$ is about 267.6 and $b$ about 86.1: on the valley floor, on the way to the lowest point at about 314.5 and 45.8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_floor():
    mo.md(r"""
    ### Exercise 3. The long walk along the floor
    """)
    return


@app.cell(hide_code=True)
def choice_floor_cell():
    choice_floor = L.choice(
        "valley-floor",
        question="In the picture, the first step lands almost right on the valley floor, but then gradient descent takes about 200 steps along the floor. Why?",
        options={
            "right": "Along the floor the slopes are small, and so are the steps",
            "lr": "The learning rate gets a little smaller with every step",
            "order": "Gradient descent first moves $b$, and only then $w$",
        },
        correct="right",
        why={
            "lr": "The learning rate stays 0.3 the whole time. What else decides how big a step is? See the formula in *The idea*.",
            "order": "Every step moves $w$ and $b$ together. Look at the lines along the floor: how far apart are they?",
            "right": "A step is the learning rate times the slope. Along the floor of the narrow valley the loss hardly changes, so the slopes, and the steps, are small.",
        },
    )
    choice_floor
    return (choice_floor,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two parameters: the gradient](/course/open/math-gradient?lang=en) you measured the gradient; now you step against it, and $w$ and $b$ move together across the loss landscape. It is the same update step as in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en), once for each parameter. With more features, a model gets more weights; in the next lesson, *Many features: the dot product*, you write their weighted sum in one short form.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_floor, choice_one, result_descend):
    L.wrap_up(choice_one, result_descend, choice_floor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
