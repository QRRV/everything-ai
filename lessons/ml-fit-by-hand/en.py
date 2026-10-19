# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-fit-by-hand. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-fit-by-hand", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mse():
    quick_mse = L.choice(
        "quick-mse",
        question="The residuals of a model on three examples are 2, −2 and 4. What is its mean squared error?",
        options={
            "mean": "1.33",
            "right": "8",
            "sum": "24",
        },
        correct="right",
        why={
            "mean": "That is the plain mean of the residuals. What happens to each residual first? See the recipe in [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).",
            "sum": "That is the sum of the squares. What is the last step of the recipe? See [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).",
            "right": "The squares are 4, 4 and 16; together 24, and divided by 3 examples, 8.",
        },
    )
    quick_mse
    return (quick_mse,)


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="On the same houses, model A has a loss of 120, and model B a loss of 85. Which model fits the houses better?",
        options={
            "b": "B",
            "a": "A",
            "cannot": "You cannot tell from the loss",
        },
        correct="b",
        why={
            "a": "Is a high loss good or bad? See the table of new words in [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).",
            "cannot": "On the same examples, the loss is exactly what compares two models. Which way is better, high or low? See [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).",
            "b": "Lower is better: on these houses, the predictions of B are off by less.",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far, someone gave you the parameters: $w = 3$ and $b = 50$. But where do they come from? You can now work out the loss of any choice of parameters, so you can compare choices. The best parameters are the ones with the lowest loss. Finding them is what training does. Before a computer does it, you do it by hand.

    **An everyday example.** The shower is too cold. You turn the tap a little, feel the water, and turn further, or back, until it is right.
    """)
    return


@app.cell(hide_code=True)
def predict_valley():
    guess_valley = L.guess(
        "valley",
        question="The model of the houses has $b = 50$. You make $w$ bigger, step by step, from 2 to 4.2. What do you think happens to the loss?",
        options={
            "down": "It keeps going down",
            "valley": "It goes down, then up again",
            "up": "It keeps going up",
        },
        reveal="**It goes down, then up again.** With a small $w$, the line lies below the houses; with a large $w$, above them. In between, the line runs through the houses, and the loss is at its lowest. Try it below.",
    )
    guess_valley
    return (guess_valley,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The loss depends on the parameters. Choose another $w$ or $b$, and the line moves, and the loss changes with it:
        """),
        L.formula(r"[[loss:\mathcal{L}]]([[param:w]], [[param:b]]) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - ([[param:w]] \cdot \mathbf{x}_i + [[param:b]])\right)^2"),
        mo.md(r"""
        Fitting a line by hand means: move the <span class="c-param">parameters</span> until the <span class="c-loss">loss</span> is as low as it gets. That lowest point gives the line that fits these examples best.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | $\mathcal{L}(w, b)$ | L of w and b | the loss as a function of the parameters: for every choice of $w$ and $b$, the loss of that model | `loss(w, b)` |

        Move the slider, or drag across the loss chart. One chart shows the houses, the <span class="c-pred">line</span> of the model and its <span class="c-residual">residuals</span>; the other, the <span class="c-loss">loss</span> for every $w$, with a <span class="c-param">point</span> at the $w$ you chose. $b$ stays 50.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_demo_cell():
    def _frame(w):
        return (C.picture("w", w, {"left": "The houses and the line", "right": "The loss for every w", "x": "area (m²)", "y": "price (thousand euros)", "loss": "loss",
                                   "alt": f"The eight houses and the line of the model with w = {L.number(w, 1)} and b = 50, with the residuals as strokes. The loss for every w from 2 to 4.2, a valley with its lowest point at w = 3.1, and a point at w = {L.number(w, 1)}."}),
                f'With <i>w</i> = {L.number(w, 1)} and <i>b</i> = 50, the <span class="c-loss">loss</span> is <span class="c-loss ea-num">{L.number(C.loss(w, 50))}</span>.')

    explore_demo = L.explore("demo-w", "w", _frame, start=2.5, low=2.0, high=4.2, step=0.1, task="Find the $w$ where the loss is lowest.")
    explore_demo
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The loss for every $w$ is a valley. Its lowest point lies at $w = 3.1$, with a loss of about 100: lower than the 172.75 of $w = 3$, the model of the earlier lessons.

    ### Recipe

    To fit one parameter by hand:

    1. Start somewhere, and note the loss.
    2. Move the parameter a little, and look at the loss again.
    3. If the loss went down, keep going that way. If it went up, go the other way.
    4. Stop where a step either way makes the loss go up: that is the lowest point.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The same search in code: the loss for five values of $w$, with $b = 50$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_search():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])
    _b = 50
    # 1. and 2. start at w = 2.8, and move w up a little each time
    for _w in [2.8, 2.9, 3.0, 3.1, 3.2]:
        _residuals = _price - (_w * _area + _b)
        _loss = (_residuals ** 2).mean()
        print(_w, _loss)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The losses are about 769, 396, 173, 100 and 177. Step 3: from 2.8 to 3.1, every step makes the loss go down, so you keep going. Step 4: from 3.1 to 3.2, it goes up, and so does a step back to 3.0. The lowest point, in steps of 0.1, is $w = 3.1$. The long numbers come from calculating with decimal numbers; 769.0550000000004 is 769.055.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The best bias term

        Now $w$ stays 3, as in the earlier lessons, and you move $b$. Find the $b$ with the lowest loss. The charts show the line and the loss for every $b$; drag across the loss chart, or move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        return (C.picture("b", b, {"left": "The houses and the line", "right": "The loss for every b", "x": "area (m²)", "y": "price (thousand euros)", "loss": "loss",
                                   "alt": f"The eight houses and the line of the model with w = 3 and b = {L.number(b, 0)}, with the residuals as strokes. The loss for every b from 0 to 120, a valley, with a point at b = {L.number(b, 0)}."}),
                f'With <i>w</i> = 3 and <i>b</i> = {L.number(b, 0)}, the <span class="c-loss">loss</span> is <span class="c-loss ea-num">{L.number(C.loss(3, b))}</span>.')

    explore_b = L.explore("best-b", "b", _frame, start=20, low=0, high=120, step=1, task="Move the slider to the $b$ with the lowest loss.")
    explore_b
    return (explore_b,)


@app.cell(hide_code=True)
def check_b(explore_b):
    result_b = L.check("best-b", explore_b)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "best-b",
        [
            "Follow the recipe: move the slider a little, and look at the loss under it each time.",
            "Watch the left chart too: where do most houses lie, above or below the line? $b$ moves the whole line up or down.",
            "The lowest point lies between 50 and 70. Go through that part one step at a time, with the arrow keys on the slider: where does the loss start to go up again?",
        ],
        why="At $b = 58$ the loss is 112.75, the lowest on the slider: at 57 it is 113.25, and at 59, 114.25. With $b = 50$, the houses lie above the line on average, so a higher $b$ fits them better.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ### Exercise 2. Which way?
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "direction",
        question="You are fitting $w$ of another model by hand. At $w = 1.4$ the loss is 52; at $w = 1.5$ it is 47. Which $w$ do you try next?",
        options={
            "back": "1.3: the loss changed, so go back",
            "on": "1.6: the loss went down, so keep going",
            "stop": "None: 47 is the lowest loss",
        },
        correct="on",
        why={
            "back": "Which way did the loss go when $w$ went from 1.4 to 1.5? See step 3 of the recipe.",
            "stop": "How do you know the loss is lowest? See step 4 of the recipe: what have you not tried yet?",
            "on": "The loss went down when $w$ went up, so you keep going up. Only when a step makes the loss go up again have you passed the lowest point.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_search():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The search as a function

        Finish `best_b(bs, area, price, w)`. It gets a list of values to try for $b$, the areas and prices of some houses, and the weight $w$. It gives back the $b$ from the list with the lowest loss. Its first two lines make arrays of the areas and prices. Replace the `...`. The check calls `best_b` with `[40, 50, 60, 70]`, the eight houses and $w = 3$: that should give `60`. It also tries other lists, houses and weights.
        """),
        L.task("Make `best_b` give back the b with the lowest loss."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("search", lines=12)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Two steps: first the loss for every b in the list, as in the worked example, then the b that goes with the lowest one.",
            "Collect the losses in a list, in the same order as `bs`. Which method from [How wrong is one prediction?](/course/open/ml-residuals?lang=en) gives the place of the smallest number?",
            "`losses = np.array(losses)`, and then `return bs[losses.___()]`.",
        ],
        why="The loop works out the loss of each b and collects them in `losses`. `.argmin()` gives the place of the lowest loss, and `bs[...]` the b at that place. With `[40, 50, 60, 70]` the losses are 427.75, 172.75, 117.75 and 262.75, so the answer is 60.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en) you worked out the loss of one model; now you compared many, and chose the one with the lowest loss. That is what training does. By hand it works for one or two parameters, but a large model has millions, and trying them all one by one would take forever. In the next lesson, *The derivative as a slope*, you learn to see from the loss itself which way to move a parameter, without trying both ways.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, result_b, result_search):
    L.wrap_up(result_b, choice_direction, result_search)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
