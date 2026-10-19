# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson opt-momentum. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-momentum", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_valley():
    quick_valley = L.choice(
        "quick-valley",
        question="Gradient descent runs in a long, narrow valley: steep across, gentle along. Why is it slow to reach the bottom?",
        options={
            "right": "The learning rate must be small for the steep direction, and those small steps hardly move it along the gentle one",
            "far": "The bottom is far away, and every step is the same size",
            "minimum": "A narrow valley has no lowest point, so it keeps going",
        },
        correct="right",
        why={
            "far": "The steps are not the same size: each is the learning rate times the slope, and the slope differs across and along the valley. See [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "minimum": "The valley has a lowest point; the trouble is getting there. See [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "right": "One learning rate for both directions: small enough not to blow up across the valley, and so far too small along it. Standardizing the features made the valley round; today you see another way.",
        },
    )
    quick_valley
    return (quick_valley,)


@app.cell(hide_code=True)
def quick_check_plus_minus():
    quick_plus_minus = L.choice(
        "quick-plus-minus",
        question="A number is +1 or −1, each with probability 0.5. What is its expected value?",
        options={
            "one": "1: the size of each value",
            "half": "0.5: the probability of each value",
            "right": "0: the two values cancel",
        },
        correct="right",
        why={
            "one": "The expected value adds each value times its probability, signs included: $0.5 \\cdot 1 + 0.5 \\cdot (-1)$. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "half": "The probabilities are the weights, not the answer: $0.5 \\cdot 1 + 0.5 \\cdot (-1)$. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "$0.5 \\cdot 1 + 0.5 \\cdot (-1) = 0$. Gradients that keep changing sign largely cancel in the same way when you add them up.",
        },
    )
    quick_plus_minus
    return (quick_plus_minus,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en), a long, narrow valley made gradient descent slow, and standardizing the features made the valley round. Inside a deep network, you cannot standardize every direction in which the loss can change: the valleys stay narrow. The learning rate must then be small enough for the steepest direction, and along the gentle directions the steps are tiny. Momentum lets gradient descent remember which way it has been going, and almost every optimizer used today has it.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "how-many-steps",
        question="Gradient descent on the loss $\\frac{1}{2}(w_1^2 + 100\\, w_2^2)$, a valley 100 times as steep across as along, starts at $(10, 1)$. With a learning rate of 0.019, close to its best, it needs 283 steps to get the loss below 0.001. Momentum, with its usual setting 0.9 and a learning rate of 0.01, about half as large. How many steps do you think it needs?",
        options={
            "more": "More than 283: its steps are about half as large",
            "same": "About 283: momentum only smooths the path",
            "right": "About 100: a third of that",
        },
        reveal="**About 100: 94 steps.** Along the valley, the gradient keeps pointing the same way, and momentum adds it up: its steps there grow to 10 times the learning rate times the gradient. Plain gradient descent with the same learning rate, 0.01, needs 539 steps.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Plain gradient descent steps by the learning rate times the gradient $g$ of this step only. **Momentum** keeps a sum $v$ of the gradients so far, in which each gradient counts $\beta$ times as much as the one after it, and steps by that:
        """),
        L.formula(r"[[grad:v]] \leftarrow [[hyper:\beta]] \, [[grad:v]] + [[grad:g]] \qquad [[param:w]] \leftarrow [[param:w]] - [[hyper:\eta]] \, [[grad:v]]"),
        mo.md(r"""
        $v$ starts at 0, and $\beta$ is usually 0.9. Written out, $v = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \dots$: the newest gradient counts fully, the one before 0.9 times, the one before that 0.81 times. Where the gradient keeps pointing the same way, as along a valley, $v$ grows until a step no longer changes it: then $v = \beta v + g$, so $v = g / (1 - \beta)$, 10 times $g$, and the steps grow tenfold. Where it keeps changing sign, as across a valley, the terms largely cancel, as $+1$ and $-1$ do in an expected value: $v$ ends up flipping between $+g / (1 + \beta)$ and $-g / (1 + \beta)$, about half of $g$, since $\beta \cdot (-g / (1 + \beta)) + g = g / (1 + \beta)$. $(1 - \beta)\, v$ is an **exponential moving average** of the gradients: a weighted average in which the weights shrink by a factor $\beta$ per step back. In PyTorch, `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` does this.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **momentum** | | stepping by a sum of the gradients so far, each counting $\beta$ times as much as the one after it | `momentum=0.9` |
        | $\beta$ | beta | how much of $v$ is kept each step; usually 0.9 | `beta` |
        | $v$ | v | the sum of the gradients so far, with older ones counting less; it starts at 0 | `v` |
        | **exponential moving average** | | a weighted average in which the weights shrink by the same factor per step back | `(1 - beta) * v` |
        """),
        L.watch_out("Momentum can overshoot: the path goes past the lowest point and comes back, as the picture shows, and the larger $\\beta$, the longer it keeps swinging. In the valley of the guess, momentum needs 707 steps with $\\beta = 0.99$, and 94 with 0.9. That is why $\\beta = 0.9$, not 0.99, is the usual choice."),
        mo.md(r"""
        The picture: 40 steps in the valley of the guess, from $(10, 1)$. The <span class="c-loss">contour lines</span> join the places with the same loss, around the lowest point; the <span class="c-param">points</span> are $(w_1, w_2)$ after each step. Without momentum, at a learning rate close to its best, every step jumps to the other side of the valley, and along it $w_1$ only gets from 10 to 4.6. With momentum, the path still swings across the valley, but slower, and those swings take longer to die out; along the valley it gets much further: it even overshoots the lowest point.
        """),
        mo.Html(C.path_picture({"plain": "without momentum, η = 0.019", "momentum": "momentum 0.9, η = 0.01", "after": "loss after 40 steps: {loss}",
                                "number": L.number,
                                "alt": "Two contour plots of a narrow valley with 40 steps from (10, 1). Without momentum, the steps zigzag across the valley and end near w1 = 4.6, loss 10.79. With momentum, the path swings more slowly, passes the lowest point and ends near (−1.4, 0.13), loss 1.73."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Gradient descent with momentum:

    1. Start with $v = 0$, one for every parameter.
    2. Each step, after the gradient $g$: $v \leftarrow \beta v + g$.
    3. Step: $w \leftarrow w - \eta v$.
    4. In PyTorch: `torch.optim.SGD(params, lr=..., momentum=0.9)`. Steps along a steady slope grow up to $1 / (1 - \beta)$ times as large, so with momentum a smaller learning rate can still go faster, as in the guess.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The valley of the guess, in NumPy: the gradient of $\frac{1}{2}(w_1^2 + 100\, w_2^2)$ is $(w_1, 100\, w_2)$. The comments with `#` say which step of the recipe each part is. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again. As it is, $v$ is just this step's gradient: plain gradient descent.
    """)
    return


@app.cell
def example_momentum():
    import numpy as np

    _lr = 0.01
    _beta = 0.9
    _w = np.array([10.0, 1.0])
    # 1. v starts at 0
    _v = np.zeros(2)
    _losses = []
    for _step in range(100):
        _g = np.array([_w[0], 100 * _w[1]])
        # 2. complete: _beta times the old _v, plus _g
        _v = _g
        # 3. the step
        _w = _w - _lr * _v
        _losses.append(0.5 * (_w[0] ** 2 + 100 * _w[1] ** 2))
    print(_losses[19], _losses[39], _losses[59], _losses[79], _losses[99])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell is plain gradient descent with learning rate 0.01: across the valley, $100 \cdot 0.01 = 1$, so one step puts $w_2$ at 0; along it, each step only takes 1% off $w_1$. The loss goes 33.4, 22.4, 15.0, 10.0 and 6.7 after 20, 40, 60, 80 and 100 steps. With momentum, it goes 8.5, 1.7, 0.047, 0.014 and 0.00044: after 100 steps, more than 10,000 times lower.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_zigzag():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Across and along
    """)
    return


@app.cell(hide_code=True)
def choice_zigzag_cell():
    choice_zigzag = L.choice(
        "zigzag",
        question="With $\\beta = 0.9$, the gradient of $w_1$ is about 1 in every step, and that of $w_2$ is about +1, −1, +1, −1 in turn. After many steps, about how large is $v$ for each?",
        options={
            "same": "About 1 for both: $v$ follows the gradient",
            "right": "About 10 for $w_1$, and about 0.5 for $w_2$, changing sign",
            "zero": "About 10 for $w_1$, and exactly 0 for $w_2$",
        },
        correct="right",
        why={
            "same": "$v$ adds the gradients up, each counting 0.9 times as much as the one after it. Along a steady slope, it grows until a step no longer changes it: $0.9 \\cdot 10 + 1 = 10$.",
            "zero": "The terms largely cancel, but not completely: the newest gradient counts fully, the one before only 0.9 times. $v$ ends up flipping between about +0.53 and −0.53.",
            "right": "For $w_1$, $v$ grows until a step no longer changes it: $0.9 \\cdot 10 + 1 = 10$. For $w_2$, it ends up flipping between $1 / 1.9 \\approx 0.53$ and $-0.53$: $0.9 \\cdot (-1 / 1.9) + 1 = 1 / 1.9$. So, compared with the gradient, the steps along the valley become about 19 times as large as those across it.",
        },
    )
    choice_zigzag
    return (choice_zigzag,)


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The steps of momentum

        Write `momentum_steps(gs, lr, beta)`: `gs` is a list of gradients of one parameter, one per step. Give back the list of steps that momentum takes, $\eta v$, by steps 1 to 3 of the recipe. The check calls `momentum_steps([1.0, 1.0, 1.0, 1.0], 0.1, 0.9)`, which should give `[0.1, 0.19, 0.271, 0.3439]`, and tries gradients that change sign, other values of `beta`, and a single gradient.
        """),
        L.task("Make `momentum_steps` give the step of momentum for each gradient."),
    ])
    return


@app.cell(hide_code=True)
def exercise_steps():
    box_steps = L.code_box("momentum-steps", lines=8)
    box_steps
    return (box_steps,)


@app.cell(hide_code=True)
def check_steps(box_steps):
    result_steps = L.check("momentum-steps", box_steps.value)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "momentum-steps",
        [
            "`v` starts at 0, before the loop over `gs`; in the loop, first the new `v`, then the step.",
            "The new `v` is `beta` times the old `v`, plus the gradient of this step; the step is `lr` times that new `v`.",
        ],
        why="`v = 0.0`, `steps = []`, and for each `g` in `gs`: `v = beta * v + g` and `steps.append(lr * v)`; then `return steps`. In the example: $v$ is 1, 1.9, 2.71 and 3.439, so the steps are 0.1, 0.19, 0.271 and 0.3439, on their way to $0.1 \\cdot 10 = 1$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_optimizer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your own optimizer with momentum

        Write a class `MomentumSGD` that does what `torch.optim.SGD(params, lr=lr, momentum=beta)` does. `MomentumSGD(params, lr, beta)` gets a list of tensors and keeps it as `self.params`, with `self.lr`, `self.beta`, and a list `self.vs` with one $v$ of zeros for each parameter: `torch.zeros_like(p)` makes a tensor of zeros with the shape of `p`. `zero_grad` is already in the box. `step()` does steps 2 and 3 of the recipe for every parameter, with `p.grad` as its gradient, inside `torch.no_grad()`, as in your training loops before you had an optimizer. The check trains a small linear model with it and compares the weights with what `torch.optim.SGD` gives: after one step, after five, and with other numbers. `.tolist()` turns a tensor into a list of numbers, for the comparison.
        """),
        L.task("Make `MomentumSGD` take the same steps as torch.optim.SGD with momentum."),
    ])
    return


@app.cell(hide_code=True)
def exercise_optimizer():
    box_optimizer = L.code_box("momentum-sgd", lines=20)
    box_optimizer
    return (box_optimizer,)


@app.cell(hide_code=True)
def check_optimizer(box_optimizer):
    result_optimizer = L.check("momentum-sgd", box_optimizer.value)
    result_optimizer
    return (result_optimizer,)


@app.cell(hide_code=True)
def hints_optimizer():
    L.hints(
        "momentum-sgd",
        [
            "In `__init__`, a loop over `params` that appends `torch.zeros_like(p)` to `self.vs`: one $v$ for each parameter, kept between steps.",
            "In `step`, inside `with torch.no_grad():`, go over the positions, `for i in range(len(self.params)):`, so that `self.params[i]` and `self.vs[i]` belong together.",
        ],
        why="`__init__`: `self.params = params`, `self.lr = lr`, `self.beta = beta`, `self.vs = []` and for each `p` in `params`: `self.vs.append(torch.zeros_like(p))`. `step`: `with torch.no_grad():`, and for each `i`: `p = self.params[i]`, `self.vs[i] = self.beta * self.vs[i] + p.grad` and `p -= self.lr * self.vs[i]`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The narrow valley of [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) is still narrow, but momentum adds up what keeps pointing the same way and lets the rest cancel, as $+1$ and $-1$ do in [Expected value and variance](/course/open/prob-expectation?lang=en). Your `MomentumSGD` has the shape of every PyTorch optimizer from [Let the optimizer take the step](/course/open/torch-optimizer?lang=en): it keeps something per parameter between steps. In the next lesson, *Adam*, the optimizer also keeps an exponential moving average of each squared gradient, so that every parameter gets steps of a size that fits it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_zigzag, result_steps, result_optimizer):
    L.wrap_up(choice_zigzag, result_steps, result_optimizer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
