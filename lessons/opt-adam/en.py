# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson opt-adam. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-adam", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Momentum](/course/open/opt-momentum?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_old_gradient():
    quick_old_gradient = L.choice(
        "quick-old-gradient",
        question="Momentum with $\\beta = 0.9$. How much does the gradient of three steps ago still count in $v$, compared with the newest one?",
        options={
            "right": "$0.9^3 \\approx 0.73$ times as much: 0.9 once more per step back",
            "nothing": "Not at all: $v$ only holds the newest gradient",
            "linear": "$1 - 3 \\cdot 0.1 = 0.7$ times as much: 0.1 less per step back",
        },
        correct="right",
        why={
            "nothing": "$v \\leftarrow \\beta v + g$ keeps the old gradients, each step back once more times $\\beta$. See [Momentum](/course/open/opt-momentum?lang=en).",
            "linear": "Each step back multiplies by 0.9 once more: how much a gradient counts shrinks by a factor, not by a fixed amount. See [Momentum](/course/open/opt-momentum?lang=en).",
            "right": "$v = g_t + 0.9\\, g_{t-1} + 0.81\\, g_{t-2} + 0.729\\, g_{t-3} + \\dots$: an exponential moving average, up to a factor. Adam keeps two of them.",
        },
    )
    quick_old_gradient
    return (quick_old_gradient,)


@app.cell(hide_code=True)
def quick_check_root_mean_square():
    quick_root_mean_square = L.choice(
        "quick-root-mean-square",
        question="The gradients of a parameter were 3, −3, 3 and −3. What is the square root of the mean of their squares?",
        options={
            "zero": "0: they cancel",
            "right": "3: the size of each gradient",
            "nine": "9: the mean of the squares",
        },
        correct="right",
        why={
            "zero": "They cancel in the mean, but squares are never negative: the mean of the squares is 9. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "nine": "9 is the mean of the squares; its square root is 3. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "The squares are all 9, their mean is 9, and its root is 3: the typical size of the gradient, whatever its sign. *The idea* below uses exactly this number.",
        },
    )
    quick_root_mean_square
    return (quick_root_mean_square,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    With gradient descent and momentum, every parameter steps by the same learning rate times its own gradient. In a network, some parameters get gradients a hundred times larger than others: the weights of other layers, of inputs with other units, of features that are seldom there. One learning rate cannot suit all of them, and it has to be small enough for the largest. Adam, the optimizer of this lesson, and its variant AdamW are what most networks are trained with today, transformers included.
    """)
    return


@app.cell(hide_code=True)
def predict_raw_pixels():
    guess_raw_pixels = L.guess(
        "raw-pixels",
        question="The MLP of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) trains for one epoch, but you forget to divide the pixels by 255: they run from 0 to 255. With SGD at a learning rate of 0.1, it read 82% of the test digits when the pixels ran from 0 to 1; now it stays at 10%. With Adam at its usual learning rate, 0.001, it read 84%. What do you think it reads now, with Adam?",
        options={
            "ten": "About 10%: the same problem as SGD",
            "half": "About 50%: it gets there, but much more slowly",
            "right": "About 80%: only a little less than before",
        },
        reveal="**About 80%: 801 of the 1,000.** At the start, the gradients of the first layer are now hundreds of times larger, and Adam divides every step by the size of its parameter's own gradients, so its steps hardly change. SGD needs a learning rate 100 times smaller, 0.001, to read 77%.",
    )
    guess_raw_pixels
    return (guess_raw_pixels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Adam** keeps two exponential moving averages per parameter: $m$ of its gradients, as momentum does, and $s$ of its squared gradients:
        """),
        L.formula(r"\begin{aligned} [[grad:m]] &\leftarrow [[hyper:\beta_1]] [[grad:m]] + (1 - [[hyper:\beta_1]]) [[grad:g]] \\ [[grad:s]] &\leftarrow [[hyper:\beta_2]] [[grad:s]] + (1 - [[hyper:\beta_2]]) [[grad:g]]^2 \end{aligned}"),
        mo.md(r"""
        $\sqrt{s}$ is the typical size of this parameter's gradient, whatever its sign, as in the quick check. Adam divides the step by it:
        """),
        L.formula(r"\begin{aligned} [[param:w]] &\leftarrow [[param:w]] - [[hyper:\eta]] \, \frac{[[grad:\hat{m}]]}{\sqrt{[[grad:\hat{s}]]} + [[hyper:\varepsilon]]} \\ [[grad:\hat{m}]] &= \frac{[[grad:m]]}{1 - [[hyper:\beta_1]]^t} \qquad [[grad:\hat{s}]] = \frac{[[grad:s]]}{1 - [[hyper:\beta_2]]^t} \end{aligned}"),
        mo.md(r"""
        Where the gradient keeps pointing the same way, $\hat{m} \approx g$ and $\sqrt{\hat{s}} \approx |g|$, so every step is about $\eta$, whether $g$ is 0.01 or 10. Where it keeps changing sign, $\hat{m}$ is small and so is the step. $m$ and $s$ start at 0, so in the first steps they are too small; $s$ much more so than $m$, because $\beta_2$ is closer to 1. Dividing by $1 - \beta^t$, with $t$ the number of the step, corrects that: at $t = 1$, $\hat{m} = g$ and $\hat{s} = g^2$. The usual numbers are $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\varepsilon = 10^{-8}$, and `torch.optim.Adam(model.parameters(), lr=0.001)` uses them.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Adam** | | an optimizer that divides each parameter's momentum by the root of the average of its squared gradients | `torch.optim.Adam` |
        | $\beta_1$, $\beta_2$ | beta one, beta two | how much of $m$ and of $s$ is kept each step: usually 0.9 and 0.999 | `betas=(0.9, 0.999)` |
        | $m$, $s$ | m, s | the moving averages of the gradient and of its square, per parameter | `m`, `s` |
        | $\hat{m}$, $\hat{s}$ | m hat, s hat | $m$ and $s$ corrected for their start at 0 | `m / (1 - beta1 ** t)` |
        | $t$ | t | the number of the step: 1, 2, 3, … | `t` |
        | $\varepsilon$ | epsilon | a tiny number, $10^{-8}$, that keeps the division from blowing up where $\hat{s}$ is 0 | `1e-8` |
        """),
        L.watch_out("Adam's learning rate means something different from SGD's: it is about the size of every step, not a factor on the gradient. So 0.1, fine for SGD on the digits, is far too large for Adam; start at 0.001. And with a penalty on large weights in the loss, its part of the gradient is divided by $\\sqrt{\\hat{s}}$ like the rest, so it works much less on parameters with large gradients. The next lesson, *AdamW: weight decay done right*, is about that."),
        mo.md(r"""
        The picture: for 1,500 weights of the hidden layer of the digit MLP, the size of the gradient and the size of the step, at step 51 of training, with SGD at learning rate 0.1 and with Adam at 0.001, on log scales. With SGD, every step is 0.1 times its gradient: a straight line, and the weights with the smallest gradients hardly move. With Adam, the steps lie between about 0.00001 and 0.001, for gradients that differ a thousandfold.
        """),
        mo.Html(C.step_picture({"sgd": "SGD, η = 0.1", "adam": "Adam, η = 0.001", "x": "size of the gradient", "y": "size of the step",
                                "alt": "Two scatter plots on log scales of the size of the step against the size of the gradient, for 1,500 weights. With SGD, the points lie on a straight line: the step is 0.1 times the gradient. With Adam, the points form a flat band between about 0.00001 and 0.001, for gradients from about 0.00001 to 0.01."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Adam, for every parameter:

    1. Start with $m = 0$, $s = 0$ and the step counter $t = 0$.
    2. Each step, after the gradient $g$: $t$ one up, $m \leftarrow \beta_1 m + (1 - \beta_1) g$ and $s \leftarrow \beta_2 s + (1 - \beta_2) g^2$.
    3. Correct both: $\hat{m} = m / (1 - \beta_1^t)$ and $\hat{s} = s / (1 - \beta_2^t)$.
    4. Step: $w \leftarrow w - \eta \, \hat{m} / (\sqrt{\hat{s}} + \varepsilon)$. In PyTorch: `torch.optim.Adam(params, lr=0.001)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two parameters whose gradients are always 0.01 and 10: a thousandfold apart. With SGD and learning rate 0.001, their steps would be 0.00001 and 0.01. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again. As it is, the corrections are left out.
    """)
    return


@app.cell
def example_adam():
    import numpy as np

    _lr = 0.001
    _beta1 = 0.9
    _beta2 = 0.999
    # 1. m and s start at 0, one for each parameter
    _m = np.zeros(2)
    _s = np.zeros(2)
    for _t in range(1, 6):
        _g = np.array([0.01, 10.0])
        # 2. the two moving averages
        _m = _beta1 * _m + (1 - _beta1) * _g
        _s = _beta2 * _s + (1 - _beta2) * _g**2
        # 3. complete: divide _m by 1 - _beta1 ** _t, and _s by 1 - _beta2 ** _t
        _m_hat = _m
        _s_hat = _s
        # 4. the step of each parameter
        print(_t, _lr * _m_hat / (_s_hat**0.5 + 1e-8))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    For both parameters, the steps are the same, up to the last digits: the gradient of 10 is a thousand times larger, but so is its root of $s$. As it is, without the corrections, the first step is 0.0032, more than three times the learning rate, and the next ones grow: 0.0042, 0.0050, 0.0054, 0.0058. $s$ starts further below its value than $m$, so the first steps come out too large. With the corrections, every step is exactly 0.001 for both: the learning rate, whether the gradient is 0.01 or 10.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The first step
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-step",
        question="At step $t = 1$, a parameter gets the gradient $g = -0.004$. With the corrections and $\\eta = 0.001$, by how much does Adam change the parameter?",
        options={
            "sgd": "$-\\eta g = 0.000004$: the learning rate times the gradient, as in SGD",
            "zero": "Almost 0: $m$ and $s$ have only just started",
            "right": "$+0.001$: the learning rate, against the sign of $g$",
        },
        correct="right",
        why={
            "sgd": "Adam divides by the root of $\\hat{s}$, the size of the gradient itself, so the size of $g$ drops out.",
            "zero": "$m$ and $s$ are small after one step, but the step divides one by the root of the other, so their size drops out; with the corrections, $\\hat{m} = g$ and $\\hat{s} = g^2$ after the first step.",
            "right": "$\\hat{m} = -0.004$ and $\\sqrt{\\hat{s}} = 0.004$, so the step is $0.001 \\cdot (-0.004 / 0.004) = -0.001$, and $w$ minus that step goes up by $\\eta$: against the sign of the gradient, downhill on the loss. In exercise 2, a step is that number itself: here $-0.001$.",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The steps of Adam

        Write `adam_steps(gs, lr, beta1, beta2)`: `gs` is a list of gradients of one parameter, one per step. Give back the list of steps that Adam takes, $\eta \hat{m} / (\sqrt{\hat{s}} + \varepsilon)$ with $\varepsilon = 10^{-8}$, by the recipe. The check calls `adam_steps([4.0, 1.0, -2.0], 0.1, 0.9, 0.999)`, which should give about `[0.1, 0.08306, 0.02986]`, and tries tiny and huge gradients, a gradient of 0, and other betas.
        """),
        L.task("Make `adam_steps` give the step of Adam for each gradient."),
    ])
    return


@app.cell(hide_code=True)
def exercise_steps():
    box_steps = L.code_box("adam-steps", lines=14)
    box_steps
    return (box_steps,)


@app.cell(hide_code=True)
def check_steps(box_steps):
    result_steps = L.check("adam-steps", box_steps.value)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "adam-steps",
        [
            "`m`, `s` and `t` start at 0, before the loop; in the loop, `t` one up first, then steps 2 to 4 of the recipe, as in the worked example.",
            "The corrections use the number of this step: `m / (1 - beta1 ** t)` and `s / (1 - beta2 ** t)`. The step is `lr * m_hat / (s_hat ** 0.5 + 1e-8)`.",
        ],
        why="`m = s = 0.0`, `t = 0`, `steps = []`, and for each `g`: `t = t + 1`, `m = beta1 * m + (1 - beta1) * g`, `s = beta2 * s + (1 - beta2) * g ** 2`, `m_hat = m / (1 - beta1 ** t)`, `s_hat = s / (1 - beta2 ** t)` and `steps.append(lr * m_hat / (s_hat ** 0.5 + 1e-8))`; then `return steps`. The first step is $0.1 \\cdot 4 / 4 = 0.1$; at the third, the gradient has changed sign, so $\\hat{m}$ is small and so is the step.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_optimizer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your own Adam

        Write a class `AdamOptimizer` that does what `torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))` does, in the shape of your `MomentumSGD` from [Momentum](/course/open/opt-momentum?lang=en). `AdamOptimizer(params, lr, beta1, beta2)` keeps `self.params`, `self.lr`, `self.beta1`, `self.beta2`, a step counter `self.t = 0`, and two lists, `self.ms` and `self.ss`, with a tensor of zeros for each parameter. `zero_grad` is already in the box. `step()` raises `self.t` by one and then, inside `torch.no_grad()`, does steps 2 to 4 of the recipe for every parameter, with $\varepsilon = 10^{-8}$. The check trains a small linear model with it and compares the weights with what `torch.optim.Adam` gives: after one step, after five, and with other numbers.
        """),
        L.task("Make `AdamOptimizer` take the same steps as torch.optim.Adam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_optimizer():
    box_optimizer = L.code_box("adam-optimizer", lines=26)
    box_optimizer
    return (box_optimizer,)


@app.cell(hide_code=True)
def check_optimizer(box_optimizer):
    result_optimizer = L.check("adam-optimizer", box_optimizer.value)
    result_optimizer
    return (result_optimizer,)


@app.cell(hide_code=True)
def hints_optimizer():
    L.hints(
        "adam-optimizer",
        [
            "`__init__` is that of your `MomentumSGD`, with two lists instead of one, both filled with `torch.zeros_like(p)`, and `self.t = 0`.",
            "`self.t` goes up once per `step()`, not once per parameter: before the loop over the parameters. In the loop, `self.ms[i]` and `self.ss[i]` belong to `self.params[i]`.",
        ],
        why="`__init__`: `self.params = params`, `self.lr = lr`, `self.beta1 = beta1`, `self.beta2 = beta2`, `self.t = 0`, `self.ms = []`, `self.ss = []`, and for each `p` in `params`: `self.ms.append(torch.zeros_like(p))` and `self.ss.append(torch.zeros_like(p))`. `step`: `self.t = self.t + 1`, then `with torch.no_grad():` and for each `i`: `p = self.params[i]`, `self.ms[i] = self.beta1 * self.ms[i] + (1 - self.beta1) * p.grad`, `self.ss[i] = self.beta2 * self.ss[i] + (1 - self.beta2) * p.grad ** 2`, `m_hat = self.ms[i] / (1 - self.beta1 ** self.t)`, `s_hat = self.ss[i] / (1 - self.beta2 ** self.t)` and `p -= self.lr * m_hat / (s_hat ** 0.5 + 1e-8)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** $m$ is the momentum of [Momentum](/course/open/opt-momentum?lang=en), as an average; $\sqrt{s}$ is the root of a mean of squares, as a standard deviation is in [Expected value and variance](/course/open/prob-expectation?lang=en). Dividing one by the other gives every parameter steps of about the learning rate, so the scale of the inputs, as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en), matters much less. In the next lesson, *AdamW: weight decay done right*, you see what that does to a penalty on large weights, and why transformers are trained with AdamW.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_steps, result_optimizer):
    L.wrap_up(choice_first, result_steps, result_optimizer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
