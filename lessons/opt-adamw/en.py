# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson opt-adamw. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow regularization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-adamw", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Adam](/course/open/opt-adam?lang=en) and [Regularization and weight decay](/course/open/ml-regularization?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hundredfold():
    quick_hundredfold = L.choice(
        "quick-hundredfold",
        question="Every gradient of one parameter becomes 100 times larger. Why does Adam's step for it hardly change?",
        options={
            "lr": "It lowers its own learning rate a hundredfold to make up for it",
            "right": "It divides by $\\sqrt{\\hat{s}}$, which grows a hundredfold too",
            "clip": "It cuts every gradient off at 1 before it uses it",
        },
        correct="right",
        why={
            "lr": "The learning rate stays as you set it. What changes is $\\sqrt{\\hat{s}}$, by which Adam divides. See [Adam](/course/open/opt-adam?lang=en).",
            "clip": "Adam cuts nothing off: it divides $\\hat{m}$ by $\\sqrt{\\hat{s}}$. See [Adam](/course/open/opt-adam?lang=en).",
            "right": "$\\hat{m}$ and $\\sqrt{\\hat{s}}$ both grow a hundredfold, so their ratio stays the same. Today you see what that does to a penalty that is part of the gradient.",
        },
    )
    quick_hundredfold
    return (quick_hundredfold,)


@app.cell(hide_code=True)
def quick_check_l2():
    quick_l2 = L.choice(
        "quick-l2",
        question="L2 regularization adds $\\lambda \\sum_j w_j^2$ to the loss. With plain gradient descent, what does that do to each weight in every update step?",
        options={
            "zero": "It sets the smallest weights to exactly 0 and leaves the rest alone",
            "lr": "It makes the learning rate a little smaller in every step",
            "right": "It pulls each weight toward 0, in proportion to its size",
        },
        correct="right",
        why={
            "zero": "L2 makes weights smaller, not exactly 0: its pull, $2\\lambda w_j$, gets smaller as the weight does. See [Regularization and weight decay](/course/open/ml-regularization?lang=en).",
            "lr": "The learning rate stays the same; the slope gets $2\\lambda w_j$ extra. See [Regularization and weight decay](/course/open/ml-regularization?lang=en).",
            "right": "The slope gets $2 \\lambda w_j$ extra, so each step shrinks $w_j$ by a fraction: weight decay. With gradient descent, L2 and weight decay are the same thing.",
        },
    )
    quick_l2
    return (quick_l2,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    With gradient descent, an L2 penalty and weight decay are the same: every step shrinks every weight by the same fraction. With Adam they are not, and much code used the L2 penalty with Adam without noticing what it did. In 2017, Ilya Loshchilov and Frank Hutter showed the difference and how to fix it. Their fix, AdamW, is the optimizer that most transformers are trained with today.
    """)
    return


@app.cell(hide_code=True)
def predict_quiet():
    guess_quiet = L.guess(
        "quiet-weight",
        question="A weight starts at 1 and gets no gradient from the data: it only gets the L2 penalty, $\\lambda w$, as its gradient. Adam trains it with a learning rate of 0.01. With $\\lambda = 0.01$, the weight is below 0.01 after 213 steps. With $\\lambda = 0.0001$, a penalty a hundred times weaker, after how many steps do you think?",
        options={
            "hundred": "About 21,300: a hundred times weaker, a hundred times slower",
            "right": "After 213 steps as well: λ makes no difference here",
            "never": "Never: a penalty that weak hardly does anything",
        },
        reveal="**After 213 steps as well.** Adam divides the gradient by the root of its own mean square. When the penalty is the whole gradient, that division takes its size out completely: the steps are the same whatever $\\lambda$ is, the first ones about the learning rate, 0.01. The strength of the penalty has no effect at all on this weight.",
    )
    guess_quiet
    return (guess_quiet,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An L2 penalty $\frac{\lambda}{2} w^2$ adds $\lambda w$ to the gradient of each weight. Adam then treats it like the rest of the gradient: it goes into $m$ and $s$, and is divided by $\sqrt{\hat{s}}$. For a weight whose gradient is mostly the penalty, that division takes the penalty's size away: the weight takes the same steps toward 0, the first ones about $\eta$, whatever $\lambda$ is. For a weight with large gradients from the data, $\sqrt{\hat{s}}$ is large, and the penalty is drowned: that weight is hardly pulled toward 0 at all. How much a weight is regularized then depends on the size of its gradients, not only on $\lambda$.

        **AdamW** takes the penalty out of the gradient. Adam steps with the gradient of the loss alone, and separately every weight shrinks by the same fraction $\eta \lambda$:
        """),
        L.formula(r"\begin{aligned} [[param:w]] &\leftarrow (1 - [[hyper:\eta]] [[hyper:\lambda]]) \, [[param:w]] \\ [[param:w]] &\leftarrow [[param:w]] - [[hyper:\eta]] \, \frac{[[grad:\hat{m}]]}{\sqrt{[[grad:\hat{s}]]} + [[hyper:\varepsilon]]} \end{aligned}"),
        mo.md(r"""
        That is **decoupled weight decay**: the decay no longer passes through Adam's division, so $\lambda$ means the same for every weight. In PyTorch, `torch.optim.Adam(params, weight_decay=0.1)` adds the penalty to the gradient, and `torch.optim.AdamW(params, weight_decay=0.1)` decays separately; AdamW first shrinks, then takes Adam's step. Weight decay is usually only given to the weight matrices: bias terms and the $\gamma$ and $\beta$ of normalization layers get none, because they set a shift or a scale, not a connection between neurons. For that, you give AdamW two **parameter groups**, each with its own `weight_decay`.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **AdamW** | Adam W | Adam with decoupled weight decay: every weight shrinks by the fraction $\eta\lambda$ per step, apart from Adam's step | `torch.optim.AdamW` |
        | **decoupled weight decay** | | shrinking each weight directly, instead of adding a penalty to its gradient | `w = w - lr * wd * w` |
        | **parameter groups** | | a list of dicts, each with its own parameters and settings, given to an optimizer | `[{"params": decay, "weight_decay": 0.1}, ...]` |
        | `p.dim()` | p dim | the number of dimensions of a tensor: 2 for a weight matrix, 1 for bias terms | `p.dim() >= 2` |
        """),
        L.watch_out("PyTorch's `weight_decay` is the $\\lambda$ in $\\lambda w$, the gradient of $\\frac{\\lambda}{2} w^2$. [Regularization and weight decay](/course/open/ml-regularization?lang=en) wrote the penalty as $\\lambda w^2$, with gradient $2\\lambda w$, so its $\\lambda$ is half of PyTorch's `weight_decay`. PyTorch's default for AdamW is 0.01; for transformers, 0.1 is usual. Weight decay with AdamW needs many steps to matter: with $\\eta = 0.001$ and $\\lambda = 0.1$, a weight loses one ten-thousandth per step."),
        mo.md(r"""
        The picture: the weight of the guess, without a gradient from the data, over 1,000 steps at learning rate 0.01, for $\lambda = 0.01$ and $\lambda = 0.0001$. With the L2 penalty inside Adam, the two lines lie on top of each other: the weight is almost 0 after about 200 steps either way. With AdamW, it shrinks by the fraction $\eta\lambda$ per step: to 0.905 after 1,000 steps with $\lambda = 0.01$, and to 0.999 with $\lambda = 0.0001$.
        """),
        mo.Html(C.quiet_picture({"l2": "L2 penalty inside Adam", "adamw": "AdamW", "x": "step", "y": "w",
                                 "big": "λ = 0.01", "small": "λ = 0.0001", "both": "both values of λ",
                                 "alt": "Two panels over 1,000 steps. With an L2 penalty inside Adam, the weight falls from 1 to 0 in about 200 steps, the same for both values of lambda. With AdamW, it falls slowly to 0.905 for lambda 0.01 and stays at 0.999 for lambda 0.0001."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    AdamW, for every parameter:

    1. Start with $m = 0$, $s = 0$ and $t = 0$.
    2. Each step: the gradient $g$ of the loss without a penalty; $t$ one up, $m$ and $s$ as in Adam, and both corrected.
    3. Shrink the weight: $w \leftarrow (1 - \eta\lambda)\, w$.
    4. Adam's step: $w \leftarrow w - \eta\, \hat{m} / (\sqrt{\hat{s}} + \varepsilon)$.
    5. In PyTorch: `torch.optim.AdamW(groups, lr=...)`, with weight decay only for the weight matrices.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two weights, both starting at 1: a quiet one that gets no gradient from the data, and a busy one whose gradient is random noise, from a generator with a fixed seed: `_rng.normal()` gives a number from the normal distribution with mean 0 and standard deviation 1, as `torch.randn` does. 1,000 steps of AdamW with $\eta = 0.01$ and $\lambda = 0.1$. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again. As it is, the cell is Adam without any weight decay.
    """)
    return


@app.cell
def example_adamw():
    import numpy as np

    _rng = np.random.default_rng(0)
    _lr = 0.01
    _wd = 0.1
    _w = np.array([1.0, 1.0])
    # 1. m and s start at 0
    _m = np.zeros(2)
    _s = np.zeros(2)
    for _t in range(1, 1001):
        # 2. the gradient from the data: none for the quiet weight, noise for the busy one; m and s, corrected
        _g = np.array([0.0, _rng.normal()])
        _m = 0.9 * _m + 0.1 * _g
        _s = 0.999 * _s + 0.001 * _g**2
        _m_hat = _m / (1 - 0.9**_t)
        _s_hat = _s / (1 - 0.999**_t)
        # 3. complete: shrink _w by the fraction _lr * _wd
        # 4. Adam's step
        _w = _w - _lr * _m_hat / (_s_hat**0.5 + 1e-8)
    print(_w)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the quiet weight stays at exactly 1: Adam has no gradient to step with, so nothing moves it. The busy one wanders with its noise, to 1.45. With step 3, both shrink by the fraction $0.01 \cdot 0.1 = 0.001$ per step: the quiet one to $0.999^{1000} \approx 0.368$, and the busy one, which also takes Adam's steps, to 0.75. The decay is the same for both, whatever their gradients.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_arg():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two optimizers, one argument
    """)
    return


@app.cell(hide_code=True)
def choice_arg_cell():
    choice_arg = L.choice(
        "weight-decay-arg",
        question="Your code says `torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)`. What does `weight_decay` do there?",
        options={
            "adamw": "What AdamW does: every weight shrinks by the fraction `lr * 0.1` per step",
            "right": "It adds `0.1 * w` to each gradient: Adam divides that by $\\sqrt{\\hat{s}}$ too",
            "ignored": "Nothing: plain Adam has no weight decay, so it ignores the argument",
        },
        correct="right",
        why={
            "adamw": "Only `torch.optim.AdamW` decays apart from the gradient. `torch.optim.Adam` with `weight_decay` is an L2 penalty inside Adam.",
            "ignored": "It does something, only not what the name suggests: it adds the penalty to the gradient.",
            "right": "It is an L2 penalty inside Adam, so it is divided by $\\sqrt{\\hat{s}}$ with the rest: weights with large gradients hardly feel it. For decoupled weight decay, use `torch.optim.AdamW`.",
        },
    )
    choice_arg
    return (choice_arg,)


@app.cell(hide_code=True)
def text_exercise_path():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One weight under AdamW

        Write `adamw_path(gs, w, lr, wd)`: `gs` is a list of gradients of the loss for one weight, one per step, and `w` its starting value. Give back the list of `w` after each step of AdamW, by steps 1 to 4 of the recipe, with $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\varepsilon = 10^{-8}$: first shrink, then Adam's step. The check calls `adamw_path([0.3, 0.1, 0.0], 1.0, 0.1, 0.5)`, which should give about `[0.85, 0.72039, 0.61704]`, and tries a weight without gradients, a weight decay of 0, and other numbers.
        """),
        L.task("Make `adamw_path` give where the weight is after each step of AdamW."),
    ])
    return


@app.cell(hide_code=True)
def exercise_path():
    box_path = L.code_box("adamw-path", lines=15)
    box_path
    return (box_path,)


@app.cell(hide_code=True)
def check_path(box_path):
    result_path = L.check("adamw-path", box_path.value)
    result_path
    return (result_path,)


@app.cell(hide_code=True)
def hints_path():
    L.hints(
        "adamw-path",
        [
            "Your `adam_steps` from [Adam](/course/open/opt-adam?lang=en) gives the steps; here `w` changes with each one, and the list holds `w` itself.",
            "In each step, shrink first, with the `w` from the step before: `w = w - lr * wd * w`. Then take Adam's step from that `w`.",
        ],
        why="`m = s = 0.0`, `t = 0`, `ws = []`, and for each `g`: `t = t + 1`, `m = 0.9 * m + 0.1 * g`, `s = 0.999 * s + 0.001 * g ** 2`, `m_hat = m / (1 - 0.9 ** t)`, `s_hat = s / (1 - 0.999 ** t)`, `w = w - lr * wd * w`, `w = w - lr * m_hat / (s_hat ** 0.5 + 1e-8)` and `ws.append(w)`; then `return ws`. In the example, the first step shrinks 1 to $1 - 0.1 \\cdot 0.5 = 0.95$, and Adam's first step takes 0.1 off: 0.85.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_groups():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Weight decay only for the weight matrices

        Write `param_groups(model, wd)`: go over `model.parameters()` and put every parameter with 2 or more dimensions, the weight matrices, in one list, and the others, bias terms and the $\gamma$ and $\beta$ of normalization layers, in another. Give back a list of two dicts: `{"params": ..., "weight_decay": wd}` for the first list and `{"params": ..., "weight_decay": 0.0}` for the second. `torch.optim.AdamW(param_groups(model, 0.1), lr=0.001)` then decays only the weight matrices. The check makes a small network with two `nn.Linear` layers and an `nn.LayerNorm`, and counts, for each group, how many numbers its parameters hold (`p.numel()` gives that for one tensor) and which weight decay it has.
        """),
        L.task("Make `param_groups` give the weight matrices weight decay, and the rest none."),
    ])
    return


@app.cell(hide_code=True)
def exercise_groups():
    box_groups = L.code_box("param-groups", lines=10)
    box_groups
    return (box_groups,)


@app.cell(hide_code=True)
def check_groups(box_groups):
    result_groups = L.check("param-groups", box_groups.value)
    result_groups
    return (result_groups,)


@app.cell(hide_code=True)
def hints_groups():
    L.hints(
        "param-groups",
        [
            "Two empty lists, then a loop over `model.parameters()` with an `if` on `p.dim() >= 2`.",
            "A group is a dict with the keys `\"params\"` and `\"weight_decay\"`; the function gives back a list of the two.",
        ],
        why="`decay = []`, `no_decay = []`; for each `p` in `model.parameters()`: `decay.append(p)` if `p.dim() >= 2`, else `no_decay.append(p)`; then `return [{\"params\": decay, \"weight_decay\": wd}, {\"params\": no_decay, \"weight_decay\": 0.0}]`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** With the gradient descent of [Regularization and weight decay](/course/open/ml-regularization?lang=en), an L2 penalty is weight decay. [Adam](/course/open/opt-adam?lang=en) divides every part of the gradient by $\sqrt{\hat{s}}$, the penalty included, and that breaks the equality; AdamW puts the decay back outside the division. In the next lesson, *Learning-rate schedules*, the learning rate itself changes during training: warmup at the start, a slow decay to the end.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_arg, result_path, result_groups):
    L.wrap_up(choice_arg, result_path, result_groups)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
