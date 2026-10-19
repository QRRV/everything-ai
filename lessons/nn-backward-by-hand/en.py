# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-backward-by-hand. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-backward-by-hand", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) and [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="$u = 5 - x$ and $y = u^2$. What is $\\frac{dy}{dx}$ at $x = 2$?",
        options={
            "outer": "6",
            "right": "−6",
            "inner": "−1",
        },
        correct="right",
        why={
            "outer": "That is the outer slope $2u$ alone. What does $u$ do when $x$ goes up? See [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "inner": "That is the inner slope alone. The chain rule multiplies it by what? See [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "right": "At $x = 2$, $u = 3$, so the outer slope $2u$ is 6. The inner slope is −1: $u$ goes down when $x$ goes up. Together: 6 × (−1) = −6.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_passes():
    quick_passes = L.choice(
        "quick-passes",
        question="You measure every slope of a model with $N$ parameters by moving each parameter in turn. How many forward passes does that take?",
        options={
            "right": "N + 1",
            "two": "2",
            "square": "N²",
        },
        correct="right",
        why={
            "two": "That is enough for one slope. How many slopes do you need? See [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en).",
            "square": "N² is how the FLOPs grow, not the number of forward passes. See [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en).",
            "right": "One where you are, and one for each parameter you move: N + 1 forward passes, each of about 2N FLOPs.",
        },
    )
    quick_passes
    return (quick_passes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en), measuring every slope by moving cost a forward pass per parameter. But the slopes are not unrelated: the loss reaches every parameter through the same few steps, such as the prediction, the residual and the square. [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) tells you how slopes combine along such steps. This lesson uses that, step by step, on a calculation you can follow by hand. Every library that trains networks, such as PyTorch, works this way.

    **An everyday example.** One euro buys 1.1 dollars, and one dollar buys 150 yen. So one euro buys 1.1 × 150 = 165 yen: along the way, you multiply the rates.
    """)
    return


@app.cell(hide_code=True)
def predict_reuse():
    guess_reuse = L.guess(
        "reuse",
        question="The loss of one student is $\\mathcal{L} = r^2$, with the residual $r = 7 - \\hat{y}$ and the prediction $\\hat{y} = w \\cdot 4 + b$. Both $\\frac{\\partial \\mathcal{L}}{\\partial w}$ and $\\frac{\\partial \\mathcal{L}}{\\partial b}$ go through $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}}$. Do you think you have to work out $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}}$ twice, once for each?",
        options={
            "twice": "Yes: once for $w$ and once for $b$",
            "right": "No: once, and both use it",
            "equal": "Only when $w$ and $b$ are equal",
        },
        reveal="**Once.** Work back from the loss: first $\\frac{\\partial \\mathcal{L}}{\\partial r}$, then $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}}$, and from that one number both $\\frac{\\partial \\mathcal{L}}{\\partial b}$ and $\\frac{\\partial \\mathcal{L}}{\\partial w}$. Every slope behind it reuses it. That is why one walk back gives every slope.",
    )
    guess_reuse
    return (guess_reuse,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Write the calculation as a **computational graph**: one node per value, with arrows from the values that go into it. The forward pass works out the values from the inputs to the loss, and keeps them. For one student who studied $x = 4$ hours and got a 7, at $w = 1$ and $b = 1$:
        """),
        L.formula(r"m = [[param:w]] \cdot x \qquad [[pred:\hat{y}]] = m + [[param:b]] \qquad [[residual:r]] = 7 - [[pred:\hat{y}]] \qquad [[loss:\mathcal{L}]] = [[residual:r]]^2"),
        mo.md(r"""
        Each node only knows its own operation. Its **local gradient** is its slope with respect to one input: for $m = w \cdot x$ it is $\frac{\partial m}{\partial w} = x$; for $\hat{y} = m + b$ it is 1 for both inputs; for $r = 7 - \hat{y}$ it is −1; for $\mathcal{L} = r^2$ it is $2r$.

        The **backward pass** walks from the loss back to the parameters. It starts with $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$, and at each arrow multiplies by the local gradient: the slope of an input is the slope of the node it goes into, times the local gradient between them. That is the chain rule, one step at a time:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w}]] = [[grad:\frac{\partial \mathcal{L}}{\partial r}]] \cdot \frac{\partial r}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial m} \cdot \frac{\partial m}{\partial w} = 4 \cdot (-1) \cdot 1 \cdot 4 = -16"),
        mo.md(r"""
        A forward pass and then a backward pass is **backpropagation**. The backward pass does a multiplication or two per arrow, so it gives every slope for a few times the cost of a forward pass: $O(N)$, not $O(N^2)$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **computational graph** | | a calculation drawn as nodes, one per value, with arrows from the values that go into each | |
        | **local gradient** | | the slope of one node with respect to one of its inputs, from its own operation alone, such as $\frac{\partial m}{\partial w} = x$ | `dm_dw` |
        | **backward pass** | | from the loss back to the parameters: the slope of each input is the slope after it times the local gradient | |
        | **backpropagation** | | a forward pass that keeps every value, then a backward pass that gives every slope | |
        """),
        L.watch_out("When a value goes into more than one node, it gets a slope back along each arrow. Its slope is the sum of them: add them up, do not keep only one."),
        mo.md(r"""
        Move the slider: it walks the backward pass, one step at a time. Each node shows its value from the forward pass, and, once the backward pass has reached it, its <span class="c-grad">slope</span>; until then, the step at which it gets it. The <span class="c-grad">numbers at the arrows</span> are the local gradients.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    _say = [
        "Step 0: the forward pass is done, and every value is kept. The backward pass has not started.",
        "Step 1: the loss with respect to itself: $\\frac{\\partial \\mathcal{L}}{\\partial \\mathcal{L}} = 1$.",
        "Step 2: $\\mathcal{L} = r^2$ has the local gradient $2r = 4$, so $\\frac{\\partial \\mathcal{L}}{\\partial r} = 1 \\cdot 4 = 4$.",
        "Step 3: $r = 7 - \\hat{y}$ has the local gradient −1, so $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}} = 4 \\cdot (-1) = -4$.",
        "Step 4: $\\hat{y} = m + b$ has the local gradient 1 for both, so $\\frac{\\partial \\mathcal{L}}{\\partial m} = -4$ and $\\frac{\\partial \\mathcal{L}}{\\partial b} = -4$.",
        "Step 5: $m = w \\cdot x$ has the local gradient $x = 4$ for $w$, so $\\frac{\\partial \\mathcal{L}}{\\partial w} = -4 \\cdot 4 = -16$. Every slope is known.",
    ]

    def _frame(step):
        step = int(step)
        return (C.picture(step, {"title": "The backward pass, step by step", "later": "at step {n}",
                                 "alt": f"A computational graph from top to bottom: w = 1 and x = 4 go into m = w · x = 4; m and b = 1 go into ŷ = m + b = 5; ŷ goes into r = 7 − ŷ = 2; r goes into L = r² = 4. The backward pass reaches L at step 1, r at step 2, ŷ at step 3, m and b at step 4, and w at step 5. At step {step} of 5: {', '.join(C.known(step)) or 'no slope yet'}."}).text,
                _say[step])

    explore_step = L.explore("step", "step", _frame, start=0, low=0, high=5, step=1, role="", task="Why does $b$ get its slope a step before $w$, though both are parameters?")
    explore_step
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    $b$ gets its slope at step 4, $w$ only at step 5. $b$ goes straight into $\hat{y}$, while $w$ first goes into $m$: one arrow further from the loss. The backward pass goes back one node at a time, so the more arrows lie between a value and the loss, the later its slope comes.

    ### Recipe

    To get every slope of a calculation:

    1. Forward pass: work out every value, from the inputs to the loss, and keep them.
    2. Start at the loss: its slope with respect to itself is 1.
    3. Walk back, node by node: the slope of an input is the slope of the node after it, times the local gradient between them.
    4. Where a value goes into more than one node, add up the slopes that come back.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The student of the picture, in code: the forward pass, then the backward pass. A name such as `dloss_dr` stands for $\frac{\partial \mathcal{L}}{\partial r}$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_backward():
    _x = 4
    _y = 7
    _w = 1
    _b = 1
    # 1. the forward pass, every value kept
    _m = _w * _x
    _y_pred = _m + _b
    _r = _y - _y_pred
    _loss = _r ** 2
    print(_m, _y_pred, _r, _loss)
    # 2. the loss with respect to itself
    _dloss_dloss = 1
    # 3. back, node by node: the slope after it times the local gradient
    _dloss_dr = _dloss_dloss * 2 * _r
    _dloss_dy_pred = _dloss_dr * -1
    _dloss_dm = _dloss_dy_pred * 1
    _dloss_db = _dloss_dy_pred * 1
    _dloss_dw = _dloss_dm * _x
    print(_dloss_dr, _dloss_dy_pred, _dloss_dw, _dloss_db)
    # 4. r ** 2 is also r * r: a node that r goes into twice, so add up both arrows
    _dloss_dr_twice = _dloss_dloss * _r + _dloss_dloss * _r
    print(_dloss_dr_twice)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The slope for $w$ is −16, the same as $2r \cdot (-x) = 2 \cdot 2 \cdot (-4)$ in [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en). The slope for $b$, −4, came almost for free: it reused $\frac{\partial \mathcal{L}}{\partial \hat{y}}$. Both are negative, so making $w$ and $b$ larger lowers the loss: the prediction 5 is below the grade 7.

    The last line is step 4 of the recipe. $r^2$ is also $r \cdot r$: a node that $r$ goes into twice. Along each arrow comes back the other input, $r = 2$, and added up that is 4: the same as $2r$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_backward():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Backwards through a neuron with ReLU

        A neuron with ReLU, for one example: $z = w \cdot x + b$, $\hat{y} = \mathrm{ReLU}(z)$, $r = y - \hat{y}$ and $\mathcal{L} = r^2$. The forward pass is in the box. Replace each `...` with one step of the backward pass, from the loss back to `w` and `b`. ReLU has the local gradient 1 where $z$ is above 0, and 0 elsewhere: it is flat there. The last line is `[dloss_dw, dloss_db]`, so the check sees it. The check also tries other numbers for `x`, `y`, `w` and `b`.
        """),
        L.task("Finish the backward pass, so that `dloss_dw` and `dloss_db` are the slopes of the loss."),
    ])
    return


@app.cell(hide_code=True)
def exercise_backward():
    box_backward = L.code_box("backward", lines=24)
    box_backward
    return (box_backward,)


@app.cell(hide_code=True)
def check_backward(box_backward):
    result_backward = L.check("backward", box_backward.value)
    result_backward
    return (result_backward,)


@app.cell(hide_code=True)
def hints_backward():
    L.hints(
        "backward",
        [
            "Step 3 of the recipe, as in the worked example, with one more node: ReLU, between $z$ and $\\hat{y}$.",
            "Each line is the slope of the line before it times one local gradient: $2r$, then −1, then 1 or 0 for ReLU, then $x$ for `w` and 1 for `b`.",
            "`dloss_dr = 2 * r`, `dloss_dy_pred = dloss_dr * ___`, `dloss_dz = dloss_dy_pred * ___` where `z > 0`, `dloss_dw = dloss_dz * ___` and `dloss_db = dloss_dz * ___`.",
        ],
        why="With $x = 2$, $y = 7$, $w = 1.5$ and $b = 1$: $z = 4$, $\\hat{y} = 4$, $r = 3$. Back: $\\frac{\\partial \\mathcal{L}}{\\partial r} = 6$, $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}} = -6$, and $z$ is above 0, so $\\frac{\\partial \\mathcal{L}}{\\partial z} = -6$. Then $\\frac{\\partial \\mathcal{L}}{\\partial w} = -6 \\cdot 2 = -12$ and $\\frac{\\partial \\mathcal{L}}{\\partial b} = -6$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_branch():
    mo.md(r"""
    ### Exercise 2. One value, two arrows
    """)
    return


@app.cell(hide_code=True)
def choice_branch_cell():
    choice_branch = L.choice(
        "branch",
        question="$s = a + b$ and $f = s \\cdot a$, at $a = 2$ and $b = 3$: then $s = 5$ and $f = 10$. The value $a$ goes into two nodes. What is $\\frac{\\partial f}{\\partial a}$?",
        options={
            "direct": "5: only along the arrow from $a$ into $f$",
            "right": "7: along both arrows, added up",
            "via": "2: only along the arrow from $a$ into $s$",
        },
        correct="right",
        why={
            "direct": "That is what comes back along one arrow. $a$ also goes into $s$: what comes back along that one? See the watch-out in *The idea*.",
            "via": "That is what comes back through $s$. $a$ also goes straight into $f$: what comes back along that arrow? See the watch-out in *The idea*.",
            "right": "Straight into $f$: $\\frac{\\partial f}{\\partial a} = s = 5$. Through $s$: $\\frac{\\partial f}{\\partial s} \\cdot \\frac{\\partial s}{\\partial a} = a \\cdot 1 = 2$. Together 7, as $f = a^2 + ab$ has the slope $2a + b = 7$.",
        },
    )
    choice_branch
    return (choice_branch,)


@app.cell(hide_code=True)
def text_exercise_check():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Check it by moving

        In [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en), moving each parameter by a tiny step was too slow to train with, but good for checking. Check the slopes of the worked example: replace the two `...` with the slopes of `loss` at $w = 1$ and $b = 1$, measured by moving $w$, and then $b$, by `h`. The last line is `[slope_w, slope_b]`, so the check sees it.
        """),
        L.task("Measure both slopes by moving, and compare them with −16 and −4 from the backward pass."),
    ])
    return


@app.cell(hide_code=True)
def exercise_check():
    box_check = L.code_box("check", lines=10)
    box_check
    return (box_check,)


@app.cell(hide_code=True)
def check_check(box_check):
    result_check = L.check("check", box_check.value)
    result_check
    return (result_check,)


@app.cell(hide_code=True)
def hints_check():
    L.hints(
        "check",
        [
            "As in [Two parameters: the gradient](/course/open/math-gradient?lang=en): the loss with one parameter moved, minus the loss where you are, divided by `h`.",
            "Move only `w` for `slope_w`, and only `b` for `slope_b`; the other stays at 1.",
            "`slope_w = (loss(1 + h, 1) - loss(___, ___)) / h`, and the same for `b`.",
        ],
        why="Moving $w$ by $h$ changes the loss by about −16 · $h$, and moving $b$ by about −4 · $h$; divided by $h$, the slopes are −15.9984 and −3.9999. They are not exactly −16 and −4, because $h$ is not 0, but they are close to what the backward pass gives.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) multiplied two slopes; the backward pass does that at every arrow of a graph, from the loss back, and reuses what it has. So backpropagation gives every slope for a few times the cost of a forward pass, where moving cost a forward pass per parameter, as in [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en). In the next lesson, *Build your own autograd: a Value that remembers*, you write a small class that builds the graph while it calculates, and does the backward pass for you.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_backward, choice_branch, result_check):
    L.wrap_up(result_backward, choice_branch, result_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
