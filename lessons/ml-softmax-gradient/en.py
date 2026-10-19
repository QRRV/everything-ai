# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-softmax-gradient. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-softmax-gradient", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) and [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cross_entropy():
    quick_cross_entropy = L.choice(
        "quick-cross-entropy",
        question="A model gives an example the probabilities 0.5, 0.25 and 0.25, and its right class is 0. About how big is its cross-entropy?",
        options={
            "other": "1.39",
            "right": "0.69",
            "all": "3.47",
        },
        correct="right",
        why={
            "other": "That is $-\\log(0.25)$, the loss if class 1 or 2 were the right one. See *The idea* in [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "all": "That is the surprise of all three classes added up. Which class counts? See *The idea* in [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "right": "Only the right class counts: $-\\log(0.5) \\approx 0.69$.",
        },
    )
    quick_cross_entropy
    return (quick_cross_entropy,)


@app.cell(hide_code=True)
def quick_check_log_loss_gradient():
    quick_log_loss_gradient = L.choice(
        "quick-log-loss-gradient",
        question="A mail is spam, $y = 1$, and has one feature, $x = 2$. The model gives it $\\hat{y} = 0.3$. What is $\\frac{\\partial \\mathcal{L}}{\\partial w}$ for this mail?",
        options={
            "right": "−1.4",
            "sign": "1.4",
            "no-x": "−0.7",
        },
        correct="right",
        why={
            "sign": "Which comes first, $\\hat{y}$ or $y$? See the formula in [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en).",
            "no-x": "That is the slope for the bias term. What does a weight add? See the formula in [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en).",
            "right": "$(\\hat{y} - y) \\cdot x = (0.3 - 1) \\cdot 2 = -1.4$.",
        },
    )
    quick_log_loss_gradient
    return (quick_log_loss_gradient,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To train a model for many classes with gradient descent, you need the slope of the cross-entropy for every logit. With ten classes there are ten logits, and in a model each comes from its own weights. Changing each logit by a tiny step and working out the loss again works, but that is one extra calculation per logit, and each slope is only an estimate. The chain rule gives all slopes at once, exactly, and for softmax with cross-entropy they turn out as simple as for log loss.

    **An everyday example.** Friends split a bill. Everyone compares what they paid with their share: who paid too much gets the difference back, who paid too little pays it. One number per person, and together the numbers make 0.
    """)
    return


@app.cell(hide_code=True)
def predict_cat():
    guess_cat = L.guess(
        "cat",
        question="A model sees a photo of a dog. It gives cat 0.7, dog 0.2 and bird 0.1. The logit of cat goes up a little. What do you think happens to the cross-entropy?",
        options={
            "up": "It goes up",
            "same": "Nothing: cat is not the right class",
            "down": "It goes down",
        },
        reveal="**It goes up.** Cross-entropy only looks at the probability of dog, but softmax shares out 1: more for cat leaves less for dog. The slope for the logit of cat turns out to be its probability, 0.7. Dog's slope is its probability minus 1, $0.2 - 1 = -0.8$: raising dog's logit is what lowers the loss.",
    )
    guess_cat
    return (guess_cat,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The cross-entropy of one example is minus the log of the probability of its right class $c$, and softmax makes that probability from the logits $z_0, \dots, z_{K-1}$: $\hat{y}_c = e^{z_c} / S$, with $S = e^{z_0} + \dots + e^{z_{K-1}}$, the sum softmax divides by. The log of a quotient is the log of the top minus the log of the bottom. That follows from $\log(a \cdot b) = \log(a) + \log(b)$ in [Exponents and logarithms](/course/open/math-exp-log?lang=en): $S$ times $e^{z_c}/S$ is $e^{z_c}$, so $\log(S) + \log(e^{z_c}/S) = \log(e^{z_c})$. And the log undoes $e$, so:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\log \frac{e^{z_c}}{S} = -z_c + \log S"),
        mo.md(r"""
        Now the slope of each part for a logit $z_k$:

        - **The slope of $-z_c$.** It is $-1$ for the logit of the right class and 0 for the others. That is minus the one-hot label: $-y_k$.
        - **The slope of $\log S$.** By the chain rule: the slope of $\log$ at $S$, which is $1/S$, times the slope of $S$ for $z_k$. Only the term $e^{z_k}$ of $S$ changes with $z_k$, and the slope of $e^z$ is $e^z$ itself. So it is $e^{z_k}/S$: that is $\hat{y}_k$.

        The slopes of $\log$ and of $e^z$ you take as given here, as [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en) did for $\log$ and the sigmoid; a later lesson derives them. Together:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial z_k}]] = [[pred:\hat{y}_k]] - y_k"),
        mo.md(r"""
        The gradient for the logits is the probabilities minus the one-hot label: the same pattern as $\hat{y} - y$ for log loss, now one slope per class. The right class gets $\hat{y}_c - 1$, which is negative: gradient descent, which steps against the slope, raises its logit. Every other class gets its own probability, which is positive: its logit goes down, the most for the class that took most. The slopes add up to 0, because the probabilities and the one-hot label both add up to 1.

        In a model, each logit is a dot product of its own weights with the features, plus its own bias term: $z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k$. The chain rule then multiplies once more by the feature, as for log loss: the slope for weight $j$ of class $k$ is $(\hat{y}_k - y_k)\,x_j$, and for its bias term $\hat{y}_k - y_k$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **gradient of softmax with cross-entropy**, $\hat{y}_k - y_k$ | y hat k minus y k | the slopes of the cross-entropy for the logits, when softmax makes the probabilities: the probabilities minus the one-hot label | `softmax(z) - np.eye(K)[label]` |
        | $S$ | S | the sum $e^{z_0} + \dots + e^{z_{K-1}}$ that softmax divides by | `np.exp(z).sum()` |
        """),
        L.watch_out("Mind the order: the probabilities minus the one-hot label, as $\\hat{y} - y$ for log loss. Swap them, and every slope gets the wrong sign: gradient descent then lowers the logit of the right class."),
        mo.md(r"""
        Move the slider: it sets $z_1$, the logit of dog, the right class, while cat keeps the logit 2 and bird the logit 0. One chart shows the <span class="c-pred">probabilities</span> as bars, dog's with a heavier outline. The other shows the <span class="c-grad">slope</span> for each logit for every $z_1$, dog's line the heavier one, with a dot at each slope for the $z_1$ of the slider and dashed lines at −0.1 and 0.1; drag across it to set $z_1$. The numbers are rounded.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        shown_p = [L.number(p, 2) for p in C.probabilities(z)]
        shown_g = [L.number(g, 2) for g in C.gradient(z)]
        return (C.picture(z, {"bars": "Probabilities from softmax", "slopes": "Slope for every z₁", "class": "class: 0 cat, 1 dog, 2 bird",
                              "p": "probability", "slope": "slope", "names": ["cat", "dog", "bird"],
                              "alt": f"The probabilities of cat, dog and bird as three bars: {shown_p[0]}, {shown_p[1]} and {shown_p[2]}. A second chart shows the slope for each logit for every z₁ from −3 to 5: dog's rises from −0.99 towards 0, cat's and bird's fall towards 0. At z₁ = {L.number(z, 1)} the slopes are {shown_g[0]}, {shown_g[1]} and {shown_g[2]}."}, shown_p),
                f"""The model gives dog <span class="c-pred ea-num">{shown_p[1]}</span>, so the slope for dog's logit is that minus 1: <span class="c-grad ea-num">{shown_g[1]}</span>. Cat and bird get their probabilities as slopes, <span class="c-grad ea-num">{shown_g[0]}</span> and <span class="c-grad ea-num">{shown_g[2]}</span>.""")

    explore_z = L.explore("logit-dog", "z₁", _frame, start=0, low=-3, high=5, step=0.5, role="", task="From which $z_1$ are all three slopes within 0.1 of 0?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    From $z_1 = 4.5$ all three slopes are within 0.1 of 0: once the model gives the right class nearly all of the probability, 0.91 there, the example hardly pushes the logits any more. At $z_1 = -3$ dog's slope is −0.99 and cat's 0.88: the biggest push comes where the model is most wrong.

    ### Recipe

    To work out the slopes of the cross-entropy for the logits `z` of one example with the label `label`:

    1. Make the probabilities with softmax: `p = softmax(z)`.
    2. Take the one-hot label off: `p - np.eye(K)[label]`.
    3. Check: the right class has a negative slope, the others a positive one, and together they make 0.
    4. Check one slope by changing only its logit by a tiny step `h`: the change in the loss, divided by `h`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The logits 2, 1 and 0 for cat, dog and bird, and the photo is of a dog: the label is 1. First the slopes from the formula, then each one measured with a tiny step. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_softmax_gradient():
    import numpy as np

    _z = np.array([2.0, 1.0, 0.0])
    _label = 1

    def _loss(z):
        _e = np.exp(z - z.max())
        return -np.log(_e[_label] / _e.sum())

    # 1. the probabilities from softmax
    _e = np.exp(_z - _z.max())
    _p = _e / _e.sum()
    print(_p)
    # 2. minus the one-hot label
    _grad = _p - np.eye(3)[_label]
    print(_grad)
    # 3. the check: negative only for the right class
    print(_grad < 0)
    # 4. each logit changed by a tiny step h
    _h = 0.0001
    for _k in range(3):
        _step = np.eye(3)[_k] * _h
        print((_loss(_z + _step) - _loss(_z)) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The probabilities are 0.665, 0.245 and 0.090. Minus the one-hot label $(0, 1, 0)$, the slopes are 0.665, −0.755 and 0.090: only dog's is negative, as `[False  True False]` shows, and together they make 0. The tiny steps measure 0.66525, −0.75526 and 0.09003: they differ from the formula only in the fifth decimal. `np.eye(3)[_k] * _h` is a vector with `h` at place `_k` and 0 elsewhere, so each step changes one logit only.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one_step():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One step
    """)
    return


@app.cell(hide_code=True)
def choice_one_step_cell():
    choice_one_step = L.choice(
        "one-step",
        question="A photo of a bird. The model gives cat 0.5, dog 0.3 and bird 0.2. Gradient descent takes one step on the logits. What happens?",
        options={
            "only": "Only the logit of bird changes: it goes up",
            "equal": "The logit of bird goes up; those of cat and dog go down by the same amount",
            "right": "The logit of bird goes up; those of cat and dog go down, cat's the most",
        },
        correct="right",
        why={
            "only": "Do the other classes get a slope of 0? See the formula in *The idea*.",
            "equal": "What slope does each wrong class get? See the formula in *The idea*.",
            "right": "The slopes are 0.5 for cat, 0.3 for dog and $0.2 - 1 = -0.8$ for bird. A step against them raises bird and lowers cat most, because cat took most.",
        },
    )
    choice_one_step
    return (choice_one_step,)


@app.cell(hide_code=True)
def text_exercise_logit_gradient():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The gradient as a function

        Finish `logit_gradient(z, label)`: it gets the logits of one example, as a list, and the number of its right class, and gives back the slope of the cross-entropy for each logit. `softmax` is already there. Replace the `...`. The check calls `logit_gradient([2, 1, 0], 1)`, the worked example: that should give about `[0.665, -0.755, 0.090]`. It also tries other logits and labels, also with another number of classes.
        """),
        L.task("Make `logit_gradient` give back the slopes of the cross-entropy for the logits `z`, with the label `label`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logit_gradient():
    box_logit_gradient = L.code_box("logit-gradient", lines=10)
    box_logit_gradient
    return (box_logit_gradient,)


@app.cell(hide_code=True)
def check_logit_gradient(box_logit_gradient):
    result_logit_gradient = L.check("logit-gradient", box_logit_gradient.value)
    result_logit_gradient
    return (result_logit_gradient,)


@app.cell(hide_code=True)
def hints_logit_gradient():
    L.hints(
        "logit-gradient",
        [
            "Steps 1 and 2 of the recipe: the probabilities, minus the one-hot label.",
            "`softmax(z)` gives the probabilities, and `np.eye(K)[label]` the one-hot label of `K` classes. How many classes are there?",
            "`return softmax(z) - np.eye(___)[label]`.",
        ],
        why="`softmax(z)` gives 0.665, 0.245 and 0.090 for `[2, 1, 0]`; the one-hot label of class 1 is $(0, 1, 0)$; the difference is 0.665, −0.755 and 0.090. `len(z)` makes it work for any number of classes.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_nudge():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Measure one slope

        Three logits, and the right class is 0. The formula says the slope for the logit of class 0 is $\hat{y}_0 - 1$. Check it as in [The derivative as a slope](/course/open/math-slope?lang=en): change only the logit of class 0 by the tiny step `h`, and put the slope of the loss in `slope`. `loss(z, label)` is already there. Replace the `...`. The last line is `slope`, so the check sees it. The check also tries other logits.
        """),
        L.task("Put the slope of `loss` for the logit of class 0 in `slope`, measured with the tiny step `h`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nudge():
    box_nudge = L.code_box("nudge", lines=12)
    box_nudge
    return (box_nudge,)


@app.cell(hide_code=True)
def check_nudge(box_nudge):
    result_nudge = L.check("nudge", box_nudge.value)
    result_nudge
    return (result_nudge,)


@app.cell(hide_code=True)
def hints_nudge():
    L.hints(
        "nudge",
        [
            "Step 4 of the recipe: the loss with the tiny step, minus the loss without, divided by the size of the step.",
            "`z + np.array([h, 0, 0])` changes only the logit of class 0. Which two losses do you take apart, and what do you divide by?",
            "`slope = (loss(z + np.array([h, 0, 0]), label) - ___) / ___`.",
        ],
        why="The tiny step lowers the loss by about 0.00008; divided by `h`, 0.0001, that is about −0.82. The formula gives the same: the probability of class 0, 0.1753, minus 1, is −0.8247.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en) the slope for the logit was $\hat{y} - y$; with softmax and cross-entropy it is the same pattern, one slope per class. In [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) you measured how good the probabilities are; now you know which way each logit has to go to make them better. In *Project: a linear classifier for handwritten digits*, gradient descent uses this gradient to train a model for the ten digits. In the next lesson, *Will it work on new data? Training and test sets*, you find out whether a trained model also works on examples it has never seen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one_step, result_logit_gradient, result_nudge):
    L.wrap_up(choice_one_step, result_logit_gradient, result_nudge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
