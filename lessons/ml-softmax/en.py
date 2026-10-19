# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-softmax. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-softmax", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) and [Sums and means along an axis](/course/open/np-axis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sigmoid():
    quick_sigmoid = L.choice(
        "quick-sigmoid",
        question="A logit of −1 goes through the sigmoid. About which probability comes out?",
        options={
            "plus": "0.73",
            "right": "0.27",
            "same": "−1",
        },
        correct="right",
        why={
            "plus": "That is $\\sigma(1)$. Is −1 above or below 0? See the picture in [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "same": "The sigmoid turns every number into one between 0 and 1. See *The idea* in [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "right": "$\\sigma(-1) = 1 / (1 + e^{1}) \\approx 0.27$: a negative logit gives less than 0.5.",
        },
    )
    quick_sigmoid
    return (quick_sigmoid,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="`m = np.array([[1, 2], [3, 4]])`. What does `m.sum(axis=1)` give?",
        options={
            "right": "`array([3, 7])`",
            "columns": "`array([4, 6])`",
            "all": "`10`",
        },
        correct="right",
        why={
            "columns": "That adds up each column: `axis=0`. Which direction does `axis=1` add up? See the picture in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "all": "That is `m.sum()`, without an axis. See *The idea* in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "right": "`axis=1` adds up along each row: 1 + 2 and 3 + 4.",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The sigmoid gives a probability for two classes. A model that reads handwritten digits has ten classes, and it gives ten logits, one per class: how much the picture looks like a 0, a 1, and so on. It needs ten probabilities: all positive, adding up to 1, and the biggest for the biggest logit. Softmax does exactly that, and nearly every model for many classes ends with it, all the way to a language model, which picks its next word out of tens of thousands.

    **An everyday example.** A pizza shared by how hungry everyone is. Everyone gets a slice, the hungriest the biggest one, and the slices together make exactly one pizza.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share",
        question="Three classes get the logits 3, 1 and 0. What do you think is the probability of the first class?",
        options={
            "linear": "0.75: 3 of the 3 + 1 + 0",
            "softmax": "About 0.84",
            "all": "1: it has the biggest logit",
        },
        reveal="**About 0.84.** Softmax takes $e$ to the power of each logit first: $e^3 \\approx 20.1$, $e^1 \\approx 2.7$ and $e^0 = 1$, together about 23.8. The first class gets $20.1 / 23.8 \\approx 0.84$. Dividing the logits themselves by their sum would give the class with the logit 0 the probability 0, and a negative logit a negative number. And the biggest logit still leaves room for the others.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Softmax** turns a vector of $K$ logits $\mathbf{z} = (z_0, \dots, z_{K-1})$ into $K$ probabilities, with the classes numbered from 0, as Python counts. Take $e$ to the power of each logit, which makes every number positive, and divide each by their sum, so that they add up to 1:
        """),
        L.formula(r"[[pred:\mathrm{softmax}(\mathbf{z})_k]] = \frac{e^{z_k}}{e^{z_0} + e^{z_1} + \dots + e^{z_{K-1}}}"),
        mo.md(r"""
        The biggest logit gets the biggest probability, and a bigger difference between logits pushes the probabilities further apart. Only the differences count: add the same number $c$ to every logit, and every $e^{z_k}$ is multiplied by $e^c$, top and bottom, so nothing changes. That gives a safe way to work it out: take the largest logit off first. Then the biggest power is $e^0 = 1$, and $e$ to a big logit, which a computer cannot hold, never comes up.

        With two classes and the logits $z$ and 0, softmax gives the first class $e^z$ divided by $e^z + e^0 = e^z + 1$. Divide top and bottom by $e^z$, and as $1 / e^z = e^{-z}$, that is the sigmoid of $z$:
        """),
        L.formula(r"\frac{e^{z}}{e^{z} + 1} = \frac{1}{1 + e^{-z}} = \sigma(z)"),
        mo.md(r"""
        Softmax is the sigmoid for many classes.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **softmax**, $\mathrm{softmax}(\mathbf{z})$ | softmax of z | turns $K$ logits into $K$ probabilities: $e$ to the power of each, divided by their sum | `e = np.exp(z - z.max())`, `e / e.sum()` |
        | $\mathbf{z}$ | z | the vector of logits, one per class; bold, because it is a vector | `z` |
        | $K$ | K | the number of classes: the length of $\mathbf{z}$ | `len(z)` |
        | $z_k$, $\mathrm{softmax}(\mathbf{z})_k$ | z k, softmax of z at place k | the logit of class $k$, from $z_0$ to $z_{K-1}$, and the probability softmax gives that class | `z[k]` |
        | **taking the largest logit off** | | subtracting the largest logit from all logits before `np.exp`: the probabilities stay the same, and no power gets too big | `z - z.max()` |
        """),
        L.watch_out("`np.exp(1000)` is too big for the computer: it becomes `inf`, infinitely large, and `inf / inf` is `nan`. Big logits do occur in real models. Taking the largest logit off first prevents it, and does not change the probabilities."),
        mo.md(r"""
        Move the slider: it sets the logit of class 0, while class 1 keeps the logit 1 and class 2 the logit 0. One chart has the <span class="c-pred">probabilities</span> as bars; together they always make 1 (the numbers on the bars are rounded). The other shows the three probabilities for every $z_0$: class 0 solid, classes 1 and 2 dashed, each with its number, and a grey dashed line at 0.5; drag across it to set $z_0$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        ps = C.probabilities(z)
        shown = [L.number(p, 2) for p in ps]
        return (C.picture(z, {"title": "Three probabilities from softmax", "x": "class", "y": "probability", "every": "For every z₀",
                              "alt": f"Three bars, one per class, as high as their probability: {shown[0]}, {shown[1]} and {shown[2]}, for the logits {L.number(z, 1)}, 1 and 0. And the three probabilities for every z₀ from −3 to 5: that of class 0 rises from near 0 to near 1 and crosses the dashed line at 0.5 at about z₀ = 1.3, while those of classes 1 and 2 fall, class 1 always above class 2; three dots at z₀ = {L.number(z, 1)}."}, shown),
                f"""With the logits {L.number(z, 1)}, 1 and 0, softmax gives <span class="c-pred ea-num">{shown[0]}</span>, <span class="c-pred ea-num">{shown[1]}</span> and <span class="c-pred ea-num">{shown[2]}</span>.""")

    explore_z = L.explore("logit-0", "z₀", _frame, start=1, low=-3, high=5, step=0.5, role="", task="From which $z_0$ does class 0 get more than half of the probability? What happens to the ratio of class 1 to class 2?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The solid line of class 0 crosses the dashed line at 0.5 at about $z_0 = 1.3$, so on the slider class 0 gets more than half from $z_0 = 1.5$: 0.55. Classes 1 and 2 share what is left. Both are divided by the same sum, so their ratio stays $e^1 / e^0 = e$, about 2.7 to 1, whatever $z_0$ is: it depends only on the difference between their two logits, 1 − 0, and that never changes.

    ### Recipe

    To turn logits into probabilities with softmax:

    1. Take the largest logit off every logit: `z - z.max()`.
    2. Take $e$ to the power of each: `np.exp(...)`.
    3. Divide each by their sum: `e / e.sum()`.
    4. Check: all positive, together 1, and the biggest logit has the biggest probability.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    First the logits 3, 1 and 0, step by step. Then the logits 1000, 999 and 998: once without taking the largest off, and once with. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_softmax():
    import numpy as np

    _z = np.array([3, 1, 0])
    # 1. the largest logit off
    _shifted = _z - _z.max()
    print(_shifted)
    # 2. e to the power of each
    _e = np.exp(_shifted)
    print(_e)
    # 3. divided by their sum
    _p = _e / _e.sum()
    print(_p)
    # 4. the check: together 1
    print(_p.sum())

    # big logits: without step 1, and with it
    _big = np.array([1000, 999, 998])
    print(np.exp(_big))
    _e_big = np.exp(_big - _big.max())
    print(_e_big / _e_big.sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With the largest logit off, the logits become 0, −2 and −3, and their powers 1, 0.135 and 0.050. Divided by their sum, 1.185, that gives 0.844, 0.114 and 0.042: together 1. The same as $e^3$, $e^1$ and $e^0$ divided by their sum, as in the guess. For 1000, 999 and 998, `np.exp` alone gives `inf` three times. NumPy also shows a warning, the line with `RuntimeWarning: overflow encountered in exp`: the result is too big. A warning is not an error; the cell runs on. With the largest off, they become 0, −1 and −2, and the probabilities 0.665, 0.245 and 0.090.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_add():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Ten more
    """)
    return


@app.cell(hide_code=True)
def choice_add_cell():
    choice_add = L.choice(
        "add-ten",
        question="A model gives three classes the logits 2, 1 and 0. What happens to the softmax probabilities if you add 10 to every logit?",
        options={
            "grow": "They all get bigger",
            "first": "The first gets bigger, the others smaller",
            "right": "Nothing: they stay the same",
        },
        correct="right",
        why={
            "grow": "Can they all get bigger, while they add up to 1? See *The idea*.",
            "first": "The differences between the logits stay the same. What does softmax look at? See *The idea*.",
            "right": "Every $e^{z_k}$ is multiplied by $e^{10}$, on top and at the bottom, so it cancels: only the differences count.",
        },
    )
    choice_add
    return (choice_add,)


@app.cell(hide_code=True)
def text_exercise_softmax():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Softmax as a function

        Finish `softmax(z)`: it gets a list of logits and gives back their probabilities, as an array. Replace the `...` with steps 1 to 3 of the recipe. The check calls `softmax([3, 1, 0])`: that should give about `array([0.844, 0.114, 0.042])`. It also tries other logits, also very big ones.
        """),
        L.task("Make `softmax` give back the probabilities of the logits `z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_softmax():
    box_softmax = L.code_box("softmax", lines=7)
    box_softmax
    return (box_softmax,)


@app.cell(hide_code=True)
def check_softmax(box_softmax):
    result_softmax = L.check("softmax", box_softmax.value)
    result_softmax
    return (result_softmax,)


@app.cell(hide_code=True)
def hints_softmax():
    L.hints(
        "softmax",
        [
            "Steps 1 to 3 of the recipe, in that order, each in one line.",
            "Which NumPy method gives the largest number of an array? And after `np.exp`, what do you divide by?",
            "`e = np.exp(z - ___)` and `return e / ___`.",
        ],
        why="`z - z.max()` takes the largest logit off, `np.exp` makes every number positive, and `e / e.sum()` makes them add up to 1. For `[3, 1, 0]`: 0.844, 0.114 and 0.042.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Two classes

        Two classes, with the logits `z` and 0. Put their softmax probabilities in `p`, as an array of two. Replace the `...`. The last line is `p`, so the check sees it. The check also tries another `z`. Compare the first probability with what the sigmoid gives for `z`.
        """),
        L.task("Put the softmax of `logits` in `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_two():
    box_two = L.code_box("two-classes", lines=7)
    box_two
    return (box_two,)


@app.cell(hide_code=True)
def check_two(box_two):
    result_two = L.check("two-classes", box_two.value)
    result_two
    return (result_two,)


@app.cell(hide_code=True)
def hints_two():
    L.hints(
        "two-classes",
        [
            "Softmax on an array of two logits, as in steps 2 and 3 of the recipe; with these small logits, step 1 is not needed.",
            "Both probabilities at once: $e$ to the power of each logit, divided by the sum of both. Which NumPy function gives the powers of a whole array?",
            "`p = np.exp(logits) / ___`.",
        ],
        why="`np.exp(logits)` is $e^{1.5} \\approx 4.48$ and $e^0 = 1$; divided by their sum, 5.48, that gives 0.818 and 0.182. The first is the sigmoid of 1.5.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) one logit became a probability for two classes; softmax does the same for any number of classes, and with two it is the sigmoid. In [Sums and means along an axis](/course/open/np-axis?lang=en) `.max()` and `.sum()` summarized an array; softmax uses both. In the next lesson, *Cross-entropy loss for many classes*, you measure how good such probabilities are, as log loss did for two classes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add, result_softmax, result_two):
    L.wrap_up(choice_add, result_softmax, result_two)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
