# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-scaling-laws. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-scaling-laws", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Log scales and power laws](/course/open/math-log-scales?lang=en) and [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="What does $y = 5 \\cdot x^{-0.5}$ look like on a log-log plot?",
        options={
            "right": "A straight line with slope $-0.5$",
            "five": "A straight line with slope 5",
            "curve": "A curve that flattens to the right",
        },
        correct="right",
        why={
            "five": "The 5 sets the height of the line, $\\log 5$; the exponent is its slope. See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "curve": "On a log-log plot a power law is straight: $\\log y = \\log 5 - 0.5 \\log x$. See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "right": "$\\log y = \\log 5 - 0.5 \\cdot \\log x$: the exponent is the slope. This lesson fits such lines through the losses of real models.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="A model of 100 million parameters trains on 2 billion tokens. About how many FLOPs does that take?",
        options={
            "two": "$2 \\times 10^{17}$",
            "shift": "$1.2 \\times 10^{17}$",
            "right": "$1.2 \\times 10^{18}$",
        },
        correct="right",
        why={
            "two": "That is $N \\cdot D$. Training costs about 6 FLOPs per parameter per token: see [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).",
            "shift": "$6 \\times 2$ is 12, so you get $12 \\times 10^{17}$, and that is not $1.2 \\times 10^{17}$: moving the decimal point one place to the left adds 1 to the power of ten.",
            "right": "$6 \\times 10^8 \\times 2 \\times 10^9$. This lesson asks what loss that compute buys.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A big training run costs millions, and you get one try. Since 2020 we have known that the loss of language models falls very regularly as they get more parameters, more data and more compute: as a power law, a straight line on a log-log plot, some of them over more than seven factors of 10. So you can train a few small models, fit a law through their losses, and predict the loss of a model a thousand times bigger before you train it. OpenAI predicted the final loss of GPT-4 this way, from runs that each used at most a ten-thousandth of its compute.
    """)
    return


@app.cell(hide_code=True)
def predict_factor():
    guess_factor = L.guess(
        "ten-percent",
        question="Kaplan and others (2020) found that the loss of a language model falls as $N^{-0.076}$ with its number of parameters $N$, if data and compute are no limit. How many times more parameters does a model need for a loss that is 10% lower?",
        options={
            "little": "1.1 times",
            "right": "4 times",
            "lots": "About 100 times",
        },
        reveal="**4 times.** A loss 0.9 times as big needs $0.9^{-1/0.076} \\approx 4.0$ times as many parameters. The exponent is small, so every step costs a big factor: halving the loss would take about 9,000 times more parameters.",
    )
    guess_factor
    return (guess_factor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: power laws in size, data and compute.** Kaplan and others trained language models from under a thousand to 1.5 billion parameters, not counting the embeddings. Their loss follows a power law: $N^{-0.076}$ in the parameters $N$, as long as data is no limit; $D^{-0.095}$ in the training tokens $D$, as long as the model is big enough; and $C^{-0.050}$ in the compute $C$, for the best model at each amount of compute. A **scaling law** is such a formula: it predicts the loss from size, data or compute.

        **Step 2: fit it on the logs.** A power law $L = a \cdot x^k$ is a straight line in the logs:
        """),
        L.formula(r"\log L = \log a + k \cdot \log x"),
        mo.md(r"""
        So take the log of the sizes and of the losses, and fit a straight line through them: its slope is the exponent $k$, its intercept $\log a$.

        **Step 3: the floor.** Part of what comes next in a text is truly open: even a perfect model cannot predict it. So the loss cannot fall below a floor $E$. Hoffmann and others (2022, the Chinchilla paper) fit
        """),
        L.formula(r"L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}"),
        mo.md(r"""
        with $E = 1.69$, $A = 406.4$, $\alpha = 0.34$, $B = 410.7$ and $\beta = 0.28$, for the loss per token on their own text. With endless data, the part above the floor, $L - E$, is a power law in $N$; $L$ itself bends on a log-log plot and flattens towards $E$. Their exponents belong to $L - E$, so they cannot be compared directly with Kaplan's, which belong to $L$ itself.

        The picture: Chinchilla's loss with endless data, purple, from 10 million to 100 billion parameters. A straight line through three runs of 10 to 100 million parameters, dashed, predicts 0.95 for 100 billion; the law gives 1.76.
        """),
        mo.Html(C.floor_picture({"title": "Chinchilla's loss and a straight line", "x": "parameters", "y": "loss", "floor": "floor E = 1.69",
                                 "curve": "the law", "line": "straight line", "lang": L.lang,
                                 "alt": "A log-log plot of loss against parameters, from 10 million to 100 billion. The loss of Chinchilla's law with endless data falls from 3.38 at 10 million to 2.46 at 100 million and 1.76 at 100 billion, bending towards a dashed floor at 1.69. Three dots mark runs of 10, 32 and 100 million parameters. A dashed straight line through them goes on falling, to 0.95 at 100 billion, far below the law."})),
        mo.md(r"""
        **Step 4: what it is for.** With a fitted law you predict the loss of a run before you start it, and you see whether a new idea moves the whole line down or only helps small models. The further you predict beyond your runs, the more the floor matters, and the less you can trust a line without it.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **scaling law** | a formula, usually a power law, that predicts a model's loss from its size, its data or its compute | |
        | $N$, $D$, $C$ | the parameters, the training tokens, and the compute in FLOPs | |
        | $E$ | the floor: the loss that no model gets below | `floor` |
        | **intercept** | the height of a straight line where $\log x = 0$; `np.polyfit` gives it after the slope, as the bias term | `b` |
        """),
        L.watch_out("A scaling law holds for one recipe: one kind of data, one tokenizer, one architecture, one training schedule. Change one of them and the line moves, so fit it again. And it predicts the loss, not what a model can do."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fit a scaling law and use it:

    1. Train a few runs that differ only in size, and note the loss of each.
    2. Subtract the floor $E$ from every loss: 0 when you are far above it, or a value from a bigger fit.
    3. Take the log of the sizes and of the losses above the floor, and fit a straight line: `k, b = np.polyfit(log_x, log_rest, 1)`, and $a = 10^b$.
    4. Predict $E + a \cdot x^k$ for a bigger $x$, and trust it less the further $x$ lies beyond your runs.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    New is a straight line through logs, with `np.polyfit`: three sizes and the losses Kaplan's law gives them.
    """)
    return


@app.cell
def example_fit():
    import numpy as np

    _n = np.array([1e6, 1e7, 1e8])  # parameters
    _loss = np.array([4.02, 3.37, 2.83])  # Kaplan's law at those sizes
    _k, _b = np.polyfit(np.log10(_n), np.log10(_loss), 1)
    print("k =", round(_k, 3), " a =", round(10 ** _b, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `np.polyfit` with degree 1 gives the slope first, $k = -0.076$, Kaplan's exponent, and then the intercept $b$, the log of $a$: $a = 10^b \approx 11.5$. Three runs, each 10 times bigger than the one before, are enough to find the line, as long as the losses lie far above the floor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Fit a law with a floor

        Complete `fit_power_law(x, loss, floor)`: fit $\text{loss} = \text{floor} + a \cdot x^k$ and give back `a, k`. The check calls `fit_power_law(np.array([1e6, 1e7, 1e8, 1e9]), np.array([3.7, 2.7, 2.2, 1.95]), 1.7)`, which should give about (128, −0.301), and tries losses that wobble, a floor of 0, and the losses of Chinchilla's law.
        """),
        L.task("Complete `fit_power_law`: a straight line through the logs of `x` and of the loss above the floor."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-power-law", lines=12)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit-power-law", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-power-law",
        [
            "Steps 2 and 3 of the recipe: the worked example, with `loss - floor` in the place of the loss.",
        ],
        why="`log_x = np.log10(x)`, `log_rest = np.log10(loss - floor)`, then `k, b = np.polyfit(log_x, log_rest, 1)` and `a = 10 ** b`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Exercise 2. Which runs
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "which-runs",
        question="You have compute for four short runs, to fit a scaling law for your GPT. Which sizes do you train?",
        options={
            "right": "1, 3, 10 and 30 million parameters",
            "steps": "10, 20, 30 and 40 million parameters",
            "close": "37, 38, 39 and 40 million parameters",
        },
        correct="right",
        why={
            "right": "Equal factors are equal steps on a log-log plot: the four points spread out over the line, and its slope is measured over a factor of 30.",
            "steps": "On a log axis, 10 to 40 is only a factor of 4, and 20, 30 and 40 crowd together at the end. Sizes an equal factor apart spread out over the line.",
            "close": "37 to 40 is hardly a step on a log axis: a little wobble in the losses changes the slope completely.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_exercise_floor():
    mo.md(r"""
    ### Exercise 3. Why there is a floor
    """)
    return


@app.cell(hide_code=True)
def explain_floor():
    explain_floor_cell = L.explain(
        "why-floor",
        question="Why can the loss of a language model not keep falling along a straight line on a log-log plot, however big the model gets? And what does that mean for what each factor of 10 in size wins?",
        model_answer="**Why there is a floor:** part of what comes next in a text is truly open, such as which name a story gives a new character. Even a perfect model can only give probabilities for it, so its loss stays above a floor $E$, 1.69 in Chinchilla's fit. **What that means:** only the part above the floor shrinks as a power law, by the same factor for each factor of 10 in size. As the loss gets close to $E$, that part is a smaller and smaller share of the loss, so each factor of 10 cuts a smaller and smaller fraction off the loss. On a log-log plot, where equal factors are equal steps, the curve flattens towards $E$, and a straight line through small models predicts too low a loss for big ones.",
        points=[
            "you say that part of the next token cannot be predicted, so even a perfect model has a loss above 0",
            "you say that only the part of the loss above the floor falls as a power law",
            "you say that each factor of 10 in size cuts a smaller and smaller fraction off the loss, so the curve flattens on a log-log plot",
        ],
    )
    explain_floor_cell
    return (explain_floor_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The power laws and log-log plots of [Log scales and power laws](/course/open/math-log-scales?lang=en), now for the loss of language models, with the compute $C$ of [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en). The next lesson, *Project: fit your own scaling law*, trains four small versions of your GPT, fits a law through them, and predicts the loss of a bigger one before it trains it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_runs, explain_floor_cell, result_fit):
    L.wrap_up(result_fit, choice_runs, explain_floor_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
