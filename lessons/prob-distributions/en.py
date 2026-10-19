# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson prob-distributions. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-distributions", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Expected value and variance](/course/open/prob-expectation?lang=en) and [Many classes: softmax](/course/open/ml-softmax?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="Bill lengths have a variance of 9 mm². What is their standard deviation?",
        options={
            "right": "3 mm: the square root of it",
            "same": "9 mm: the same number",
            "half": "4.5 mm: half of it",
        },
        correct="right",
        why={
            "same": "The variance is in squared units, mm²; the standard deviation is back in mm. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "half": "The standard deviation is not half the variance. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "√9 = 3: the standard deviation is the square root of the variance, in the unit of the values.",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def quick_check_classes():
    quick_classes = L.choice(
        "quick-classes",
        question="A softmax over three classes gives the first two the probabilities 0.7 and 0.2. What does it give the third?",
        options={
            "zero": "0: the first class has won",
            "mean": "0.45: the mean of the two",
            "right": "0.1: all three add up to 1",
        },
        correct="right",
        why={
            "zero": "Softmax gives every class a probability above 0. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "mean": "The probabilities are not averaged: together they add up to 1. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "1 − 0.7 − 0.2 = 0.1: the probabilities of a softmax always add up to 1.",
        },
    )
    quick_classes
    return (quick_classes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Whether an email is spam, which digit an image shows, how long a penguin's bill is: deep learning describes each with a probability distribution, and the losses you train with come from them. Three distributions come back again and again. Know them, and you can read what a model's output means, check whether data looks the way a model assumes, and draw new values that look like real ones.

    **An everyday example.** A weather app gives a chance of rain: yes or no. A poll gives a share for each party: one of many. A doctor puts your height next to the curve of your age group: a measurement.
    """)
    return


@app.cell(hide_code=True)
def predict_band():
    _x = C.bills()
    guess_band = L.guess(
        "within-one",
        question=f"The {len(_x)} Adélie penguins in the data have a mean bill length of {L.number(_x.mean(), 1)} mm, with a standard deviation of {L.number(_x.std(), 1)} mm. What share of them do you think has a bill within one standard deviation of the mean, between {L.number(_x.mean() - _x.std(), 1)} and {L.number(_x.mean() + _x.std(), 1)} mm?",
        options={
            "third": "About a third: the band is narrow",
            "right": "About two thirds: most lie near the mean",
            "all": "Almost all of them: few penguins stand out from the rest",
        },
        reveal=f"**About two thirds: {L.number(100 * C.within(1), 1)}%.** Within two standard deviations it is {L.number(100 * C.within(2), 1)}%. A normal distribution predicts about 68% and 95%: the bill lengths of one species follow the bell curve closely.",
    )
    guess_band
    return (guess_band,)


@app.cell(hide_code=True)
def text_idea():
    _d = C.penguins()
    _males = int((_d["sex"] == "male").sum())
    _known = int(_d["sex"].notna().sum())
    _shares = _d["species"].value_counts() / len(_d)
    mo.vstack([
        mo.md(f"""
        ## The idea

        A probability distribution gives every possible outcome its probability, as in [Probability: counting how often](/course/open/prob-basics?lang=en). Three kinds come back everywhere.

        The **Bernoulli distribution** is for one yes-or-no outcome: 1 with probability $p$, 0 with probability $1 - p$. Its expected value is $p$ and its variance $p(1 - p)$, with the formulas of [Expected value and variance](/course/open/prob-expectation?lang=en). Of the {_known} penguins whose sex is known, {_males} are male: $p$ = {L.number(_males / _known, 3)}.

        The **categorical distribution** is for one of $K$ classes, each with its own probability; together they add up to 1. The softmax output of a classifier is one, as in [Many classes: softmax](/course/open/ml-softmax?lang=en), and the Bernoulli distribution is the case $K = 2$. The species of the penguins: Adélie {L.number(_shares["Adelie"], 2)}, Gentoo {L.number(_shares["Gentoo"], 2)}, Chinstrap {L.number(_shares["Chinstrap"], 2)}.

        The **normal distribution** $\\mathcal{{N}}(\\mu, \\sigma^2)$ is for measurements that cluster around a mean: the bell curve, with the mean $\\mu$ and the standard deviation $\\sigma$. Its **probability density**, or density for short, says how densely the values lie near each $x$:
        """),
        L.formula(r"[[pred:p(x)]] = \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2\sigma^2}}"),
        mo.md(r"""
        A density is not a probability: the probability of a range of values is the area under the curve there, and the whole area is 1. So a density can be above 1, when the values lie close together: $\mathcal{N}(0, 0.2^2)$ is about 2 at its mean. About 68% of the area lies within one $\sigma$ of the mean, and about 95% within two. Formulas write the variance in $\mathcal{N}(\mu, \sigma^2)$; Python's `random.gauss` and NumPy take the standard deviation $\sigma$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Bernoulli distribution** | ber-NOO-lee | one yes-or-no outcome: 1 with probability $p$, 0 with probability $1 - p$ | `random.random() < p` |
        | **categorical distribution** | | one of $K$ classes, each with its own probability, adding up to 1 | `random.choices(classes, weights=probs)` |
        | **normal distribution**, $\mathcal{N}(\mu, \sigma^2)$ | normal with mean mu and variance sigma squared | the bell curve of measurements around the mean $\mu$, with standard deviation $\sigma$ | `random.gauss(mu, sigma)` |
        | **probability density**, $p(x)$ | p of x | how densely the values lie near $x$; the area under it over a range is the probability of that range | |
        | $\mu$, $\sigma$ | mu, sigma | the mean and the standard deviation of a normal distribution | `mu`, `sigma` |

        The code in the table draws one value from each distribution. `random.random()` gives a number from 0 up to 1, every value equally likely, so it is below `p` with probability `p`: one draw of a Bernoulli distribution, as `True` or `False`. `random.choices(classes, weights=probs)` picks one class, each in proportion to its number in `weights`, and gives it back in a list; `weights` may also hold counts. `random.gauss(mu, sigma)` gives one number from $\mathcal{N}(\mu, \sigma^2)$.
        """),
        L.watch_out("Not every measurement is normal. Incomes, the sizes of cities or the number of words in a text have a long tail to one side, and the 68% rule does not hold for them. And a mix of groups, such as the bills of all three species together, can have two or three humps. Look at a histogram before you assume a bell curve."),
        mo.md(r"""
        The picture: the bill lengths of the Adélie penguins in bins of 1 mm, and the <span class="c-pred">bell curve</span> with their mean and standard deviation. The dashed lines are one standard deviation from the mean.
        """),
        mo.Html(C.picture({"title": "Bill length of Adélie penguins", "x": "bill length (mm)", "y": "penguins", "band": "within one σ",
                           "alt": f"A histogram of the bill lengths of {len(C.bills())} Adélie penguins, from about 32 to 46 mm, highest at 37 to 40 mm, with a bell curve over it that follows the bars closely. Dashed lines at one standard deviation on each side of the mean."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To describe data with a distribution:

    1. Look at the kind of outcome: yes or no is Bernoulli, one of several classes is categorical, a measurement may be normal.
    2. Estimate the parameters from the data: $p$ is the share of yes, each class gets its share, $\mu$ is the mean and $\sigma$ the standard deviation.
    3. Use the distribution: its probabilities, its expected value and variance, or new values drawn from it.
    4. Check it against the data, for a normal distribution with the share within one and two standard deviations.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins with all three distributions. `C.penguins()` is the table, `C.bills()` the bill lengths of the Adélie penguins, and `C.within(k)` the share of those within `k` standard deviations, which you write yourself in exercise 3. The comments with `#` say which step of the recipe each part is. Steps 3 and 4 are not finished: run the cell as it is, then complete them where the comment says, and run it again.
    """)
    return


@app.cell
def example_distributions():
    import random
    import numpy as np

    _d = C.penguins()
    # 1. sex is yes or no: Bernoulli. The species is one of three: categorical.
    #    The bill length is a measurement: normal?
    # 2. the parameters: p, the shares of the species, mu and sigma
    _males = np.sum(_d["sex"] == "male")
    _females = np.sum(_d["sex"] == "female")
    _p = _males / (_males + _females)
    print(_p)
    print(_d["species"].value_counts() / len(_d))
    _x = C.bills()
    _mu = np.mean(_x)
    _sigma = np.std(_x)
    print(_mu, _sigma)
    # 3. use them: the expected value of Bernoulli is here; add its variance. Then one draw from each
    print(_p)
    random.seed(1)
    print(random.random() < _p)
    print(random.choices(["Adelie", "Gentoo", "Chinstrap"], weights=[152, 124, 68]))
    print(random.gauss(_mu, _sigma))
    # 4. check the normal distribution: the share within one σ is here; add the share within two
    print(C.within(1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Of the penguins whose sex is known, 50.5% are male: $p$ = 0.505, with variance 0.25, the largest a Bernoulli distribution can have. The `weights` of `random.choices` are the counts of the species, 152, 124 and 68: `value_counts` without dividing. The draws give a male, a Chinstrap penguin and a bill of 39.0 mm. Step 4: 68.2% within one standard deviation and 96.7% within two, close to the 68% and 95% of a normal distribution.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which distribution?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="A model looks at an image and gives a probability for each of the 10 digits. Which distribution do those 10 probabilities form?",
        options={
            "right": "Categorical: one of 10 classes",
            "bernoulli": "Bernoulli: the answer is right or wrong",
            "normal": "Normal: the digits lie around a mean",
        },
        correct="right",
        why={
            "bernoulli": "A Bernoulli distribution has two outcomes. Here the image shows one of 10 digits, each with its own probability.",
            "normal": "The digits are classes, not measurements: a 7 is not between a 6 and an 8 in any useful way.",
            "right": "One of 10 classes, each with its own probability, adding up to 1: a categorical distribution.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_density():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The bell curve as a function

        Write `normal_density(x, mu, sigma)`: the density of $\mathcal{N}(\mu, \sigma^2)$ at `x`, by the formula. The box imports `math`: use `math.exp`, `math.sqrt` and `math.pi`. The check calls `normal_density(4, 1, 3)`, which should give about `0.0807`, and tries other values.
        """),
        L.task("Make `normal_density` give the density of the normal distribution."),
    ])
    return


@app.cell(hide_code=True)
def exercise_density():
    box_density = L.code_box("density", lines=6)
    box_density
    return (box_density,)


@app.cell(hide_code=True)
def check_density(box_density):
    result_density = L.check("density", box_density.value)
    result_density
    return (result_density,)


@app.cell(hide_code=True)
def hints_density():
    L.hints(
        "density",
        [
            "The formula has two parts: the number in front, 1 divided by $\\sigma \\sqrt{2\\pi}$, and $e$ to the power $-\\frac{(x - \\mu)^2}{2\\sigma^2}$.",
            "`math.exp(v)` is $e$ to the power `v`, and `**` is a power. Put brackets around everything you divide by.",
        ],
        why="At $x = 4$, $\\mu = 1$, $\\sigma = 3$: the number in front is $\\frac{1}{3 \\cdot 2.507} \\approx 0.133$, and $e^{-9/18} = e^{-1/2} \\approx 0.6065$. Together about 0.0807.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_within():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Check against the bell curve

        Write `share_within(x, k)`: the share of the values of the array `x` that lie within `k` standard deviations of their mean, so more than the mean minus `k` standard deviations and less than the mean plus `k`. That is step 4 of the recipe. The check calls `share_within(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1)`, which should give `0.6`, and tries other arrays and other values of `k`.
        """),
        L.task("Make `share_within` give the share of the values within `k` standard deviations of the mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_within():
    box_within = L.code_box("share-within", lines=10)
    box_within
    return (box_within,)


@app.cell(hide_code=True)
def check_within(box_within):
    result_within = L.check("share-within", box_within.value)
    result_within
    return (result_within,)


@app.cell(hide_code=True)
def hints_within():
    L.hints(
        "share-within",
        [
            "Work out the mean and the standard deviation, and from them the two edges of the band.",
            "A boolean mask keeps the values above the lower edge; a second mask, on what is left, keeps those below the upper edge. Divide how many are left by how many there were.",
        ],
        why="The mean is 5.5 and the standard deviation about 2.87, so the band runs from about 2.63 to 8.37. The values 3 to 8 lie inside: 6 of 10.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The probabilities of [Probability: counting how often](/course/open/prob-basics?lang=en) and the expected value and variance of [Expected value and variance](/course/open/prob-expectation?lang=en) now come in three shapes: Bernoulli for yes or no, categorical for the classes of [Many classes: softmax](/course/open/ml-softmax?lang=en), and normal for measurements. In the next lesson, *Entropy: average surprise*, you measure how uncertain a distribution is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_density, result_within):
    L.wrap_up(choice_which, result_density, result_within)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
