# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson stats-mle. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("stats-mle", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en) and [How good is a language model?](/course/open/lm-nll?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kl():
    quick_kl = L.choice(
        "quick-kl",
        question="Is $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ always the same as $D_{\\mathrm{KL}}(q \\,\\|\\, p)$?",
        options={
            "right": "No: the truth and the expectation play different parts",
            "yes": "Yes: it measures how far apart two distributions are",
            "coin": "Yes, for a coin: with two outcomes the order does not matter",
        },
        correct="right",
        why={
            "yes": "KL is not a distance: the truth weighs the outcomes, and swapping the two usually gives another number. See [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).",
            "coin": "Even for a coin it matters: for a fair coin $p$ and an expected $q$ of 90% heads, $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ is about 0.74 bits, and swapped, $D_{\\mathrm{KL}}(q \\,\\|\\, p)$ is about 0.53. See [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).",
            "right": "In $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ the truth $p$ weighs the outcomes; swap them and the weights change, and so does the number.",
        },
    )
    quick_kl
    return (quick_kl,)


@app.cell(hide_code=True)
def quick_check_nll():
    quick_nll = L.choice(
        "quick-nll",
        question="Two language models are tested on the same real text. One has an average NLL of 2.1, the other 2.4. Which predicts the text better?",
        options={
            "high": "The one with 2.4: it is more sure",
            "right": "The one with 2.1: it is less surprised",
            "same": "Neither: the NLL says nothing about quality",
        },
        correct="right",
        why={
            "high": "A higher NLL means more surprise by what really came: worse. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "same": "The average NLL is exactly the measure of how well a model predicts real text. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "right": "Lower is better: the model with 2.1 gave the real next characters higher probabilities.",
        },
    )
    quick_nll
    return (quick_nll,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model has parameters, and data comes in. Which values of the parameters should you choose? Much of machine learning gives the same answer: the values that make the data you saw most likely. That answer explains why language models, classifiers and many other models are all trained on the same kind of loss.

    **An everyday example.** You find a die on the street and throw it 60 times: 20 sixes, where a fair die would give about 10. You no longer believe it is fair. But which chance of a six should you believe in instead, and why that one?
    """)
    return


@app.cell(hide_code=True)
def predict_coin():
    guess_coin = L.guess(
        "seven-of-ten",
        question="A coin landed heads 7 times in 10 tosses. You may choose its probability of heads, $p$. Which $p$ do you think makes these 10 tosses most likely?",
        options={
            "fair": "0.5: coins are usually fair",
            "right": "0.7: the share of heads",
            "sure": "1: heads came most",
        },
        reveal="**0.7.** The probability of these tosses is $p^7 (1 - p)^3$, seven heads and three tails. At $p$ = 0.5 that is about 0.00098; at 0.7 about 0.0022, more than twice as likely; at $p$ = 1 it is 0, because then a tail could never have come. The share of heads is the $p$ that makes the data most likely.",
    )
    guess_coin
    return (guess_coin,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The likelihood of [How good is a language model?](/course/open/lm-nll?lang=en) is the probability that a model gives the data you saw, as a function of the parameters $\theta$: the probabilities $p_\theta(x_i)$ of all examples $x_i$, multiplied. **Maximum likelihood** chooses the $\theta$ that makes it largest:
        """),
        L.formula(r"\begin{aligned} [[param:\hat{\theta}]] &= \arg\max_\theta \prod_i p_\theta(x_i) \\ &= \arg\min_\theta \; [[loss:-\frac{1}{n} \sum_i \log p_\theta(x_i)]] \end{aligned}"),
        mo.md(r"""
        The two forms give the same $\hat{\theta}$: the log keeps the order of numbers, so the largest product has the largest sum of logs; a minus sign turns the largest into the smallest, and dividing by $n$ does not move it. The second form is the average negative log-likelihood, and that is the cross-entropy of [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en) between the data and the model. For the coin: 7 of the 10 terms are $-\log p$ and 3 are $-\log(1 - p)$, so the average is $-(0.7 \log p + 0.3 \log(1 - p))$, the cross-entropy with the shares in the data as the truth. So training on the cross-entropy loss is maximum likelihood; and since the entropy of the data does not change, it makes the KL divergence from the data to the model as small as it can be.

        For a coin, the maximum likelihood estimate of $p$ is the share of heads. For one of several classes, it is the share of each class: the bigram table of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en), each row of counts divided by its sum, is the maximum likelihood estimate, and no other table makes the text it was counted on more likely. Add a belief about the parameters before you see any data, and a penalty joins the loss.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **maximum likelihood** | | choosing the parameters that make the data most likely; the same as choosing the ones with the lowest average negative log-likelihood | |
        | $\theta$ | theta | all parameters of a model together; for the coin only $p$ | `p` |
        | $p_\theta(x_i)$ | p theta of x i | the probability the model with parameters $\theta$ gives example $x_i$ | |
        | $\prod_i$ | product over i | multiply the expression for every $i$, as $\sum_i$ adds it up | |
        | $\hat{\theta}$ | theta hat | the estimate of the parameters | `best` |
        | $\arg\max_\theta$, $\arg\min_\theta$ | arg max, arg min over theta | the $\theta$ at which something is largest or smallest, not that value itself | `grid[np.argmin(values)]` |
        | `np.inf` | infinity | a number larger than every other number, a good start when you look for the smallest | `np.inf` |
        """),
        L.watch_out("Maximum likelihood fits the data you have, also its accidents. A coin that landed heads 3 times in 3 tosses gets $p$ = 1: tails would be impossible. With little data, a penalty or more data keeps the estimate sensible."),
        mo.md(r"""
        The picture: the average negative log-likelihood of $p$ for 7 heads in 10 tosses. It is lowest at the share of heads, 0.7.
        """),
        mo.Html(C.coin_picture({"title": "7 heads in 10 tosses", "x": "probability of heads, p", "y": "average NLL",
                                "best": "lowest at 0.7: {v}", "fair": "0.5: {v}", "number": L.number,
                                "alt": "A curve of the average negative log-likelihood of p, high near 0 and near 1, with its lowest point at p = 0.7, about 0.611. A grey point at p = 0.5, about 0.693."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the maximum likelihood estimate:

    1. Write the probability the model gives one example, as a function of the parameter.
    2. The average negative log-likelihood: minus the log of that probability, averaged over all examples.
    3. Find the parameter with the lowest average: here by trying a grid of values.
    4. Compare with the share in the data: for a coin, the share of heads.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The coin with 7 heads in 10 tosses, as an array of ones and zeros, and a grid of values of $p$. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_mle():
    import numpy as np

    _data = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    _grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    _nlls = []
    for _p in _grid:
        # 1. the probability p gives each toss: p for heads, 1 - p for tails
        _probs = _data * _p + (1 - _data) * (1 - _p)
        # 2. the average negative log-likelihood
        _nlls.append(-np.mean(np.log(_probs)))
    print(np.array(_nlls))
    # 3. the p with the lowest average
    _best = _grid[np.argmin(_nlls)]
    print(_best)
    # 4. compare with the share of heads in the data: add it here, with np.mean
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The lowest of the five averages is at 0.7, and with step 4 complete you see that the share of heads in the data is 0.7 too. Its average NLL, about 0.611, is the entropy of a coin with $p$ = 0.7, in nats: at the best $p$, the model's cross-entropy equals the entropy of the data, and the KL divergence is 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why the log?
    """)
    return


@app.cell(hide_code=True)
def choice_log_cell():
    choice_log = L.choice(
        "why-log",
        question="Maximum likelihood looks for the largest product of probabilities. Why do models minimize the average negative log-likelihood instead?",
        options={
            "different": "It gives a better answer: the log makes small probabilities count more",
            "faster": "The product cannot be worked out at all, not even for a few examples",
            "right": "Its best point is in the same place, and a sum of logs does not round to 0",
        },
        correct="right",
        why={
            "different": "It gives the same answer: the log keeps the order, so the best parameters do not change.",
            "faster": "For a few examples the product works fine; for millions it becomes 0 in a computer, and that is the problem.",
            "right": "The log keeps the order, so the best point stays where it was; and a sum of millions of logs is a normal number, where the product would round to 0.",
        },
    )
    choice_log
    return (choice_log,)


@app.cell(hide_code=True)
def text_exercise_nll():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The average negative log-likelihood

        Write `avg_nll(p, data)`: the average negative log-likelihood of `p`, with the natural log, for an array `data` of ones and zeros, by steps 1 and 2 of the recipe. The check calls `avg_nll(0.75, np.array([1, 0, 1, 1]))`, which should give about `0.5623`, and tries other values.
        """),
        L.task("Make `avg_nll` give the average surprise of `data` under `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nll():
    box_nll = L.code_box("avg-nll", lines=6)
    box_nll
    return (box_nll,)


@app.cell(hide_code=True)
def check_nll(box_nll):
    result_nll = L.check("avg-nll", box_nll.value)
    result_nll
    return (result_nll,)


@app.cell(hide_code=True)
def hints_nll():
    L.hints(
        "avg-nll",
        [
            "Step 1 of the recipe: a one has probability `p`, a zero has probability `1 - p`. Multiplying by `data` and by `1 - data` picks the right one for each example.",
            "Step 2: minus the log of each probability, and then the mean.",
        ],
        why="A one has probability 0.75 and costs $-\\log(0.75) \\approx 0.288$; the zero has 0.25 and costs $-\\log(0.25) \\approx 1.386$. Three ones and one zero: $(3 \\cdot 0.288 + 1.386) / 4 \\approx 0.562$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The most likely p

        Write `best_p(data, grid)`: the `p` from the array `grid` with the lowest average negative log-likelihood for `data`, by step 3 of the recipe. The check calls `best_p(np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0]), np.array([0.1, 0.3, 0.5, 0.7, 0.9]))`, which should give `0.3`, and tries other data and other grids. This box does not know `avg_nll` from exercise 2: work out the average in the loop, or copy your `avg_nll` into the box, above `best_p`.
        """),
        L.task("Make `best_p` give the `p` of the grid that makes the data most likely."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-p", lines=12)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-p", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-p",
        [
            "Go over the grid with a loop, and work out the average negative log-likelihood of each `p`, as in exercise 2.",
            "Keep the best so far: start with `best_nll = np.inf`, and replace it, with its `p`, each time you find a lower one. Or collect them in a list and use `np.argmin`.",
        ],
        why="3 ones in 10: the share is 0.3, and that is the value of the grid where the average NLL is lowest.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The average negative log-likelihood of [How good is a language model?](/course/open/lm-nll?lang=en) is the cross-entropy of [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en), and making it as small as you can is maximum likelihood. That is why so many models train on the same loss. In the next lesson, *Floating point and the log-sum-exp trick*, you see what happens when a computer works out those logs and exponents, and how to keep them from overflowing.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_log, result_nll, result_best):
    L.wrap_up(choice_log, result_nll, result_best)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
