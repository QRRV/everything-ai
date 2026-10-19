# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson info-cross-entropy-kl. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("info-cross-entropy-kl", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Entropy: average surprise](/course/open/info-entropy?lang=en) and [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_die():
    quick_die = L.choice(
        "quick-die",
        question="A die has 8 sides, all equally likely. What is the entropy of one throw?",
        options={
            "eight": "8 bits: one for each side",
            "one": "1 bit: one throw",
            "right": "3 bits: three yes-or-no questions",
        },
        correct="right",
        why={
            "eight": "Each question halves what is left: 8 sides need $\\log_2 8 = 3$ questions, not 8. See [Entropy: average surprise](/course/open/info-entropy?lang=en).",
            "one": "One bit is one fair yes-or-no question, enough for a coin but not for 8 sides. See [Entropy: average surprise](/course/open/info-entropy?lang=en).",
            "right": "With 8 equally likely sides, the entropy is $\\log_2 8 = 3$ bits.",
        },
    )
    quick_die
    return (quick_die,)


@app.cell(hide_code=True)
def quick_check_onehot():
    quick_onehot = L.choice(
        "quick-onehot",
        question="In the cross-entropy loss $-\\sum_k y_k \\log(\\hat{y}_k)$, the label $y$ is a one-hot vector. What is left of the sum?",
        options={
            "all": "The terms of all classes, each weighted by the model",
            "right": "Only the term of the right class",
            "wrong": "Only the terms of the wrong classes",
        },
        correct="right",
        why={
            "all": "The one-hot vector is 0 for every class but the right one, so the other terms drop out. See [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "wrong": "The wrong classes have $y_k = 0$, so their terms are 0. See [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en).",
            "right": "Only the right class has $y_k = 1$: the loss is $-\\log(\\hat{y}_c)$.",
        },
    )
    quick_onehot
    return (quick_onehot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model never knows the true distribution of its data; it has its own estimate of it. How much does a wrong estimate cost? The cross-entropy loss you train with answers that, and the answer splits into two parts: the surprise that the data has anyway, and the extra that comes from the model's wrong estimate. Training can remove only the second part.

    **An everyday example.** You pack for a week of sun, and it rains. Some trouble you would have had anyway, packed or not; the rest comes from packing for the wrong weather.
    """)
    return


@app.cell(hide_code=True)
def predict_coin():
    guess_coin = L.guess(
        "fair-coin",
        question="A coin is fair, but you expect it to land heads 90% of the time. A fair coin has 1 bit of entropy. How many bits of surprise per toss do you think you get, on average?",
        options={
            "right": "About 1.74 bits: the tails, half of the tosses, surprise you a lot",
            "heads": "About 1.15 bits: the heads, half of the tosses, hardly surprise you",
            "tails": "About 3.32 bits: a tail, which you gave only 0.1, costs that much",
        },
        reveal=f"**About 1.74 bits: {L.number(C.coin(0.5, 0.9), 3)}.** Half of the tosses are heads, which you expected with 0.9: {L.number(C.coin(1.0, 0.9), 2)} bits each. The other half are tails, which you gave only 0.1: {L.number(C.coin(0.0, 0.9), 2)} bits each. On average {L.number(C.coin(0.5, 0.9), 2)} bits, {L.number(C.coin(0.5, 0.9) - 1, 2)} more than the 1 bit a fair coin has anyway.",
    )
    guess_coin
    return (guess_coin,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Say the truth is the distribution $p$, and you expect $q$. Each outcome comes as often as $p$ says, but surprises you as much as $q$ says. The **cross-entropy** is the expected surprise:
        """),
        L.formula(r"[[loss:H(p, q)]] = -\sum_i p_i \log_2 q_i"),
        mo.md(r"""
        It is the general form of the cross-entropy of [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en): there, $p$ was the one-hot label, so only the right class counted, and the log was the natural one. When $q$ is $p$, the cross-entropy is the entropy $H(p)$ of [Entropy: average surprise](/course/open/info-entropy?lang=en). For any other $q$ it is more: expecting the wrong distribution always costs something on average.

        The **KL divergence**, after Kullback and Leibler, is that extra:
        """),
        L.formula(r"[[loss:D_{\mathrm{KL}}(p \,\|\, q)]] = H(p, q) - H(p) = \sum_i p_i \log_2 \frac{p_i}{q_i}"),
        mo.md(r"""
        It is 0 when $q$ is $p$, and never negative. It is not a distance: $D_{\mathrm{KL}}(p \,\|\, q)$ and $D_{\mathrm{KL}}(q \,\|\, p)$ are usually different. When you train a model with the cross-entropy loss, $p$ is the data and $H(p)$ does not change, so lowering the loss lowers the KL divergence: it brings the model's $q$ closer to the data. And when $q$ gives probability 0 to an outcome that $p$ has, the cross-entropy and the KL divergence are infinite.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | $H(p, q)$ | H of p and q | the cross-entropy: the expected surprise when the truth is $p$ and you expect $q$ | `-np.sum(p * np.log2(q))` |
        | **KL divergence**, $D_{\mathrm{KL}}(p \,\|\, q)$ | KL divergence from q to p | the extra surprise above the entropy of $p$ when you expect $q$; 0 only when $q$ is $p$ | `np.sum(p * np.log2(p / q))` |
        """),
        L.watch_out("The order matters. In $H(p, q)$ and $D_{\\mathrm{KL}}(p \\,\\|\\, q)$, $p$ is the truth that weighs the outcomes, and $q$ is what you expect. Swap them and you get another number, and in a loss another training."),
        mo.md(r"""
        The picture: a fair coin, so $p$ is half heads, half tails, and the $q$ you expect. The cross-entropy is lowest, 1 bit, when you expect what is true; at $q$ = 0.9 the gap above 1 bit is the KL divergence.
        """),
        mo.Html(C.coin_picture({"title": "Cross-entropy of a fair coin", "x": "expected probability of heads, q", "y": "bits",
                                "floor": "entropy of the fair coin: 1 bit", "gap": "KL = {kl} bits", "number": L.number,
                                "alt": "A U-shaped curve of the cross-entropy, lowest at q = 0.5 with 1 bit, rising steeply towards q = 0 and q = 1. A dashed line at 1 bit. At q = 0.9 an arrow from 1 bit up to the curve, at about 1.74 bits: the KL divergence of 0.74 bits."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure what a wrong expectation costs:

    1. Write the true distribution $p$ and the expected $q$, over the same outcomes.
    2. The cross-entropy: the surprise under $q$, $-\log_2 q_i$, weighted by $p_i$, added up; leave out the outcomes with $p_i = 0$.
    3. The entropy of $p$, as in [Entropy: average surprise](/course/open/info-entropy?lang=en).
    4. The KL divergence: the cross-entropy minus the entropy, the extra bits for expecting $q$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The species of the 344 Palmer penguins as $p$, and as $q$ the estimate that all three species are equally common. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_kl():
    import numpy as np

    # 1. the truth, from the counts, and an expectation of three equal shares
    _p = np.array(C.species_counts()) / 344
    _q = np.array([1, 1, 1]) / 3
    # 2. the cross-entropy, in bits, over the outcomes that p has (here all three)
    _keep = _p > 0
    _cross = -np.sum(_p[_keep] * np.log2(_q[_keep]))
    # 3. the entropy of p: the 0 holds the place; replace it, as step 2 but with p in the log
    _entropy = 0
    # 4. the extra: the KL divergence
    print(_cross, _entropy, _cross - _entropy)
    return


@app.cell(hide_code=True)
def text_after_example():
    _p = [n / 344 for n in C.species_counts()]
    _biscoe = C.species_counts("Biscoe")
    _q = [n / sum(_biscoe) for n in _biscoe]
    mo.md(f"""
    With the 0 in step 3, the last number is the cross-entropy itself, and that is too much. With step 3 complete: expecting three equal shares costs {L.number(C.bits(_p, [1 / 3] * 3), 3)} bits per penguin, against {L.number(C.bits(_p, _p), 3)} for the true shares, the two numbers the worked example of [Entropy: average surprise](/course/open/info-entropy?lang=en) printed: a KL divergence of {L.number(C.bits(_p, [1 / 3] * 3) - C.bits(_p, _p), 3)} bits, small, because the species are not far from equally common. Take as $q$ the shares on Biscoe island instead, where the data has no Chinstrap: $q$ gives Chinstrap probability 0, the 68 Chinstraps are infinitely surprising, and the cross-entropy and the KL divergence are infinite.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_equal():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. When is there no extra?
    """)
    return


@app.cell(hide_code=True)
def choice_equal_cell():
    choice_equal = L.choice(
        "when-equal",
        question="When is the cross-entropy $H(p, q)$ exactly the entropy $H(p)$?",
        options={
            "uniform": "When $p$ gives every outcome the same probability",
            "sure": "When $q$ is sure of one outcome",
            "right": "When $q$ is the same as $p$",
        },
        correct="right",
        why={
            "uniform": "A uniform $p$ still costs extra when $q$ is not uniform too: what matters is whether $q$ matches $p$.",
            "sure": "A sure $q$ gives probability 0 to the other outcomes; if $p$ has any of those, the cross-entropy is infinite.",
            "right": "The KL divergence is 0 only when $q$ is $p$; any other $q$ costs extra bits on average.",
        },
    )
    choice_equal
    return (choice_equal,)


@app.cell(hide_code=True)
def text_exercise_cross():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Cross-entropy as a function

        Write `cross_entropy(p, q)`: the cross-entropy in bits when the truth is `p` and you expect `q`, by step 2 of the recipe. The check calls `cross_entropy(np.array([0.5, 0.25, 0.125, 0.125]), np.array([0.25, 0.125, 0.5, 0.125]))`, which should give `2.25`, and tries other pairs, also with probabilities of 0.
        """),
        L.task("Make `cross_entropy` give the expected surprise under `q` when the truth is `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cross():
    box_cross = L.code_box("cross-entropy", lines=7)
    box_cross
    return (box_cross,)


@app.cell(hide_code=True)
def check_cross(box_cross):
    result_cross = L.check("cross-entropy", box_cross.value)
    result_cross
    return (result_cross,)


@app.cell(hide_code=True)
def hints_cross():
    L.hints(
        "cross-entropy",
        [
            "Step 2 of the recipe: which distribution goes in the log, and which one weighs?",
            "Keep only the outcomes where `p` is above 0, with a boolean mask, and use it on both arrays.",
        ],
        why="The surprises under `q` are 2, 3, 1 and 3 bits; weighted by `p`: $0.5 \\cdot 2 + 0.25 \\cdot 3 + 0.125 \\cdot 1 + 0.125 \\cdot 3 = 2.25$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kl():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The KL divergence

        Write `kl(p, q)`: the KL divergence in bits, the extra surprise above the entropy of `p`, by steps 2 to 4 of the recipe. The check calls `kl(np.array([0.5, 0.25, 0.125, 0.125]), np.array([0.25, 0.125, 0.5, 0.125]))`, which should give `0.5`, and tries other pairs, both ways round.
        """),
        L.task("Make `kl` give the extra surprise when you expect `q` and the truth is `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kl():
    box_kl = L.code_box("kl", lines=7)
    box_kl
    return (box_kl,)


@app.cell(hide_code=True)
def check_kl(box_kl):
    result_kl = L.check("kl", box_kl.value)
    result_kl
    return (result_kl,)


@app.cell(hide_code=True)
def hints_kl():
    L.hints(
        "kl",
        [
            "Steps 3 and 4 of the recipe: the cross-entropy, minus the entropy of `p`. Or in one sum, with $\\log_2 \\frac{p_i}{q_i}$.",
            "Leave out the outcomes where `p` is 0, as in exercise 2; keep `p` as the weight.",
        ],
        why="The cross-entropy is 2.25 bits and the entropy of `p` is 1.75 bits, so the KL divergence is 0.5 bits.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The entropy of [Entropy: average surprise](/course/open/info-entropy?lang=en) is what the truth costs anyway; the cross-entropy of [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) is what you pay with your own expectation, and the KL divergence is the difference. Training with the cross-entropy loss shrinks that difference. In the next lesson, *Maximum likelihood*, you see that this is the same as making the data as likely as you can, and why so many models train on this loss.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_equal, result_cross, result_kl):
    L.wrap_up(choice_equal, result_cross, result_kl)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
