# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson info-entropy. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("info-entropy", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sure():
    quick_sure = L.choice(
        "quick-surprise",
        question="A weather model gave rain the probability 0.8 on Monday and 0.2 on Tuesday. It rained on both days. On which day was the surprise larger?",
        options={
            "right": "Tuesday: the model found rain unlikely then",
            "monday": "Monday: the probability was the larger number",
            "same": "Neither: it rained on both days, so the surprise is the same",
        },
        correct="right",
        why={
            "monday": "The surprise is minus the log of the probability: the larger the probability, the smaller the surprise. See [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "same": "The surprise depends on the probability the model gave to what happened, not only on what happened. See [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "right": "$-\\log(0.2) \\approx 1.61$ and $-\\log(0.8) \\approx 0.22$: what the model found unlikely surprises it more.",
        },
    )
    quick_sure
    return (quick_sure,)


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="A number is 1, 2 or 3, with the probabilities 0.2, 0.5 and 0.3. What is its expected value?",
        options={
            "middle": "2: the middle value",
            "likely": "2, because 2 is the most likely",
            "right": "2.1: each value times its probability, added up",
        },
        correct="right",
        why={
            "middle": "The middle value counts only when the probabilities are balanced around it; here 3 is more likely than 1. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "likely": "The most likely value is not the long-run average: the others count too. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "1 · 0.2 + 2 · 0.5 + 3 · 0.3 = 0.2 + 1 + 0.9 = 2.1.",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    How much information is in a message? A coin toss tells you less than the roll of a die, and a forecast of sun in the desert tells you almost nothing. Information theory measures this in bits. The same number says how far a file can be compressed, and in the next lessons it leads to the loss that language models are trained and compared with.

    **An everyday example.** In the game Twenty Questions you find out what someone has in mind with yes-or-no questions. A good question splits what is left into two halves, so twenty questions can tell about a million things apart.
    """)
    return


@app.cell(hide_code=True)
def predict_bits():
    guess_bits = L.guess(
        "plays-bits",
        question=f"The plays of Shakespeare use {C.kinds()} different characters. A computer stores each in 8 bits, and if all {C.kinds()} were equally common, a code would need {L.number(C.uniform_bits(), 2)} bits per character. A smarter code gives the common characters short codes and the rare ones long codes. How many bits per character do you think it needs, at least, on average?",
        options={
            "six": "About 5.9: how often a character appears hardly matters",
            "right": "About 4.8: the common characters save a lot",
            "three": "About 3.1: the common characters save even more",
        },
        reveal=f"**About 4.8: {L.number(C.plays_entropy(), 2)} bits.** That is the entropy of the characters of the plays: the space alone is about 15% of them, and `e` about 8%. No code that looks at one character at a time can do better on average. gzip, which also looks at the characters before, gets {L.number(C.gzip_bits(), 2)} bits per character: the context helps.",
    )
    guess_bits
    return (guess_bits,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **bit** is the answer to one yes-or-no question whose two answers are equally likely. The surprise $-\log(p)$ of [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en), with the log to base 2, is measured in bits: an outcome with probability $p$ carries $-\log_2(p)$ bits. A fair coin gives 1 bit; one of 8 equally likely outcomes gives 3 bits, the three yes-or-no questions that halve 8 to 4, 2 and 1.

        **Entropy** is the expected surprise of a distribution: each outcome's surprise, weighted by its probability, added up, as an expected value in [Expected value and variance](/course/open/prob-expectation?lang=en).
        """),
        L.formula(r"H(p) = -\sum_i p_i \log_2 p_i"),
        mo.md(r"""
        Entropy is 0 when one outcome is sure, and largest, $\log_2 K$, when all $K$ outcomes are equally likely. In bits, it is a floor: by a theorem of Claude Shannon from 1948, no code that codes each symbol on its own can use fewer bits per symbol on average, and no strategy of yes-or-no questions needs fewer questions on average. The best codes come within 1 bit of it, and reach it exactly when every probability is 1/2, 1/4, 1/8 and so on. With the natural log, as in log loss, the unit is the **nat**; 1 nat is about 1.44 bits.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **bit** | | the unit of information: the answer to one fair yes-or-no question | |
        | **entropy**, $H(p)$ | H of p | the expected surprise of a distribution, in bits: how uncertain you are before you see the outcome | `-np.sum(p * np.log2(p))` |
        | $\log_2$ | log base 2 | the exponent of 2 that gives a number: $\log_2 8 = 3$ | `np.log2(x)` |
        | **nat** | | the unit of information with the natural log instead of $\log_2$; 1 nat is about 1.44 bits | `np.log(x)` |
        """),
        L.watch_out("Entropy measures uncertainty, not value: a string of random characters, each of the 65 equally likely, has more entropy per character than a page of Shakespeare. And it treats each symbol on its own; when symbols depend on each other, as letters in words do, a code that uses the context can go below it, as gzip did."),
        mo.md(r"""
        The picture: the entropy of a coin that lands heads with probability $p$. It is 0 for a coin that always or never lands heads, and largest, 1 bit, for a fair coin.
        """),
        mo.Html(C.coin_picture({"title": "The entropy of a coin", "x": "probability of heads", "y": "entropy (bits)",
                                "point": "p = {p}: {h} bits", "number": L.number,
                                "alt": "A curve shaped like an arch, from 0 bits at probability 0, up to 1 bit at probability 0.5, and down to 0 bits at probability 1. Marked: 1 bit at 0.5, and 0.47 bits at 0.9."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the entropy of a distribution:

    1. List the probabilities of all outcomes; together they make 1.
    2. Leave out the outcomes with probability 0: they never happen and add no surprise.
    3. Work out the surprise of each outcome in bits, $-\log_2(p)$.
    4. Weight each surprise by its probability and add them up; compare with $\log_2 K$, the most there can be.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The entropy of the species of the 344 Palmer penguins: Adélie, Gentoo and Chinstrap. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_entropy():
    import numpy as np

    # 1. the probabilities, from the counts of each species
    _counts = np.array(C.species_counts())
    _p = _counts / _counts.sum()
    # 2. no species has probability 0 here, but the line keeps the recipe whole
    _p = _p[_p > 0]
    # 3. the surprise of each species, in bits
    _surprise = -np.log2(_p)
    print(_surprise)
    # 4. the most there can be with 3 species is here; weight the surprises by _p, add them up, and print that too
    print(np.log2(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    _p = C.species_probs()
    mo.md(f"""
    Chinstrap is the rarest, so it surprises most: {L.number(_p[2], 2)} as a probability gives about {L.number(C.surprise_bits(_p[2]), 2)} bits. With step 4 complete, the entropy is {L.number(C.species_entropy(), 3)} bits, a little below the {L.number(C.surprise_bits(1 / 3), 3)} of three equally common species: the species are not quite equally common, so a guess is a little easier.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_coins():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The most uncertain outcome
    """)
    return


@app.cell(hide_code=True)
def choice_coins_cell():
    choice_coins = L.choice(
        "most-entropy",
        question="Which of these has the highest entropy?",
        options={
            "coin": "A fair coin: two sides, each half of the time",
            "skewed": "A spinner with four parts: one comes up 70% of the time, the other three 10% each",
            "right": "A spinner with three equal parts, each a third of the time",
        },
        correct="right",
        why={
            "coin": "A fair coin has 1 bit, the most two outcomes can have. With three equally likely outcomes you know even less in advance.",
            "skewed": "Four outcomes, but one comes up most of the time, so you often guess right: about 1.36 bits. Three equal parts give more.",
            "right": "Three equally likely outcomes: $\\log_2 3 \\approx 1.585$ bits. More outcomes give more entropy only when they are not lopsided: the four-part spinner has about 1.36 bits, the coin 1.",
        },
    )
    choice_coins
    return (choice_coins,)


@app.cell(hide_code=True)
def text_exercise_entropy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Entropy as a function

        Write `entropy(p)`: the entropy in bits of the probabilities in the array `p`, by the recipe. The check calls `entropy(np.array([0.5, 0.25, 0.25]))`, which should give `1.5`, and tries other distributions, also one with a probability of 0.
        """),
        L.task("Make `entropy` give the expected surprise of `p`, in bits."),
    ])
    return


@app.cell(hide_code=True)
def exercise_entropy():
    box_entropy = L.code_box("entropy", lines=7)
    box_entropy
    return (box_entropy,)


@app.cell(hide_code=True)
def check_entropy(box_entropy):
    result_entropy = L.check("entropy", box_entropy.value)
    result_entropy
    return (result_entropy,)


@app.cell(hide_code=True)
def hints_entropy():
    L.hints(
        "entropy",
        [
            "Steps 2 to 4 of the recipe: leave out the zeros first, then the surprise in bits, then weigh and add.",
            "A boolean mask, `p[p > 0]`, leaves out the zeros. `np.log2` gives the log to base 2 of every number at once.",
        ],
        why="The surprises are 1, 2 and 2 bits. Weighed: $0.5 \\cdot 1 + 0.25 \\cdot 2 + 0.25 \\cdot 2 = 1.5$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bits():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How small can a text get?

        Write `min_bits(text)`: the least number of bits that a code needs for the whole text when it uses only how often each character appears. Count each character, turn the counts into probabilities, and multiply the entropy by the number of characters. The check calls `min_bits("aaab")`, which should give about `3.245`, and tries other texts.
        """),
        L.task("Make `min_bits` give the entropy of the characters of `text` times its length."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bits():
    box_bits = L.code_box("min-bits", lines=16)
    box_bits
    return (box_bits,)


@app.cell(hide_code=True)
def check_bits(box_bits):
    result_bits = L.check("min-bits", box_bits.value)
    result_bits
    return (result_bits,)


@app.cell(hide_code=True)
def hints_bits():
    L.hints(
        "min-bits",
        [
            "Count with a dictionary, as in [Dictionaries](/course/open/py-dicts?lang=en): one key per character, the count as its value. Then each count divided by the length of the text is a probability.",
            "The entropy of those probabilities is the number of bits per character; the whole text needs that many times its length.",
        ],
        why="In \"aaab\", `a` has probability 0.75 and `b` 0.25. The entropy is $0.75 \\cdot 0.415 + 0.25 \\cdot 2 \\approx 0.811$ bits per character, and 4 characters need about 3.245 bits.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The surprise of [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en), in bits, and averaged as an expected value as in [Expected value and variance](/course/open/prob-expectation?lang=en), is the entropy: how uncertain a distribution is, and how few bits a code can get away with. In the next lesson, *Cross-entropy and KL divergence*, you measure the surprise when you expect one distribution and the truth is another.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_coins, result_entropy, result_bits):
    L.wrap_up(choice_coins, result_entropy, result_bits)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
