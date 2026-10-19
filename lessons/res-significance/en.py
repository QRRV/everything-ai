# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-significance. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-significance", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en) and [Fair comparisons](/course/open/res-fair-baselines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_resample():
    quick_resample = L.choice(
        "quick-resample",
        question="You bootstrap the mean of 8 runs. What is one resample?",
        options={
            "right": "8 of your runs, drawn with replacement",
            "half": "4 of your runs, drawn without replacement",
            "new": "8 new runs, trained with other seeds",
        },
        correct="right",
        why={
            "half": "A resample is as big as your sample, and a run can come twice. See [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en).",
            "new": "The bootstrap trains nothing new: it draws from the runs you have. See [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en).",
            "right": "As many as you have, with replacement, so some runs come twice and others not at all.",
        },
    )
    quick_resample
    return (quick_resample,)


@app.cell(hide_code=True)
def quick_check_baseline_rate():
    quick_rate = L.choice(
        "quick-rate",
        question="You compare your method with a baseline. Which learning rate does the baseline get?",
        options={
            "yours": "The rate that was best for your method",
            "paper": "The one its own paper used",
            "right": "Its own best, from the same tries",
        },
        correct="right",
        why={
            "yours": "A rate that suits your method may not suit the baseline. See [Fair comparisons](/course/open/res-fair-baselines?lang=en).",
            "paper": "Its paper had another setup, and maybe another tuning effort. See [Fair comparisons](/course/open/res-fair-baselines?lang=en).",
            "right": "The same learning rates tried for both, and each at its own best: the same tuning effort.",
        },
    )
    quick_rate
    return (quick_rate,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en), the small GPT without warmup ended lower than with warmup for every one of five seeds, and yet the two intervals overlapped a lot. Those intervals treated the runs as two separate groups. But the runs come in pairs: with the same seed, both runs start from the same weights and see the mini-batches in the same order. If one run of a seed ends high, the other mostly does too. Compare the runs pair by pair, and that shared part falls away.

    **An everyday example.** You want to know which of two shoe soles wears out less. Give ten children sole A and ten other children sole B, and the differences between the children, how much they run and jump, drown out the soles. Give each child one sole of each kind, one on each foot, and compare the two feet of each child: the differences between children cancel out.
    """)
    return


@app.cell(hide_code=True)
def predict_chance():
    guess_chance = L.guess(
        "all-ten",
        question="Suppose warmup makes no difference at all. Then for each seed it is a coin flip which of its two runs ends lower. In the course's runs, all ten seeds ended lower without warmup. How likely is that by chance alone?",
        options={
            "ten": "About 1 in 10",
            "hundred": "About 1 in 100",
            "right": "About 1 in 1000",
        },
        reveal="**About 1 in 1000.** All ten lower is like ten coin flips that all come up heads: $(1/2)^{10} = 1/1024$. Even counting both directions, all lower or all higher, it is 2 in 1024, 0.2%. Chance alone explains it badly. The course trained seeds 5 to 9 for this lesson; seeds 0 to 4 are those of One run is not a result, and they came out exactly the same again.",
    )
    guess_chance
    return (guess_chance,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Give both settings the same seeds, and take the **difference** of each pair: the run with warmup minus the run without. Then ask whether these differences could come from chance alone.

        The **null hypothesis** is that the change makes no difference. If it holds, the two runs of a seed could just as well swap places, and a swap flips the sign of their difference. A **permutation test** does that: it flips the sign of each difference at random, many times, and takes the mean each time. Those means are what chance alone gives. The **p-value** is the share of them that lies at least as far from 0 as the mean you measured, on either side: before you measured, the change could have gone either way. If p is small, usually below 0.05, chance alone explains your result badly, and the difference is called **statistically significant**.

        With few pairs there are few patterns of signs: $2^n$ for $n$ pairs. With five pairs that is 32, and the most extreme pattern, all positive or all negative, still gives p = 2/32 = 0.0625. With this test, five seeds can never get below 0.05, however clear the difference.

        A **paired bootstrap** gives an interval instead: it resamples the differences, with replacement, and takes the mean each time; the middle 95% of those means is the interval of the difference. If it does not contain 0, chance alone is a poor explanation too. As with any bootstrap, the interval is too narrow when there are only a few pairs. When the two runs of a seed are alike, as here, resampling the two settings each on its own breaks up the pairs: the interval gets much wider, because then the differences between seeds count as noise.

        The first picture: each of the ten seeds as a line from its run with warmup to its run without. The second: the means of all 1,024 patterns of signs of the ten differences, and the mean you measured: no pattern lies further from 0, and only its mirror image, with all signs flipped, lies as far.
        """),
        mo.Html(C.pairs_picture({"left": "Seed by seed", "right": "Chance alone", "y": "validation loss", "with": "with warmup", "without": "without",
                                 "x": "mean difference", "count": "patterns", "yours": "measured", "lang": L.lang,
                                 "alt_left": "Ten lines, one per seed, from the run with warmup to the run without; all ten go down, by 0.002 to 0.014.",
                                 "alt_right": "A histogram of the means of all 1,024 sign patterns of the ten differences, around 0, from about −0.006 to 0.006; the measured mean, 0.0059, lies at the far right edge."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **null hypothesis** | the assumption that a change makes no difference, which a test checks |
        | **permutation test** | a test that shuffles what the null hypothesis says does not matter, here the signs of the differences, to see what chance alone gives; NLP papers call this paired version the approximate randomization test | <!-- termen:ok -->
        | **p-value** | the share of chance results at least as extreme as yours, if the null hypothesis is true |
        | **statistically significant** | with a p-value below a limit chosen in advance, usually 0.05 |
        | **paired bootstrap** | the bootstrap of the differences of pairs, which keeps each pair together |
        """),
        L.watch_out("A p-value is not the chance that the null hypothesis is true: it is how often chance alone would give a result at least as extreme as yours if it were true. A p-value above 0.05 does not show that there is no difference: with few seeds, even a real one stays hidden. And a tiny p-value does not show that a difference is big: here it is 0.006."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To test whether a difference between two settings could be chance:

    1. Pair the runs: the same seeds for both settings, and the difference per seed.
    2. Permutation test: flip the signs of the differences at random, many times, and take the mean each time; p is the share of those means at least as far from 0 as yours.
    3. Paired bootstrap: resample the differences with replacement, many times, and take the mean each time; the middle 95% is the interval of the difference.
    4. Report the mean difference, its interval, p and the number of seeds.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 and 2 with the course's differences, with warmup minus without. The comments say which step each part is. It is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_permutation():
    import numpy as np

    _all = np.array([0.0036, 0.0083, 0.0059, 0.0022, 0.005, 0.0135, 0.0048, 0.0053, 0.0071, 0.0036])
    # 1. the differences of seeds 0 to 4
    _diffs = _all[:5]  # complete: the differences of all ten seeds, _all
    _observed = np.mean(_diffs)
    # 2. the permutation test: random signs, 10,000 times, and how often the mean lies at least as far from 0 as ours
    _rng = np.random.default_rng(0)
    _means = []
    for _ in range(10000):
        _signs = _rng.choice([-1, 1], size=len(_diffs))
        _means.append(np.mean(_signs * _diffs))
    _p = np.mean(np.abs(_means) >= abs(_observed))
    print("seeds:", len(_diffs), " mean difference:", round(_observed, 4), " p:", _p)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With five seeds, p is about 0.06: all five positive is the most extreme pattern there is, and still 2 of the 32 patterns of signs reach it. With all ten seeds, p is about 0.002. The difference hardly changed, from about 0.005 to 0.006; what changed is how much chance can explain.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bootstrap():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A paired bootstrap

        Complete `paired_bootstrap(a, b, n_boot, seed)`: step 3 of the recipe as a function. `a` and `b` are the runs of two settings, with the same seeds in the same order. It gives back three numbers: the mean of the differences `a` minus `b`, and the low and the high end of the 95% bootstrap interval of that mean, from `n_boot` resamples with a random number generator made with `seed`. The loop has the shape of the worked example, with a resample of the differences in place of random signs. Replace the two `...`. The check calls `paired_bootstrap(with_warmup, without_warmup, 10000, 0)` with the course's ten seeds, and tries other runs. A bootstrap is random, so the check allows the small differences that another way of drawing gives.
        """),
        L.task("Make `paired_bootstrap(a, b, n_boot, seed)` give back the mean of the differences `a` minus `b`, pair by pair, and the two ends of the 95% bootstrap interval of that mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bootstrap():
    box_bootstrap = L.code_box("paired-bootstrap", lines=10)
    box_bootstrap
    return (box_bootstrap,)


@app.cell(hide_code=True)
def check_bootstrap(box_bootstrap):
    result_bootstrap = L.check("paired-bootstrap", box_bootstrap.value)
    result_bootstrap
    return (result_bootstrap,)


@app.cell(hide_code=True)
def hints_bootstrap():
    L.hints(
        "paired-bootstrap",
        [
            "The differences come first: subtract the two lists as NumPy arrays, each run in `a` minus the run in `b` with the same seed. Then resample those differences the way you resampled runs in [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
            "`diffs = np.array(___) - np.array(___)`, and in the loop `means.append(np.mean(rng.choice(___, size=len(diffs))))`.",
        ],
        why="```python\nimport numpy as np\n\n\ndef paired_bootstrap(a, b, n_boot, seed):\n    diffs = np.array(a) - np.array(b)\n    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n        means.append(np.mean(rng.choice(diffs, size=len(diffs))))\n    low, high = np.quantile(means, [0.025, 0.975])\n    return [np.mean(diffs), low, high]\n```\n\nEach resample draws whole pairs, as their differences, so the two runs of a seed stay together. The middle 95% of the means of the resamples is the interval of the mean difference.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Paired or not

    Your `paired_bootstrap` on the course's pairs, next to the bootstrap that resamples each setting on its own, and the exact p-value of the permutation test over all patterns of signs. Until Exercise 1 is right, the course's `paired_bootstrap` works out the paired intervals.

    In the first column, pairing makes the interval much narrower: the two runs of a seed are alike. In the second, it does not: with another number of layers, the same seed no longer gives the same starting weights or the same order of the mini-batches, so the two runs of a seed have less in common. And with three pairs, the paired interval runs only from the smallest difference to the largest: much too narrow to trust.
    """)
    return


@app.cell(hide_code=True)
def demo_paired(box_bootstrap, result_bootstrap):
    _boot, _yours = C.paired_bootstrap_of(box_bootstrap.value if result_bootstrap.state == "correct" else None)
    _s = lambda x: C.signed(x, 4, L.lang)
    # one column per comparison, so that the table fits on a phone
    _cols = []
    for _a, _b in [(C.WARMUP, C.NO_WARMUP), (C.FOUR_LAYERS, C.TWO_LAYERS)]:
        _m, _lo, _hi = _boot(list(_a), list(_b), 10000, 0)
        _u = C.bootstrap(_a, _b, paired=False)
        _p = C.flip_p(C.differences(_a, _b))
        _cols.append([str(len(_a)), _s(_m), f"{_s(_lo)} to {_s(_hi)}", f"{_s(_u[1])} to {_s(_u[2])}", C.number(_p, 3, L.lang)])
    _names = ["seeds", "mean difference", "paired interval", "unpaired interval", "p"]
    _lines = [f"| {_names[_i]} | {_cols[0][_i]} | {_cols[1][_i]} |" for _i in range(len(_names))]
    _who = "" if _yours else "\n\nExercise 1 is not right yet, so the course's `paired_bootstrap` made the paired intervals."
    mo.md("| | with warmup minus without | 4 layers minus 2 layers, both at 0.03 |\n|---|---|---|\n" + "\n".join(_lines) + _who)
    return


@app.cell(hide_code=True)
def text_exercise_three():
    mo.md(r"""
    ### Exercise 2. Three seeds
    """)
    return


@app.cell(hide_code=True)
def choice_three_cell():
    choice_three = L.choice(
        "three-seeds",
        question="The 4-layer and the 2-layer GPT, each at its best learning rate, with seeds 0 to 2: the differences are −0.0114, +0.0003 and +0.0038, and the permutation test gives p = 1. What do you conclude?",
        options={
            "same": "The 2-layer and 4-layer GPT are equally good here",
            "right": "No difference shows, yet one may exist",
            "better": "4 layers wins: its mean is 0.002 lower",
        },
        correct="right",
        why={
            "same": "A large p-value does not show that there is no difference. With three pairs there are only 8 patterns of signs, so p can never be below 2/8 = 0.25.",
            "better": "Its mean is 0.002 lower, but one seed goes one way and two the other way: chance explains that easily.",
            "right": "These runs give no reason to think 4 layers helps here. But with only 8 patterns of signs, p cannot go below 0.25 with three seeds, however big the difference: to show a small one, you need many more seeds.",
        },
    )
    choice_three
    return (choice_three,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Exercise 3. Report the effect of warmup
    """)
    return


@app.cell(hide_code=True)
def explain_report():
    explain_report_cell = L.explain(
        "report",
        question="Write down the result of warmup as you would in a paper: the setup, the seeds, the mean difference with its interval, the p-value and the test, and what it means.",
        model_answer="**Setup:** the small GPT of Change one thing at a time, 6 epochs on the first quarter of Tiny Shakespeare, with a warmup of 100 steps and without, seeds 0 to 9 for both. **Result:** without warmup, the validation loss ended lower for all ten seeds, by 0.006 on average (95% paired bootstrap interval 0.004 to 0.008; permutation test over the signs of the differences, p = 0.002). **Meaning:** chance alone explains this difference badly, but it is small: a learning rate of 0.003 instead of 0.001 lowers the loss by 0.3. With only the first five seeds, the same test could not have gone below p = 0.0625.",
        points=[
            "you name the setup and that both settings had the same ten seeds",
            "you give the mean difference, its paired interval, the p-value and which test gave it",
            "you say what it means: chance alone explains it badly, yet it is small next to what the learning rate does",
        ],
    )
    explain_report_cell
    return (explain_report_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The bootstrap comes from [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en), the seeds from [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en), and the 2-layer and 4-layer runs from [Fair comparisons](/course/open/res-fair-baselines?lang=en). Pairing works here because both settings used the same seeds; the same idea works for two models on the same test examples, pair by pair. The next lesson, *Figures that show something*, puts a result like this in a figure.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_three, explain_report_cell, result_bootstrap):
    L.wrap_up(result_bootstrap, choice_three, explain_report_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
