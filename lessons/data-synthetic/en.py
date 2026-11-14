# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-synthetic. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-synthetic", "en")
    C = L.checks   # the 10,000 runs of AI4I, the log and the test, the simulated runs, k-NN, the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Fix the data, not the model](/course/open/data-centric-ai?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fixed():
    quick_fixed = L.choice(
        "quick-fixed",
        question="Data-centric AI keeps one thing the same and improves another. Which?",
        options={
            "model": "The data stays the same, and you improve the model",
            "test": "Model and data stay the same; only the test set changes",
            "right": "The model stays the same, and you improve the data",
        },
        correct="right",
        why={
            "model": "That is the model-centric way: choosing another k, for example. See *The idea* in [Fix the data, not the model](/course/open/data-centric-ai?lang=en).",
            "test": "The test set stays the same, so that the numbers before and after can be compared. See *The idea* in [Fix the data, not the model](/course/open/data-centric-ai?lang=en).",
            "right": "Right: fixing labels made the data better. Today you make it better by adding examples that were missing.",
        },
    )
    quick_fixed
    return (quick_fixed,)


@app.cell(hide_code=True)
def quick_check_vote():
    quick_vote = L.choice(
        "quick-vote",
        question="With 41 of the 274 penguin labels wrong, k = 1 got 14 test penguins wrong and k = 9 only 3. Why does k = 9 suffer less?",
        options={
            "right": "Most of the 9 nearest have the right label, and they outvote a wrong one",
            "fixed": "k = 9 fixes the wrong labels while it trains",
            "fewer": "k = 9 stores fewer penguins, and so fewer wrong labels",
        },
        correct="right",
        why={
            "fixed": "k-NN changes no label: it only votes with the labels it has. See *Your turn* in [Fix the data, not the model](/course/open/data-centric-ai?lang=en).",
            "fewer": "Every k stores all training penguins; k only says how many of them vote. See [Fix the data, not the model](/course/open/data-centric-ai?lang=en).",
            "right": "Right: a vote of 9 hides a wrong label among the right ones. Today k = 5 votes between two classes, failure or not: with an odd k, a vote cannot tie.",
        },
    )
    quick_vote
    return (quick_vote,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Some examples are rare because they are bad news: a machine that breaks down, a fraud, a rare disease. You cannot wait for a hundred breakdowns before you train a model on them. When there is too little data, you can make more.

    This lesson uses AI4I 2020: 10,000 runs of a milling machine, with temperatures, speed, torque and failures. AI4I is itself made: Stephan Matzka (2020) built it with a simulation that looks like real maintenance data, and wrote down the rules by which its machine fails. Here you treat its runs as if they came from a real machine. The first 3000 runs are your log, and not one of them failed by heat. The next 2000, in a warmer stretch, hold 115 heat failures.

    **An everyday example.** Pilots practice an engine fire in a flight simulator. Waiting for a real one would be a poor way to learn.
    """)
    return


@app.cell(hide_code=True)
def predict_missed_cell():
    guess_missed = L.guess(
        "missed-guess",
        question="You add 1000 simulated runs to the log, labeled with a rule for heat failures, and train k-NN with k = 5 on the log and the simulated runs together. How many of the 115 real heat failures in the next 2000 runs does it miss?",
        options={
            "all": "Nearly all 115: made-up runs teach nothing about the real machine",
            "right": "About 10",
            "half": "About half",
        },
        reveal="**About 10.** With the lesson's simulated runs, k-NN misses 10 of the 115 and raises 9 false alarms; with other simulated runs, drawn at random, it misses 2 to 16. With the log alone it misses all 115: no run in it ever failed by heat.",
    )
    guess_missed
    return (guess_missed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Three ways to make data.** Change examples you have: a photo of a cat, turned a little or cut smaller, is still a cat; a later lesson does that with images. Imitate the process that makes the data: a **simulation** is a program that imitates how the data comes about, so that it can make examples, also of cases you never measured. Or let a model write examples and their labels: a language model can write questions and answers, as the language domain shows. What you make this way is **synthetic data**. Today you simulate: a run of the machine is a temperature difference and a speed.

        **A label from a rule.** A simulated run needs a label, and it gets one from a rule: the one AI4I's maker wrote down for a heat failure. When the difference between process and air temperature is below 8.6 K and the tool turns slower than 1380 rpm, the heat cannot get away, and the machine stops. A label that a rule gives is a **synthetic label**. It is only as right as the rule.

        **Like the real data where it matters.** Simulated runs only help if they look like real ones on the points that matter. Here they take their speeds from the log: speeds the machine really runs at. Only the temperature difference goes where the log never went: between 7.5 and 9.5 K, while the log stays above 8.6 K.

        **Test on real data only.** A test set of simulated runs gets its labels from the same rule. It checks whether the model learned the rule, not whether the rule is right. Only real runs, used neither to train nor to simulate, say how the model does on the machine.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **synthetic data** | data a program made, instead of data collected from the world | `_sim_diff` |
        | **simulation** | a program that imitates the process that makes the data | |
        | **synthetic label** | a label that a rule or a program gives an example, instead of the real outcome | `_sim_heat` |
        | `start + width * rng.random(n)` | `n` random numbers from `start` to `start + width`: `rng.random(n)`, short for `rng.random(size=n)`, gives numbers between 0 and 1, multiplying stretches them to the width, and adding `start` moves them up | `7.5 + 2 * _rng.random(1000)` |
        | an `if` inside an `if` | the inner `if` only runs when the outer condition holds, so both must be true | `if _sim_diff[_i] < 8.6:` and below it `if _sim_speed[_i] < 1380:` |
        """),
        L.watch_out("The rule works so well here because it is the rule that made AI4I. A rule from a real engineer is a guess, close or a little off, and a synthetic label is never more right than its rule."),
        mo.md(r"""
        The slider sets the speed limit of the rule: the 1000 simulated runs get their labels from it, and k-NN trains on them and the log. Left, the heat failures k-NN misses and the false alarms it raises on the next 2000 real runs; right, the same on 1000 other simulated runs, labeled by the same rule. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_limit_cell():
    def _frame(limit):
        said = C.said(limit, L.lang)
        alt = ("Left: the heat failures k-NN misses and its false alarms on the next 2000 real runs, for every speed limit "
               "of the rule. Right: the same on 1000 other simulated runs, labeled by the same rule. " + said)
        return (C.picture(limit, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_limit = L.explore("limit", "speed limit (rpm)", _frame, start=1380, values=C.LIMITS, role="hyper",
                              task="Move the slider from 1260 to 1500. At which limit does k-NN do well on the real runs, and at which on the simulated ones?")
    explore_limit
    return


@app.cell(hide_code=True)
def text_after_limit():
    mo.md(r"""
    On the simulated test runs, k-NN makes 5 to 15 mistakes at every limit: it learned the rule, whatever the rule says. On the real runs only 1380 rpm works, the limit that made AI4I: 10 missed, 9 false alarms. At 40 rpm lower, the simulation makes too few failures, and k-NN misses 76 of the 115; at 40 rpm higher, it calls runs that went fine failures, with 83 false alarms. A test on simulated runs would have called every one of these rules good.

    Why not use the rule itself? Here you could: with two features and one kind of failure, the rule does the job alone. Simulated runs earn their place when a model also has to learn what the log shows and no rule you know describes. On a real machine that is most of what happens; in AI4I even that came from rules, so this lesson keeps to one kind of failure, to show what the simulation does.

    ### Recipe

    When your data has too few examples of what matters:

    1. Write down what you know about how that data comes about: here the rule for a heat failure.
    2. Draw runs at random where the machine can be, also where your log has none, and take from the log what it measured well, such as the speeds.
    3. Label every simulated run with the rule: synthetic labels.
    4. Train on your real data and the simulated runs together.
    5. Test on real data only, data that served neither for training nor for the simulation.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads AI4I, takes the first 3000 runs as the log and the next 2000 as the test, and trains k-NN with k = 5 twice: on the log alone, and on the log with 1000 simulated runs. Each time it prints how many of the 115 heat failures it misses and how many false alarms it raises. Run the cell.
    """)
    return


@app.cell
def example_simulate():
    import numpy as np
    import pandas as pd

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _temps = np.array(_df[["Process temperature [K]", "Air temperature [K]"]])
    _diff = _temps[:, 0] - _temps[:, 1]              # process minus air temperature, in K
    _speed = np.array(_df["Rotational speed [rpm]"])
    _heat = np.array(_df["HDF"])                     # 1 for a heat failure


    def _runs(diff, speed):
        # one row for every run: its temperature difference and its speed
        runs = np.zeros((len(diff), 2))
        runs[:, 0] = diff
        runs[:, 1] = speed
        return runs


    def _knn(train, labels, runs, k):
        # k-NN on standardized values: 1 when most of the k nearest failed
        mean, std = train.mean(axis=0), train.std(axis=0)
        out = []
        for x in runs:
            dists = np.linalg.norm((train - mean) / std - (x - mean) / std, axis=1)
            nearest = np.argsort(dists, kind="stable")[:k]
            out.append(int(np.mean(labels[nearest]) > 0.5))
        return np.array(out)


    _log = _runs(_diff[:3000], _speed[:3000])
    _test, _real = _runs(_diff[3000:5000], _speed[3000:5000]), _heat[3000:5000]

    # 1. the log alone: 3000 runs, not one heat failure
    _pred = _knn(_log, _heat[:3000], _test, 5)
    print("log alone:", np.sum(_real[_pred == 0]), "missed,", np.sum(_pred[_real == 0]), "false alarms")

    # 2. simulate 1000 runs: a temperature difference between 7.5 and 9.5 K, a speed from the log
    _rng = np.random.default_rng(4)
    _sim_diff = 7.5 + 2 * _rng.random(1000)
    _sim_speed = _speed[:3000][_rng.integers(0, 3000, 1000)]

    # 3. label them with the rule: synthetic labels
    _sim_heat = np.zeros(1000)
    for _i in range(1000):
        if _sim_diff[_i] < 8.6:
            if _sim_speed[_i] < 1380:
                _sim_heat[_i] = 1
    print("simulated heat failures:", int(np.sum(_sim_heat)))

    # 4. train on the log and the simulated runs together, test on the next 2000 real runs
    _train = np.concatenate([_log, _runs(_sim_diff, _sim_speed)])
    _labels = np.concatenate([_heat[:3000], _sim_heat])
    _pred = _knn(_train, _labels, _test, 5)
    print("log and simulation:", np.sum(_real[_pred == 0]), "missed,", np.sum(_pred[_real == 0]), "false alarms")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The log alone misses all 115: not one of its runs failed by heat, so no neighbor ever votes for a failure. With 1000 simulated runs, 76 of which fail by the rule, k-NN misses 10 and raises 9 false alarms. All 19 mistakes lie at the edge of the rule, with a temperature difference near 8.6 K or a speed near 1380 rpm, where runs that went fine and runs that failed are neighbors.

    Why from 7.5 K? It is a guess at how far a warm day can push the difference down. The exact lower end matters less than the rule: starting from 6.5 or 8.0 K instead, k-NN misses 2 to 26 of the 115 over twenty random simulations, against 2 to 16 from 7.5 K.

    `_speed[:3000][_rng.integers(0, 3000, 1000)]` picks 1000 speeds from the log, at random positions; the same speed can be picked twice.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_power():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A rule for power failures

        AI4I's maker wrote a rule for every kind of failure. A power failure: the power the tool needs, torque times speed, is below 3500 W or above 9000 W. `power_failure(torque, speed)` already computes the power in watts. Replace the `...` with the rule: 1 for a power failure, and 0 otherwise. The check calls `power_failure(20.0, 1500)`, which should give 1, and seven more; under it, your rule labels all 10,000 runs of AI4I.
        """),
        L.task("Let `power_failure` give 1 below 3500 W or above 9000 W, and 0 otherwise."),
    ])
    return


@app.cell(hide_code=True)
def exercise_power():
    box_power = L.code_box("power", lines=10)
    box_power
    return (box_power,)


@app.cell(hide_code=True)
def check_power(box_power):
    result_power = L.check("power", box_power.value)
    result_power
    return (result_power,)


@app.cell(hide_code=True)
def hints_power():
    L.hints(
        "power",
        [
            "There are two ways to fail. What does your function give back in each, and what at the end, when neither holds?",
            "An `if` with a `return` in it stops the function at once. After two such `if`s, only the runs without a power failure are left.",
            "`if power < ___:` with `return 1` below it, the same for `power > ___`, and at the end `return ___`.",
        ],
        why="For the example, the power is 20.0 × 1500 × 2π / 60 ≈ 3142 W, below 3500 W: the first `if` gives 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_between():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Numbers between two limits

        `between(rng, low, high, n)` should give `n` random numbers between `low` and `high`, from `rng.random(n)`, the way the worked example draws its temperature differences between 7.5 and 9.5 K. Replace the `...`. The check calls `between(np.random.default_rng(1), 7.5, 9.5, 3)` and five more, each time with a new generator, so that it can compare your numbers with its own.
        """),
        L.task("Let `between` give `n` numbers between `low` and `high`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_between():
    box_between = L.code_box("between", lines=5)
    box_between
    return (box_between,)


@app.cell(hide_code=True)
def check_between(box_between):
    result_between = L.check("between", box_between.value)
    result_between
    return (result_between,)


@app.cell(hide_code=True)
def hints_between():
    L.hints(
        "between",
        [
            "`rng.random(n)` gives `n` numbers between 0 and 1. How wide must they become, and where must they start?",
            "Multiplying stretches them, to between 0 and `high - low`. Then move them up, so that they start at `low`.",
            "`return ___ + (___ - ___) * rng.random(n)`.",
        ],
        why="For `low` 7.5 and `high` 9.5 the width is 2: `rng.random(n)` times 2 lies between 0 and 2, and 7.5 plus that between 7.5 and 9.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test_on():
    mo.md(r"""
    ### Exercise 3. Which test do you trust?
    """)
    return


@app.cell(hide_code=True)
def choice_test_on_cell():
    choice_test_on = L.choice(
        "test-on",
        question="A colleague's rule says a heat failure happens below 1460 rpm. Trained on the log and 1000 runs simulated with that rule, k-NN misses 8 of the 207 failures in 1000 other simulated runs, labeled the same way, and raises 3 false alarms. What does that say about the real machine?",
        options={
            "good": "The model will do well: 11 mistakes in 1000 runs",
            "right": "Little: the test labels come from the same rule, so the test checks the rule against itself",
            "strict": "It will miss many real failures: 1460 rpm is too strict",
        },
        correct="right",
        why={
            "good": "The simulated test runs got their labels from the colleague's rule. On the next 2000 real runs, the same model raises 176 false alarms.",
            "strict": "1460 rpm is wider than AI4I's 1380, not stricter: more runs count as failures. On the real runs, the model misses 6 and raises 176 false alarms.",
            "right": "Right: the model learned the rule, and the test grades it by that same rule. On the next 2000 real runs it misses 6 of the 115 and raises 176 false alarms: the rule is too wide.",
        },
    )
    choice_test_on
    return (choice_test_on,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Fix the data, not the model](/course/open/data-centric-ai?lang=en) made the data better by fixing labels; this lesson makes it better by adding the examples that were missing. k-NN on standardized features comes from [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), and a test set that took part in nothing else from [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en). Changing the examples you have comes back with images, and a model that writes its own examples in the language domain.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_on, result_between, result_power):
    L.wrap_up(result_power, result_between, choice_test_on)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
