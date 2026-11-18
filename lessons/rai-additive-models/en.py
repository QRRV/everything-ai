# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-additive-models. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-additive-models", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One short question about [Glass-box models](/course/open/rai-glass-box-models?lang=en) and one about [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_odds():
    quick_odds = L.choice(
        "quick-odds",
        question="In a logistic regression, education has the weight 0.345. What does one more level of education do to the odds of a yes, with the other features fixed?",
        options={
            "add": "It adds 0.345 to the odds",
            "right": "It multiplies them by $e^{0.345}$",
            "chance": "It multiplies the probability by 1.412",
        },
        correct="right",
        why={
            "add": "The 0.345 is added to the logit, the log of the odds. The odds themselves are multiplied by $e^{0.345}$, about 1.412. See [Glass-box models](/course/open/rai-glass-box-models?lang=en).",
            "right": "Right: the weight adds to the logit, so the odds are multiplied by $e^{0.345}$, about 1.412. This lesson reads a whole curve the same way: by what it adds to the score.",
            "chance": "1.412 is the factor on the odds, not on the probability; what it does to a probability depends on where you start. See [Glass-box models](/course/open/rai-glass-box-models?lang=en).",
        },
    )
    quick_odds
    return (quick_odds,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="In gradient boosting, what does each new small tree learn?",
        options={
            "labels": "The labels themselves, all over again",
            "right": "What the trees before it got wrong",
            "half": "A random half of the features each time",
        },
        correct="right",
        why={
            "labels": "Each tree learns what is still wrong: the residuals of the trees before it. See [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en).",
            "right": "Right: each tree learns what the trees before it still get wrong, and a part of what it predicts, the learning rate, is added. In this lesson each tree also gets only one column.",
            "half": "Choosing features at random is what a random forest does. In boosting each tree learns the residuals of the trees before it. See [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Glass-box models](/course/open/rai-glass-box-models?lang=en) the logistic regression gave age a single weight: every year of age multiplies the odds of earning more than 50,000 dollars by 1.047, from 25 to 26 just as from 70 to 71. One weight per feature is one straight line on the log-odds. If what you predict goes up and then down again, a straight line cannot show it, and you would not see that from the weight. A generalized additive model gives each feature its own curve, which may bend, and adds the curves up. Each curve can be drawn, so the model stays a glass box.

    **An everyday example.** Postage for a parcel: one table for its weight, one for its size, one for where it goes. You look up each part and add them up. The weight table may bend, so that the first kilo costs more than each kilo after it, and you can still see what each part costs, because the parts are only added, never mixed.
    """)
    return


@app.cell(hide_code=True)
def predict_seventy():
    guess_seventy = L.guess(
        "seventy",
        question="Of the 50-year-olds in UCI Adult, 43.1 percent earn more than 50,000 dollars. What about the 70-year-olds?",
        options={
            "more": "More, about 55 percent",
            "same": "About the same",
            "fewer": "Fewer, about 19 percent",
        },
        reveal="**Fewer: 18.8 percent.** Of the 25-year-olds it is 6.4 percent. The share rises with age and then falls again. The logistic regression of [Glass-box models](/course/open/rai-glass-box-models?lang=en) cannot show that: its weight for age, 0.046, says that each year multiplies the odds by 1.047, after 50 too. These shares also mix in education and hours; the curves of this lesson hold those fixed.",
    )
    guess_seventy
    return (guess_seventy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One curve per feature.** Trevor Hastie and Robert Tibshirani (1986): "We introduce the class of generalized additive models which replaces the linear form ∑ βjXj by a sum of smooth functions ∑ sj(Xj)." A linear model adds up a weight times each feature. An additive model adds up, for each feature, a curve read at its value, a curve of whatever shape the data asks for. For a yes or no, the sum is a score on the log-odds, as in a logistic regression, and the sigmoid turns it into a chance. Yin Lou, Rich Caruana and Johannes Gehrke (2012) call the curves shape functions: "Since the shape functions can be arbitrarily complex, GAMs are more accurate than simple linear models. But since they do not contain any interactions between features, they can be easily interpreted by users."

        **Learning the curves with boosting.** Lou and colleagues learn the curves the way you know from [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en), with one change: each small tree sees one feature only. "in each iteration of boosting we cycle sequentially through all features." Whatever a tree adds then belongs to the curve of its feature. In their tests, "bagged trees with 2-4 leaves as shape functions in combination with gradient boosting" did best on most datasets; a tree of depth 2 has at most four leaves.

        **Explainable Boosting Machines.** The library InterpretML (Harsha Nori and colleagues, 2019) calls its version an Explainable Boosting Machine, "a fast implementation of the GA2M algorithm" (GA2M adds some pairs of features; see *Watch out* below). It learns the same way: "The boosting procedure is carefully restricted to train on one feature at a time in round-robin fashion using a very low learning rate so that feature order does not matter." And it predicts by adding up: "each function fj acts as a lookup table per feature, and returns a term contribution. These term contributions are simply added up, and passed through the link function g", for a yes or no the sigmoid. Its documentation says: "EBMs are often as accurate as state-of-the-art blackbox models while remaining completely interpretable." Note the "often". On UCI Adult, Lou and colleagues measured an error rate of 16.04 percent for a logistic regression, 14.85 for a random forest and about 14.6 for their boosted additive models.

        **Reading the curves.** Molnar: "The interpretation of smooth effects requires a visual check of the fitted curve." You read a curve by its differences, as you read a weight: in the worked example, going from age 25 to 50 adds 2.17 to the score, so the odds become $e^{2.17}$, about 8.8 times as large, with education and hours fixed. Like a weight, a curve says what the model learned from the data, not what causes what.

        **The asthma surprise.** In the 1990s, a rule-based system learned from data on pneumonia patients that asthma lowers their risk. Rich Caruana and colleagues (2015) tell why: "patients with a history of asthma who presented with pneumonia usually were admitted not only to the hospital but directly to the ICU (Intensive Care Unit)." That care "was so effective that it lowered their risk of dying from pneumonia compared to the general population", so "models trained on the data incorrectly learn that asthma lowers risk, when in fact asthmatics have much higher risk (if not hospitalized)". On one of the datasets of that time a neural network predicted better than a logistic regression, but it was not used: "if the rule-based system had learned that asthma lowers risk, certainly the neural nets had learned it, too", and "the lack of intelligibility made it difficult to know what other problems might also need fixing." In 2015 their additive model with pairs, on data of 14,199 pneumonia patients, "has found the same pattern discovered back then". Because each term can be drawn, they could see it and fix it: "We can 'repair' the model by eliminating this term (effectively setting the weight on this graph to zero), or by using human expertise to redraw the graph so that the risk score for asthma=1 is positive, not negative."

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **generalized additive model** (GAM) | a model that adds up one curve per feature, each read at the feature's value; for a yes or no the sum is a score on the log-odds |
        | **Explainable Boosting Machine** (EBM) | InterpretML's additive model, whose curves are learned by boosting small trees one feature at a time; by default it also adds some pairs of features |
        | $f_j(x_j)$ | what the curve of feature $j$ adds to the score at the value $x_j$ |
        """),
        L.watch_out("An EBM is not only one curve per feature: \"EBMs include pairwise interactions by default.\" Lou and colleagues (2013) call such a model GA2M, \"Generalized Additive Models plus Interactions\", with \"a small number of pairwise interaction terms\"; a pair is a table over two features, which you can still draw. And when features go together, such as a person's age and years of work experience, the curves are harder to read: Hastie and Tibshirani warn that this may be even worse for the curves of an additive model than for the weights of a linear one."),
    ])
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    explore_rounds = L.explore("rounds", "rounds", lambda n: C.rounds_picture(n, "en"), start=1, values=C.STOPS,
                               role="hyper",
                               task="Move the number of rounds from 1 to 100. After how many rounds is the model right more often than always saying no, and where does the age curve end up highest?")
    mo.vstack([
        mo.md("**The curves, round after round**"),
        explore_rounds,
        mo.md(r"""
        The model of the worked example, on the 48,842 people of UCI Adult: what each curve adds to the score, on one scale. Each round adds three small trees, one per column. At first the curves are almost flat and the model still says no to everyone; then they take their shape.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To learn an additive model with boosting, for a yes or no:

    1. Start every person's score at the log-odds of the share of yes: `np.log(share / (1 - share))`.
    2. Take the features one at a time. For each: the chance `p = 1 / (1 + np.exp(-score))`, a tree of depth 2 that learns `y - p` from that one column, and `rate` times what it predicts added to the score.
    3. Repeat step 2 for many rounds, and keep every tree with its feature.
    4. The curve of a feature is `rate` times the sum of what its trees predict, at every value of the feature.
    5. To predict for a person: the start, plus each curve at the person's value, then the sigmoid.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The three columns of [Reasons for a decision](/course/open/rai-reason-codes?lang=en), now with a curve each. `np.log` is the natural logarithm, which undoes `np.exp`. `people[[name]]`, with two pairs of brackets, keeps one column as a table, the form a tree learns from; `.reshape(-1, 1)` turns a row of numbers into such a column. For a yes or no, the trees learn `y - p`, the label minus the chance: it says which way and how far the chance is off, and the score moves that way. These differences in chance are small next to the score, so each step adds half of what a tree predicts instead of a tenth. InterpretML takes more care: bagging, and in the settings its paper recommends a learning rate of 0.01 and 5,000 epochs, rounds through all the features. Run the cell: it learns the curves, prints a few points of each, and the chance of the applicant aged 30 of [Reasons for a decision](/course/open/rai-reason-codes?lang=en). It takes a second or two.
    """)
    return


@app.cell
def example_gam():
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _y = (_people["income"] == ">50K").to_numpy()

    # 1. every score starts at the log-odds of the share who earn more
    _start = np.log(_y.mean() / (1 - _y.mean()))
    _score = np.zeros(len(_y)) + _start
    _trees = []
    # 3. step 2, 100 rounds
    for _r in range(100):
        # 2. one column at a time: a small tree learns what is still wrong, half of it is added
        for _name in _names:
            _column = _people[[_name]].to_numpy()
            _p = 1 / (1 + np.exp(-_score))
            _tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(_column, _y - _p)
            _score = _score + 0.5 * _tree.predict(_column)
            _trees.append((_name, _tree))
    print("start", round(_start, 3))
    print("right", round(np.mean((_score >= 0) == _y), 3))

    # 4. the curve of each column, at every value from 0 to its largest
    _curves = []
    for _name in _names:
        _values = np.arange(_people[_name].max() + 1).reshape(-1, 1)
        _curve = np.zeros(len(_values))
        for _tree_name, _tree in _trees:
            if _tree_name == _name:
                _curve = _curve + 0.5 * _tree.predict(_values)
        _curves.append(_curve)
    print("age 25, 50, 70:", _curves[0][25].round(2), _curves[0][50].round(2), _curves[0][70].round(2))
    print("education 9, 13, 16:", _curves[1][9].round(2), _curves[1][13].round(2), _curves[1][16].round(2))
    print("hours 20, 40, 60:", _curves[2][20].round(2), _curves[2][40].round(2), _curves[2][60].round(2))

    # 5. the applicant aged 30 with education level 9, working 40 hours a week
    _s = _start + _curves[0][30] + _curves[1][9] + _curves[2][40]
    print("chance", round(1 / (1 + np.exp(-_s)), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Every score starts at −1.157, the log-odds of the 23.9 percent who earn more. After 100 rounds of three trees, the model is right for 80.0 percent of the people; the logistic regression of [Glass-box models](/course/open/rai-glass-box-models?lang=en) for 78.9 percent, the tree of depth 2 for 78.7, always saying no for 76.1. The age curve is lowest around 20, rises to its highest point at 50, 0.76, and falls after that: 0.51 at 70. The education curve climbs in steps, with big ones at 13, a bachelor's degree, and at 15. The hours curve rises up to 50 hours a week and falls a little after that. For the applicant aged 30 with education level 9 who works 40 hours, the curves add −0.28, −0.56 and −0.15 to the start, and the chance of a yes is 0.104.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chance():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Add up the curves

        Write `chance(person, start, curves)`: the chance of a yes for one person. `person` holds one value per feature, `start` is the score everyone begins with, and `curves` holds one curve per feature, so `curves[i][person[i]]` is what feature `i` adds for this person. The check first calls `chance((2, 1), -1.0, [[0.0, 0.5, 1.5], [-0.5, 0.5]])`: two features with small made-up curves. Then it tries other people, curves and numbers of features; the picture runs your function with the curves of the worked example.
        """),
        L.task("Write `chance(person, start, curves)`: the start plus each curve at the person's value, through the sigmoid."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chance():
    box_chance = L.code_box("chance", lines=10)
    box_chance
    return (box_chance,)


@app.cell(hide_code=True)
def check_chance(box_chance):
    result_chance = L.check("chance", box_chance.value)
    result_chance
    return (result_chance,)


@app.cell(hide_code=True)
def hints_chance():
    L.hints(
        "chance",
        [
            "Start from `start`. For each feature, add what its curve gives at the person's value. Then turn the sum into a chance, as a logistic regression does.",
            "Go over the features with `for i in range(len(person)):`. Feature `i` has the value `person[i]`, and its curve is `curves[i]`: which number in it do you need? After the loop comes the sigmoid, `1 / (1 + np.exp(-score))`.",
            "`score = score + curves[i][___]`",
        ],
        why="`score = start`, then `for i in range(len(person)):` with `score = score + curves[i][person[i]]` inside, and after the loop `return 1 / (1 + np.exp(-score))`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_round():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One round, one column at a time

        The box has everything except three lines inside the loop over the three columns. Fill them in: the chance `p` from the score as it is now, a tree of depth 2 with `random_state=0` that learns `y - p` from this one column, and the new score. The check compares `score` after one round and also tries another learning rate; the picture shows what each of your trees adds.
        """),
        L.task("Replace the three `...`: `p`, a tree on `column` that learns `y - p`, and the new `score`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_round():
    box_round = L.code_box("round", lines=18)
    box_round
    return (box_round,)


@app.cell(hide_code=True)
def check_round(box_round):
    result_round = L.check("round", box_round.value)
    result_round
    return (result_round,)


@app.cell(hide_code=True)
def hints_round():
    L.hints(
        "round",
        [
            "Three steps for each column, as in one round of gradient boosting: what is still wrong, a small tree that learns it, and a part of what the tree predicts added to the score.",
            "The chance comes from the score with the sigmoid. The tree is `DecisionTreeRegressor(max_depth=2, random_state=0)`, fitted with `.fit(column, y - p)`.",
            "`score = score + rate * tree.predict(___)`",
        ],
        why="`p = 1 / (1 + np.exp(-score))`, `tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(column, y - p)` and `score = score + rate * tree.predict(column)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.md(r"""
    ### Exercise 3. Reading a curve
    """)
    return


@app.cell(hide_code=True)
def choice_read_cell():
    choice_read = L.choice(
        "read",
        question="In the worked example, the age curve rises from 25 to its highest point at 50 and falls after that. What does that say?",
        options={
            "cause": "Turning 50 makes a person earn more",
            "right": "With the rest equal, the model scores 50 highest",
            "most": "Most people of 50 in the table earn over 50,000 dollars",
        },
        correct="right",
        why={
            "cause": "A curve is what the model learned from the table, not a cause. People of 50 differ from people of 25 in many ways the three columns do not hold. See *Reading the curves* in *The idea*.",
            "right": "With education and hours the same, the age curve adds the most to the score at 50: 2.17 more than at 25. It says what the model learned from the table, not that age causes income; the next exercise shows how a curve can mislead.",
            "most": "Of the 50-year-olds in the table, 43.1 percent earn more: not most. The curve gives what age adds to the score with education and hours fixed; it is not a share of people. See *Reading the curves* in *The idea*.",
        },
    )
    choice_read
    return (choice_read,)


@app.cell(hide_code=True)
def text_exercise_treated():
    mo.md(r"""
    ### Exercise 4. Another surprise
    """)
    return


@app.cell(hide_code=True)
def choice_treated_cell():
    choice_treated = L.choice(
        "treated",
        question="Caruana and colleagues imagine a second case: \"perhaps pregnant women with pneumonia also receive aggressive treatment that lowers their risk compared to the general population.\" If so, what would a model learn from such data?",
        options={
            "right": "That pregnancy lowers the risk of dying",
            "raises": "That pregnancy raises the risk of dying",
            "nothing": "Nothing: it only learns what doctors tell it",
        },
        correct="right",
        why={
            "right": "Right. The data show who died, after the treatment they got. If pregnant patients were treated more aggressively and died less often, the model learns that pregnancy lowers the risk, as it did with asthma. In an additive model you can see that term, ask a doctor, and remove or redraw it.",
            "raises": "Without that treatment their risk might be higher, but the model only sees who died after the care they got. With aggressive treatment, fewer of them died, so the model learns that pregnancy lowers the risk, as with asthma. See *The asthma surprise* in *The idea*.",
            "nothing": "A model learns from the data, not from what doctors say: whatever goes with fewer deaths in the table looks like lower risk. With aggressive treatment, the model learns that pregnancy lowers the risk, as with asthma. See *The asthma surprise* in *The idea*.",
        },
    )
    choice_treated
    return (choice_treated,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Glass-box models](/course/open/rai-glass-box-models?lang=en) you read a weight by what it adds to the logit; here each feature has a whole curve, and you read it the same way, by its differences. The curves were learned as in [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en), one column at a time. The next lesson, [Many good models](/course/open/rai-rashomon?lang=en), asks why a model you can read is often about as accurate as one you cannot. Later in this module, [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en) draws a curve like these for any model, a black box too, made afterwards instead of read from the model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_read, choice_treated, result_chance, result_round):
    L.wrap_up(result_chance, result_round, choice_read, choice_treated)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
