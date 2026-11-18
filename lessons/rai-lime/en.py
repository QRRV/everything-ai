# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-lime. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-lime", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fidelity():
    quick_fidelity = L.choice(
        "quick-fidelity",
        question="A tree of depth 3 is grown on the forest's answers. What does its fidelity on the test half measure?",
        options={
            "right": "How often it says what the forest says",
            "accuracy": "How often it is right about who earns more",
            "leaves": "How many of its leaves say yes",
        },
        correct="right",
        why={
            "right": "Right: fidelity compares the copy with the model it copies, not with the truth, on rows it did not learn from. In this lesson a copy is made again, but it only has to agree with the forest close to one person.",
            "accuracy": "That is accuracy, which compares with the truth. Fidelity compares the copy with the forest: how often they give the same answer. See [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en).",
            "leaves": "Fidelity is about answers, not about the tree's shape: the share of rows on which the copy says what the forest says. See [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en).",
        },
    )
    quick_fidelity
    return (quick_fidelity,)


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="The copy of depth 3 agrees with the forest on 98.8% of the people with education up to 12, and on 86.3% of those with 13 or more. You must explain the forest's answer for one person with education 13. How much does the copy help?",
        options={
            "fully": "Fully: it agrees on 95.7% overall",
            "right": "Less: it misses one in seven like them",
            "none": "Not at all: a copy says nothing about anyone",
        },
        correct="right",
        why={
            "right": "Right: for people with education 13 or more, the copy gives another answer than the forest for 13.7%, about one in seven. A copy for everyone can be off just where your person is. This lesson copies the forest only close to that one person.",
            "fully": "95.7% is over everyone. For people with education 13 or more the copy agrees on 86.3%, so it is off for about one in seven like this person. See [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en).",
            "none": "For people with education 13 or more the copy still agrees with the forest on 86.3%: it helps, but it is off for about one in seven like this person. See [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en).",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The bank's supervisor now has a complaint on the desk. A person aged 41, with a bachelor's degree (education_num 13), who works 40 hours a week, got a no: the forest gives a chance of 0.475 that they earn more than 50,000 dollars, just under one half. The person asks which of the three features weighed most. The copy of the last lesson summarizes the forest for everyone, and for people with a degree it is off for about one in seven. So copy the forest again, but only close to this person: make many rows, ask the forest about each, and fit a straight line in which rows near the person count most. That is LIME, and its slopes say how the forest's chance moves near this one person. How near "near" is turns out to be a choice, and the answer changes with it.

    **An everyday example.** To find out what a pot of soup needs, you taste spoonfuls with small changes: a pinch more salt, a drop of lemon, a little water. What you learn holds for this pot, not for every soup. And the size of your changes matters: with a pinch you learn about this pot; with a cup of salt you learn only that a cup of salt is too much.
    """)
    return


@app.cell(hide_code=True)
def predict_order():
    guess_order = L.guess(
        "order",
        question="With lime's own setting for how near rows must be, the kernel width 1.3, education_num weighs most for this person. With a width of 0.2, only rows very close to the person count. Which feature weighs most then?",
        options={
            "age": "age",
            "education": "education_num, as before",
            "hours": "hours_per_week",
        },
        reveal="**hours_per_week, in 9 of 10 runs.** Very close to the person, the forest's chance jumps with the hours: from 0.394 at 39 hours to 0.610 at 42. With width 1.3, education_num weighs most in all 10 runs: 0.144 on average, against 0.108 for hours_per_week and 0.107 for age. The forest is the same; only how near \"near\" is changed.",
    )
    guess_order
    return (guess_order,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One prediction, one small model.** Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin (2016) called their method LIME, for Local Interpretable Model-agnostic Explanations: "The overall goal of LIME is to identify an interpretable model over the interpretable representation that is locally faithful to the classifier." Locally faithful is fidelity, as in the last lesson, but only close to one example: the small model is a local surrogate. They weigh two things against each other: how unfaithful the small model is near the example, where rows count more the closer they are, and how complex it is. In their version the small model is a straight line over a few features. They warn that "local fidelity does not imply global fidelity", and that "if the underlying model is highly non-linear even in the locality of the prediction, there may not be a faithful explanation."

        **What it found.** Ribeiro et al. trained a classifier on 20 images, "hand selected such that all pictures of wolves had snow in the background, while pictures of huskies did not." Of 27 graduate students with at least one graduate course in machine learning, 10 trusted it before seeing explanations, and 12 named snow as a feature it might use; after LIME's explanations, 3 trusted it and 25 named snow. The authors add: "Although our sample size is small". In another test, a classifier that sorted texts into Christianity and atheism had 94% held-out accuracy, yet "The word 'Posting' appears in 22% of examples in the training set, 99% of them in the class 'Atheism'." On a new set of web pages it scored 57.3%, and a cleaned version 69.0%, although on the old test split the cleaned one scored less, 88.6% against 94.0%.

        **How near is near?** Each made-up row counts by its closeness to the person, $\exp(-d^2 / (2w^2))$: $d$ is the row's distance to the person, with each column in standard deviations as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en), and $w$ is the kernel width. A row at the person counts 1; a row far away, almost 0. The lime package takes 0.75 times the square root of the number of columns, 1.3 for three. (The paper writes $\exp(-d^2/w^2)$; the package takes its square root, which is the formula above.) There is no good rule for it. Christoph Molnar: "The choice of neighborhood is an unsolved problem when using LIME with tabular data." In their analysis of LIME on tables, for a model that is itself a straight line, Damien Garreau and Ulrike von Luxburg show that with a very large width "all the examples receive positive weights", that with too small a width the line "would just be a constant fit, missing all the relevant information", and that "poor choices of parameters can lead LIME to miss important features."

        **Where the rows come from.** The paper makes its rows by keeping a random part of the example, such as some of the words of a text. For a table, the package decides. By default it cuts every column into four bins at its quartiles, the values a quarter, half and three quarters of the way through the sorted training rows, and draws each column from the training data; a weight then means "in the same bin as this person". Without bins (`discretize_continuous=False`) it draws each column at random around the column's mean, most values near it and few far away, with the column's own spread: a normal distribution. With either setting, unless you pass `sample_around_instance=True`, the rows are not drawn around the person; Molnar: "LIME samples are not taken around the instance of interest, but from the training data's mass center, which is problematic." The person comes in through the closeness. This lesson's LIME does what the package does without bins, except that it fits a plain straight line where the package fits ridge regression, a straight line that keeps its weights smaller. At lime's width 1.3, where the rows count as much as 1965 rows at the person, the two give the same weights here, to three decimals. At small widths ridge pulls the weights toward 0: at width 0.1 it gives 0.004, 0.004 and 0.013, close to the constant fit Garreau and von Luxburg describe.

        **Run it again.** Ribeiro et al. call their method "fairly robust to sampling noise since the samples are weighted". Others disagree. Molnar: "in my experience, if you repeat the sampling process, then the explanations that come out can be different." Giorgio Visani and four colleagues ran LIME 10 times for one credit applicant, with a model on more columns than here; their index for whether the same 7 most important features came back was 89.44% with one setting (kernel width 3, ridge penalty 1), 14.17% with another (width 1.3, penalty 0.001). And David Alvarez-Melis and Tommi Jaakkola found that on a made-up dataset with two columns, for a small neural network, LIME and SHAP, the method of the next lesson, "yield explanations that vary considerably for some neighboring inputs".

        **Fooled on purpose.** LIME's rows need not look like real people: in the worked example, 324 of the 5000 lie outside what the table holds, such as people younger than 17. Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh and Himabindu Lakkaraju (2020) used that. Their model decides on race alone for real people, and on a column that means nothing for rows that look made up; a random forest tells the two apart. On data about defendants (COMPAS), LIME then put the meaningless column on top instead of race "in 100% of held-out points". Kernel widths from 0.15 to 0.95 times the root of the number of columns did not help: "the attacks remain equally effective". Like this lesson, they used lime without bins. Molnar: "LIME explanations can be manipulated by the data scientist to hide biases".

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **LIME** | a method that explains one prediction: make many rows, ask the model about them, and fit a straight line in which rows close to the example count most; its weights are the explanation |
        | **local surrogate** | a model you can read that copies a black box only close to one example |
        | **kernel width** | how far from the example rows still count: small, and only very close rows count; large, and all rows count almost alike |
        | $\exp(-d^2 / (2w^2))$ | the closeness of a row at distance $d$ with kernel width $w$: 1 at the example, near 0 far away |
        | `predict_proba` | in scikit-learn: for each row, the chance of each class; column 1 is the chance of yes |
        | `sample_weight` | in scikit-learn's `fit`: how much each row counts; a row with 0.5 counts half |
        """),
        L.watch_out("A LIME weight is a slope of a straight line near one person, made from random rows and a kernel width you chose. It is not the forest's reason, it changes with the width and between runs, and a model can be built to hide what it does from it. Run it several times and with several widths, and trust only what stays."),
    ])
    return


@app.cell(hide_code=True)
def explore_width_cell():
    explore_width = L.explore("width", "kernel width", lambda v: C.width_picture(v, "en"), start=1.3,
                              values=C.WIDTHS, role="hyper",
                              task="Move the slider from width 0.1 to 10. At which widths does hours_per_week weigh most, and at which do the 10 runs disagree with each other?")
    mo.vstack([
        mo.md("**LIME's weights for one person, by kernel width**"),
        explore_width,
        mo.md(r"""
        The course's LIME, made as in the worked example below, for the person aged 41 with education_num 13 who works 40 hours: 10 runs, each with its own 5000 random rows. Each dot is the weight one run gives a column; the short line is their average. A weight is how much the line's chance of yes rises for one standard deviation more of that column: 13.7 years of age, 2.6 levels of education, 12.5 hours a week.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To explain one prediction with LIME:

    1. Take the row to explain, and the model's answer for it.
    2. Make many rows the model can answer, and ask the model about each of them.
    3. Give each row a closeness, from its distance to the row you explain and a kernel width.
    4. Fit a straight line through the rows, each counted by its closeness: its weights are the explanation.
    5. Read a weight as how much the answer moves near this row, per standard deviation of its column.
    6. Run it again with other random rows and other widths, and trust only what stays.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forest of the last lessons, and the person at position 5 of the test half. `.predict_proba(rows)[:, 1]` gives the forest's chance of yes for each row. `_mean` and `_std` hold the mean and the standard deviation of each column of the training half: `.mean(axis=0)` and `.std(axis=0)` work column by column. `_lime(seed, width, person)` is one run for one person. The random generator's `.normal(size=(5000, 3))` gives 5000 rows of 3 numbers, each drawn around 0 with spread 1, so `_mean + z * _std` are rows around the mean of each column; `z` is the same rows in standard deviations. `(person - _mean) / _std` is the person in standard deviations too, and `np.linalg.norm(..., axis=1)` gives the length of each row, here its distance to the person. `LinearRegression().fit(z, chance, sample_weight=closeness)` fits the straight line, each row counted by its closeness, and `.coef_` holds its three weights. Run the cell. It takes a few seconds.
    """)
    return


@app.cell
def example_lime():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression

    # 1. the forest of the last lessons, and the person to explain
    _people = pd.read_csv("data/adult.csv.gz")
    _columns = ["age", "education_num", "hours_per_week"]
    _X = _people[_columns].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _person = _X[_test][5]
    print("person:", _person, " chance of yes:", _forest.predict_proba([_person])[0, 1].round(3))
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)

    def _lime(seed, width, person):
        # 2. 5000 made-up rows, each column drawn around its mean, and the forest's chance of yes for each
        z = np.random.default_rng(seed).normal(size=(5000, 3))
        chance = _forest.predict_proba(_mean + z * _std)[:, 1]
        # 3. the person in standard deviations, each row's distance to them, and how much the row counts
        d = np.linalg.norm(z - (person - _mean) / _std, axis=1)
        closeness = np.exp(-d ** 2 / (2 * width ** 2))
        # 4. a straight line through the rows, each counted by its closeness: its weights explain
        line = LinearRegression().fit(z, chance, sample_weight=closeness)
        return line.coef_, closeness.sum()

    # 5. lime's own width for three columns
    _weights, _total = _lime(0, 0.75 * 3 ** 0.5, _person)
    print("width 1.3: the rows count as much as", _total.round(1), "rows at the person")
    for _j in range(3):
        print("  weight of", _columns[_j], _weights[_j].round(3))

    # 6. other random rows and widths, and how many rows at the person they count as
    for _width in [0.1, 0.2, 0.75 * 3 ** 0.5, 10]:
        for _seed in range(3):
            _weights, _total = _lime(_seed, _width, _person)
            print("width", round(_width, 1), " seed", _seed, " weights", _weights.round(3),
                  " rows", _total.round(1))

    # 7. a very wide kernel for another person: the first of the test half
    _other = _X[_test][0]
    print("another person:", _other, " width 10, seed 0, weights", _lime(0, 10, _other)[0].round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With lime's width, the 5000 rows count as much as 1965.1 rows at the person, and education_num weighs most: 0.144, so near this person the line's chance of yes rises by 0.144 for 2.6 levels more education. hours_per_week and age come next with 0.106 and 0.105, so close that in 4 of the 10 runs of the slider age comes before hours_per_week. With width 0.2 hours_per_week weighs most in every one of the three runs here, and with width 0.1 the rows count as much as only about 2 rows at the person, and the weight of education_num jumps from 0.081 to 0.199 to 0.242 between runs. With width 10 nearly every row counts fully: the weights are 0.082, 0.111 and 0.073, and in the first run they are the same, to three decimals, for another person: a 47-year-old with education_num 10 who works 45 hours. Then LIME no longer explains this person, but fits one line for everyone.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_closeness():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How much does a row count?

        Write `closeness(distances, width)`: `distances` is a list of distances to the person, in standard deviations, and the function returns a list with the closeness of each distance `d`, $\exp(-d^2 / (2w^2))$ with $w$ the `width`. `math.exp(x)` is $e^x$, and `d ** 2` is $d^2$. The check first calls `closeness([0, 1, 2], 1)`, then other lists. The picture draws your function for three widths, and adds it up over the 5000 rows of the worked example.
        """),
        L.task("Write `closeness(distances, width)`: for each distance `d` in `distances`, append its closeness to a list `out`, and return `out`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_closeness():
    box_closeness = L.code_box("closeness", lines=10)
    box_closeness
    return (box_closeness,)


@app.cell(hide_code=True)
def check_closeness(box_closeness):
    result_closeness = L.check("closeness", box_closeness.value)
    result_closeness
    return (result_closeness,)


@app.cell(hide_code=True)
def hints_closeness():
    L.hints(
        "closeness",
        [
            "A row at the person, at distance 0, counts 1; the farther away, the less. Work out the formula for each distance, and collect the results in a list.",
            "Start with `out = []` and go over the list with `for d in distances:`. For each `d`, append `math.exp(...)` with the formula inside; after the loop, return the list.",
            "`out.append(math.exp(-d ** 2 / (2 * ___)))`",
        ],
        why="`out = []`, then `for d in distances:` with `out.append(math.exp(-d ** 2 / (2 * width ** 2)))`. After the loop, `return out`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_slope():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The slope near the person

        LIME's line has a slope for each column. With one column you can work it out by hand. With $c_i$ the closeness of row $i$, and $\bar x$ and $\bar y$ the means of `xs` and `ys` in which each row counts by its closeness:

        $$\text{slope} = \frac{\sum_i c_i\,(x_i - \bar x)(y_i - \bar y)}{\sum_i c_i\,(x_i - \bar x)^2}$$

        The box already works out the two means, `mean_x` and `mean_y`. Finish `local_slope(xs, ys, closeness)`: sum the top of the fraction in `up` and the bottom in `down`, going over the rows with `i`, and return `up / down`. The check first calls `local_slope([1, 2, 3, 4], [1, 3, 2, 4], [1, 1, 1, 0])`, where the last row counts 0. The picture runs your function on 5000 rows for hours_per_week alone, with the person's age and education kept: for widths 0.3 and 1.3, and with every row counting fully.
        """),
        L.task("Replace the `...`: start `up` and `down` at 0, in a loop over `i` add each row's term of the top to `up` and its term of the bottom to `down`, and return `up / down`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slope():
    box_slope = L.code_box("slope", lines=15)
    box_slope
    return (box_slope,)


@app.cell(hide_code=True)
def check_slope(box_slope):
    result_slope = L.check("slope", box_slope.value)
    result_slope
    return (result_slope,)


@app.cell(hide_code=True)
def hints_slope():
    L.hints(
        "slope",
        [
            "The top multiplies, for each row, its closeness by how far its x is from `mean_x` and how far its y is from `mean_y`; the bottom uses how far its x is, squared. Sum each over all rows.",
            "Start `up = 0` and `down = 0`, then `for i in range(len(xs)):`. In the loop, add the row's term of the top to `up` and its term of the bottom to `down`; after the loop, return `up / down`.",
            "`up = up + closeness[i] * (xs[i] - mean_x) * ___`",
        ],
        why="`up = 0` and `down = 0`, then `for i in range(len(xs)):` with `up = up + closeness[i] * (xs[i] - mean_x) * (ys[i] - mean_y)` and `down = down + closeness[i] * (xs[i] - mean_x) ** 2`. After the loop, `return up / down`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_wide():
    mo.md(r"""
    ### Exercise 3. A very wide kernel
    """)
    return


@app.cell(hide_code=True)
def choice_wide_cell():
    choice_wide = L.choice(
        "wide",
        question="With kernel width 10, nearly every one of the 5000 rows counts fully. What does LIME then say about the person aged 41 of the worked example?",
        options={
            "right": "Nearly what it says for anyone else",
            "exact": "The forest's own reasons for this person",
            "none": "Nothing: no row is close enough",
        },
        correct="right",
        why={
            "right": "Right. With width 10 the closeness hardly depends on the distance, so the line is fitted to all rows alike: in the first run of the worked example, the weights are 0.082, 0.111 and 0.073, both for this person and for a 47-year-old with education_num 10 who works 45 hours. Garreau and von Luxburg: with a very large width \"all the examples receive positive weights\". The explanation is then a global one, one line for everyone.",
            "exact": "LIME's weights are slopes of a straight line fitted to the forest's answers, never the forest's own reasons. And with width 10 that line is fitted to all rows alike, so it is nearly the same for anyone. See *How near is near?* in *The idea*.",
            "none": "A wider kernel makes rows count more, not less: with width 10 a row 3 standard deviations away still counts 0.956. The line is then fitted to all rows alike, nearly the same for anyone. See *How near is near?* in *The idea*.",
        },
    )
    choice_wide
    return (choice_wide,)


@app.cell(hide_code=True)
def text_exercise_fooled():
    mo.md(r"""
    ### Exercise 4. A model that hides
    """)
    return


@app.cell(hide_code=True)
def choice_fooled_cell():
    choice_fooled = L.choice(
        "fooled",
        question="A model decides on race alone for real applicants, and on a column that means nothing for rows that do not look real. LIME explains one real applicant. What does it show?",
        options={
            "race": "Race, because the model decides on it",
            "right": "The column that means nothing",
            "both": "Both, each with about half the weight",
        },
        correct="right",
        why={
            "right": "Right. LIME asks the model about made-up rows, and those need not look like real people: in the worked example, 324 of the 5000 lie outside what the table holds. So the model answers LIME with the column that means nothing, and LIME reports that column. In the model of Slack et al., that happened for 100% of the people they explained in COMPAS.",
            "race": "The model uses race only for rows that look real. LIME's rows are made up, and the model can tell: it answers them with the other column, so that is the column LIME sees. See *Fooled on purpose* in *The idea*.",
            "both": "LIME only sees the model's answers on its own made-up rows, and on most of those the model does not use race. See *Fooled on purpose* in *The idea*.",
        },
    )
    choice_fooled
    return (choice_fooled,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en) copied the forest for everyone; LIME copies it close to one person, and is a local explanation in the terms of [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en). Its distance is the one of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), measured in standard deviations as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en), and its line is the linear regression of [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en), with some rows counting more than others. The next lesson, [Shapley values](/course/open/rai-shapley-values?lang=en), also explains one prediction, but shares it out over the features with a rule that does not depend on a kernel width.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fooled, choice_wide, result_closeness, result_slope):
    L.wrap_up(result_closeness, result_slope, choice_wide, choice_fooled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
