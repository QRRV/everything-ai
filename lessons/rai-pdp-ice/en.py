# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-pdp-ice. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-pdp-ice", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Which features matter?](/course/open/rai-permutation-importance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_which():
    quick_which = L.choice(
        "quick-which",
        question="Shuffling education_num on the test half costs the forest 4.4 points of accuracy. What does that number tell you?",
        options={
            "up": "That more education means more yeses",
            "right": "How much the forest leans on it, not which way",
            "world": "How much education adds to income in the real world",
        },
        correct="right",
        why={
            "right": "Right. Permutation importance says how much worse the forest gets without the column's information, not whether a higher value pushes the prediction up or down. Molnar: \"Feature importance doesn't tell you how the feature influences the prediction, only how much it affects the loss.\" This lesson shows which way.",
            "up": "4.4 points says how much the forest leans on education, not which way it moves the prediction: that could even differ from person to person. This lesson draws which way. See [Which features matter?](/course/open/rai-permutation-importance?lang=en).",
            "world": "Permutation importance describes one fitted model, not the world: scikit-learn says it shows \"how important this feature is for a particular model\". See [Which features matter?](/course/open/rai-permutation-importance?lang=en).",
        },
    )
    quick_which
    return (quick_which,)


@app.cell(hide_code=True)
def quick_check_copies():
    quick_copies = L.choice(
        "quick-copies",
        question="Two columns hold the same information, such as education_num twice. You shuffle each one on its own. What does permutation importance show?",
        options={
            "copy": "The copy gets all the importance",
            "right": "Each copy looks less important than both are",
            "double": "Each copy gets twice the importance it should",
        },
        correct="right",
        why={
            "right": "Right. Shuffle one copy and the forest still has the other: in the worked example of that lesson, one copy cost 1.9 points, the other 1.4, both together 6.1. Features that go together cause trouble in this lesson too.",
            "copy": "Both copies get a small share: shuffled on its own, one copy cost 1.9 points, the other 1.4, and both together 6.1, because the forest falls back on the copy it still has. See [Which features matter?](/course/open/rai-permutation-importance?lang=en).",
            "double": "Shuffling one copy hurts less, not more: the forest still has the other. One copy cost 1.9 points, the other 1.4, both together 6.1. See [Which features matter?](/course/open/rai-permutation-importance?lang=en).",
        },
    )
    quick_copies
    return (quick_copies,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Which features matter?](/course/open/rai-permutation-importance?lang=en) the supervisor of the bank learned that the forest leans on education and age; as before, a prediction of earning more than 50,000 dollars is the bank's yes. The next question is which way. Does the chance of a yes rise with age, and keep rising? For everyone in the same way? Permutation importance cannot say: it measures how much worse the model gets, not where the prediction goes. This lesson asks the model directly: give every person the same age, see what it says, and do that for every age. The average over all people is a partial dependence plot; one line per person shows who reacts differently.

    **An everyday example.** A baker wonders what 50 cents more for a loaf would do. Ask all customers and average, and you hear: a few loaves fewer a week. Ask them one by one, and most do not care, while a few stop coming altogether. The average answer is the partial dependence; the answers one by one are what this lesson calls ICE curves.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="A tree of depth 2 asks first about education and then about age: at 33.5 for education up to 12.5, at 28.5 above it. Give everyone age 17, then 18, up to 90, and average what the tree says: that average has two steps, at 29 and at 34. How many steps does the line of one person have?",
        options={
            "two": "Two, like the average",
            "one": "One",
            "none": "None: one person's line is flat",
        },
        reveal="**One.** A person with education up to 12.5 only meets the question about 33.5, and steps at 34; a person above 12.5 steps at 29. The average has two steps because it mixes both kinds of people: no single person has a line like the average. Exercise 1 draws them.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Set a feature, average the predictions.** Jerome Friedman introduced partial dependence plots in 2001, to look at a model with many inputs through one or two of them at a time. Pick a feature $S$ and a value $v$. Give every row of the data the value $v$ for $S$, keep the other columns $C$ as they are, ask the model $f$, and average over the $N$ rows:

        $$\mathrm{PD}_S(v) = \frac{1}{N}\sum_{i=1}^{N} f(v,\ x_{C,i})$$

        where $x_{C,i}$ are the other columns of row $i$. Do that for each value on a grid and draw the line: the partial dependence plot, PDP. Friedman wrote that partial dependence functions "can be used to help interpret models produced by any 'black box' prediction method", and advised one or two features at a time. Friedman also gave a condition: the average can be a useful summary as long as the shape of the curve does not depend too strongly on the values of the other features.

        **One line per person.** Alex Goldstein, Adam Kapelner, Justin Bleich and Emil Pitkin (2015) start from that condition. Features interact when what one feature does depends on the value of another, and they write: "In the presence of substantial interaction effects, the partial response relationship can be heterogeneous. Thus, an average curve, such as the PDP, can obfuscate the complexity of the modeled relationship." Their answer is to keep the line of every row: an individual conditional expectation curve, ICE curve. "Note that the PDP curve is the average of the N ICE curves and can thus be viewed as a form of post-processing." Friedman had suggested something like it: compute the curve for a few rows drawn at random, to judge how much it varies.

        **When the average is flat.** Goldstein and colleagues show the danger with made-up data: $Y = 0.2X_1 - 5X_2 + 10X_2 \cdot 1(X_3 \ge 0) + E$, where $1(X_3 \ge 0)$ is 1 when $X_3 \ge 0$ and 0 otherwise, $E$ is noise, and $X_1$, $X_2$, $X_3$ are drawn evenly between −1 and 1. They fit a model to 1000 rows, and its partial dependence on $X_2$ comes out flat. "Clearly X2 is associated with Y; it is simply that the averaging inherent in the PDP shields this discovery from view." For rows with $X_3 \ge 0$, the line of $X_2$ has slope $-5 + 10 = +5$; for the other rows, $-5$. About half the rows fall on each side, and in the average the slopes cancel. You compute it in exercise 2.

        **Pinch the lines together.** When the lines start at different heights, it is hard to see how much each one moves. Goldstein and colleagues then "join or 'pinch' all prediction lines" at one value, by subtracting from each line its value there: a centered ICE plot, c-ICE. Choosing the lowest or the highest value, they found, "results in the most interpretable plots"; Molnar: "Anchoring the curves at the lower end of the feature is a good choice." scikit-learn's `PartialDependenceDisplay.from_estimator` does it with `centered=True`, at the first value of the grid. The y-axis of such a plot is a change, no longer a chance.

        **Rows that do not exist.** To compute the curve at age 17, every person gets age 17, also people with a doctorate. Features are correlated when their values go together, as age and education do; scikit-learn calls the other columns $C$ the complement features, and warns that PDPs and ICE curves "assume that the input features of interest are independent from the complement features, and this assumption is often violated in practice", so that "in the case of correlated features, we will create absurd data points". Molnar's example: at a height of 200 cm, averaging over weight "might include a weight below 50 kg, which is unrealistic for a 2 meter person." Daniel Apley and Jingyu Zhu (2020) write that partial dependence plots "can produce erroneous results if the predictors are strongly correlated, because they require extrapolation of the response at predictor values that are far outside the multivariate envelope of the training data." Where there are no real rows, the model can only guess. So show the data under the curve. Molnar: "Omitting the distribution can be misleading because you might overinterpret regions with almost no data." Goldstein and colleagues mark each line at the row's own value.

        **What it does not tell you.** Molnar: "The relationship is causal for the model – because we explicitly model the outcome as a function of the features – but not necessarily for the real world!" A curve that rises with education says that the forest gives more yeses when you raise education in a row; it does not say that more schooling raises a person's income. And a curve shows one or two features at a time; Molnar again: "The realistic maximum number of features in a partial dependence function that can be meaningfully visualized is two."

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **partial dependence plot (PDP)** | for each value of one feature: give every row that value, ask the model, and average; drawn as a line over the values |
        | **ICE curve** | individual conditional expectation: the same for one row, without averaging; the PDP is the average of the ICE curves |
        | **centered ICE (c-ICE)** | ICE curves minus their value at one point, usually the lowest, so that all of them start at 0 |
        | **grid** | the values of the feature at which the curves are computed |
        | **interaction** | what one feature does depends on the value of another; then the ICE curves do not run side by side |
        | $1(X_3 \ge 0)$ | 1 when $X_3 \ge 0$, and 0 otherwise |
        | `kind` | in scikit-learn's `partial_dependence`: `"average"` for the PDP, `"individual"` for the ICE curves, `"both"` for both |
        """),
        L.watch_out("A partial dependence plot shows what the model does when you change one column of a row, not what happens to a person in the world. And where the curve runs through rows that do not occur, such as 17-year-olds with a doctorate, the model is guessing: look at where the real people are before you read the curve."),
    ])
    return


@app.cell(hide_code=True)
def explore_education_cell():
    explore_education = L.explore("education", "education_num", lambda v: C.level_picture(v, "en"), start=9,
                                  values=C.EDUCATION, role="data",
                                  task="Move the slider from 1 to 16. How much do the thin lines rise from age 25 to 45 for people with little education, and for people with a degree? And the dashed average of everyone?")
    mo.vstack([
        mo.md("**ICE curves for age, by education**"),
        explore_education,
        mo.md(r"""
        The forest of the worked example: 100 trees with leaves of at least 20 people, grown on the training half. Thin lines: what it says for 12 people of the test half with the chosen education_num, when their age is set to each age from 17 to 90. Thick line: their average. Dashed: the average over the 1000 people of the worked example, the partial dependence plot. education_num 9 is a high-school diploma (HS-grad), 13 a bachelor's degree.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To see how a model's prediction moves with one feature:

    1. Choose the feature and the grid: each value it takes, or evenly spaced values between two percentiles.
    2. For each row and each value on the grid, copy the row, set the feature to that value, and ask the model. Each row gives an ICE curve.
    3. Average the curves at each value on the grid: that is the partial dependence.
    4. Draw the curves with their average. Where they fan out, look for the feature that sorts them; pinch them at the lowest value to compare how much each one moves.
    5. Mark where the real rows are, and do not read the curve where the feature's value does not occur together with the rest of the row.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forest of [Which features matter?](/course/open/rai-permutation-importance?lang=en) with leaves of at least 20 people, on the three columns of [Reasons for a decision](/course/open/rai-reason-codes?lang=en), with the same halves. `.astype(float)` turns whole numbers into decimal numbers, which `partial_dependence` asks for. `partial_dependence(model, X, [0], kind="both")` takes column 0, age, as the feature, and gives three things: `grid_values`, the ages on the grid; `average`, the partial dependence at each of them; and `individual`, the ICE curve of each row of `X`. Each comes as a list with one entry per feature, hence the `[0]`. scikit-learn asks the model about every row at every age, so the example takes the first 1000 people of the test half; `PartialDependenceDisplay.from_estimator(model, X, [0], kind="both")`, which draws the plot with matplotlib, draws at most 1000 ICE curves (`subsample=1000`) unless you ask for more. By default the grid holds 100 evenly spaced values between the 5th and the 95th percentile of the feature, or each value the feature takes when there are fewer than 100. `np.isin(a, values)` says for each element of `a` whether it is one of `values`, `np.allclose(a, b)` whether two arrays are equal up to rounding, `_lines[:, _ages == 45]` takes from every line the column at age 45, `[:, 0]` makes that one column a flat array, column 0 of `_lines` is the first age on the grid, 17, and `&` between two conditions holds where both hold. Run the cell. It takes a few seconds.
    """)
    return


@app.cell
def example_pdp():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import partial_dependence

    # 1. three columns of UCI Adult as decimal numbers, in two halves; a forest, leaves of at least 20 people
    _people = pd.read_csv("data/adult.csv.gz")
    _X = _people[["age", "education_num", "hours_per_week"]].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _some = _X[_test][:1000]

    # 2 and 3. each age on the grid for all 1000 people: one line per person, and their average
    _pd = partial_dependence(_forest, _some, [0], kind="both")
    _ages, _average, _lines = _pd["grid_values"][0], _pd["average"][0], _pd["individual"][0]
    print("grid:", _ages[0], "to", _ages[-1], "with", len(_ages), "ages")
    print("average at 20, 30, ..., 90:", _average[np.isin(_ages, [20, 30, 40, 50, 60, 70, 80, 90])].round(3))
    print("the average of the lines:", np.allclose(_average, _lines.mean(axis=0)))

    # 4. how much each line rises from age 25 to 45, by education
    _rise = _lines[:, _ages == 45][:, 0] - _lines[:, _ages == 25][:, 0]
    for _low, _high in [(1, 9), (10, 12), (13, 16)]:
        _them = (_some[:, 1] >= _low) & (_some[:, 1] <= _high)
        print("education", _low, "to", _high, "-", _them.sum(), "people, rise:", _rise[_them].mean().round(3))

    # 5. who is in the average at age 17
    _degree = _some[:, 1] >= 13
    print("with education 13 or more:", _degree.sum(), "of", len(_some))
    print("at age 17, with:", _lines[_degree, 0].mean().round(3))
    print("at age 17, without:", _lines[_some[:, 1] < 13, 0].mean().round(3))
    _young = _X[:, 0] == 17
    print("aged 17 in the table:", _young.sum(), "- with 13 or more:", (_young & (_X[:, 1] >= 13)).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Among these 1000 people, age takes 62 values, from 17 to 90, so the grid holds each of them. The average rises from 0.038 at age 20 to 0.354 at 50, and falls to about 0.28 after 70; it is exactly the average of the 1000 lines. But the lines do not all rise alike. From 25 to 45, a line rises by 0.154 on average for the 452 people with education up to 9, by 0.204 for the 296 with 10 to 12, and by 0.476 for the 252 with 13 or more. The forest has learned that age matters more for people with more education, and the average, which rises by 0.25, hides that. Then the left end. At age 17 the average includes those 252 people with education 13 or more, while not one of the 595 people aged 17 in the whole table has that much. For them the forest says 0.102 at age 17, for the others 0.016: a guess about people who do not occur.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ice():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One person's line

        The box holds `model`, a tree of depth 2 grown on the training half, with the share of yes in each of its four leaves, rounded. Write `ice(row, j, grid)`: one person's ICE curve, a list with what `model` says for `row` with column `j` set to each value of `grid` in turn. `list(row)` makes a copy, so you can change the copy and leave `row` as it is. The check first calls `ice([30, 10, 40], 0, [20, 30, 40, 50])`, then other rows, columns and grids. The picture runs your function for 40 people of the test half, at every age from 17 to 90.
        """),
        L.task("Write `ice(row, j, grid)`: for each value `v` in `grid`, what `model` says for a copy of `row` with column `j` set to `v`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ice():
    box_ice = L.code_box("ice", lines=20)
    box_ice
    return (box_ice,)


@app.cell(hide_code=True)
def check_ice(box_ice):
    result_ice = L.check("ice", box_ice.value)
    result_ice
    return (result_ice,)


@app.cell(hide_code=True)
def hints_ice():
    L.hints(
        "ice",
        [
            "Make an empty list for the line. Go through the values of `grid` one by one: for each, make a copy of the row in which column `j` has that value, and add what `model` says for the copy.",
            "Start with `line = []` and `for v in grid:`. Inside the loop, make `changed = list(row)`, put `v` into column `j` of `changed`, and add `model(changed)` to `line` with `append`. After the loop, return `line`.",
            "`changed[___] = v`",
        ],
        why="`line = []`, then `for v in grid:` with `changed = list(row)`, `changed[j] = v` and `line.append(model(changed))`. After the loop, `return line`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A flat average

        The box holds the example of Goldstein and colleagues, with their formula as the model, without its noise: for a row `[x1, x2, x3]`, it gives 0.2·x1 − 5·x2, plus 10·x2 when x3 ≥ 0. `random.seed(0)` makes the random numbers the same each time, and `random.uniform(-1, 1)` draws a number between −1 and 1, so `rows` holds 1000 rows like theirs. Fill in the loop, so that `average` holds the partial dependence on X2, column 1, at each value of `grid`: for each value, give every row that value in column 1, and average what `model` says. The picture shows your average under the lines of 40 rows.
        """),
        L.task("Replace the `...`: for each `v`, add up `model` for every row with column 1 set to `v`, and append the total divided by `len(rows)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average():
    box_average = L.code_box("average", lines=25)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_average(box_average):
    result_average = L.check("average", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average",
        [
            "This is the formula of *The idea*: for each value of the grid, the average over all rows of what the model says when column 1 has that value. So you need a loop over the rows inside the loop over the grid, and a total that starts at 0 for each value.",
            "Inside `for v in grid:`, start `total = 0`, then `for row in rows:` with `changed = list(row)`, column 1 of `changed` set to `v`, and `total = total + model(changed)`. After the loop over the rows, add the average to `average`.",
            "`average.append(total / ___)`",
        ],
        why="`total = 0`, then `for row in rows:` with `changed = list(row)`, `changed[1] = v` and `total = total + model(changed)`; after that loop, `average.append(total / len(rows))`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flat():
    mo.md(r"""
    ### Exercise 3. A flat line
    """)
    return


@app.cell(hide_code=True)
def choice_flat_cell():
    choice_flat = L.choice(
        "flat",
        question="A colleague draws the partial dependence plot of one feature, and it is flat. Can they conclude that the model ignores that feature?",
        options={
            "scale": "Yes, if it shows probabilities",
            "right": "No: lines that rise and fall can cancel out",
            "flat": "Yes: if the average is flat, so is every line",
        },
        correct="right",
        why={
            "right": "Right. In the example of Goldstein and colleagues, which you averaged in exercise 2, the partial dependence on X2 is almost flat, while every line has a slope of +5 or −5. Draw the ICE curves before you conclude anything from a flat average.",
            "scale": "The scale does not matter here: in exercise 2 the model gives plain numbers, and the average over X2 is flat while every line is steep. See *When the average is flat* in *The idea*.",
            "flat": "An average can be flat while no line is: in exercise 2, about half the lines rise by 10 and the others fall by 10, and your average hardly moves. See *When the average is flat* in *The idea*.",
        },
    )
    choice_flat
    return (choice_flat,)


@app.cell(hide_code=True)
def text_exercise_absurd():
    mo.md(r"""
    ### Exercise 4. The left end
    """)
    return


@app.cell(hide_code=True)
def choice_absurd_cell():
    choice_absurd = L.choice(
        "absurd",
        question="In the worked example, the average at age 17 runs over 1000 people, 252 of whom have education 13 or more. Not one of the 595 people aged 17 in the table has that much. What does that mean for the left end of the curve?",
        options={
            "few": "It rests on too few people to be trusted",
            "right": "It is partly made of rows that never occur",
            "causal": "It shows what a degree does for a 17-year-old",
        },
        correct="right",
        why={
            "right": "Right. Setting age to 17 keeps each person's education, so a quarter of the rows are 17-year-olds with a degree, and the forest never saw one: for them it says 0.102, for the others 0.016. scikit-learn: with correlated features \"we will create absurd data points\". Read the curve where the real people are.",
            "few": "The average runs over 1000 people at every age, 17 included. The trouble is which rows: a quarter of them are 17-year-olds with a degree, who do not occur in the table. See *Rows that do not exist* in *The idea*.",
            "causal": "A curve shows what the model does, not what happens in the world, and here the model is guessing: no 17-year-old in the table has that much education. See *Rows that do not exist* and *What it does not tell you* in *The idea*.",
        },
    )
    choice_absurd
    return (choice_absurd,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Which features matter?](/course/open/rai-permutation-importance?lang=en) measured how much the forest leans on a feature; this lesson draws which way the prediction moves, and for whom. Both describe the whole model from its answers alone: they are global explanations, and model-agnostic, as sorted in [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en). The forest is that of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en), and the tree of depth 2 asks its questions as in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en). The next lesson, [A small model that copies a big one](/course/open/rai-global-surrogate?lang=en), copies the whole forest into one small tree that you can read, and measures how often the copy agrees.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_absurd, choice_flat, result_average, result_ice):
    L.wrap_up(result_ice, result_average, choice_flat, choice_absurd)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
