# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-project-tabular. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-project-tabular", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [The machine learning workflow](/course/open/ml-workflow?lang=en), [The same model with scikit-learn](/course/open/ml-sklearn?lang=en) and [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="A team decides that a model has to be right about at least 95 of 100 penguins. Which step of the workflow is that?",
        options={
            "data": "3\\. Data",
            "right": "2\\. Metric",
            "model": "5\\. Model",
        },
        correct="right",
        why={
            "data": "The data step is about where the examples come from and how you split them. Which step says how good is good enough? See *The idea* in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "model": "The model step trains and chooses; the goal it is judged by is agreed before that. See *The idea* in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "right": "A metric, accuracy, and a goal for it, 0.95: agreed before any training.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="`model` is a scikit-learn model, `X` holds the features and `y` the labels. Which lines let it learn, and then predict for the rows of `X_new`?",
        options={
            "swap": "`model.predict(X, y)`, then `model.fit(X_new)`",
            "one": "`model.fit(X)`, then `model.predict(X_new, y)`",
            "right": "`model.fit(X, y)`, then `model.predict(X_new)`",
        },
        correct="right",
        why={
            "swap": "Which of the two learns, and which only asks? See the recipe in [The same model with scikit-learn](/course/open/ml-sklearn?lang=en).",
            "one": "To learn, a model needs the features and the labels; to predict, only the features. See the recipe in [The same model with scikit-learn](/course/open/ml-sklearn?lang=en).",
            "right": "`fit` learns from the features and the labels; `predict` gives one prediction per row of new features.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="A feature has a mean of 4200 and a standard deviation of 800 in the training set. What is the standardized value of 5000?",
        options={
            "right": "1",
            "raw": "800",
            "ratio": "1.19",
        },
        correct="right",
        why={
            "raw": "That is 5000 minus the mean. What do you divide it by? See the recipe in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "ratio": "That is 5000 divided by the mean. Standardizing takes the mean off first. See the recipe in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "right": "$(5000 - 4200) / 800 = 1$: one standard deviation above the mean.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You have learned the pieces one at a time: the workflow, a baseline, scikit-learn, standardizing, a validation set and a test set. Today you put them together on a real table, from the question to a number you can report, and you write down what you found. Nothing is new; what is new is doing it all in one go, in the right order.

    **An everyday example.** In driving lessons you practise steering, braking and parking one at a time. The test is one drive through real traffic.
    """)
    return


@app.cell(hide_code=True)
def predict_right():
    guess_right = L.guess(
        "how-many",
        question="Logistic regression gets four measurements of each penguin: the length and depth of its bill, the length of its flipper and its body mass. How many of 68 new penguins do you think it puts in the right species?",
        options={
            "thirty": "About 30: as many as always guessing the most common species",
            "fifty": "About 50",
            "most": "About 66",
            "all": "All 68",
        },
        reveal="**About 66.** On the validation set it gets 66 of the 68 right, an accuracy of 0.97, where always guessing the most common species, Adelie, gets 30. The four measurements say a lot about the species. The two it misses are both Chinstraps with a short bill, which look like Adelies.",
    )
    guess_right
    return (guess_right,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        There is no new idea today: you run [The machine learning workflow](/course/open/ml-workflow?lang=en) on a real table. The **Palmer penguins** are 344 penguins of three species, Adelie, Chinstrap and Gentoo, from three islands near Palmer Station in Antarctica. Kristen Gorman and the Palmer Station Long Term Ecological Research program measured them from 2007 to 2009; Allison Horst, Alison Hill and Kristen Gorman put them in one table. Its license is CC0: you may use it for anything. You met its datasheet in [Where does this data come from?](/course/open/data-provenance?lang=en). The course has it as `data/penguins.csv`: one row per penguin, with its species, its island, four measurements, its sex and the year.

        The six steps, for this table:

        1. **Question.** Which species is a penguin, from its four measurements? A classification with three classes; the label is the column `species`. Here the label is in the table. When what you want to know is not, you need a stand-in that you can measure, and the model then learns the stand-in, not what you meant.
        2. **Metric.** Accuracy: every penguin counts the same, and every mistake costs the same. The goal: at least 0.95.
        3. **Data.** Two penguins have no measurements: they go. The other 342 are shuffled with a seed and split: 206 to train on, 68 to choose on and 68 for the test.
        4. **Baseline.** `DummyClassifier` from scikit-learn, the classifier counterpart of the `DummyRegressor` you met: always the most common species of the training set.
        5. **Model.** `LogisticRegression` from scikit-learn: the same `fit` and `predict`, and with three classes it uses softmax. By default it also adds a small penalty that keeps the weights small. The features are standardized with the mean and standard deviation of the training set.
        6. **Round again.** Look at the penguins it gets wrong, and at each species on its own. Only when everything is chosen: once on the test set.
        """),
        L.watch_out("The mean and the standard deviation for standardizing come from the training set only, also for the validation set and the test set. Those stand for new penguins, which you do not have yet while you train."),
        mo.Html(C.picture({"title": "The 206 penguins of the training set", "x": "bill length (mm)", "y": "bill depth (mm)",
                           "alt": "The 206 training penguins by bill length and bill depth, a colour and shape per species. Adelies, circles, have short, deep bills; Gentoos, squares, long and shallow ones; Chinstraps, triangles, long and deep ones. Adelies and Chinstraps overlap a little, at a bill length of about 41 to 46 mm."})),
        mo.md(r"""
        Here colour and shape mean a species. Two measurements already separate the three species almost entirely: the Gentoos have long, shallow bills, the Adelies short, deep ones, and the Chinstraps long, deep ones. Where the Adelies and Chinstraps overlap, at a bill length between about 41 and 46 mm, mistakes are most likely.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To run the workflow on a table with scikit-learn:

    1. Write down the question and the label; agree on the metric and the goal.
    2. Read the table, keep the rows with every feature, and split it with a seed: `df.iloc[order[:206]]` and so on.
    3. Measure a baseline on the validation set: `DummyClassifier().fit(X, y)`, then `predict`.
    4. Standardize with the training set: `(X - mean) / std`, with `mean` and `std` from `train[features]`.
    5. Train `LogisticRegression()` on the training set, and measure its accuracy on the validation set.
    6. Look at what it gets wrong, `val[pred != val["species"]]`, and go round again. At the end, measure once on the test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 to 5 with two of the four measurements: flipper length and body mass. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_two():
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    # 1. which species is a penguin? Accuracy, with a goal of at least 0.95
    # 2. read the table, keep the penguins with a body mass, and split them with a seed
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    print(len(_df), len(_train), len(_val))
    _features = ["flipper_length_mm", "body_mass_g"]
    # 3. the baseline: always the most common species of the training set
    _baseline = DummyClassifier()
    _baseline.fit(_train[_features], _train["species"])
    print(np.mean(_baseline.predict(_val[_features]) == _val["species"]))
    # 4. standardize with the mean and standard deviation of the training set
    _mean = _train[_features].mean()
    _std = _train[_features].std()
    # 5. logistic regression, measured on the validation set
    _model = LogisticRegression()
    _model.fit((_train[_features] - _mean) / _std, _train["species"])
    _pred = _model.predict((_val[_features] - _mean) / _std)
    print(np.mean(_pred == _val["species"]))
    return DummyClassifier, LogisticRegression, np, pd


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    342 penguins are left, 206 of them to train on and 68 to choose on. The baseline always says Adelie, the most common species of the training set, and is right about 0.44 of the validation set: 30 of the 68. With flipper length and body mass, logistic regression gets 0.84: far above the baseline, but below the goal of 0.95. Per species, the 0.84 hides more: all 25 Gentoos and 29 of the 30 Adelies are right, but only 3 of the 13 Chinstraps. Those two measurements tell the Gentoos apart, which are big, but not the Adelies from the Chinstraps, which weigh about the same. So: round again, with the bill too.

    `_df.iloc[_order[:206]]` takes the rows at those 206 positions, as an array of positions did in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en); `isna() == False` is `True` where the body mass is there. pandas' `.std()` divides by one less than the number of values, one of the other tools from [Describing data: mean, median and spread](/course/open/stats-describe?lang=en); with 206 penguins that makes no difference here.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. All four measurements

        The box reads and splits the table as in the worked example, and puts all four measurements in `features`. Replace the `...` with steps 4 and 5: standardize with the mean and standard deviation of the training set, let `LogisticRegression` learn from `train`, put the predicted species of the penguins in `val` in `pred`, and put the accuracy in `accuracy`. The last line is `accuracy`, so the check sees it. The check also tries other features.
        """),
        L.task("Put in `accuracy` the accuracy on the validation set of logistic regression on all four measurements."),
    ])
    return


@app.cell(hide_code=True)
def exercise_model():
    box_model = L.code_box("model", lines=17)
    box_model
    return (box_model,)


@app.cell(hide_code=True)
def check_model(box_model):
    result_model = L.check("model", box_model.value)
    result_model
    return (result_model,)


@app.cell(hide_code=True)
def hints_model():
    L.hints(
        "model",
        [
            "Steps 4 and 5 of the worked example, with `features` instead of the two measurements.",
            "`mean` and `std` come from `train[features]` only. Which sets do you divide by them before `fit` and before `predict`?",
            "`mean = train[features].mean()`, `std = ___`, then `model.fit((train[features] - mean) / std, ___)`, `pred = model.predict(___)` and `accuracy = np.mean(pred == ___)`.",
        ],
        why="With all four measurements, logistic regression gets 66 of the 68 validation penguins right: an accuracy of about 0.97, above the goal of 0.95. The bill measurements tell the Adelies from the Chinstraps.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Which penguins does it get wrong?

        Step 6: look at the mistakes. The box trains the model of exercise 1, with its predictions for the validation set in `pred`. Replace the `...` with one line: put in `wrong` the rows of `val` whose species differs from the prediction. The last line is `wrong`, so the check sees it. The check also tries other features.
        """),
        L.task("Put in `wrong` the penguins of the validation set that the model puts in the wrong species."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_mistakes = L.code_box("mistakes", lines=18)
    box_mistakes
    return (box_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_mistakes):
    result_mistakes = L.check("mistakes", box_mistakes.value)
    result_mistakes
    return (result_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "mistakes",
        [
            "A comparison of `pred` with the species gives `True` or `False` per penguin. How did a mask select rows of a table?",
            "Which comparison is `True` exactly where the model is wrong?",
            "`wrong = val[pred ___ val[\"species\"]]`.",
        ],
        why="`pred != val[\"species\"]` is `True` for the two penguins the model gets wrong, and `val[...]` keeps those rows. Both are Chinstraps from Dream with a bill of 42.4 and 43.5 mm: short for a Chinstrap, long for an Adelie, right where the two species meet in the picture.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proxy():
    mo.md(r"""
    ### Exercise 3. A label that is not in the table
    """)
    return


@app.cell(hide_code=True)
def choice_proxy_cell():
    choice_proxy = L.choice(
        "proxy",
        question="The researchers also want to know which penguins do not get enough to eat. The table has no column for that. What can they use as the label?",
        options={
            "mass": "Body mass as it is: the heaviest penguins eat the best",
            "right": "Whether a penguin is light for its species and sex: a stand-in for not enough to eat, which can also be wrong",
            "none": "Nothing: without such a column, no model is possible",
        },
        correct="right",
        why={
            "mass": "Gentoos are heavier than Adelies whatever they eat, and males heavier than females. What does the model then learn: food, or species? See step 1 in *The idea*.",
            "none": "A label you cannot measure can often be replaced by one you can: a stand-in. What is the catch? See step 1 in *The idea*.",
            "right": "A label made from other columns, compared within the species and sex, is a stand-in for what you really want to know. A penguin can also be light for other reasons, so write down what the label really measures.",
        },
    )
    choice_proxy
    return (choice_proxy,)


@app.cell(hide_code=True)
def text_test():
    mo.md(r"""
    ### Once on the test set

    The model with four measurements reaches the goal on the validation set, so the choosing is done. In the order of the work, only now does the test set come in, once. Run the cell: it trains the same model and measures it, and the baseline, on the 68 penguins of the test set.
    """)
    return


@app.cell
def example_test(DummyClassifier, LogisticRegression, np, pd):
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _test = _df.iloc[_order[274:]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _mean = _train[_features].mean()
    _std = _train[_features].std()
    _model = LogisticRegression()
    _model.fit((_train[_features] - _mean) / _std, _train["species"])
    _pred = _model.predict((_test[_features] - _mean) / _std)
    _baseline = DummyClassifier()
    _baseline.fit(_train[_features], _train["species"])
    print(np.mean(_pred == _test["species"]), np.mean(_baseline.predict(_test[_features]) == _test["species"]))
    print(_test[_pred != _test["species"]][["species", "island", "bill_length_mm", "bill_depth_mm"]])
    return


@app.cell(hide_code=True)
def text_after_test():
    mo.md(r"""
    On the test set the model gets 67 of the 68 right, an accuracy of 0.99, against 0.43 for the baseline. The one it misses is again a Chinstrap from Dream. That the test set comes out a little higher than the validation set is most likely chance: with 68 penguins, one penguin more or less is 0.015. The number you report is 0.99 on the test set, next to the baseline.
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Write down what you found, in three or four sentences, for the researchers: the question, the metric and the goal, the baseline, the model and its number on the test set, and what goes wrong.",
        model_answer="We predicted the species of a penguin from four measurements, with accuracy as the metric and 0.95 as the goal. Always guessing the most common species, Adelie, gets 0.43 on the test set; logistic regression on the standardized measurements gets 0.99 (0.97 on the validation set). Its mistakes are Chinstraps with a short bill, which it takes for Adelies. With 68 test penguins, one penguin is 0.015, so these numbers are rough.",
        points=[
            "the question and the label: the species, from four measurements",
            "the metric and the goal: accuracy, at least 0.95",
            "the baseline next to the model, both on the test set: about 0.43 against 0.99",
            "what goes wrong: Chinstraps with short bills, taken for Adelies",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The machine learning workflow](/course/open/ml-workflow?lang=en) gave the order, [The same model with scikit-learn](/course/open/ml-sklearn?lang=en) the `fit` and `predict`, [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) the standardizing, and [Where does this data come from?](/course/open/data-provenance?lang=en) the datasheet of these penguins. Together they took a real table from a question to a reported number. In the next lesson, *Who is in the data, and who is missing?*, you ask what such a number does not tell you: who the data leaves out, and who a model could harm.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_proxy, explain_report, result_mistakes, result_model):
    L.wrap_up(result_model, result_mistakes, choice_proxy, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
