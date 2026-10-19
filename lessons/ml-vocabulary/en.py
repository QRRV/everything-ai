# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-vocabulary. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-vocabulary", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en) and [Asking questions of data](/course/open/data-explore?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_coat():
    quick_coat = L.choice(
        "quick-coat",
        question="In the lesson about writing rules and learning from examples, the computer found the rule for when to wear a coat. What did it use to find that rule?",
        options={
            "person": "A rule that a person wrote down first",
            "temps": "Only the temperatures of the mornings",
            "examples": "Mornings with their temperature, and whether a coat was worn",
        },
        correct="examples",
        why={
            "person": "Then nobody would have to learn anything. What did the computer get instead? See the table of new words in [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en).",
            "temps": "With only the temperatures, how could it count the mistakes of a rule? See the recipe in [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en).",
            "examples": "Each morning had a temperature and the answer, coat or no coat. The computer tried rules and kept the one with the fewest mistakes on those examples.",
        },
    )
    quick_coat
    return (quick_coat,)


@app.cell(hide_code=True)
def quick_check_isna():
    quick_isna = L.choice(
        "quick-isna",
        question="`df` has a column `price`, with some prices missing. Which line keeps the rows where the price is missing?",
        options={
            "rows": "`df[df[\"price\"].isna()]`",
            "count": "`df[\"price\"].isna().sum()`",
            "mask": "`df[\"price\"].isna()`",
        },
        correct="rows",
        why={
            "count": "That counts the missing prices: one number. How do you keep rows with a mask? See the recipe in [Tables with pandas](/course/open/data-pandas?lang=en).",
            "mask": "That is the mask: `True` or `False` for every row. Where does a mask go, to keep rows? See the recipe in [Tables with pandas](/course/open/data-pandas?lang=en).",
            "rows": "`df[\"price\"].isna()` is `True` where the price is missing, and between the square brackets of `df` it keeps those rows.",
        },
    )
    quick_isna
    return (quick_isna,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Most tasks in machine learning, from a spam filter to a language model, are described with the same few words. Which examples are there? What do you know about each one? What should the model predict? Which examples does it learn from, and which does it predict? With these words, you can say for such a task what goes in, what comes out, and where the right answers come from.

    **An everyday example.** Learning to tell birds apart: you look at many birds, at their size, color and beak, and a guide tells you each bird's name. Later you see a new bird, and you name it yourself.
    """)
    return


@app.cell(hide_code=True)
def predict_houses():
    guess_houses = L.guess(
        "house-features",
        question="A table of houses has the columns `area`, `rooms`, `garden` and `price`. A model should predict the price of houses that are for sale. Which columns do you think the model can use to predict the price of a house that is for sale?",
        options={
            "three": "`area`, `rooms` and `garden`",
            "price": "Only `price`",
            "all": "All four columns",
        },
        reveal="**`area`, `rooms` and `garden`.** Those are known for every house, even for one that is for sale. The price is what the model should predict: for a house that is for sale, it is not known yet. So the model learns from the first three, and from the prices of houses that were sold.",
    )
    guess_houses
    return (guess_houses,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## The idea

    Here, a dataset for machine learning is a table of examples, one per row. The columns you know beforehand are the features: what the model gets. The column you want to predict is the label: the right answer. Here, training is learning from examples that have a label, and inference is using the trained model on new examples, which have features but no label yet.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **example** | one item of data, one row of the table: here, one house | `df.iloc[0]` |
    | **feature** | one value you know about an example beforehand, one column: here, the area | `df["area"]` |
    | **label** | the right answer that belongs to an example, the column you want to predict: here, the price | `df["price"]` |
    | **training** | a model learns from examples whose label is known | |
    | **inference** | a trained model predicts the label of new examples, from their features | |

    `df.iloc[0]` is the first row, the first example: `iloc` takes rows by their place, from 0, as an index does in a list, not by their row label. The same table can be used for another task with another label: which column is the label depends on what you want to predict.
    """),
    L.watch_out("Here, a *label* is the right answer of an example. In [Tables with pandas](/course/open/data-pandas?lang=en), the row labels were the numbers to the left of each row, and on a chart a label is the text along an axis: those are other things. A *feature* is not a feature of a product, and *inference* here only means using a trained model."),
    ])
    return


@app.cell(hide_code=True)
def picture_houses():
    mo.vstack([
        draw.data_tables(C.picture(["the features", "the label"]), title="Features and label",
                         alt="The table of 10 houses twice. On the left, the columns area, rooms and garden are marked: the features. On the right, the column price is marked: the label. The last two houses have a question mark for their price."),
        mo.md(r"""
        The same 10 houses twice. On the left, the features are marked: the area in square meters, the number of rooms, and whether there is a garden (1) or not (0). On the right, the label is marked: the price, in thousands of euros. Houses 0 to 7 were sold, so their price is known: they are for training. Houses 8 and 9 are for sale: their price is a question mark, and predicting it is inference.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A feature has to be known at the moment you predict. The price a house sold for cannot be a feature for predicting that price, and neither can anything that is only known afterwards: a model that learned from it would be useless for new examples."),
        mo.md(r"""
        ### Recipe

        To describe a machine learning task with a table:

        1. Say what you want to predict: that column is the label.
        2. Choose the features: the columns that are known at the moment you predict, and that may help.
        3. Take them out of the table: the features with a list of names, the label with one name.
        4. The rows with a label are for training; the rows without one are for inference.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The 10 houses, in `lessons/ml-vocabulary/houses.csv`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_houses():
    import pandas as pd

    _houses = pd.read_csv("lessons/ml-vocabulary/houses.csv")
    # 1. the label: what we want to predict
    _labels = _houses["price"]
    # 2. and 3. the features: what we know beforehand
    _features = _houses[["area", "rooms", "garden"]]
    print(_features.shape)
    print(_labels.shape)
    # 4. which houses have a label, and which do not
    print(_labels.isna())
    # one example: the first house, its features and its label
    print(_houses.iloc[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The features are a table of 10 rows and 3 columns, `(10, 3)`; the labels are one column of 10 values, `(10,)`. The mask is `False` for houses 0 to 7, which have a price, and `True` for houses 8 and 9: those are the houses to predict. The first example is a house of 62 square meters, with 3 rooms and no garden, sold for 245 thousand euros. It prints as 62.0, 3.0 and 0.0, because one row has one dtype, and the missing prices make the table use floats.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spam():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The label of a spam filter
    """)
    return


@app.cell(hide_code=True)
def choice_spam_cell():
    choice_spam = L.choice(
        "spam-label",
        question="An email program should sort new emails into spam and not spam. It learns from old emails that people marked as spam or not. In this task, what is the label?",
        options={
            "words": "The words in the email",
            "marked": "Spam or not spam, as people marked it",
            "sender": "Who sent the email",
        },
        correct="marked",
        why={
            "words": "The words are known for every email, even a new one. Is that what the program should predict, or what it predicts from?",
            "sender": "The sender is known for every email, even a new one. Is that what the program should predict, or what it predicts from?",
            "marked": "Spam or not is what the program should predict for a new email, and people gave that answer for the old ones: that is the label. The words and the sender are features.",
        },
    )
    choice_spam
    return (choice_spam,)


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Before the exam

        A group of students wants to predict, on the morning of an exam, who will pass. The box reads their table as `df`, from `lessons/ml-vocabulary/students.csv`: for each student, the hours they studied (`hours`), the hours they slept the night before (`sleep`), the grade they got for the exam, from 1 to 10 (`grade`), and whether they passed (`passed`). Make `features`: the columns that can be features. Replace the `...`. The last line is `features`, so the check sees it.
        """),
        L.task("Make `features`: the columns of `df` that are known before the exam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    box_features = L.code_box("features", lines=6)
    box_features
    return (box_features,)


@app.cell(hide_code=True)
def check_features(box_features):
    result_features = L.check("features", box_features.value)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def hints_features():
    L.hints(
        "features",
        [
            "Step 2 of the recipe: a feature has to be known at the moment you predict, here before the exam.",
            "`passed` is the label. Of the other three, which does a student know on the morning of the exam, and which only afterwards?",
            "`features = df[[___, ___]]`.",
        ],
        why="`hours` and `sleep` are known before the exam. `passed` is the label, the answer to predict, and `grade` is only known after the exam, so neither can be a feature.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Which examples to predict?

        Finish `to_predict(path, label)`. It reads the CSV file at `path` as `df`, and gives back the rows whose label is missing, with all their columns: the examples a trained model has to predict. `label` is the name of the column with the label. Replace the `...`. The check calls `to_predict("lessons/ml-vocabulary/houses.csv", "price")`: that should give the two houses for sale. It also tries other tables and labels.
        """),
        L.task("Make `to_predict(path, label)` give back the rows of the table whose label is missing."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("to-predict", lines=7)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("to-predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "to-predict",
        [
            "Step 4 of the recipe: the rows without a label are for inference. Keep them with a mask, as in the second question of the quick check.",
            "`.isna()` on the column whose name is in `label` gives the mask: `True` where the label is missing.",
            "`return df[___]`, with the mask between the brackets.",
        ],
        why="`df[label].isna()` is `True` for every row whose label is missing, and `df[...]` keeps those rows, with all their columns: the features a model predicts from.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en), the mornings were the examples, the temperature the feature, and coat or no coat the label; now you have the words for them. In [Asking questions of data](/course/open/data-explore?lang=en) you looked for missing values; a missing label marks an example to predict. In the next lesson, *A model is a function with parameters*, you see what a model does with the features of an example: it turns them into a prediction.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spam, result_features, result_predict):
    L.wrap_up(choice_spam, result_features, result_predict)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
