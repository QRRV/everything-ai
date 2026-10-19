# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-classification. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-classification", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_photos():
    quick_photos = L.choice(
        "quick-photos",
        question="A model learns from 5,000 photos that people marked as cat or dog. Which kind of learning is that?",
        options={
            "right": "Supervised learning",
            "unsupervised": "Unsupervised learning",
            "reinforcement": "Reinforcement learning",
        },
        correct="right",
        why={
            "unsupervised": "Every photo has an answer here: people marked it. See step 1 of the recipe in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "reinforcement": "The model does not act and get rewards; it learns from examples with answers. See the recipe in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "right": "Every photo has a label that people gave it: cat or dog.",
        },
    )
    quick_photos
    return (quick_photos,)


@app.cell(hide_code=True)
def quick_check_next_word():
    quick_next_word = L.choice(
        "quick-next-word",
        question="A language model learns to predict the next word of a text. Where does each label come from?",
        options={
            "people": "People wrote each next word down as a label",
            "rewards": "From rewards for good words",
            "right": "From the text itself",
        },
        correct="right",
        why={
            "people": "Nobody has to write these labels. Where is the next word already? See *The idea* in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "rewards": "Rewards belong to reinforcement learning. What does the model predict here? See *The idea* in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "right": "Hide the word that comes next, and that word is the label: self-supervised learning.",
        },
    )
    quick_next_word
    return (quick_next_word,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The table of students in [Examples, features and labels](/course/open/ml-vocabulary?lang=en) had two columns you could predict: the grade, and whether they passed. If a student got a 6.5 and a model predicts 6.3, that is almost right. But if the answer is "yes" and the model predicts "no", that is simply wrong: there is nothing in between. Those are two kinds of tasks, and later they need different models and different ways to count mistakes. So before you build anything, you ask which of the two your task is.

    **An everyday example.** A weather forecast says it will be 18 degrees tomorrow, and that it will rain. The temperature is a number, and 17 is close to 18. Rain is yes or no.
    """)
    return


@app.cell(hide_code=True)
def predict_digit():
    guess_digit = L.guess(
        "digit",
        question="A model reads a handwritten digit, 0 to 9, from a picture. Which kind of task do you think that is?",
        options={
            "number": "A number to predict: the answer is a digit",
            "answers": "One of ten fixed answers",
            "both": "Both at once",
        },
        reveal="**One of ten fixed answers.** The answer is written with a digit, but a prediction of 7.5 means nothing, and a 7 read as an 8 is no better than a 7 read as a 1: both are simply wrong. The ten digits are ten classes, which only look like numbers. This task is classification.",
    )
    guess_digit
    return (guess_digit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In supervised learning, the label decides the task.

        - In **regression**, the label is a number, such as a price, a grade or a temperature. Values in between make sense, and a prediction close to the label is almost right.
        - In **classification**, the label is one of a fixed set of answers, the **classes**: spam or not spam, yes or no, the digits 0 to 9. A prediction is right or wrong; there is nothing in between.

        Classes are often written as numbers, such as 0 = cat, 1 = dog and 2 = bird. They are still classes: the numbers are only names, and a cat and a bird do not make a dog on average.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **regression** | | a task whose label is a number, such as a price | `df["grade"]` |
        | **classification** | | a task whose label is one of a fixed set of answers, such as spam or not spam | `df["passed"]` |
        | **class** | | one of the answers of classification, such as spam, or yes | `"yes"` |
        """),
        L.watch_out("*Regression* and *class* also mean something else. In software, a regression is a mistake in a program that comes back after a change; here it is a task that predicts a number. And a class here is one of the answers, not a class at school, and not the `class` of Python code."),
        draw.data_tables(C.picture(["Regression: the label is a number", "Classification: the label is a class"]),
                         alt="The table of 7 students twice, with the columns hours, sleep, grade and passed. On the left, the column grade is marked: numbers such as 4.5, 7.5 and 6.0. On the right, the column passed is marked: yes or no."),
        mo.md(r"""
        The same seven students twice, with the same features: the hours they studied and slept. On the left, the label is the grade, a number: regression. On the right, the label is passed, yes or no: classification, with two classes. The table is the same; the label you choose makes the task.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To tell whether a task is regression or classification:

    1. Find the label: the column you want to predict.
    2. Look at its values. A number, where values in between make sense and close is almost right? Regression.
    3. One of a fixed set of answers? Classification: those answers are the classes. `.value_counts()` lists them.
    4. Classes written as numbers stay classes: ask whether a value in between would mean anything.

    A number label can also be turned into classes:

    5. Choose a threshold, and give every value at or above it one class and every value below it the other.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The students' table, with two tasks: predict the grade, or predict whether a student passes. At the end, the grade becomes classes with a threshold: 5.5 or more is a pass. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_tasks():
    import pandas as pd

    _df = pd.read_csv("lessons/ml-vocabulary/students.csv")
    # 1. the first task: the label is grade
    # 2. numbers: the lowest and the highest grade
    print(_df["grade"].min(), _df["grade"].max())
    # 1. the second task: the label is passed
    # 3. a fixed set of answers: the classes, and how many students are in each
    print(_df["passed"].value_counts())
    # 5. from a number to classes: the first four grades, with 5.5 as the threshold
    _classes = []
    for _grade in [4.5, 7.5, 6.5, 3.0]:
        if _grade >= 5.5:
            _classes.append("yes")
        else:
            _classes.append("no")
    print(_classes)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The grades go from 3.0 to 8.5, and a 6.8 or a 7.25 would make sense too: predicting the grade is regression. `passed` has only two values, `yes` for 4 students and `no` for 3: predicting it is classification, with the classes yes and no. The 3 students without a grade are not counted; their labels are not known yet. With 5.5 as the threshold, the grades 4.5, 7.5, 6.5 and 3.0 become no, yes, yes and no: exactly what `passed` says for those students. The number label of one task gives the classes of the other.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_regression():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which one is regression?
    """)
    return


@app.cell(hide_code=True)
def choice_regression_cell():
    choice_regression = L.choice(
        "which-regression",
        question="Which of these tasks is regression?",
        options={
            "spam": "Whether an email is spam",
            "language": "Which language a text is written in",
            "right": "How many minutes a train will be late",
            "digit": "Which digit, 0 to 9, is in a picture",
        },
        correct="right",
        why={
            "spam": "Spam or not spam: is that a number, or one of two answers? See step 3 of the recipe.",
            "language": "Dutch, English, French: a fixed set of answers. See step 3 of the recipe.",
            "digit": "The digits are written as numbers, but would a digit of 7.5 mean anything? See step 4 of the recipe.",
            "right": "The minutes are a number: 4.5 minutes makes sense, and 6 minutes predicted for 7 is almost right.",
        },
    )
    choice_regression
    return (choice_regression,)


@app.cell(hide_code=True)
def text_exercise_numbers():
    mo.md(r"""
    ### Exercise 2. Classes written as numbers
    """)
    return


@app.cell(hide_code=True)
def choice_numbers_cell():
    choice_numbers = L.choice(
        "class-numbers",
        question="Four photos have the labels 0, 0, 2 and 2, with 0 = cat, 1 = dog and 2 = bird. The mean of the labels is 1. What does that tell you?",
        options={
            "dog": "The typical photo shows a dog",
            "right": "Nothing: the numbers are only names of classes",
            "half": "Half the photos show a dog",
        },
        correct="right",
        why={
            "dog": "There is no dog among the four photos at all. Is a mean of class numbers a class? See *The idea*.",
            "half": "Two photos show a cat and two a bird: none a dog. See *The idea*, on classes written as numbers.",
            "right": "Two cats and two birds, and no dog: this mean of class numbers says nothing about the photos. To describe classes, count them.",
        },
    )
    choice_numbers
    return (choice_numbers,)


@app.cell(hide_code=True)
def text_exercise_delays():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. From minutes to classes

        The minutes a train is late are a number: predicting them is regression. A travel app only wants to show whether a train is late: a train that is 5 minutes late or more counts as late. Finish `delay_classes(delays)`. It gets a list of delays in minutes and gives back a list with a class for each delay: `"late"` or `"on time"`. Replace the `...`. The check calls `delay_classes([2, 12, 0, 7])`: that should give `["on time", "late", "on time", "late"]`. It also tries other lists.
        """),
        L.task("Make `delay_classes` give back the class of each delay."),
    ])
    return


@app.cell(hide_code=True)
def exercise_delays():
    box_delays = L.code_box("delay-classes", lines=9)
    box_delays
    return (box_delays,)


@app.cell(hide_code=True)
def check_delays(box_delays):
    result_delays = L.check("delay-classes", box_delays.value)
    result_delays
    return (result_delays,)


@app.cell(hide_code=True)
def hints_delays():
    L.hints(
        "delay-classes",
        [
            "Go through the delays one by one, and for each one add a class to `classes`.",
            "Inside the loop, an `if` with an `else`. Which condition makes a delay late? Mind a delay of exactly 5.",
            "`for delay in delays:`, then `if delay ___ 5:` with `classes.append(\"late\")`, and under `else:` `classes.append(___)`.",
        ],
        why="`delay >= 5` is `True` for 5 minutes or more: then the class is `\"late\"`, and otherwise `\"on time\"`. For `[2, 12, 0, 7]`: on time, late, on time, late.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en), supervised learning had labels; now you know that the kind of label decides the task: a number for regression, a class for classification. In [Examples, features and labels](/course/open/ml-vocabulary?lang=en) the students' table had two labels to choose from; each gives one of the two tasks. Later in the course, a model for classification does not just pick a class: it gives a probability for each class. The next lesson, *Exponents and logarithms*, gives you two functions such a model is built from.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_numbers, choice_regression, result_delays):
    L.wrap_up(choice_regression, choice_numbers, result_delays)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
