# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-limits-of-prediction. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-limits-of-prediction", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en), and one about [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fact():
    quick_fact = L.choice(
        "quick-fact",
        question="Which of these decisions is the safest to automate?",
        options={
            "right": "Checking whether someone is 18 or older",
            "fraud": "Deciding whether someone committed fraud",
            "child": "Deciding which parent a child lives with",
        },
        correct="right",
        why={
            "right": "It checks a fact, the same way every time, as long as people can still have a wrong record corrected.",
            "fraud": "Whether someone meant to deceive is a judgment about circumstances. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
            "child": "That weighs the circumstances of one family. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
        },
    )
    quick_fact
    return (quick_fact,)


@app.cell(hide_code=True)
def quick_check_r2():
    quick_r2 = L.choice(
        "quick-r2",
        question="A model's R² on its test set is 0. What does that mean?",
        options={
            "perfect": "Its predictions match every label exactly",
            "half": "It gets about half of the labels right",
            "right": "No better than always guessing the mean",
        },
        correct="right",
        why={
            "perfect": "That would be an R² of 1. See *The idea* in [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).",
            "half": "R² is not a share of right answers, but the share of the spread of the labels that the model explains. See *The idea* in [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).",
            "right": "R² compares the MSE with the variance, the MSE of always predicting the mean: at 0 they are the same.",
        },
    )
    quick_r2
    return (quick_r2,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Schools, courts and youth care would like to know in advance who will drop out, commit a crime again or need help: then they could act early. Systems that promise this are sold and used. Before a prediction decides something about one person, two questions come first: how well can this outcome be predicted at all, and is the thing the model predicts the thing you care about?

    **An everyday example.** A weather forecast for tomorrow is often right. A forecast for one day three months from now is hardly better than the usual weather for that time of year.
    """)
    return


@app.cell(hide_code=True)
def predict_challenge():
    guess_challenge = L.guess(
        "challenge",
        question="160 teams got about 13,000 facts about each of thousands of families, from a child's birth on, and tried to predict the child's school grades at 15. What R² do you think the best team reached?",
        options={
            "right": "About 0.2",
            "half": "About 0.5",
            "high": "About 0.9",
        },
        reveal="**About 0.2**, for school grades and for hardship in the family, and close to 0 for the four other outcomes. And the best teams were only a little better than a simple model with four variables chosen by an expert (Salganik and colleagues, 2020).",
    )
    guess_challenge
    return (guess_challenge,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The *Fragile Families Challenge* asked how far a life can be predicted. A study had followed about 4,000 families since the birth of their child. In 2017, 160 teams got its data up to the child's ninth year and tried to predict six outcomes at age 15, such as the child's school grades and whether the family was evicted. The best R² was about 0.2 for grades and for hardship, and close to 0 for the other four. The teams were not bad: predicting from the past simply worked poorly here. Why is still open; one reading is that a life depends on much that no data holds.

        The computer scientist Arvind Narayanan puts AI tasks in three kinds. *Perception*, such as speech to text or reading a scan, has made real progress: the answer is in the data. *Automating judgment*, such as spam filters, is far from perfect but improving. *Predicting social outcomes*, such as who will commit a crime again or do well in a job, is the kind he calls fundamentally dubious.

        There is a second problem. What you can measure is often not what you mean. US hospitals used an algorithm that chose patients for extra care by predicting their health costs, standing in for how sick they were. But less was spent on Black patients who were just as ill, so at the same score they were sicker. If patients with the same score had been equally sick, the share of Black patients picked for extra care would have risen from 17.7 to 46.5 percent (Obermeyer and colleagues, 2019). That gap between the label a model is trained on and the question you want answered is a **target-goal mismatch**.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **target-goal mismatch** | the gap between what a model is trained to predict, because it can be measured, and what you actually want to know, such as health costs instead of health needs |
        """),
        L.watch_out("An R² of 0.2 does not mean that every prediction is a little off. For some people the prediction is far off, and a decision about one person rests on that one prediction."),
        mo.Html(C.picture({"title": "Ten children: real grade at 15 and a model's prediction", "xlabel": "child", "ylabel": "grade point average (1 to 4)",
                           "mean": "mean 2.8",
                           "alt": "Ten made-up children, with their real grade point average at 15 and a model's prediction. The predictions stay between 2.4 and 3.1, close to the mean of 2.8, while the real grades run from 1.75 to 4.0. For children 4, 5, 8 and 9 the prediction is 0.85 or 0.9 off."})),
        mo.md(r"""
        Made up by the course. The grade is a grade point average from 1 to 4. A filled dot is a child's real grade, an open dot the model's prediction; the line between them is the miss.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Before a prediction decides something about a person:

    1. Write down the goal you mean, and the target the data can measure.
    2. Ask how the two differ, and for whom.
    3. Find out how well anyone has predicted an outcome like this, and compare with a simple model.
    4. Look at the misses for single people, not only the average.
    5. When the score is low, let the prediction help plan for groups at most, not decide about one person.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A function that gives the MAE of a model's predictions, so you can compare two models with one line each. Four made-up grades: the first model predicts something close to each, the second always predicts 3. Run the cell.
    """)
    return


@app.cell
def example_mae():
    import numpy as np

    def _mae(y, pred):
        return np.mean(abs(y - pred))

    _y = np.array([2.0, 3.0, 4.0, 3.0])
    print(_mae(_y, np.array([2.5, 3.0, 3.5, 3.0])))
    print(_mae(_y, np.array([3.0, 3.0, 3.0, 3.0])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first model is off by 0.25 on average, the second by 0.5. A function does the same work for any two arrays. Step 3 of the recipe compares a careful model with a simple one, with R²: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_r2():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How much does the model explain?

        The children of the picture. `actual` holds their real grades, `best` the predictions of a careful model, `simple` those of a model with four variables. Replace the `...` with the body of `r2`, as in the worked example, but with R² from [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en): put the MSE in `mse` and the variance of `y` in `variance`, and give back R² with `return`. The last line gives R² for both models, so the check sees it. The check also tries another simple model.
        """),
        L.task("Make `r2(y, pred)` give the R² of the predictions `pred` for the real values `y`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_r2():
    box_r2 = L.code_box("r2", lines=12)
    box_r2
    return (box_r2,)


@app.cell(hide_code=True)
def check_r2(box_r2):
    result_r2 = L.check("r2", box_r2.value)
    result_r2
    return (result_r2,)


@app.cell(hide_code=True)
def hints_r2():
    L.hints(
        "r2",
        [
            "Inside the function, `y` and `pred` are arrays, as in the worked example. Work out R² from them, in three lines.",
            "The MSE is the mean of the squared misses. The variance is the MSE of always predicting the mean of `y`. R² is 1 minus the one divided by the other.",
            "`variance = np.mean((y - ___) ** 2)`",
        ],
        why="`mse = np.mean((y - pred) ** 2)`, `variance = np.mean((y - y.mean()) ** 2)` and `return 1 - mse / variance` give `[0.2, 0.15]`: the careful model explains a fifth of the spread, the simple model a little less.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 2. Which kind of task?

    Six tasks for AI. Put each one with the kind it belongs to, in Narayanan's three kinds.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Ask for each task: is the answer already in the data, is it a judgment people make, or does it lie in someone's future?",
            "Two tasks are about a recording and a picture. Two others ask what will happen to a person years from now.",
            "Perception: speech to text and \\_\\_\\_. Judgment: spam and \\_\\_\\_. Social outcomes: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Perception:** speech to text and a scan. **Automating judgment:** spam and an essay. **Predicting social outcomes:** again? and a good hire?",
    )
    return


@app.cell(hide_code=True)
def text_exercise_target():
    mo.md(r"""
    ### Exercise 3. The target and the goal
    """)
    return


@app.cell(hide_code=True)
def choice_target_cell():
    choice_target = L.choice(
        "target",
        question="A town wants to predict which children need extra help at school, and trains a model on which children got extra help in the past. What is the main problem?",
        options={
            "size": "The data has too few children in it to learn from",
            "right": "It learns who got help, not who needed it",
            "fine": "Nothing, as long as the past data are accurate",
        },
        correct="right",
        why={
            "size": "More children would not fix it: the label itself asks the wrong question. See *The idea*.",
            "right": "Who got help depends on who asked, who was noticed and what there was to give: a target-goal mismatch, as with health costs for health needs.",
            "fine": "Accurate data about who got help still says little about who needed it. See the hospital example in *The idea*.",
        },
    )
    choice_target
    return (choice_target,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en) you asked what may be automated. Here you saw a limit that no law sets: some outcomes in people's lives can hardly be predicted, and the label you can measure is often not the question you mean. R² from [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en) tells you how much room a prediction leaves.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_target, result_kinds, result_r2):
    L.wrap_up(result_r2, result_kinds, choice_target)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
