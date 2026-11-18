# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-project-audit-explain. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rai-project-audit-explain", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Project: measure the bias of your own classifier](/course/open/rai-project-audit-bias?lang=en) and one about [What would have to change?](/course/open/rai-counterfactuals?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_why():
    quick_why = L.choice(
        "quick-why",
        question="In step one, your `report` gave the women of the test set a yes rate of 0.070 and the men 0.248. What does that table tell you about why?",
        options={
            "right": "Nothing yet: it counts outcomes per group",
            "sex": "That the classifier looks at sex",
            "size": "That there are fewer women in the test set",
        },
        correct="right",
        why={
            "right": "An audit measures who gets a yes, per group. Why the outcomes differ, the table does not say. This lesson asks the classifier itself.",
            "sex": "The classifier does not get the column sex at all, and still gives women far fewer yeses. The table counts outcomes per group; why they differ, it does not say. See *The idea* in [Project: measure the bias of your own classifier](/course/open/rai-project-audit-bias?lang=en).",
            "size": "There are fewer women, 5,421 against 10,860 men, but a yes rate is a share within each group, so the size of a group does not explain it. The table counts outcomes; why they differ, it does not say. See [Project: measure the bias of your own classifier](/course/open/rai-project-audit-bias?lang=en).",
        },
    )
    quick_why
    return (quick_why,)


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="With the median absolute deviations as steps, being 81 was the closest counterfactual of the applicant aged 30; with the standard deviations as steps, education level 16. What made the difference?",
        options={
            "right": "The distance, which picks the closest",
            "model": "The model was trained again in between",
            "person": "The two used different applicants",
        },
        correct="right",
        why={
            "right": "The same model and the same applicant, with another distance, give another closest counterfactual. In this lesson a counterfactual may change one column only, the hours a week, and that choice of what may change does the same work: write it down.",
            "model": "The model stayed the same, and so did the applicant; only the steps the distance counts in changed. See [What would have to change?](/course/open/rai-counterfactuals?lang=en).",
            "person": "It was the same applicant, aged 30, and the same model; only the steps the distance counts in changed. See [What would have to change?](/course/open/rai-counterfactuals?lang=en).",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Project: measure the bias of your own classifier](/course/open/rai-project-audit-bias?lang=en) you measured your classifier on the test set: it gives a yes to 7.0% of the women and 24.8% of the men, an impact ratio of 0.28, although it never gets the column sex. That audit showed where the outcomes differ, not why. Since then, this module taught you to ask a model why: which columns it leans on, how its answer moves with one column, what pushes one person's answer up or down, and what a person would have to change. Today you put those questions to the same classifier, keep three new functions in your project, and write the explanations part of your audit report: what the explanations show about the bias you measured, and what they cannot show.

        **An everyday example.** A shop sees its sales fall by a fifth in one month. A breakdown per department shows that almost all of the drop is in umbrellas. That tells the owner where the drop sits, not why: a dry month, a shop nearby that sells them cheaper, or a shelf moved to the back. And whether the owner should do something about it is yet another question. Explanations of a model are such a breakdown: they show where a gap sits in the model, not why the world looks the way it does, nor whether the model may decide the way it does.
        """),
        draw.project_steps(["Measure the bias of your classifier", "Explain your classifier"], 1,
                           here_label="← today", title="Where your project stands: your audit"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_gap():
    guess_gap = L.guess(
        "gap",
        question="Your classifier does not get the column sex. Of the columns it does get, which carries the largest part of the gap between women and men, do you think?",
        options={
            "relationship": "relationship, with Husband and Wife",
            "hours": "hours_per_week",
            "right": "marital_status",
            "occupation": "occupation",
        },
        reveal="**marital_status**, by far. In the logit, the number before the sigmoid, the women of the test set average −3.063 and the men −1.647, a gap of 1.416, about 1.42. Of that gap, marital_status carries 1.01, hours_per_week 0.18 and occupation 0.09; relationship, which holds Husband and Wife among its values, carries only 0.04. *The idea* shows how to split a gap like this.",
    )
    guess_gap
    return (guess_gap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Four questions.** Step one measured outcomes per group. The explanations of this module each ask the classifier behind those outcomes another question:

        - Which columns does it lean on, over all people? Permutation importance, from [Which features matter?](/course/open/rai-permutation-importance?lang=en): shuffle one column and see how much less often the classifier is right.
        - How does its answer move with one column? Partial dependence, from [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en): give everyone the same value and take the mean probability. Here for women and men apart.
        - What pushes one person's answer up or down? Shapley values, from [Shapley values](/course/open/rai-shapley-values?lang=en). Averaged over a group, they show where a gap between groups sits.
        - What would a person have to change? A counterfactual, from [What would have to change?](/course/open/rai-counterfactuals?lang=en), with only a change in something the person can act on: here more hours of work a week.

        **Shapley values of a logistic regression, exactly.** In [Shapley values](/course/open/rai-shapley-values?lang=en) you went through every coalition of four columns. Your classifier needs no coalitions. A logistic regression adds up: its logit is the bias term plus every column times its weight: the three number columns, divided by their largest value, and the 0/1 columns. In a sum, a column adds the same to every coalition: its weight times how far the person lies from the average of the background, here the people to train on. That is the column's Shapley value on the logit, exactly. For a word column such as `marital_status`, add up its 0/1 columns. The values of one person add up to that person's logit minus the mean logit of the people to train on. For a logistic regression, that mean is also the logit of the average person, a made-up row with every column at its average: −2.105 for your classifier, a probability of 0.109. That is not the mean probability of those people, which is 0.241. On the probability itself, after the sigmoid, this shortcut does not hold.

        **What the explanations show about your classifier.** Shuffle `marital_status` in the test set, and the classifier is right for 76.5% of the people instead of 83.4%: a permutation importance of 0.070 (unrounded, 0.8344 − 0.7648 = 0.0697), the largest of the eight columns, ahead of `education_num` with 0.043 and `occupation` with 0.021 (seed 0). Averaged over the women of the test set, the Shapley values add up to −0.96; over the men, to +0.46. Of that gap of 1.42, `marital_status` carries 1.01. Of the women in the test set, 15.2% are `Married-civ-spouse`, against 60.6% of the men, and between `Never-married` and `Married-civ-spouse` lie 2.46 in the logit. If every woman of the test set were `Married-civ-spouse`, their mean probability of a yes would be 0.318 instead of 0.126; for the men, 0.381 instead of 0.295. And of the 5,039 women with a no, 19.8% would get a yes by working more hours a week, at most 99, with everything else the same; of the 8,168 men with a no, 43.9%. Most of them would need 60 hours a week or more: 78.6% of those women and 74.0% of those men. Whether a person can really work that much belongs in the report too. Berk Ustun, Alexander Spangher and Yang Liu warned that recourse need not be equal: "the cost of recourse can differ between males and females even when models ignore gender". Here even whether recourse exists differs.
        """),
        mo.Html(C.course_gap_figure("en")),
        mo.md(r"""
        **What they cannot show.** The explanations explain the classifier, not the world. That it leans on marital status says nothing about why married people in the survey of 1994 more often earn more, nor about whether a decision about people may lean on marital status. Those questions are for the people who decide with the classifier, and they belong in the report as open questions. An explanation can also look reassuring where it should not: permutation importance gives `race` −0.001, so shuffling it hardly changes how often the classifier is right. Yet averaged over the people marked `Other`, the Shapley value of `race` is −0.64, over `Black` −0.11, and over `White` +0.03. Exercise 4 asks what to make of that.

        **Explanations can be made to look good.** Dylan Slack and colleagues showed how a builder who wants to hide a bias can fool LIME and SHAP. Both explain a model by sending it made-up inputs, perturbed versions of real ones, which often look unlike real data. A "scaffolding" classifier first asks whether an input looks real: real inputs go to the biased model, made-up ones to a harmless model. With LIME, a harmless column took the top place from the sensitive one for 100% of the held-out cases in two datasets, and a loan column did for 91% in a third; with SHAP and one harmless column, the sensitive column lost the top place for 84% to 100%, depending on the data, and with two harmless columns the attack worked less often. Fooling SHAP had a cost: the scaffold then agreed with the biased model on only 75% to 91% of the real inputs. Their conclusion: explanations "that rely on input perturbations, such as LIME and SHAP, are not reliable". The attack needs a builder who sets out to fool; your own classifier is no such scaffold. Ulrich Aïvodji and colleagues have a name for making a model look fairer than it is, fairwashing: "promoting the false perception that a machine learning model respects some ethical values". On UCI Adult, a random forest had a demographic-parity unfairness of 0.13 on the people it was to be explained for, and 0.18 on the test set; the simple rule lists their method found to mimic it looked fairer, and the best one agreed with the forest for 90.8%: "On Adult Income, the best model has a fidelity of 0.908 and an unfairness of 0.058."

        **Leave a trail.** Margaret Mitchell and colleagues name a limitation of model cards themselves: "the usefulness and accuracy of a model card relies on the integrity of the creator(s) of the card itself." The same holds for your audit, a first-party audit by someone who shares the builder's interest. You can make it checkable: Inioluwa Deborah Raji and colleagues ask an internal audit to leave a "transparency trail", the documents of each stage. So write down how each explanation was made: for which classifier, on which people, with which seeds and which background.

        **The audit report.** An audit ends in an **audit report**: what was measured and explained, how, what came out, and what was not measured. Model cards suggest "interpretability approaches" as one of the details a card may give about a model, and ask for results that are "disaggregated", per group: so explain per group too, as you do here per sex.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **audit report** | the document an audit ends in: what was measured and explained, how, with what result, and what was not measured |
        """),
        L.watch_out("An explanation is evidence about a model, not a verdict on it. It can show where a gap sits, and it can look reassuring where it should not. Report every explanation with how it was made, and never as proof that the classifier is fair."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The second part of your audit:

    1. Start from what step one measured: the gap in the yes rate, and the classifier it was measured on.
    2. Global: which columns does the classifier lean on? Permutation importance on the test set, with a few seeds. That is `importance`.
    3. How does its mean answer move with one column, per group? Partial dependence for women and for men. That is `dependence`.
    4. Local: what pushes one person, or a group on average, up or down? Shapley values; for a logistic regression, exactly on the logit, each weight times how far the person lies from the background's average.
    5. What would a person have to change, and can they? A counterfactual with only a change in something the person can act on, here the hours a week; compare how many women and men have one. That is `fewest_hours`.
    6. Write down what the explanations show, how each was made, and what they cannot show.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4 for one person: the woman in row 34982 of the table, in the test set: 40 years old, a bachelor's degree, 50 hours a week in `Exec-managerial`, never married, and a no from the classifier. The cell trains the classifier of step one, as in your `model.py`. `_model.coef_[0]` holds its weights, one per column, in the order of the columns of `_X`. `_X.loc[34982]` is the row with the label 34982, and `_X[_train].mean()` the average of the people to train on: subtracting gives how far the person lies from it, column by column, and multiplying by the weights gives the Shapley value of every column of `_X`. `_shares.index.str.startswith(_column)` is True for the 0/1 columns whose name starts with the name of the column, such as `marital_status_Never-married`, and for a number column such as `age`, for the column itself. Run the cell; it takes a second or two.
    """)
    return


@app.cell
def example_shapley():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # the classifier of step one, as in work/projects/audit/model.py
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")

    # the Shapley values on the logit of one person: each weight times how far the person lies from the average
    _shares = _model.coef_[0] * (_X.loc[34982] - _X[_train].mean())
    for _column in _columns:
        print(_column, round(_shares[_shares.index.str.startswith(_column)].sum(), 3))
    print("sum", round(_shares.sum(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The classifier gives this person a probability of 0.278, a logit of −0.954, against −2.105 for the average person; the eight values add up to the difference, 1.151. `education_num` pushes up by 0.863 and `occupation` by 0.759; `marital_status` pushes down by 1.211, more than any other column. With `Married-civ-spouse` and the rest the same, the probability would be 0.818, a yes. That changed row is one the data hardly holds, with `relationship` still `Not-in-family`: 23 of the 48,842 rows are `Married-civ-spouse` and `Not-in-family`. A what-if like this changes one column and leaves the rest. Exercise 3 asks instead what this person could change.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_importance():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. importance: which columns does your classifier lean on?

        `importance(scores_of, people, column, seed)` gets a function `scores_of`, which gives the classifier's probability of a yes for every row of a table, a table `people` with a column `rich`, 1 for the people who earn more than 50,000 dollars a year and 0 for the others, the name `column` of a column to shuffle, and a `seed`. `shuffled` is a copy of `people`, made with `.copy()`, whose column `column` is shuffled: `.sample(frac=1, random_state=seed)` puts the values of the column in a random order, and `.to_numpy()` takes them without their row labels, so that they land in that new order. Replace the `...` with three lines: `right`, the share of `people` for whom the classifier is right, a yes (a probability above 0.5) when `rich` is 1 and a no when it is 0; `after`, the same for `shuffled`; and `return right - after`. The check calls `importance(scores_of, PEOPLE, "marital_status", 0)`, with `PEOPLE` the test set and `scores_of` the course's classifier, then other columns, seeds and tables. When it is right, the check saves it in your project, in `work/projects/audit/importance.py`.
        """),
        L.task("Return how much less often the classifier is right on `people` after the shuffle."),
    ])
    return


@app.cell(hide_code=True)
def exercise_importance():
    box_importance = L.code_box("importance", lines=7)
    box_importance
    return (box_importance,)


@app.cell(hide_code=True)
def check_importance(box_importance):
    result_importance = L.check("importance", box_importance.value)
    result_importance
    return (result_importance,)


@app.cell(hide_code=True)
def hints_importance():
    L.hints(
        "importance",
        [
            "Step 2 of the recipe. In step one, `report` compared each yes with `rich`. Here you ask for every person whether the classifier's yes, a probability above 0.5, matches `rich == 1`, and take the share of matches.",
            "`(scores_of(people) > 0.5) == (people[\"rich\"] == 1)` is True for every person the classifier gets right; you want the share of True, as in step one. Do the same for `shuffled`, and subtract.",
            "`after = ((scores_of(shuffled) > 0.5) == (shuffled[\"rich\"] == 1)).___()`",
        ],
        why="`right = ((scores_of(people) > 0.5) == (people[\"rich\"] == 1)).mean()`, `after = ((scores_of(shuffled) > 0.5) == (shuffled[\"rich\"] == 1)).mean()` and `return right - after`. For `marital_status` with seed 0, the share goes from 0.8344 to 0.7648, a drop of 0.0697, which the check shows as 0.070.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dependence():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. dependence: how does the answer move with one column?

        `dependence(scores_of, people, column, value)` gives the partial dependence at one value: the mean probability of a yes when everyone in `people` has `value` in the column `column`. `changed` is already a copy of `people`. Replace the `...` with two lines: give every row of `changed` the value `value` in the column `column`, and return the mean of `scores_of(changed)`. The check calls `dependence(scores_of, WOMEN, "marital_status", "Married-civ-spouse")`, with `WOMEN` the women of the test set, then the men, another value and a column of numbers. It saves your function in `work/projects/audit/dependence.py`.
        """),
        L.task("Return the mean probability of a yes with everyone at `value` in the column `column`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dependence():
    box_dependence = L.code_box("dependence", lines=5)
    box_dependence
    return (box_dependence,)


@app.cell(hide_code=True)
def check_dependence(box_dependence):
    result_dependence = L.check("dependence", box_dependence.value)
    result_dependence
    return (result_dependence,)


@app.cell(hide_code=True)
def hints_dependence():
    L.hints(
        "dependence",
        [
            "Step 3 of the recipe. In [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en) you gave every person the same value and took the mean of the model's answers. Here the answers come from `scores_of`.",
            "`changed[column] = value` puts `value` in every row of that column. Then `scores_of(changed)` gives one probability per row.",
            "`return scores_of(changed).___()`",
        ],
        why="`changed[column] = value` and `return scores_of(changed).mean()`. If every woman of the test set were `Married-civ-spouse`, their mean probability would be 0.318 instead of 0.126.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. fewest_hours: what could one person change?

        A counterfactual with one change a person can act on: the hours of work a week, with everything else the same. `fewest_hours(scores_of, person, top)` gets a table `person` with one row and gives the fewest hours a week, from 1 to `top`, at which the classifier gives that person a yes, or `None` when no number up to `top` does. `changed` is a copy of `person`, and the loop goes through `hours` from 1 to `top`. Replace the `...` with three lines: give `changed` the hours of this round in the column `hours_per_week`, ask in an `if` whether the probability of `changed` is above 0.5, and if so return `hours`. `scores_of(changed)` gives an array with one probability, and `[0]` takes it. The first number that gives a yes is the fewest, so the function can stop there. The check calls `fewest_hours(scores_of, ONE, 99)`, with `ONE` the person of the worked example, then other people and other values of `top`. It saves your function in `work/projects/audit/recourse.py`.
        """),
        L.task("Return the fewest hours a week, from 1 to `top`, that give `person` a yes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("hours", lines=8)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("hours", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "hours",
        [
            "Step 5 of the recipe. In step one, `threshold_for` went through the thresholds in a loop and kept the last one that passed. Here you stop at the first number of hours that passes, with `return`.",
            "`changed[\"hours_per_week\"] = hours` gives the copy the hours of this round. `scores_of(changed)` holds one probability, at `[0]`; compare it with 0.5, as everywhere in this lesson.",
            "`if scores_of(changed)[0] ___ 0.5:`",
        ],
        why="`changed[\"hours_per_week\"] = hours`, then `if scores_of(changed)[0] > 0.5:` and under it `return hours`. The person of the worked example would get a yes from 82 hours a week on, 32 more than now.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell explains your own classifier, the one in `work/projects/audit/model.py`, with your own functions: your `report` of step one per sex, your `importance` for every column over three seeds, your `dependence` for women and men at every value of `marital_status`, and your `fewest_hours` for 100 women and 100 men with a no, drawn at random. The Shapley values per sex the course works out itself, from your classifier's weights, as the worked example does. It takes a few seconds, and it runs again after each exercise.
    """)
    return


@app.cell(hide_code=True)
def demo_project(result_dependence, result_hours, result_importance):
    _r = (result_importance, result_dependence, result_hours)   # so that this cell runs after every check
    _audit = C.demo()
    if _audit is None:
        _out = mo.md("*Finish exercise 1 first: then your `importance` is in your project, and this cell uses it.*")
    elif isinstance(_audit, str):
        _out = mo.md(C.broken(_audit, L.lang))
    else:
        _parts = C.demo_parts(_audit, L.lang)
        _out = mo.vstack([mo.md(_p) if _kind == "md" else mo.Html(_p) for _kind, _p in _parts])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_race():
    mo.md(r"""
    ### Exercise 4. Is race unimportant?
    """)
    return


@app.cell(hide_code=True)
def choice_race_cell():
    choice_race = L.choice(
        "race",
        question="Your `importance` gives `race` −0.001: shuffle the column, and the classifier is right about as often as before. A colleague writes in the report: \"The classifier does not use race.\" What is wrong with that sentence?",
        options={
            "nothing": "Nothing: shuffling race changes nothing that counts",
            "right": "It counts right answers, not who gets pushed down",
            "seed": "Another seed might give race a larger number",
            "small": "The test set holds too few people to tell",
        },
        correct="right",
        why={
            "nothing": "The classifier does use race: it has a weight for each value, and averaged over the people marked `Other`, race pushes the logit down by 0.64, over `Black` by 0.11. Shuffling the column barely changes how often it is right, but that is not the same as not using it.",
            "right": "Permutation importance asks how much less often the classifier is right with the column shuffled. Race pushes some people down, yet moves few answers across 0.5, and 85.7% of the test set is `White`, so a shuffle mostly hands out `White` again. The Shapley values show the push: −0.64 on average for `Other`, −0.11 for `Black`, +0.03 for `White`. Unimportant for how often it is right, and not used, are different claims.",
            "seed": "With seeds 0 to 4, race gets from −0.001 to 0.000: always about 0. More seeds would not change that, because the number answers another question: how often the classifier is right, not whom it pushes down.",
            "small": "The test set holds 16,281 people, and the Shapley values show race moving the logit clearly, −0.64 on average for the people marked `Other`. The trouble is not the number of people: permutation importance answers another question, how often the classifier is right.",
        },
    )
    choice_race
    return (choice_race,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Exercise 5. The explanations in your audit report
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Write the part of your audit report about the explanations, for someone who did not take this lesson: what do the explanations show about the gap between women and men that step one measured, what can they not show, and how were they made? Use the numbers of your own project, or those of the course.",
        model_answer="Step one measured a yes rate of 0.070 for the women of the test set and 0.248 for the men, an impact ratio of 0.28, although the classifier does not get sex. The explanations show where that gap sits in the classifier. Shuffling marital_status lowers how often it is right from 0.834 to 0.765, the largest permutation importance (seed 0, test set). Averaged per sex, the Shapley values on the logit, against the people to train on, put 1.01 of the gap of 1.42 in marital_status: 15.2% of the women are Married-civ-spouse against 60.6% of the men. With everyone Married-civ-spouse, the mean probability of a yes would be 0.318 for the women and 0.381 for the men, instead of 0.126 and 0.295. Of the people with a no, 19.8% of the women and 43.9% of the men could get a yes by working more hours a week, at most 99, most of them 60 hours or more. The explanations do not show why marital status goes with income in the data of 1994, nor whether a decision may lean on it; and race, with a permutation importance of about 0, still pushes the logit down, by 0.64 on average for the people marked Other. All of this holds for this classifier on this test set; the seeds and the background are written down so that someone else can check them.",
        points=[
            "you say where the gap sits according to the explanations, with numbers: which column carries most of it, by permutation importance and by the Shapley values per sex",
            "you say what the explanations cannot show: why the data look like this, whether the classifier may decide this way, and that a column with an importance of about 0, such as race, can still push people down",
            "you say how the explanations were made, for which classifier, on which people and with which seeds or background, so that someone else can check them",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The explanations of this module, [Which features matter?](/course/open/rai-permutation-importance?lang=en), [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en), [Shapley values](/course/open/rai-shapley-values?lang=en) and [What would have to change?](/course/open/rai-counterfactuals?lang=en), each asked one question of the classifier of [Project: measure the bias of your own classifier](/course/open/rai-project-audit-bias?lang=en). [Tools for explanations](/course/open/rai-xai-tools?lang=en) showed SHAP and DiCE doing such work at scale; for a logistic regression, the Shapley values on the logit needed no tool at all. Your project `audit` now holds the functions of both steps, with `importance.py`, `dependence.py` and `recourse.py` new today. The next lesson, *Can a model explain itself?*, asks whether a language model's explanation of its own answer can be trusted.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_race, explain_report, result_dependence, result_hours, result_importance):
    L.wrap_up(result_importance, result_dependence, result_hours, choice_race, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
