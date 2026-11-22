# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-counterfactuals. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-counterfactuals", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Reasons for a decision](/course/open/rai-reason-codes?lang=en) and one about [Shapley values](/course/open/rai-shapley-values?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reason():
    quick_reason = L.choice(
        "quick-reason",
        question="Against the 1,608 applicants just above the cut-off, education was the principal reason for the no of the applicant aged 30, with a gap of −1.344. What does that reason tell the applicant?",
        options={
            "amount": "How many more levels would give a yes",
            "right": "What weighed most against them",
            "only": "That education alone decided the no",
        },
        correct="right",
        why={
            "right": "Right: a reason measures the applicant against a reference group and names what weighed most. It gives no amount: how much would have to change for a yes is another question, and this lesson's.",
            "amount": "A reason is a gap against a reference group, −1.344 here, in units of the logit. It says what weighed most, not how much would have to change: that is a counterfactual question, this lesson's. See [Reasons for a decision](/course/open/rai-reason-codes?lang=en).",
            "only": "Age and hours weighed against too, −0.789 and −0.302 against the same group; education weighed most. See [Reasons for a decision](/course/open/rai-reason-codes?lang=en).",
        },
    )
    quick_reason
    return (quick_reason,)


@app.cell(hide_code=True)
def quick_check_direction():
    quick_direction = L.choice(
        "quick-direction",
        question="In the last lessons, education_num got a Shapley value of +0.181 for the person aged 41. Does that say that one more level of education would raise the person's chance?",
        options={
            "more": "Yes: a positive value means more is better",
            "step": "Yes, by 0.181 for each extra level",
            "right": "No: it says nothing about a change",
        },
        correct="right",
        why={
            "right": "Right. The value shares out the gap between this person's chance and the background's average; it does not say what a change would do. What would change the answer is this lesson's question.",
            "more": "A positive Shapley value does not mean that raising the feature raises the prediction: it shares out the gap between this person and a background. See [Shapley values](/course/open/rai-shapley-values?lang=en).",
            "step": "+0.181 is education_num's share of the gap between this person's chance, 0.451, and the background's average, 0.225; it is no rate per level. See [Shapley values](/course/open/rai-shapley-values?lang=en).",
        },
    )
    quick_direction
    return (quick_direction,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The applicant aged 30 from [Reasons for a decision](/course/open/rai-reason-codes?lang=en) got a no: the bank's logistic regression gives a logit of −2.315, a chance of yes of 0.090. The reasons told the applicant what weighed most against them. The applicant's next question is another one: what would I have to change to get a yes? Sandra Wachter, Brent Mittelstadt and Chris Russell proposed to answer exactly that, without opening the model: "You were denied a loan because your annual income was £30,000. If your income had been £45,000, you would have been offered a loan." Such a sentence is easy to follow and needs nothing from the model but its answers. But there are many such sentences for one person, which one you get depends on how you measure a change, and some ask for changes nobody can make.

    **An everyday example.** A web shop says: "Add €12 to your basket and shipping is free." It does not tell you how it works out shipping, only a small change that gets you the other outcome. Whether you want to buy €12 more is up to you, and next month the threshold may be different.
    """)
    return


@app.cell(hide_code=True)
def predict_closest():
    guess_closest = L.guess(
        "closest",
        question="The applicant changes one column. Measured in steps of each column's typical spread, as Wachter et al. suggest, which single change that gives a yes lies closest?",
        options={
            "age": "Being 81 instead of 30",
            "education": "Education level 16 instead of 9",
            "hours": "96 hours a week instead of 40",
        },
        reveal="**Being 81.** In steps of each column's median absolute deviation, 10 years, 1 level and 3 hours, 51 more years is 5.1 steps, 7 more levels of education 7 steps, and 56 more hours 18.7 steps. The closest counterfactual asks the applicant to be 51 years older: true of the model, of no use to the applicant. *The idea* shows where these numbers come from.",
    )
    guess_closest
    return (guess_closest,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A counterfactual explanation.** A **counterfactual explanation** says how the input would have had to differ for the model to give another answer: if this had been that, the answer would have been yes. In Wachter et al.'s words, counterfactuals "describe a dependency on the external facts that led to that decision." It asks only for the model's answers, so it works for any model. It is not unique: "Multiple counterfactuals are possible, as multiple desirable outcomes can exist", and the one they prefer is the "closest possible world" (all p. 6).

        **Finding one.** For a model with gradients, such as their neural network, Wachter et al. search: they move a row by gradient descent until the model gives the wanted answer while the row stays close to the person, and different starting rows end in different counterfactuals. For a logistic regression no search is needed. The bank says yes from a logit of 0. Changing column $k$ alone by an amount $\Delta_k$ moves the logit $z$ by $w_k \Delta_k$, so the logit reaches 0 when

        $$\Delta_k = -\frac{z}{w_k}.$$

        For the applicant aged 30, $z = -2.315$, and education_num, with weight 0.345, needs 6.71 more levels. You write this in exercise 1.

        **Close in what?** To weigh a change in years against a change in levels and in hours, each column needs a scale. Wachter et al.: "we suggest use of the L1 norm, or Manhattan distance, weighted by the inverse median absolute deviation" (p. 17). The **median absolute deviation** (MAD) of a column is the median of how far its values lie from their median, and the distance adds each column's change in steps of its MAD:

        $$d(x, x') = \sum_k \frac{|x_k - x'_k|}{\mathrm{MAD}_k}.$$

        In UCI Adult the MADs are 10 years of age, 1 level of education and 3 hours a week. You write this distance in exercise 2. Another distance can give another answer, in Wachter et al.'s own experiment too: with squared distances their counterfactuals changed the grade point average far more readily than the exam score, "and this is down to our choice of distance function" (p. 19). Solon Barocas, Andrew D. Selbst and Manish Raghavan call scaling by the data "somewhat arbitrary": one lender might count 5,000 dollars more income a year as worth one more year on the job, and a competing lender, with other training data, 10,000 dollars (p. 7). The slider below changes one such scale.

        **Changes nobody can make.** The closest counterfactual may change what a person cannot change. In Wachter et al.'s experiment on law-school data, with a plain squared distance, race, coded 0 or 1, first got meaningless values, fractional or negative; after they held it to 0 or 1 in two runs: "The algorithm now suggests always changing the race to 'white' as part of the counterfactual." (p. 19). Berk Ustun, Alexander Spangher and Yang Liu ask for **recourse** instead, now often called **algorithmic recourse**: "the ability of a person to change the decision of a model by altering actionable input variables (e.g., income vs. age or marital status)" (p. 1). Amir-Hossein Karimi and colleagues sort the columns into three kinds: actionable, such as a bank balance; mutable but not actionable, such as a credit score; and immutable, such as a birthplace (p. 5). Age changes, but not by anything the applicant does. A counterfactual offers recourse when it changes only actionable columns, which in DiCE you ask for with `features_to_vary`.

        **Many answers.** There "may exist zero, one, or multiple" best counterfactuals (Karimi et al., p. 7), and the one a method finds depends on the distance, where the search starts and the random seed. Which one a bank shows is a choice as well: Karimi et al. warn that someone could pick "the most socially acceptable explanation out of many equally plausible ones" (p. 10).

        **What it does not show.** A counterfactual says what would change the answer of this model for this person, and no more. "Counterfactuals which do not modify a protected attribute cannot be used as evidence that the attribute was irrelevant to the decision." (Wachter et al., p. 15). And it holds only while the model stays the same. Kaivalya Rawal, Ece Kamar and Himabindu Lakkaraju gave recourse under one model, retrained it on newer data and counted how many recourses the new model rejected: with bail data from 1978 and 1980, 96.6% of the recourses from Ustun et al.'s method for a logistic regression. Across models, methods and shifts the share ran from 0% to 100%, with 0% or close to it for several tree and neural models on the same bail data (p. 7).

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **counterfactual explanation** | how the input would have had to differ for the model to give another answer, often the closest such change, by a chosen distance |
        | **algorithmic recourse** | whether a person can change a model's decision by changing what they can act on, such as income, not age |
        | **actionable** | a column the person can change by acting, such as hours of work; not age or a birthplace |
        | **median absolute deviation** (MAD) | the median of how far a column's values lie from their median: a typical step in that column |
        | $z$ | the applicant's logit; the bank says yes from $z = 0$ |
        | $\Delta_k$ | how much column $k$ alone has to change to bring the logit to 0 |
        | $d(x, x')$ | Wachter et al.'s distance: each column's change in steps of its MAD, added up |
        """),
        L.watch_out("A counterfactual explanation is true of one model, one distance and one set of columns that may change. Say which columns could change and how you measured close, and never read a column it leaves alone as a column that did not matter."),
    ])
    return


@app.cell(hide_code=True)
def explore_years_cell():
    explore_years = L.explore("years", "years of age in one step", lambda v: C.years_picture(v, "en"), start=10,
                              values=C.YEARS, role="hyper",
                              task="Move the slider from 10, the MAD of age, down to 7 and then to 1. With how many years in a step is being 81 no longer the closest of the three?")
    mo.vstack([
        mo.md("**How much is a year?**"),
        explore_years,
        mo.md(r"""
        Each bar is the distance from the applicant aged 30 to one counterfactual that changes one column by the smallest whole number that gives a yes. Education and hours keep their MADs, 1 level and 3 hours; the slider sets how many years of age make one step. Of these three, being 81 is closest with 8 or more years in a step, and education level 16 with 7 or fewer. Nothing about the applicant or the model changed, only how a year is weighed against a level of education. With the standard deviations as scales instead, 13.71 years, 2.57 levels and 12.39 hours, education level 16 is closest, at 2.72 steps against 3.72 for being 81.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To give a counterfactual explanation:

    1. Write down the answer you want: here a logit of 0 or more, the bank's yes.
    2. Say which columns may change. For recourse, only the actionable ones.
    3. Choose a scale for each column, such as its MAD in the data, and say which.
    4. Find changes that give the wanted answer: for a logistic regression, $-z / w_k$ for one column; for any model, a search.
    5. Measure each with the distance, and show the closest few, not only one.
    6. Say what it does not show: a column it leaves alone may still have counted, and a new model can answer differently.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The model and the applicant of [Reasons for a decision](/course/open/rai-reason-codes?lang=en), with Wachter et al.'s distance. `np.median(..., axis=0)` gives the median of each column and `np.abs` the size of each number. `math.ceil` rounds up to a whole number, because ages, levels and hours are whole numbers in the table, and `.copy()` makes a new array, so the applicant's own row stays as it is. Run the cell.
    """)
    return


@app.cell
def example_counterfactuals():
    import math

    import numpy as np
    import pandas as pd

    # the three columns of UCI Adult, the model of the lesson on reasons, and its applicant aged 30
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _weights = np.array([0.046, 0.345, 0.042])
    _bias = -8.48
    _person = np.array([30, 9, 40])
    _logit = _person @ _weights + _bias
    print("logit", round(_logit, 3), " chance of yes", round(1 / (1 + math.exp(-_logit)), 3))

    # the scale of Wachter et al.: the median absolute deviation (MAD) of each column
    _mad = np.median(np.abs(_X - np.median(_X, axis=0)), axis=0)
    print("MAD", _mad)

    # one column at a time: the smallest whole change that brings the logit to 0, and its distance
    for _k in range(3):
        _other = _person.copy()
        _other[_k] = _person[_k] + math.ceil(-_logit / _weights[_k])
        _new = _other @ _weights + _bias
        _distance = (np.abs(_other - _person) / _mad).sum()
        print(_names[_k], "to", _other[_k], " logit", round(_new, 3), " distance", round(_distance, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The closest counterfactual is being 81, at 5.1 steps. Why age? Per step of its scale, each column moves the logit by its weight times its MAD: age by 0.046 · 10 = 0.46, education by 0.345 · 1 = 0.345 and hours by 0.042 · 3 = 0.126. Age moves the logit most per step, so spending every step on age gets there in the fewest steps, and no mix of columns comes closer: a mix spends some of its steps on a column that moves the logit less. But age is not actionable. Leave it out, and education level 16, a doctorate for someone with a high-school diploma, is closest, at 7 steps. Further away lie, among others, level 15 with 46 hours a week (8 steps), level 14 with 55 hours (10), level 13 with 63 hours (11.7) and 96 hours a week alone (18.7). Each is a true counterfactual of this model. Which one the bank sends is a choice.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_needed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How much has to change?

        Write `change_needed(weights, bias, person, k)`: `weights` and `person` are lists with a number for each column, `bias` is a number and `k` is the position of a column. Start `logit` at `bias` and, for each column `j`, add `weights[j] * person[j]`. Return how much column `k` alone has to change to bring the logit to 0: minus the logit divided by `weights[k]`. The check first calls `change_needed([0.046, 0.345, 0.042], -8.48, [30, 9, 40], 1)`: the applicant aged 30 needs 6.71 more levels of education. The picture runs your function for each column of that applicant.
        """),
        L.task("Write `change_needed(weights, bias, person, k)`: work out the logit, then return how much column `k` alone has to change to bring it to 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_needed():
    box_needed = L.code_box("needed", lines=6)
    box_needed
    return (box_needed,)


@app.cell(hide_code=True)
def check_needed(box_needed):
    result_needed = L.check("needed", box_needed.value)
    result_needed
    return (result_needed,)


@app.cell(hide_code=True)
def hints_needed():
    L.hints(
        "needed",
        [
            "The logit is a sum that starts at the bias. Changing column `k` by some amount moves the logit by `weights[k]` times that amount, and the logit has to move by minus itself to reach 0.",
            "Start with `logit = bias`, and add `weights[j] * person[j]` inside `for j in range(len(person)):`. After the loop, the change is minus the logit, divided by the weight of column `k`.",
            "`return ___ / weights[k]`",
        ],
        why="`logit = bias`, then `for j in range(len(person)):` with `logit = logit + weights[j] * person[j]`, and after the loop `return -logit / weights[k]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How far is a counterfactual?

        Write `distance(person, other, scale)`, Wachter et al.'s distance: `person` and `other` are two rows, and `scale` has a number for each column. Start `total` at 0 and, for each column `k`, add the size of its change, `abs(other[k] - person[k])`, divided by `scale[k]`; `abs` is Python's absolute value, the number without its minus sign. Return `total`. The check first calls `distance([30, 9, 40], [30, 15, 46], [10, 1, 3])`: education level 15 with 46 hours a week lies 0 + 6 + 2 = 8 steps from the applicant. The picture runs your function on six counterfactuals of the applicant, with the MADs as scale.
        """),
        L.task("Write `distance(person, other, scale)`: add up, for each column `k`, the size of its change divided by `scale[k]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_distance():
    box_distance = L.code_box("distance", lines=6)
    box_distance
    return (box_distance,)


@app.cell(hide_code=True)
def check_distance(box_distance):
    result_distance = L.check("distance", box_distance.value)
    result_distance
    return (result_distance,)


@app.cell(hide_code=True)
def hints_distance():
    L.hints(
        "distance",
        [
            "A change down is as far as a change up, and a year is not a level: take the size of each change, then count it in steps of its own column's scale.",
            "Start with `total = 0` and loop with `for k in range(len(person)):`. Each time, add the size of the change, `abs(other[k] - person[k])`, divided by that column's scale. Return `total` after the loop.",
            "`total = total + abs(other[k] - person[k]) / ___`",
        ],
        why="`total = 0`, then `for k in range(len(person)):` with `total = total + abs(other[k] - person[k]) / scale[k]`, and after the loop `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_race():
    mo.md(r"""
    ### Exercise 3. A column it leaves alone

    A counterfactual names one way to another answer. What does it say about the columns it does not change?
    """)
    return


@app.cell(hide_code=True)
def choice_race_cell():
    choice_race = L.choice(
        "race",
        question="A bank's model uses race among its columns. The counterfactual it sends an applicant changes only education. Does that show that race played no part in the no?",
        options={
            "right": "No, it says nothing about race",
            "closest": "Yes: otherwise the closest change would be in race",
            "distance": "Yes, if the bank used Wachter et al.'s distance",
        },
        correct="right",
        why={
            "right": "Right. A column a counterfactual leaves alone may still have counted; the closest change only goes where a step moves the answer most. Wachter et al. say so themselves: \"Counterfactuals which do not modify a protected attribute cannot be used as evidence that the attribute was irrelevant to the decision.\" Whether race counts is a question about many decisions, not one explanation.",
            "closest": "The closest change goes where a step moves the logit most, not to the column that weighed most against the applicant. For the applicant aged 30 the closest counterfactual changes only age, while education and hours weighed against the applicant too. Wachter et al.: a counterfactual that leaves a protected attribute alone is no evidence that it was irrelevant (p. 15).",
            "distance": "The distance decides which counterfactual is closest, not whether race mattered. A counterfactual that leaves race alone says nothing about it, whatever the distance (Wachter et al., p. 15). See *What it does not show* in *The idea*.",
        },
    )
    choice_race
    return (choice_race,)


@app.cell(hide_code=True)
def text_exercise_later():
    mo.md(r"""
    ### Exercise 4. After the model changes

    A counterfactual is a statement about one model. Banks retrain their models.
    """)
    return


@app.cell(hide_code=True)
def choice_later_cell():
    choice_later = L.choice(
        "later",
        question="The applicant gets education level 16, the recourse the bank named. Meanwhile the bank has retrained its model on newer data. Does the applicant now get a yes?",
        options={
            "promised": "Yes: the bank's counterfactual promised it",
            "right": "Not for sure: the new model may say no",
            "cutoff": "Yes, as long as the cut-off is still 0",
        },
        correct="right",
        why={
            "right": "Right. A counterfactual is true of the model that made it. Rawal, Kamar and Lakkaraju retrained models on newer data and counted how many recourses the new model rejected: with bail data from 1978 and 1980, 96.6% of the recourses from Ustun et al.'s method for a logistic regression, though for several tree and neural models it was 0% or close to it. Barocas et al. count a model that is stable over time among the hidden assumptions of counterfactual explanations.",
            "promised": "The counterfactual is a statement about the old model: with level 16, that model says yes. A retrained model has other weights. With bail data from 1978 and 1980, Rawal, Kamar and Lakkaraju found that a new logistic regression rejected 96.6% of the recourses from Ustun et al.'s method.",
            "cutoff": "The cut-off can stay at 0 while the weights change: a retrained model has other weights, so the same row gets another logit. With bail data from 1978 and 1980, Rawal, Kamar and Lakkaraju found that a new logistic regression rejected 96.6% of the recourses from Ustun et al.'s method.",
        },
    )
    choice_later
    return (choice_later,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Reasons for a decision](/course/open/rai-reason-codes?lang=en) told the applicant what weighed most against them; a counterfactual says what would change the answer, and the two need not name the same column. [Shapley values](/course/open/rai-shapley-values?lang=en) shared one prediction out over its features; a counterfactual looks for another prediction nearby. DiCE from [Tools for explanations](/course/open/rai-xai-tools?lang=en) searches for several counterfactuals at once, for any model, and its `features_to_vary` is how you ask it for recourse. The next lesson, [Project: explain your classifier](/course/open/rai-project-audit-explain?lang=en), puts this lesson and the ones before it to work on a model of your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_later, choice_race, result_distance, result_needed):
    L.wrap_up(result_needed, result_distance, choice_race, choice_later)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
