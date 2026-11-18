# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-bias-lifecycle. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-lifecycle", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_overall():
    quick_overall = L.choice(
        "quick-overall",
        question="A speech app understands 75 percent of all its test recordings. Why measure it per accent as well?",
        options={
            "higher": "Per accent, the score is always higher",
            "right": "The overall score can hide a group it fails",
            "law": "The law asks for a score per accent in every app",
        },
        correct="right",
        why={
            "higher": "Per accent some scores are higher and some lower; the overall score mixes them. See the *Worked example* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).",
            "right": "In that lesson the app understood 11 of 12 recordings with accent A, but only 1 of 3 with accent C.",
            "law": "No law asks that of every app; the reason is that one score can hide a group. See the *Worked example* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).",
        },
    )
    quick_overall
    return (quick_overall,)


@app.cell(hide_code=True)
def quick_check_past():
    quick_past = L.choice(
        "quick-past",
        question="A loan model learns from ten years of old loans, recorded without mistakes, but the decisions back then were unfair. Where does its skew come from?",
        options={
            "who": "Some groups are missing from the data",
            "labels": "The people who gave the labels were careless",
            "right": "The past that the data records",
        },
        correct="right",
        why={
            "who": "Nothing says a group is missing: the data is complete. See *Exercise 2* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).",
            "labels": "The loans were recorded without mistakes. See *Exercise 2* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).",
            "right": "The data is accurate, but what it records was not fair, so a model trained on it learns the old decisions.",
        },
    )
    quick_past
    return (quick_past,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Harini Suresh and John Guttag open their framework (2021) with two simplified scenarios. In the first, a model detects heart attacks, and it misses them more often in women. More data on women who had a heart attack makes the model better for them. In the second, a model learns, with people's ratings of past applicants as its label, whether someone is suited to work as a lab technician, and it rates women lower. Here, more labeled data of the same kind does not help. The two problems looked the same, but had different sources: the first model had too few examples of women, the second learned from human ratings that were skewed. Bias can enter at every step from the world to a model in use, and only when you name the step do you know where the fix has to go.

    **An everyday example.** A thermometer hangs in the sun and reads too high. Reading it more often does not help: every new reading has the same error. You have to move the thermometer. But if it misses the cold nights because you only read it at noon, reading it more often does help.
    """)
    return


@app.cell(hide_code=True)
def predict_shades():
    guess_shades = L.guess(
        "shades",
        question="Three commercial systems that guess gender from a photo were tested on 1,270 members of parliament from six countries. For lighter-skinned men, the highest error rate was 0.8 percent. What do you think it was for darker-skinned women?",
        options={
            "same": "About 1 percent, as for lighter-skinned men",
            "few": "Between 5 and 10 percent",
            "right": "Up to about 35 percent",
        },
        reveal="**Up to 34.7 percent.** In *Gender Shades* (2018), Joy Buolamwini and Timnit Gebru found that all three systems did worst on darker-skinned women, with error rates from 20.8 to 34.7 percent. The common benchmarks had not shown this: darker-skinned women were only 7.4 percent of the images in Adience and 4.4 percent in IJB-A. That is evaluation bias, one of the seven kinds below.",
    )
    guess_shades
    return (guess_shades,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Seven places.** Suresh and Guttag follow the way from the world to a model in use, and name seven places where harm can enter. Three are in making the data: historical, representation and measurement bias. Four are in building and using the model: aggregation, learning, evaluation and deployment bias. The kinds can overlap, but naming each one, they write, makes them less confusing and easier to tackle. And the way is a loop: decisions that a model helps make change the world that the next data comes from.
        """),
        mo.Html(C.picture({"title": "Bias at every step", "heads": ["making the data", "building and using the model"],
                           "steps": [("1. The world", ["historical bias"]), ("2. The sample", ["representation bias"]),
                                     ("3. Features and labels", ["measurement bias"]),
                                     ("4. Training", ["aggregation bias", "learning bias"]),
                                     ("5. Testing", ["evaluation bias"]), ("6. Use", ["deployment bias"])],
                           "back": "decisions change the world, and so the next data",
                           "alt": "Six boxes in a loop. Making the data, down the left: 1, the world, with historical bias; 2, the sample, with representation bias; 3, features and labels, with measurement bias. Then across to building and using the model, up the right: 4, training, with aggregation bias and learning bias; 5, testing, with evaluation bias; 6, use, with deployment bias. A dashed arrow runs from 6 back to 1: decisions change the world, and so the next data."})),
        mo.md(r"""
        **Making the data.** *Historical bias* "arises even if data is perfectly measured and sampled, if the world as it is or was leads to a model that produces harmful outcomes." Word vectors learned from Google News, three million words and phrases, completed "man is to computer programmer as woman is to x" with x = homemaker (Bolukbasi and colleagues, 2016). The news recorded the world faithfully; the world was the problem. *Representation bias* means the data has too few examples of part of the people the model is used for. In a sample of the photos with a location in the image set ImageNet, around 45 percent came from the US, 1 percent from China and 2.1 percent from India (Shankar and colleagues, 2017). It happens when the data comes from other people than the ones the model is used for, when the sample is uneven, and even with a perfect sample, when a group is small. *Measurement bias* means a feature or label is a stand-in for what you care about, and measures it worse for some groups. Arrests stand in for crime, and neighborhoods that are policed more get more arrests. A US care algorithm took costs as a stand-in for health need: at the same score from the algorithm, Black patients were considerably sicker than White patients, because less money was spent on their care. Removing that gap would raise the share of Black patients getting extra help from 17.7 to 46.5 percent (Obermeyer and colleagues, 2019).

        **Building and using the model.** *Aggregation bias* comes from one model for groups that should be treated differently: it can fit no group well, or only the largest. The blood value HbA1c, widely used to diagnose and monitor diabetes, differs in complex ways across ethnicities and genders, so one model or one threshold for everyone can fit no group well. *Learning bias* comes from choices in how a model learns that make the gap between groups larger. Making a trained model smaller by pruning it, for example, can amplify the differences in performance for rare groups: with little room, the model keeps what is most frequent. *Evaluation bias* comes from test data that does not look like the people the model is used for, and from one overall score that hides a group, as in *Gender Shades*. *Deployment bias* comes from using a model differently from what it was built for. Risk tools built to predict whether someone will reoffend have been used "off-label", to help determine the length of a sentence.

        **Where the fix goes.** Each step has its own kind of fix. Against representation bias, more data helps, if it is data of the missing people. Against historical and measurement bias, more data from the same process will likely not help, Suresh and Guttag write, because it goes through the same steps. Measurement bias needs better measuring or labeling; with historical bias, the question becomes what a model should learn from an unequal world at all. Aggregation bias needs a model that takes the differences between groups into account, learning bias another objective or way of learning, evaluation bias a metric per group or a better benchmark, and deployment bias attention to how people really use the model. Suresh and Guttag give no list of fixes on purpose, so as not to suggest that a complete set of solutions exists that works everywhere.

        **Bias in people.** Three words from psychology and statistics. *Implicit bias* is attitudes and stereotypes that people hold without being aware of them: past experience influences a judgment "in a fashion not introspectively known by the actor", Anthony Greenwald and Mahzarin Banaji wrote in 1995, and the term implicit bias later came out of that research. *In-group bias* is "the tendency to judge members of one's own group (in-group) more favorably than comparable persons who are members of another group". It appears even when people watch themselves being put into a group at random: the minimal group effect, first shown by Tajfel in 1970. When people with these biases give the labels, or made the past decisions in the data, the biases enter a model as measurement bias or historical bias. *Selection bias* means the examples in the data differ systematically from the people you want to say something about, because of how they were chosen. A hiring model only learns how the people who were hired did, and never sees how the others would have done. Selection bias is one way to get representation bias, but not the only one: a perfect sample can still hold too few examples of a small group.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **historical bias** | bias that is already in the world the data comes from, so even data that is perfectly measured and sampled leads to a harmful model |
        | **representation bias** | too few examples of part of the people a model is used for, so it works worse for them |
        | **measurement bias** | a feature or label is a stand-in for what you care about, and measures it worse for some groups |
        | **aggregation bias** | one model for groups that differ in how the features relate to the outcome, so it fits none of them well |
        | **learning bias** | choices in how a model learns make the gap between groups larger |
        | **evaluation bias** | test data that does not look like the people a model is used for, so a good score hides a group |
        | **deployment bias** | a model is used differently from what it was built for |
        | **implicit bias** | attitudes and stereotypes that people hold without being aware of them |
        | **in-group bias** | judging members of your own group more favorably than comparable others |
        | **selection bias** | the examples in the data differ systematically from the people you want to say something about, because of how they were chosen |
        """),
        L.watch_out("A test of implicit bias cannot tell whether one particular person is biased: its results vary too much from one time to the next for that, Greenwald and Calvin Lai wrote in 2020. And as researchers cannot reliably reduce implicit bias, they wrote, there is no reason to expect that a training can."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find where a bias entered:

    1. Say what goes wrong, and for whom.
    2. Walk the steps from the world to the use, and ask at each one. Was the world itself unequal? Who is missing from the data, or rare? Does each feature and label measure what you care about equally well for everyone? Does one model fit every group? Do the choices in training widen a gap? Does the test data look like the people it is used for, and do you measure per group? Is the model used as it was meant to be?
    3. Count where you can: the share of each group in the data, the metric per group, who a rule picks.
    4. Put the fix at the step you named. More data of the same kind only helps against representation bias.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A care program, with patients made up by the course, after the US care algorithm above. Each patient in `_patients` has a group, a number of chronic illnesses (how sick they are) and their care costs last year, in thousands. The program takes cost as a sign of need, and helps everyone whose cost is `_limit` or more. `_helped` keeps a count per group: `_helped[_patient["group"]]` is the count of the patient's own group, and the line under the `if` adds 1 to it. Run the cell.
    """)
    return


@app.cell
def example_helped():
    # made-up patients: group, number of chronic illnesses, care costs last year in thousands
    _patients = [
        {"group": "A", "illnesses": 3, "cost": 7},
        {"group": "B", "illnesses": 5, "cost": 5},
        {"group": "A", "illnesses": 1, "cost": 2},
        {"group": "B", "illnesses": 2, "cost": 3},
        {"group": "A", "illnesses": 6, "cost": 11},
        {"group": "B", "illnesses": 6, "cost": 6},
        {"group": "A", "illnesses": 2, "cost": 5},
        {"group": "B", "illnesses": 1, "cost": 1},
        {"group": "A", "illnesses": 5, "cost": 8},
        {"group": "B", "illnesses": 3, "cost": 4},
    ]
    # the program: help everyone whose care costs were at least this much
    _limit = 6

    _helped = {"A": 0, "B": 0}
    for _patient in _patients:
        if _patient["cost"] >= _limit:
            _helped[_patient["group"]] = _helped[_patient["group"]] + 1
    print(_helped)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Group A gets three of the four places, group B one. But look at the illnesses: in both groups the patients have 1, 2, 3, 5 and 6, so the groups are equally sick. At the same illness less was spent on group B, so its costs are lower. In this made-up list, the patient of group B with 5 illnesses gets no help, and the patient of group A with 3 does. What if the program chooses by need? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_helped():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. By need

        Let the program choose by need instead of by cost: it helps everyone with at least `need` chronic illnesses. Replace the `...` with code as in the worked example, with the names without their underscore: start with `helped = {"A": 0, "B": 0}`, loop over `patients` with `patient`, and add 1 to the count of the patient's group when `patient["illnesses"]` is `need` or more. The last line is `helped`, so the check sees it. The check also tries other values of `need`.
        """),
        L.task("Work out `helped`: per group, how many patients have at least `need` chronic illnesses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_helped():
    box_helped = L.code_box("helped", lines=22)
    box_helped
    return (box_helped,)


@app.cell(hide_code=True)
def check_helped(box_helped):
    result_helped = L.check("helped", box_helped.value)
    result_helped
    return (result_helped,)


@app.cell(hide_code=True)
def hints_helped():
    L.hints(
        "helped",
        [
            "Start from the worked example: the loop and the counting line stay the same. Only the `if` changes: it looks at the illnesses, not at the cost.",
            "The `if` compares `patient[\"illnesses\"]` with `need`. Which comparison means *or more*?",
            "`if patient[\"illnesses\"] ___ need:`, and under it the counting line of the worked example, without underscores.",
        ],
        why="`helped = {\"A\": 0, \"B\": 0}`, then `for patient in patients:` with `if patient[\"illnesses\"] >= need:` under it, and inside that `helped[patient[\"group\"]] = helped[patient[\"group\"]] + 1`. That gives `{'A': 2, 'B': 2}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ### Exercise 2. Which kind of bias?

    Six cases from the first half of the way, where the data is made. Sort each by the kind of bias in it.
    """)
    return


@app.cell(hide_code=True)
def sort_steps_cell():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps",
        [
            "Ask of each case: is the data right while the world itself was unequal, is a group missing or rare, or does a label measure something worse for one group?",
            "Another city is about who the data comes from. Which two cases are about a label that stands in for something else?",
            "Historical bias: pay and \\_\\_\\_. Representation bias: another city and \\_\\_\\_. Measurement bias: \\_\\_\\_ and inspections.",
        ],
        why="**Historical bias:** pay and CEOs; the data is right, but the world was unequal. **Representation bias:** another city and five percent. **Measurement bias:** diagnoses and inspections; the label measures the real thing worse for one group or one place.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fix():
    mo.md(r"""
    ### Exercise 3. More of the same?
    """)
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    choice_fix = L.choice(
        "fix",
        question="A colleague wants to fix the care program of the worked example by collecting twice as many patients in the same way, with the same rule on cost. What do you think?",
        options={
            "more": "Yes, more data means less bias",
            "right": "No, cost still measures need badly for B",
            "share": "Only if half of the new patients are in group B",
        },
        correct="right",
        why={
            "more": "New patients collected in the same way go through the same rule, and at the same illness group B still costs less. More data only helps against representation bias. See *Where the fix goes* in *The idea*.",
            "right": "This is measurement bias: the label, cost, is a stand-in that measures need worse for group B. The fix is in the label, as in Exercise 1, not in more data.",
            "share": "Group B is already half of the patients: it is not missing. Its costs are lower at the same illness, and more of the same data keeps that. See *Where the fix goes* in *The idea*.",
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en) you looked for skew in three places: who is in the data, who gave the labels, and the past the data records. Here they became historical, representation and measurement bias, with four more places while the model is built and used, and a rule: name the step, and put the fix there. In [The machine learning workflow](/course/open/ml-workflow?lang=en) each round went back to the step that holds the model back; with bias, you go back to the step where it came in. The next lesson, *Feedback loops*, follows the dashed arrow in the picture: how the decisions of a model change the data it learns from next.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, result_helped, result_steps):
    L.wrap_up(result_helped, result_steps, choice_fix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
