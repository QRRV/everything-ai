# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-bias-in-data. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-in-data", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The machine learning workflow](/course/open/ml-workflow?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_again():
    quick_again = L.choice(
        "quick-again",
        question="In which step of the workflow do you look at the examples a model gets wrong?",
        options={
            "baseline": "4\\. Baseline",
            "metric": "2\\. Metric",
            "right": "6\\. Round again",
        },
        correct="right",
        why={
            "baseline": "The baseline is the simplest prediction to compare with; it comes before the model. See *The idea* in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "metric": "The metric is agreed before there is a model at all. See *The idea* in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "right": "After the model: you look at what it gets wrong, and go back to the step that holds it back.",
        },
    )
    quick_again
    return (quick_again,)


@app.cell(hide_code=True)
def quick_check_goal():
    quick_goal = L.choice(
        "quick-goal",
        question="Why do you agree on the metric and the goal before you see any results?",
        options={
            "right": "So that they cannot bend towards the model you happen to have",
            "fast": "Because training goes faster with a goal",
            "train": "Because the model is trained on the metric, so it has to be there before training starts",
        },
        correct="right",
        why={
            "fast": "The goal does not change how training runs. What can go wrong if you choose it afterwards? See the watch-out in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "train": "A model is trained on its loss, not on the metric you judge it by. What can go wrong if you choose the metric afterwards? See the watch-out in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "right": "Chosen afterwards, a metric and a goal bend towards the results, as a test score does when you choose with the test set.",
        },
    )
    quick_goal
    return (quick_goal,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A score is a mean over a test set, and a test set is made of people, photos or recordings that someone collected. In 2018, Joy Buolamwini and Timnit Gebru tested three commercial systems that guess a person's gender from a photo of their face. The systems were wrong about at most 0.8% of the lighter-skinned men, and up to 34.7% of the darker-skinned women. One overall score said little about that. A model learns from its data, so it also learns who is missing from it, and the views of the people who collected and labeled it. This lesson is about that kind of bias, **harmful bias**: when a system treats groups unfairly. It has nothing to do with the bias term $b$ of a model.

    **An everyday example.** A shoe shop that only stocks sizes 38 to 44 serves most of its customers. Someone with size 35 leaves with nothing, and never shows up in the shop's figures.
    """)
    return


@app.cell(hide_code=True)
def predict_hidden():
    guess_hidden = L.guess(
        "hidden",
        question="A face detector finds 94% of the faces in its test set. 90% of those faces are of group A, 10% of group B. What do you think is the lowest share of the faces of group B it could find?",
        options={
            "close": "Not much below 94%: the overall score is a mean over all faces",
            "eighty": "About 80%",
            "right": "40%",
        },
        reveal="**40%.** If it found every face of group A, the mean over all faces would be 0.9 × 1 + 0.1 × 0.4 = 0.94. A group of 10% weighs only a tenth in the overall score, so even a bad result for it hardly moves the number. The detector of this lesson finds about 98% of group A and only 60% of group B, and still 0.9 × 0.98 + 0.1 × 0.6 ≈ 0.94 of all faces. You only see it when you measure per group.",
    )
    guess_hidden
    return (guess_hidden,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Three questions before you trust a score:

        - **Who could be harmed?** **Harm** is a bad effect of an AI system on people: a loan refused unfairly, a face a camera does not find, a group shown as a stereotype. A **stakeholder** is anyone who has an interest in the system or is affected by it: the people who use it, the people it decides about, even when they never see it, and the people who build or buy it.
        - **Who is in the data, and who is missing?** **Data bias** is a skew in a dataset: some groups are missing, too rare, or described in a distorted way, and a model trained on it learns the same skew. It can come from who was measured, from the people who gave the labels, and from the past that the data records.
        - **For whom does the score hold?** An accuracy over two groups, A and B, is a mean of their accuracies, weighted by their size:
        """),
        L.formula(r"\text{accuracy} = \frac{n_A}{n} \cdot \text{accuracy}_A + \frac{n_B}{n} \cdot \text{accuracy}_B"),
        mo.md(r"""
        A small group weighs little, however badly the model does for it. So you measure per group as well, as a confusion matrix shows each class on its own row.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **harm** | | a bad effect of an AI system on people, such as an unfair decision, not being recognized, or being shown as a stereotype | |
        | **stakeholder** | | anyone who has an interest in an AI system or is affected by it: users, the people it decides about, makers and buyers | |
        | **data bias** | | a skew in a dataset: groups missing, too rare or described in a distorted way, so that a model learns the same skew | |
        | **harmful bias** | | when a system treats groups unfairly; not the bias term $b$ | |
        | $n_A$ | n A | the number of examples of group A in the test set | `len(results[results["group"] == "A"])` |
        """),
        L.watch_out("Leaving a column such as gender or ethnicity out of the features does not remove a skew. Other columns, such as a postcode or a first name, can stand in for it, and the model learns the skew through them."),
        mo.Html(C.picture({"share": "Share of the test set", "share_y": "share", "acc": "Faces found", "acc_y": "accuracy", "all": "all",
                           "alt": "Left, the share of each group in the test set of a face detector: group A 0.9, group B 0.1. Right, the share of faces it finds: about 0.98 for group A, 0.6 for group B, and 0.94 for all faces together."})),
        mo.md(r"""
        The face detector of the guess, in numbers: 180 faces of group A and 20 of group B. The bar for all faces, on the right, is almost as high as the one for group A, because group A is nine tenths of the test set.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Before you trust a score:

    1. List the stakeholders: who uses the system, who does it decide about, and who is affected without using it?
    2. Ask what harm a mistake can do, and to whom.
    3. Count who is in the data, per group, and ask who is missing.
    4. Ask who measured and who gave the labels, and how.
    5. Measure the metric per group, not only overall.
    6. Write down what you found, next to the score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The face detector of the picture: its results on 200 test faces, 1 for a face it found and 0 for one it missed. `pd.DataFrame({...})` makes a table from a dict: each key becomes the name of a column, and its list the values in that column, one per row. `value_counts()` divided by the number of rows gives the share of each value. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_groups():
    import pandas as pd

    # 1. stakeholders: the people whose faces it should find, and everyone who relies on it
    # 2. harm: a face it misses is a person the system does not see
    _groups = ["A"] * 180 + ["B"] * 20
    _found = [1] * 176 + [0] * 4 + [1] * 12 + [0] * 8
    _results = pd.DataFrame({"group": _groups, "found": _found})
    # 3. who is in the data: the share of each group
    print(_results["group"].value_counts() / len(_results))
    # 5. the score overall, and per group
    print(_results["found"].mean())
    for _group in ["A", "B"]:
        print(_group, _results[_results["group"] == _group]["found"].mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Group A is 0.9 of the test set and group B 0.1. Overall, the detector finds 0.94 of the faces: 188 of 200. For group A that is about 0.98, for group B only 0.6. Step 4 is not in the code: who took these photos, and who decided which faces count as found? Step 6 is a sentence next to the score: "finds 94% of all faces, but only 60% of the faces of group B, which is 10% of the test set".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accents():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Per accent

        A speech app was tested on 20 recordings: 12 of people with accent A, 5 with accent B and 3 with accent C. `results` has a row per recording, with its accent and 1 in `correct` if the app understood it. Replace the `...` with one line: add the accuracy of this accent to `accuracies`, with a mask as in the worked example. The last line is `accuracies`, so the check sees it. The check also tries other results.
        """),
        L.task("Put in `accuracies` the accuracy of the app for accent A, B and C, in that order."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accents():
    box_accents = L.code_box("per-accent", lines=10)
    box_accents
    return (box_accents,)


@app.cell(hide_code=True)
def check_accents(box_accents):
    result_accents = L.check("per-accent", box_accents.value)
    result_accents
    return (result_accents,)


@app.cell(hide_code=True)
def hints_accents():
    L.hints(
        "per-accent",
        [
            "Step 5 of the recipe, as in the worked example: per group, the mean of `correct` over the rows of that group.",
            "Which mask keeps only the rows of one accent? And how does a value go onto the end of a list?",
            "`accuracies.append(results[___][\"correct\"].___())`.",
        ],
        why="`results[results[\"accent\"] == accent]` keeps the recordings of one accent, and the mean of their `correct` is the share the app understood: 11 of 12 for A, 3 of 5 for B and 1 of 3 for C. Overall it understands 15 of 20, 0.75, which hides that it gets most recordings of accent C wrong.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sources():
    mo.md(r"""
    ### Exercise 2. Where does the skew come from?

    Six cases of data bias. Put each one with the place its skew comes from: who is in the data, who gave the labels, or the past that the data records.
    """)
    return


@app.cell(hide_code=True)
def sort_sources_cell():
    sort_sources = L.sort("sources")
    sort_sources
    return (sort_sources,)


@app.cell(hide_code=True)
def check_sources(sort_sources):
    result_sources = L.check("sources", sort_sources)
    result_sources
    return (result_sources,)


@app.cell(hide_code=True)
def hints_sources():
    L.hints(
        "sources",
        [
            "Ask for each case: is a group missing or rare, did people judge and label it, or does the data faithfully record an unfair past?",
            "Two cases are about who could end up in the data at all. Which two are about a judgement that people wrote down?",
            "Who is in the data: the celebrity photos and \\_\\_\\_. Who gave the labels: rude or not and \\_\\_\\_. The past: \\_\\_\\_ and the old loans.",
        ],
        why="**Who is in the data:** the celebrity photos and the volunteers. **Who gave the labels:** rude or not, and the pain scores. **The past:** ten years of hires and the old loans; there the data is accurate, but what it records was not fair.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stakeholders():
    mo.md(r"""
    ### Exercise 3. Who has a stake?
    """)
    return


@app.cell(hide_code=True)
def choice_stakeholders_cell():
    choice_stakeholders = L.choice(
        "stakeholders",
        question="A school uses a model that flags students who might drop out, from their attendance and grades, so that a mentor can talk to them. Is a student the model does not flag one of its stakeholders?",
        options={
            "unseen": "No: that student never sees the model or what it says",
            "right": "Yes: a student it misses gets no help",
            "silent": "No: the model only decides about the students it flags",
        },
        correct="right",
        why={
            "unseen": "A stakeholder does not have to see the system. What happens to a student who needs help and is not flagged? See *The idea*.",
            "silent": "The model looks at every student, and not flagging one is a decision too. What happens to a student it misses? See *The idea*.",
            "right": "The model decides about every student, even when it stays silent. The school and its mentors, the students it flags, their parents and the makers have a stake too.",
        },
    )
    choice_stakeholders
    return (choice_stakeholders,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Where does this data come from?](/course/open/data-provenance?lang=en) you wrote down the limits of a dataset, who or what is missing; in [The machine learning workflow](/course/open/ml-workflow?lang=en) you looked at what a model gets wrong in step 6. Here you also ask for whom it goes wrong, and who is not in the data at all. These questions come back whenever a model decides about people. In the next lesson, *A neuron*, you take the first step towards neural networks.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stakeholders, result_accents, result_sources):
    L.wrap_up(result_accents, result_sources, choice_stakeholders)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
