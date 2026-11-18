# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-compas-amazon. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-compas-amazon", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) and [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_harm():
    quick_harm = L.choice(
        "quick-harm",
        question="A bank's model says no to loans more often for people from one neighborhood than for people with the same income from another. Which kind of harm is that?",
        options={
            "representational": "Representational harm",
            "right": "Allocative harm",
            "none": "No harm, as long as most answers are right",
        },
        correct="right",
        why={
            "representational": "Representational harm is about how a system shows or describes groups. This model decides who gets a loan. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "The model hands out loans, and one group gets fewer of them than it should: allocative harm.",
            "none": "A model can be right most of the time and still say no more often to one group. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_harm
    return (quick_harm,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="A spam filter is tested on 100 mails. Which share is its precision?",
        options={
            "right": "Of the mails it calls spam, the share that is spam",
            "recall": "Of the mails that really are spam, the share it calls spam",
            "accuracy": "Of all the mails, the share it gets right",
        },
        correct="right",
        why={
            "right": "Precision is $TP / (TP + FP)$: of what the filter calls positive, the share that really is positive.",
            "recall": "That is recall, $TP / (TP + FN)$. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "accuracy": "That is accuracy, which counts every mail. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2016 and 2018, two reports showed what a close look at a model's mistakes can reveal. One was about a risk score that American courts use, the other about a hiring model at Amazon. They are still the cases people bring up first, and this lesson goes through both, with the numbers.

    **An everyday example.** A store detective checks shoppers' bags. Ask *what share of the checks finds something*, and the detective looks equally fair to every group. Ask *what share of the honest shoppers was checked*, and one group may turn out to be stopped far more often. The two questions can give different answers about the same detective.
    """)
    return


@app.cell(hide_code=True)
def predict_again():
    guess_again = L.guess(
        "again",
        question="ProPublica followed 7,214 defendants in Broward County, Florida, for two years. COMPAS had rated about 3,300 of them as medium or high risk. What share of those do you think were arrested again within two years?",
        options={
            "most": "About 9 in 10",
            "right": "About 6 in 10",
            "few": "About 3 in 10",
        },
        reveal="**About 6 in 10**: of the 3,317 people rated medium or high risk, 2,035 were arrested again, 0.61. So about four in ten of the people the score flagged were not.",
    )
    guess_again
    return (guess_again,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **COMPAS** is a program of the company Northpointe, now Equivant, that gives a defendant a **risk score** from 1 to 10. Broward County in Florida uses it mainly to decide whether someone may wait for trial at home or stays in jail. Scores 1 to 4 count as low, 5 to 7 as medium and 8 to 10 as high.

        In 2016, journalists at ProPublica collected the scores of the people scored in Broward County in 2013 and 2014, and checked who was arrested again within two years. Call *rated medium or high* positive, or *flagged*, and each group gets a confusion matrix, as in the picture. Then two questions give two answers.

        - *Of the people the score flagged, what share was arrested again?* That is the precision of each group. It is almost the same for Black and White defendants, as the worked example shows, and Northpointe said that this makes the score fair: a high score means the same in both groups.
        - *Of the people who were not arrested again, what share did the score flag anyway?* You work that out in Exercise 1. ProPublica found a share almost twice as large for Black defendants, and called the score biased.

        Both answers are true at once. In this data, 0.51 of the Black defendants and 0.39 of the White defendants were arrested again: arrested, not proven to have committed a crime, and who gets arrested also depends on where the police look. When those shares differ, a score that makes mistakes cannot have the same precision, the same share of wrong flags and the same share of misses in both groups; a lesson in the module on fairness shows why.

        **Amazon** started in 2014 on a model that gave job applicants one to five stars. It learned from patterns in the CVs sent to the company over ten years, and most of them came from men: in effect, Reuters wrote, the model taught itself that male candidates were preferable. By 2015 the team saw that the model marked down CVs with the word *women's*, as in *women's chess club captain*, and graduates of two colleges for women only. The team made the model ignore those words, but could not be sure it would find no other way to sort the applicants that could prove discriminatory, and by early 2017 the team was disbanded. Reuters reported it in 2018; according to Reuters, recruiters had looked at the model's advice but never relied on it alone.

        In both cases the harm is, or would have been, allocative, about freedom and about jobs, and in both it came from the data and from what was measured, not from a bug in the code.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **risk score** | a number a model gives a person or case to say how likely something unwanted is, such as a new arrest or a missed payment; it often decides who is checked first |
        """),
        L.watch_out("*Fair* needs a measure. The same scores are fair by one question and unfair by another, so when someone calls a model fair, ask: by which numbers, and for whom?"),
        mo.Html(C.picture({"title": "COMPAS in Broward County: a confusion matrix for each group",
                           "groups": ["Black defendants", "White defendants"],
                           "rows": ["not arrested again", "arrested again"],
                           "columns": ["rated low", "rated medium or high"],
                           "wrong_flag": "flagged, but not arrested again",
                           "missed": "rated low, but arrested again",
                           "thousands": ",",
                           "alt": "Two confusion matrices. Black defendants: of those not arrested again, 990 rated low and 805 rated medium or high; of those arrested again, 532 rated low and 1,369 rated medium or high. White defendants: of those not arrested again, 1,139 rated low and 349 rated medium or high; of those arrested again, 461 rated low and 505 rated medium or high."})),
        mo.md(r"""
        The counts ProPublica published for the 6,150 Black and White defendants. In each matrix, the cell at the top right holds the people the score flagged who were not arrested again.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a risk score group by group:

    1. Choose what counts as positive: here, rated medium or high.
    2. Count the confusion matrix of each group.
    3. For each group, work out the share of the flagged people who had the outcome: its precision.
    4. For each group, work out the share of the people without the outcome who were flagged anyway.
    5. Compare the groups on both, and ask what the label really measures.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 for COMPAS. `_tp` holds for each group how many people the score flagged who were arrested again, `_fp` how many it flagged who were not. Run the cell.
    """)
    return


@app.cell
def example_precision():
    _tp = {"black": 1369, "white": 505}
    _fp = {"black": 805, "white": 349}

    for _group in _tp:
        print(_group, _tp[_group] / (_tp[_group] + _fp[_group]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Of the Black defendants the score flagged, 0.63 were arrested again; of the White defendants, 0.59. Almost the same: that was Northpointe's answer to ProPublica. Exercise 1 does step 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flagged():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Flagged, but not arrested again

        Step 4 of the recipe. `fp` holds for each group how many of the people who were not arrested again the score rated medium or high, `tn` how many of them it rated low, and the loop goes through the groups with `group`. Replace the `...` with one line, as in the worked example, but put each share in `flagged` instead of printing it. The last line is `flagged`, so the check sees it. The check also tries other numbers.
        """),
        L.task("Put in `flagged`, under each group, the share of its people who were not arrested again that the score rated medium or high."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flagged():
    box_flagged = L.code_box("flagged", lines=8)
    box_flagged
    return (box_flagged,)


@app.cell(hide_code=True)
def check_flagged(box_flagged):
    result_flagged = L.check("flagged", box_flagged.value)
    result_flagged
    return (result_flagged,)


@app.cell(hide_code=True)
def hints_flagged():
    L.hints(
        "flagged",
        [
            "The same shape as the worked example: a count divided by itself plus another count. Only which counts you use changes, and the answer goes under the group.",
            "Above the line: the people who were flagged. Below it: all the people who were not arrested again, so two counts added up. And how do you put a value under a key in a dictionary?",
            "`flagged[group] = ___[group] / (___[group] + tn[group])`",
        ],
        why="`flagged[group] = fp[group] / (fp[group] + tn[group])` gives about `{'black': 0.45, 'white': 0.23}`: of the Black defendants who were not arrested again, almost half were flagged, against fewer than a quarter of the White defendants.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sources():
    mo.md(r"""
    ### Exercise 2. Where does the problem come from?

    Six situations: the CVs and the arrest come from the two cases, the other four are made up by the course. Put each one where its problem comes from.
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
            "Ask for each situation: is it about whom the model learned from, about what it was asked to predict, or about what people do with its answer?",
            "Two situations are about people who are missing from the data, or who fill most of it. Two others are about what people do with the score once it exists.",
            "Data: the CVs and \\_\\_\\_. Label: \\_\\_\\_ and the promotion. Use: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Who is in the data:** the CVs and the survey. **What the label measures:** the arrest and the promotion. **How the score is used:** the sentence and the top five.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Exercise 3. Fair by which measure?
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="A company says: *Our fraud score is fair: in every group, the same share of the people it flags really did commit fraud.* What does that claim leave open?",
        options={
            "meaning": "Whether a flag means the same in each group",
            "total": "How many people the score looks at in a year, in total",
            "right": "How often honest people are flagged in each group",
        },
        correct="right",
        why={
            "meaning": "That is what the claim says: the same share of the flagged did commit fraud, so a flag means the same. What does it not say? See *The idea*.",
            "total": "The number of people matters for the cost, but not for whether the score treats the groups alike. See *The idea*.",
            "right": "The claim is about precision. As with COMPAS, the honest people of one group can still be flagged far more often, so ask for that share too.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) you saw that a system that hands out more of the bad to one group does allocative harm; COMPAS and Amazon's model are two real cases, about freedom and about jobs. In [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) you counted one confusion matrix; here you compared one for each group, and saw that a single number per group does not settle whether a score is fair. In the next lesson, *The childcare benefits scandal and SyRI*, you look at two Dutch cases in which the government used risk scores to look for fraud.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, result_flagged, result_sources):
    L.wrap_up(result_flagged, result_sources, choice_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
