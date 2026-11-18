# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-dutch-algorithms. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-dutch-algorithms", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The childcare benefits scandal and SyRI](/course/open/rai-case-toeslagen-syri?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_profiling():
    quick_profiling = L.choice(
        "quick-profiling",
        question="A city council checks people with a low income for fraud more often than others, whatever each of them did. What is that called?",
        options={
            "right": "Risk profiling",
            "contest": "Making a decision contestable",
            "oversight": "A person in the loop who checks a model",
        },
        correct="right",
        why={
            "right": "Choosing whom to check from something a group shares, here a low income, instead of from what one person did.",
            "contest": "Contestability is about whether people can challenge a decision, not about whom you check. See *The idea* in [The childcare benefits scandal and SyRI](/course/open/rai-case-toeslagen-syri?lang=en).",
            "oversight": "A person in the loop checks what a model decides; here the question is whom the council checks at all. See *The idea* in [The childcare benefits scandal and SyRI](/course/open/rai-case-toeslagen-syri?lang=en).",
        },
    )
    quick_profiling
    return (quick_profiling,)


@app.cell(hide_code=True)
def quick_check_nationality():
    quick_nationality = L.choice(
        "quick-nationality",
        question="On the made-up scorecard of the last lesson, applications A and D did the same things, yet only D was checked. Why?",
        options={
            "far": "D had a larger distance between home and childcare",
            "right": "D got points for not having Dutch nationality",
            "staff": "A member of staff looked at D before A",
        },
        correct="right",
        why={
            "far": "Neither got points for distance: they differed in one indicator only. Which one? See Exercise 1 in [The childcare benefits scandal and SyRI](/course/open/rai-case-toeslagen-syri?lang=en).",
            "right": "With 4 points for *not Dutch*, D reached 7 points, above the line of 6, and A stayed at 3.",
            "staff": "The scorecard decided from the points alone, before any person looked. See Exercise 1 in [The childcare benefits scandal and SyRI](/course/open/rai-case-toeslagen-syri?lang=en).",
        },
    )
    quick_nationality
    return (quick_nationality,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The childcare benefits scandal was not the only time a Dutch organization let an algorithm help judge people. Students, people on welfare, taxi drivers and whole neighborhoods got a score too. Each time, the people affected could not see the choices behind it: which features counted, whether a person really looked, and whether the system worked at all.

    **An everyday example.** A teacher checks the homework of the pupils at the back of the class more often than that of the pupils at the front. More copying is then found at the back, even if the pupils at the front copy just as often: you only find something where you look.
    """)
    return


@app.cell(hide_code=True)
def predict_lawyers():
    guess_lawyers = L.guess(
        "lawyers",
        question="Over ten years, lawyers helped 376 students whom DUO accused of cheating on their student grant. How many of them do you think had a migration background?",
        options={
            "half": "About half",
            "most": "About three in four",
            "right": "Almost all of them",
        },
        reveal="**367 of the 376**, journalists of Investico and NOS op 3 found in 2023, from the cases of 32 lawyers. Those cases need not be typical of all checks, so *The idea* uses counts of all students, from Statistics Netherlands.",
    )
    guess_lawyers
    return (guess_lawyers,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **DUO.** From 2012, DUO, the agency that pays Dutch student grants, used a risk profile to choose which students with the higher grant for living away from their parents it would check with a home visit. Three features raised the score: a vocational course (mbo) instead of college or university, a short distance to the parents' address, and a young age. Staff then picked students by hand, mostly from the high scores. None of the three features is someone's background, yet they stood in for it: a vocational course and a short distance to the parents were much more common among students with a non-European migration background. In 2024 the organization Algorithm Audit, with data from Statistics Netherlands, found that the profile labeled these students high risk 2.0 times as often as students of Dutch origin, and that they were picked for a home visit 6.2 times as often. The Dutch Data Protection Authority called the profile discriminatory and unlawful in 2024, and noted that DUO had never evaluated it. The minister apologized in March 2024, and in November 2024 the next minister announced that more than 10,000 former students get their fines and repayments back; 61 million euros was set aside for it.

        **Rotterdam.** From 2017 to 2021, the city of Rotterdam ranked about 30,000 people on welfare with a model that gave each of them a score between 0 and 1 from 315 features, and the people with the highest scores were investigated. About twenty of the features were about language, such as how well someone spoke or wrote Dutch. Journalists of Lighthouse Reports got hold of the model, and in 2023 they published that people who struggled with Dutch were about 2.2 times as common among the highest scores as among everyone on welfare. Rotterdam stopped using the model in 2021, as a precaution.

        **Uber and Ola.** Drivers for Uber and Ola from the United Kingdom went to court in Amsterdam under the GDPR. In 2023 the court of appeal found that Uber had removed three drivers from its platform by a solely automated decision: the staff in between did no more than a symbolic act, and the drivers were not heard. Uber had to explain the logic behind the decisions, on pain of 4,000 euros a day. Ola's fraud probability score, the court found, affected drivers significantly, so Ola had to explain it too. In [A person in the loop](/course/open/rai-human-oversight?lang=en) you saw that a person who mostly says yes catches few mistakes; here the court did not even count what such a person did as a decision by a person.

        **CAS.** From 2013 in Amsterdam, and later in other parts of the country, the Dutch police used CAS, the Crime Anticipation System. It split an area into squares of 125 by 125 meters and forecast in which squares crime such as burglary was to be expected, from police records such as earlier incidents and known suspects living nearby. An evaluation by the Police Academy found in 2017 no signs that it had the effects the police hoped for, and the teams could not see which data drove a forecast. The police stopped using CAS on 1 December 2025, because its added value was unclear.

        **The Algorithm Register.** Since December 2022, Dutch government bodies can describe the algorithms they use in the **Algorithm Register**, at algoritmes.overheid.nl: what each one is for, how it works, which data it uses, how people oversee it, and its risks. It started with 109 algorithms; on 18 November 2026 it listed 1,606. Registering is not yet required by law, so an algorithm that is missing from it may still be in use. CAS is in it too, now marked as out of use.

        Each case turned on choices the people affected could not see: which features counted, whether a person really decided, and whether the system worked. A register makes some of those choices visible, before anyone has to go to court for them.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **Algorithm Register** | the public list on which Dutch government bodies can describe the algorithms they use: what for, how they work, with which data, and with which oversight and risks |
        """),
        L.watch_out("A larger share among the people checked does not show that a group cheats more. You only find misuse where you look: a group that is checked more often is also caught more often."),
        mo.Html(C.picture({"title": "Who was in the group, step by step",
                           "rows": ["All students with the grant (248,650)",
                                    "Labeled high risk by the profile (42,800)",
                                    "Picked for a home visit (2,810)"],
                           "axis": "share with a non-European migration background (%)", "decimal": ".",
                           "alt": "Three bars for DUO's students in 2014. Of all 248,650 students with the grant, 22.0 percent had a non-European migration background; of the 42,800 the profile labeled high risk, 36.0 percent; of the 2,810 picked for a home visit, 63.2 percent. A dashed line marks the 22.0 percent of all students."})),
        mo.md(r"""
        The filled bar is what the profile itself gave out: it made the share larger, and the selection by hand made it larger still. The shares come from Algorithm Audit's report of 2024, for the students who had the grant on 1 February 2014.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compare how often two groups are picked:

    1. Count the people in each group.
    2. Count how many of each group were picked.
    3. For each group, divide the people picked by all the people in the group: the share picked.
    4. Divide the one share by the other: how many times as often one group is picked.
    5. Do this for each step, from the score to the check, to see where the gap grows.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The counts here are worked out by the course from the shares in Algorithm Audit's report, for the 248,650 students who had the grant on 1 February 2014. `non-European` stands for the students with a non-European migration background, and `everyone else` for all other students, so the results need not match the report's exactly: the report compares with students of Dutch origin only. `_students` holds how many students of each group had the grant, `_high_risk` how many of them the profile labeled high risk. Run the cell.
    """)
    return


@app.cell
def example_high_risk():
    _students = {"non-European": 54703, "everyone else": 193947}
    _high_risk = {"non-European": 15408, "everyone else": 27392}

    _share = {}
    for _group in _students:
        _share[_group] = _high_risk[_group] / _students[_group]
    print(round(_share["non-European"], 2), round(_share["everyone else"], 2))
    print(round(_share["non-European"] / _share["everyone else"], 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The profile labeled 0.28 of the students with a non-European migration background high risk, and 0.14 of everyone else: a share twice as large. Step 5 of the recipe asks for the next step too, the home visit: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_visited():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who got the home visit?

        `students` holds again how many students of each group had the grant, and `visited` how many of them DUO picked for a home visit. Replace the first `...` with one line, as in the worked example, that puts under `picked[group]` the share of this group's students who got a home visit; the loop goes through the groups with `group`. Replace the second `...` with how many times as large the non-European share is as the share of everyone else, without `round`. The last line is `times`, so the check sees it. The check also tries other numbers.
        """),
        L.task("Put in `times` how many times as often a student with a non-European migration background got a home visit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_visited():
    box_visited = L.code_box("visited", lines=10)
    box_visited
    return (box_visited,)


@app.cell(hide_code=True)
def check_visited(box_visited):
    result_visited = L.check("visited", box_visited.value)
    result_visited
    return (result_visited,)


@app.cell(hide_code=True)
def hints_visited():
    L.hints(
        "visited",
        [
            "The worked example does the same for whom the profile labeled high risk. Which count takes the place of `_high_risk` here?",
            "Below the line go all the students of the group, `students[group]`. For `times`, take the two shares out of `picked` by their keys.",
            "`picked[group] = ___[group] / students[group]`, and after the loop `times = picked[\"non-European\"] / ___`.",
        ],
        why="`picked[group] = visited[group] / students[group]` in the loop, and `times = picked[\"non-European\"] / picked[\"everyone else\"]` after it. That gives `6.09`, about 6.1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.md(r"""
    ### Exercise 2. Three questions

    Six facts from the four cases, each about one of three questions. Put each one under the question it is about.
    """)
    return


@app.cell(hide_code=True)
def sort_unseen_cell():
    sort_unseen = L.sort("unseen")
    sort_unseen
    return (sort_unseen,)


@app.cell(hide_code=True)
def check_unseen(sort_unseen):
    result_unseen = L.check("unseen", sort_unseen)
    result_unseen
    return (result_unseen,)


@app.cell(hide_code=True)
def hints_unseen():
    L.hints(
        "unseen",
        [
            "Ask for each fact: is it about what went into the score, about who made the decision, or about what the system achieved?",
            "Two facts are about the inputs of a model, two about Uber and Ola, and two about an evaluation.",
            "What counted: the distance and \\_\\_\\_. Whether a person decided: \\_\\_\\_ and the deductions. Whether it worked: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**What counted in the score:** the distance and the language. **Whether a person really decided:** the sign-off and the deductions. **Whether the system worked:** the evaluation and the effects.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_register():
    mo.md(r"""
    ### Exercise 3. What is in the register?
    """)
    return


@app.cell(hide_code=True)
def choice_register_cell():
    choice_register = L.choice(
        "register",
        question="A city council starts using a model that chooses whom to check for welfare fraud, and describes it in the Algorithm Register. What can you read there?",
        options={
            "score": "The score the model gave you, and where you can object to it",
            "proof": "Proof that the model does not discriminate, checked and approved by the government",
            "right": "What the model is for, which data it uses, and how people oversee it",
        },
        correct="right",
        why={
            "score": "The register describes the algorithm itself, not what it decided about one person. See *The idea*.",
            "proof": "An entry is a description by the body that uses the algorithm, not a test of it. At DUO, it took a separate study with data from Statistics Netherlands to show the gap. See *The idea*.",
            "right": "And its risks. Registering is not yet required by law, so not every algorithm is in it.",
        },
    )
    choice_register
    return (choice_register,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The childcare benefits scandal and SyRI](/course/open/rai-case-toeslagen-syri?lang=en) you saw, on a made-up scorecard, how one indicator about who someone is can decide who is checked. DUO's profile never looked at background, yet with features that stood in for it, and a selection by hand after the score, students with a non-European migration background got a home visit about six times as often; a later lesson, *Leaving out the column is not enough*, shows why leaving a feature out is not enough. In the next lesson, *Robodebt, the A-levels and more*, you look at cases abroad.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_register, result_unseen, result_visited):
    L.wrap_up(result_visited, result_unseen, choice_register)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
