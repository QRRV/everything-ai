# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-toeslagen-syri. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-toeslagen-syri", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weigh():
    quick_weigh = L.choice(
        "quick-weigh",
        question="Which of these decisions weighs circumstances, instead of checking a fact?",
        options={
            "fee": "Whether a fee was paid on time",
            "right": "Whether someone meant to deceive",
            "age": "Whether an applicant is 18 years or older",
        },
        correct="right",
        why={
            "fee": "Whether a fee was paid is a fact that a system can check the same way every time. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
            "right": "Whether someone meant to deceive depends on this person and this case: that needs someone who can see the circumstances.",
            "age": "An age is a fact that a system can check the same way every time. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
        },
    )
    quick_weigh
    return (quick_weigh,)


@app.cell(hide_code=True)
def quick_check_contest():
    quick_contest = L.choice(
        "quick-contest",
        question="What does *contestability* mean?",
        options={
            "accuracy": "How often the decisions of a system turn out to be right in the end",
            "right": "How well people can challenge a decision and get it changed",
            "speed": "How fast a decision is made",
        },
        correct="right",
        why={
            "accuracy": "A system that is often right can still leave people no way to object. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
            "right": "For that, people need to know that a decision was made and why, where to object, enough time, and someone who can change it.",
            "speed": "Speed says nothing about whether people can object. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
        },
    )
    quick_contest
    return (quick_contest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In January 2021 the Dutch cabinet resigned over the way the Tax Administration had treated parents who received a childcare benefit. A risk model was part of it, and so were the rules and the people around the model. A year earlier, a Dutch court had struck down the law behind SyRI, a system that searched for fraud by linking data of many government bodies. Together, the two cases show what goes wrong when nobody can see how people are picked, or what it costs them.

    **An everyday example.** A shop checks the bag of every customer who wears a large coat. Most customers with a large coat steal nothing, and the shop never says why it stops them. If a missing receipt then costs a fine of a year's shopping, the coat rule is no longer the only problem.
    """)
    return


@app.cell(hide_code=True)
def predict_label():
    guess_label = L.guess(
        "label",
        question="From 2012 to early 2020, how many people do you think got the label *intent or gross negligence* from the Tax Administration, which meant they could not get a payment plan?",
        options={
            "hundreds": "A few hundred people",
            "thousands": "About 3,000 people",
            "right": "25,000 to 35,000",
        },
        reveal="**25,000 to 35,000**, as the state secretary told parliament in March 2020. On 4 December 2020 parliament heard that, in a sample, 94 of every 100 of these labels could no longer be seen as justified.",
    )
    guess_label
    return (guess_label,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Risk profiling** means choosing whom to check from characteristics that a group of people share, instead of from something one person did. Tax offices, banks and city councils all do it. What matters is which characteristics count, whether anyone can see them, and what happens to the people who are picked.

        **The childcare benefits.** From 2016 to 2018, more than half a million Dutch households a year got a childcare benefit from the Tax Administration. From 2013, a self-learning model gave each application a risk score between 0 and 1, from a few dozen indicators, and staff checked the applications with the highest scores. They did not know why the model had picked an application. One indicator was whether the applicant had Dutch nationality or not. The Dutch Data Protection Authority found it in the model from at least March 2016 to October 2018, called this unlawful and discriminatory in 2020, and in 2021 fined the Tax Administration 2.75 million euros for this and two other unlawful ways of processing nationality.

        Around the model stood hard rules. A small mistake, such as a missing signature, could mean paying back the benefit of a whole year. Between 2012 and 2020, 25,000 to 35,000 people got the label *intent or gross negligence*: they could not get a payment plan, and other government bodies saw them as deliberate fraudsters. In December 2020 a committee of the Dutch parliament called it *ongekend onrecht*, unprecedented injustice. By the middle of 2026, more than 44,000 parents had been recognized as harmed.

        **SyRI**, the System Risk Indication, linked data about income, housing, debts, benefits and more, from city councils, the benefits agencies, the Tax Administration and others, and a risk model with points per indicator marked people and addresses worth looking into. From 2015 it was used in five projects, all in neighborhoods the government called problem areas. On 5 February 2020 the court in The Hague ruled that the law behind SyRI breached the right to respect for private life in the European Convention on Human Rights. The risk model and its indicators were secret, so nobody, not even the court, could check how SyRI chose people, and someone it flagged could hardly defend themselves. Whether SyRI discriminated, the court could not say, because it could not look inside. The state did not appeal.

        Both cases show what goes wrong without transparency and without a real way to object.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **risk profiling** | choosing whom to check, for example for fraud, from characteristics that a group of people share, such as income, address or nationality, instead of from what one person did |
        """),
        L.watch_out("Taking nationality out is not enough. Other indicators, such as an address or a type of school, can stand in for it. The next lesson shows such a case."),
        mo.Html(C.picture({"title": "From an application to a label",
                           "steps": ["A parent applies for the childcare benefit.",
                                     "A self-learning model gives the application a risk score between 0 and 1.",
                                     "Staff check the applications with the highest scores, without knowing why they were picked.",
                                     "A small mistake, such as a missing signature, can mean paying back the whole year.",
                                     "The label intent or gross negligence: no payment plan."],
                           "model": "the model", "rules": "rules and people",
                           "alt": "Five steps. 1, a parent applies for the childcare benefit. 2, a self-learning model gives the application a risk score between 0 and 1. 3, staff check the applications with the highest scores, without knowing why they were picked. 4, a small mistake, such as a missing signature, can mean paying back the whole year. 5, the label intent or gross negligence: no payment plan. Step 2 is the model; steps 3 to 5 are rules and people."})),
        mo.md(r"""
        Only step 2 is the model. The harm also came from the rules and the people around it. The picture shows one way an application could end in the label; the Data Protection Authority could not see what the nationality indicator did to individual applicants.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look at a risk profile before it is used:

    1. List the indicators and the points each one gives.
    2. Ask of each indicator whether it says something about what a person did, or only about who the person is.
    3. Score a few applications that differ in one indicator only, and see who crosses the line.
    4. Check that the people who are picked can learn why, and can object.
    5. Write down what being picked costs a person, also when the check finds nothing.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The real model had a few dozen indicators; only two are public, nationality and the distance between home and childcare. The scorecard here is made up by the course, with three indicators, and an application with 6 points or more goes to a person for a check. `_points` holds the points of each indicator, `_application` whether each indicator holds for one application. Run the cell.
    """)
    return


@app.cell
def example_score():
    _points = {"distance": 2, "income change": 3, "not Dutch": 4}
    _application = {"distance": True, "income change": False, "not Dutch": True}

    _score = 0
    for _indicator in _points:
        if _application[_indicator]:
            _score = _score + _points[_indicator]
    print(_score)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    This application has 6 points: enough to be checked. Step 3 of the recipe asks for more applications that differ in one indicator only: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who crosses the line?

        Four applications, made up by the course. `points` holds the points of each indicator, and `applications` holds for each application whether each indicator holds. Replace the `...` with the body of `risk_score`, which gets one `application`: the lines of the worked example, with the names without their underscore, such as `score` and `indicator`, and at the end give the score back instead of printing it. Inside the function you can use `points` as it is. The loop under it goes through the applications with `name` and puts each score in `scores`, and the last line is `scores`, so the check sees it. The check also tries other points.
        """),
        L.task("Make `risk_score` give back the sum of the points of the indicators that hold for the application."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=24)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "The worked example adds up the points of one application. Put the same lines in the function, and give the score back at the end.",
            "Start `score` at 0 before the loop. Inside the loop, an `if` asks whether the indicator holds for `application`. Where does `return` go: in the loop, or after it?",
            "`if application[indicator]:` and under it `score = score + ___[indicator]`; after the loop, `return ___`.",
        ],
        why="`score = 0`, then `for indicator in points:` with `if application[indicator]:` and `score = score + points[indicator]` inside it, and `return score` after the loop. That gives `{'A': 3, 'B': 5, 'C': 9, 'D': 7}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_failures():
    mo.md(r"""
    ### Exercise 2. What went wrong?

    Six facts from the two cases. Put each one under what went wrong.
    """)
    return


@app.cell(hide_code=True)
def sort_failures_cell():
    sort_failures = L.sort("failures")
    sort_failures
    return (sort_failures,)


@app.cell(hide_code=True)
def check_failures(sort_failures):
    result_failures = L.check("failures", sort_failures)
    result_failures
    return (result_failures,)


@app.cell(hide_code=True)
def hints_failures():
    L.hints(
        "failures",
        [
            "Ask for each fact: is it about what nobody could see, about who could be picked, or about what happened after a mistake?",
            "Two facts are about SyRI, four about the childcare benefits. Two of the facts about the benefits are about money and what followed from it.",
            "Nobody could see: the staff and \\_\\_\\_. Who was picked: \\_\\_\\_ and the neighborhoods. What a mistake cost: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Nobody could see how it worked:** the staff and the court. **Who was picked:** the nationality and the neighborhoods. **What a small mistake cost:** the signature and the label.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ruling():
    mo.md(r"""
    ### Exercise 3. Why did the court strike down SyRI's law?
    """)
    return


@app.cell(hide_code=True)
def choice_ruling_cell():
    choice_ruling = L.choice(
        "ruling",
        question="On what ground did the court in The Hague strike down the law behind SyRI in 2020?",
        options={
            "errors": "It was proven that SyRI flagged mostly innocent people",
            "gdpr": "The court found that SyRI broke the GDPR's rules on automated decisions",
            "right": "It broke the right to private life, as nobody could check it",
        },
        correct="right",
        why={
            "errors": "The court could not look inside SyRI, so it could not say how often it was wrong. See *The idea*.",
            "gdpr": "The court did not rule on the GDPR. Which right did it name? See *The idea*.",
            "right": "Article 8 of the European Convention on Human Rights: the law did not let anyone see or check how SyRI chose people, so the people it flagged could hardly defend themselves. The court also found that the law did not limit which data could be linked.",
        },
    )
    choice_ruling
    return (choice_ruling,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en) you saw that weighing circumstances needs someone who can see the case, and that people need a way to object. In the childcare benefits scandal, a label that needed someone to weigh intent went to tens of thousands of people, and most of those labels were later found unjustified; in SyRI, nobody could see what to object to. In the next lesson, *More algorithms that judged people*, you look at more Dutch cases, and at a register in which government bodies can describe the algorithms they use.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ruling, result_failures, result_score):
    L.wrap_up(result_score, result_failures, choice_ruling)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
