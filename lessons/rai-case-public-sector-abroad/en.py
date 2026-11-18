# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-public-sector-abroad. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-public-sector-abroad", "en")
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
def quick_check_room():
    quick_room = L.choice(
        "quick-room",
        question="When a rule moves from an official's desk into software, where does the room to make an exception go?",
        options={
            "desk": "It stays with the official at the desk",
            "right": "To the people who design the system",
            "person": "To the person the decision is made about",
        },
        correct="right",
        why={
            "desk": "In software the rule decides every case the same way, so the official no longer has that room. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
            "right": "Mark Bovens and Stavros Zouridis described this in 2002: the room moves to the designers, out of sight of the person the decision is about.",
            "person": "The person the decision is about sees even less of the rule than before. See *The idea* in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
        },
    )
    quick_room
    return (quick_room,)


@app.cell(hide_code=True)
def quick_check_reason():
    quick_reason = L.choice(
        "quick-reason",
        question="A decision letter gives as its only reason: *Score above threshold*. What is wrong with that?",
        options={
            "fine": "Nothing is wrong: it gives a reason",
            "right": "It gives the person nothing to argue with",
            "law": "Thresholds are not allowed in decisions about people",
        },
        correct="right",
        why={
            "fine": "A reason is not always an explanation. See the *Watch out* box in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
            "right": "What made the score high, and is that right? Without that, the person cannot object to anything.",
            "law": "A threshold can be part of a lawful decision. See the *Watch out* box in [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en).",
        },
    )
    quick_reason
    return (quick_reason,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Between November 2016 and March 2017 alone, the Australian government sent about 20,000 letters a week to people who, it said, had been paid too much in benefits, sometimes years before. The debts came from a calculation that spread a year's income evenly over fortnights, and many of them were wrong. Robodebt, as the scheme came to be called, is one of four cases in this lesson from outside the Netherlands. Each went wrong in another way, and each way has its own check.

    **An everyday example.** A teacher hears that a class scored 7 out of 10 on average, and gives every student a 7. For the class as a whole that is right; for nearly every student it is wrong.
    """)
    return


@app.cell(hide_code=True)
def predict_grades():
    guess_grades = L.guess(
        "grades",
        question="In 2020 the exams in England were canceled, and a model set the A-level grades. Of the 718,276 grades, how many do you think came out lower than the grade the teacher had given?",
        options={
            "few": "About 1 in 20",
            "quarter": "About 1 in 4",
            "right": "About 2 in 5",
        },
        reveal="**About 2 in 5**: 39.1 of every 100 grades, most of them one grade lower. Four days after the results, the students got their teachers' grades instead.",
    )
    guess_grades
    return (guess_grades,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Income averaging** means spreading what someone earned over a long period evenly over shorter periods, as if it came in at the same pace all the time. For someone with a steady wage that is close to the truth. For someone who worked half a year and then had no job, it is not.

        **Robodebt.** In Australia, benefits are paid per fortnight, and people report what they earned in each fortnight. From 2015 the government compared those reports with the income that employers had reported to the tax office, spread evenly over the fortnights of the period in the data. Where the averaged income was higher than what someone had reported, the system assumed they had been paid too much and raised a debt, sometimes with a fee of 10 percent on top, unless the person could prove otherwise, with evidence from years before. Deanna Amato was told of a debt of 3,215.38 dollars: averaging had put Amato's income in six fortnights at almost four times the real amount. When Centrelink, the agency that pays the benefits, looked again, the overpayment was only 1.48 dollars, and later it turned out that Amato had been paid about 480 dollars too little. In November 2019 the Federal Court declared the demand for the debt invalid, in orders the government agreed to. In 2021 the court approved a settlement: the government would drop debts of at least 1.763 billion dollars against about 433,000 people, refund about 751 million dollars already paid, and pay 112 million dollars in compensation, including legal costs. A Royal Commission concluded in 2023: *Robodebt was a crude and cruel mechanism, neither fair nor legal, and it made many people feel like criminals.*

        **The A-levels.** In 2020 the exams in England were canceled. Teachers gave each student the grade they would most likely have got, and ranked their students per subject. Ofqual, the exams regulator, found those grades much higher than in earlier years: the share of A* would have gone from 7.7 to 13.9 percent. So it used a model instead. For each school it predicted how the grades would be spread, from the school's results in that subject in the three years before, and then handed out those grades in the order of the teacher's ranking. A student's grade depended on the school's past and on their place in the ranking, not on the grade their teacher gave. For the smallest classes, up to about 5 students, the model used the teachers' grades, because there were too few students to compare with; from about 5 to 15 it mixed the two. Across England, the share of A and A* grades rose by only 2.4 points, but 39.1 of every 100 grades came out lower than the teacher's. On 17 August 2020, four days after the results, Ofqual gave students their teachers' grades instead, and apologized.

        **MiDAS.** From October 2013 to August 2015, a computer system of the unemployment agency of Michigan, in the United States, decided by itself that people had committed fraud. Not returning a questionnaire, for example, led to a fraud finding, with a penalty of four times the benefits. When 22,427 of those findings were reviewed, 93 of every 100 were found to involve no fraud, though a federal judge later noted that this figure has been questioned. In 2024 a court approved a settlement of 20 million dollars.

        **Allegheny.** In Allegheny County, Pennsylvania, the staff who answer calls about children who may be neglected see a score from 1 to 20 before they decide whether someone goes to investigate. The score predicts whether the child will be placed outside the home within two years, from data of 21 public sources, such as mental health services and jail records. The writer Virginia Eubanks pointed out that families who use public services leave far more data behind than families who do not. Above some scores, the tool says the call must be investigated, unless a supervisor decides otherwise. Researchers at Carnegie Mellon University found in 2022 that, for the calls from 2016 to 2018 in which the staff could choose, the score alone would have marked 67.9 of every 100 Black children and 50.0 of every 100 White children for an investigation; with the staff's own decisions it was 50.2 and 43.1.

        The four cases went wrong in different ways. In Robodebt a calculation was wrong for many people. In the A-levels a rule was right on average and wrong for many students. In MiDAS a rule took the place of a person who weighs the case; in Allegheny the score could have, and the staff who weighed the cases made smaller differences between Black and White children than the score alone.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **income averaging** | spreading what someone earned over a long period evenly over shorter periods, as if it came in at the same pace all the time; Robodebt used it to work out debts |
        """),
        L.watch_out("Figures from these cases count different things: debts, people, grades or reviewed findings. Robodebt alone has about 433,000 people with wrong debts, 470,000 debts that the government said it would refund, and 648,000 people in the class action. Before you compare two figures, check what each one counts."),
        mo.Html(C.picture({"title": "What averaging sees in each fortnight",
                           "axis": "dollars", "fortnight": "fortnight",
                           "earned": "earned", "averaged": "averaged income", "invented": "made up by averaging",
                           "alt": "Bars of what someone earned in each of the 26 fortnights of a year: 1,300 dollars in each of the first 10, nothing in the other 16. A line at 500 dollars is the averaged income. In the 16 fortnights without work, the space between nothing and the line is shaded: income that averaging made up."})),
        mo.md(r"""
        Made up by the course: someone who earned 1,300 dollars in each of the first 10 fortnights of a year, and nothing after. Averaging puts 500 dollars in every fortnight, also in the 16 fortnights without work.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a calculation or a rule that decides about people:

    1. Write down what it assumes about a person's life, such as an income that comes in at the same pace all year.
    2. Make up a few people for whom the assumption does not hold, such as someone who worked half the year.
    3. Work out what the calculation says for them, and compare it with what really happened.
    4. For a rule that is right on average, look at the people it gets wrong, not only at the total.
    5. Ask who has to prove that the result is wrong, whether they can, and who can overrule the result.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A casual worker, made up by the course, earned 400 and 600 dollars in turns over the 26 fortnights of a year. `_earned` holds what this person earned in each fortnight. The first loop adds up the year in `_total`, and `_average` is that total divided by the number of fortnights. The second loop goes through the fortnights again and, where `_average` is higher than what was earned, adds the difference to `_invented`. Run the cell.
    """)
    return


@app.cell
def example_averaging():
    _earned = [400, 600, 400, 600, 400, 600, 400, 600, 400, 600, 400, 600, 400,
               600, 400, 600, 400, 600, 400, 600, 400, 600, 400, 600, 400, 600]

    _total = 0
    for _amount in _earned:
        _total = _total + _amount
    _average = _total / len(_earned)

    _invented = 0
    for _amount in _earned:
        if _average > _amount:
            _invented = _invented + (_average - _amount)
    print(_average, _invented)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Averaging puts 500 dollars in every fortnight. In the 13 fortnights in which this person earned 400, it sees 100 dollars too much each time: 1,300 dollars that were not earned in those fortnights. Where someone also got a benefit in such a fortnight, Robodebt counted that income against it. Step 2 of the recipe asks for someone for whom the assumption does not hold at all: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_averaging():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Half a year of work

        A person made up by the course earned 1,000 dollars in each of the first 13 fortnights of a year, then lost their job and lived on benefits. `earned` holds what this person earned in each fortnight. Replace the `...` with the lines of the worked example, with the names without their underscore: `total`, `amount`, `average` and `invented`. Both loops go through `earned` with `amount`. Leave out the `print`: the last line is `invented`, so the check sees it. The check also tries someone whose income goes up and down.
        """),
        L.task("Work out `average`, the averaged income per fortnight, and `invented`, the income that averaging puts in fortnights in which it was not earned."),
    ])
    return


@app.cell(hide_code=True)
def exercise_averaging():
    box_averaging = L.code_box("averaging", lines=16)
    box_averaging
    return (box_averaging,)


@app.cell(hide_code=True)
def check_averaging(box_averaging):
    result_averaging = L.check("averaging", box_averaging.value)
    result_averaging
    return (result_averaging,)


@app.cell(hide_code=True)
def hints_averaging():
    L.hints(
        "averaging",
        [
            "The worked example does the same for the casual worker. Copy its lines, without the `print`, and take the underscore off every name.",
            "First add up `earned` in `total`, and divide by `len(earned)` for `average`. Then a second loop: in which fortnights does averaging see more than was earned?",
            "`if average > amount:` and under it `invented = invented + (___ - ___)`.",
        ],
        why="`total = 0` and `for amount in earned:` with `total = total + amount`, then `average = total / len(earned)`. Then `invented = 0` and a second `for amount in earned:` with `if average > amount:` and `invented = invented + (average - amount)` inside it. That gives an `average` of `500.0` and `invented` of `6500.0`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_failures():
    mo.md(r"""
    ### Exercise 2. Three ways to go wrong

    Six facts from the four cases. Put each one under the way it went wrong.
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
            "Ask of each fact: was the arithmetic itself wrong, was the outcome right for the group but not for the person, or did a number or a rule decide where a person should have weighed the case?",
            "Two facts are about Robodebt, two about the A-levels, one about Michigan and one about Allegheny.",
            "A calculation: the fortnights and \\_\\_\\_. Right on average: the school's past and \\_\\_\\_. In place of judgment: \\_\\_\\_ and the required investigation.",
        ],
        why="**A calculation that was wrong for many people:** the fortnights and the 3,215.38 dollars. **A rule that was right on average:** the school's past and the totals. **A number or a rule in place of judgment:** the questionnaire and the required investigation.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.md(r"""
    ### Exercise 3. Why did small classes keep their teachers' grades?
    """)
    return


@app.cell(hide_code=True)
def choice_small_cell():
    choice_small = L.choice(
        "small",
        question="Why did the A-level model give the smallest classes, up to about 5 students, their teachers' grades?",
        options={
            "right": "Too few students to compare with the past",
            "trust": "Ofqual trusted the teachers of small classes more",
            "private": "Most small classes were at private schools",
        },
        correct="right",
        why={
            "right": "With so few students, the school's past says too little about this year: Ofqual wanted to avoid adjustments it could not defend. Small classes are far more common in some subjects than in others, so the same rule did not work out the same for every student.",
            "trust": "The model did not trust the teachers' grades in general: Ofqual found them much higher than in earlier years. What did the size of the class change? See *The idea*.",
            "private": "The rule was about the size of the class, not the kind of school. Why would size matter for a model that compares with the past? See *The idea*.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Deciding about people automatically](/course/open/rai-automated-decisions?lang=en) you saw that weighing circumstances needs someone who can see the case, and that people need a real way to object. Robodebt put a calculation where that weighing should have been and left people to prove it wrong with evidence from years before; MiDAS let a rule decide on fraud; the A-levels showed that a rule can be right on average and wrong for many students. In the next lesson, *When the label measures the wrong thing*, you look at a health algorithm that learned from the wrong label, and at a credit card whose decisions nobody outside could check.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_small, result_averaging, result_failures):
    L.wrap_up(result_averaging, result_failures, choice_small)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
