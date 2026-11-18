# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-automated-decisions. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-automated-decisions", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A person in the loop](/course/open/rai-human-oversight?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bias():
    quick_bias = L.choice(
        "quick-bias",
        question="A reviewer misses a problem because the system did not point at it. What is that called?",
        options={
            "right": "Automation bias",
            "oversight": "Human oversight, working as intended",
            "power": "Too little power to say no",
        },
        correct="right",
        why={
            "oversight": "Human oversight is the person watching. Here the watching failed: what is that failure called? See *The idea* in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
            "right": "Missing what the system did not flag, and following its wrong suggestions, are the two faces of automation bias.",
            "power": "Power is about saying no without paying for it. Here the reviewer did not see the problem at all. See *The idea* in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
        },
    )
    quick_bias
    return (quick_bias,)


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="What tells you best whether a person in the loop works?",
        options={
            "agree": "How often the person agrees with what the system says",
            "speed": "How fast the person gets through each case",
            "right": "How many of the system's mistakes the person catches",
        },
        correct="right",
        why={
            "agree": "A person who agrees 9 times out of 10 may be catching nothing. See Exercise 1 in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
            "speed": "Speed says something about time, not about whether the check works. See Exercise 1 in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
            "right": "Only the cases where the system is wrong show whether the person adds anything.",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    More and more decisions about people are made by software, or prepared by it: who gets a benefit, a fine, a permit or a check by an inspector. That can be fairer than a person deciding. It can also take away the one thing that helped a person in an unusual situation: someone who listens. Whether an automated decision is acceptable depends on what it decides, and on what someone can do when it is wrong.

    **An everyday example.** A ticket machine that only takes cards is the same for everyone and never in a bad mood. For someone with only cash, there is nobody to ask.
    """)
    return


@app.cell(hide_code=True)
def predict_judges():
    guess_judges = L.guess(
        "judges",
        question="In one US immigration court, the same kind of asylum cases from one country went to different judges by chance. How far apart do you think the judges' grant rates were?",
        options={
            "close": "Within about 5 percentage points of each other",
            "some": "About 20 percentage points apart",
            "right": "From about 5 to about 88 percent",
        },
        reveal="**From about 5 to about 88 percent.** In the Miami immigration court, asylum seekers from Colombia had about a 5 percent chance before one judge and about 88 percent before another judge in the same building (Ramji-Nogales, Schoenholtz and Schrag, 2007).",
    )
    guess_judges
    return (guess_judges,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        People who decide are not all the same: some are strict, some are mild, some decide differently in the morning than at the end of the day. A rule in software decides every case the same way, and that can be fairer. But it also takes away room. An official at a desk could see that a case was unusual and make an exception. In 2002 Mark Bovens and Stavros Zouridis described how that room moves from the official at the desk to the people who design the system, out of sight of the person it decides about.

        Whether an automated decision is acceptable depends first on *what* it decides. Checking a fact, such as an age or whether a fee was paid, can be done the same way every time. Weighing circumstances, such as whether someone meant to deceive or whether a debt is too heavy to bear, needs someone who can see this case.

        And it depends on what happens when the decision is wrong. **Contestability** is how well people can challenge a decision and get it looked at again and changed. For that, a person needs to know that a decision was made and why, where to object, enough time, and someone who listens and can change it. Where the GDPR allows a fully automated decision with legal or similarly serious effects at all, it gives people the right to contest it. In the Netherlands, someone who disagrees with a decision of a government body has six weeks to object.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **contestability** | how well people can question a decision made with an AI system, and get it looked at again and changed |
        """),
        L.watch_out("A reason is not always an explanation. *Score above threshold* is a reason, but it gives the person nothing to argue with: what made the score high, and is that right?"),
        mo.Html(C.picture({"title": "Five decision letters, against three tests", "letter": "letter",
                           "tests": ["reason", "where to object", "42 days"], "yes": "can object", "no": "hard to object",
                           "alt": "Five decision letters against three tests: a reason, where to object, and at least 42 days. Letters 1 and 5 pass all three. Letter 2 has no reason, letter 3 says nothing about where to object, and letter 4 gives only 14 days."})),
        mo.md(r"""
        A benefits office made up by the course. A filled dot means that the letter passes the test.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To decide whether to automate a decision about people:

    1. Write down what the system decides, and whether it checks a fact or weighs circumstances.
    2. Leave weighing circumstances to a person, with the system as help at most.
    3. Give every decision a reason the person can argue with.
    4. Say where to object, and give enough time.
    5. Make sure an objection reaches someone who can change the decision, and count how often they do.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    # in mo.vstack, so that marimo's editor keeps the "" in the text as it is
    mo.vstack([
        mo.md(r"""
        ### Worked example

        A function that tests one thing about a decision letter: does it give a reason? A letter is a dictionary; an empty reason is `""`. Run the cell.
        """),
    ])
    return


@app.cell
def example_reason():
    def _has_reason(letter):
        if letter["reason"] == "":
            return False
        return True

    print(_has_reason({"reason": "income above the limit", "contact": "objections desk", "days": 42}))
    print(_has_reason({"reason": "", "contact": "objections desk", "days": 42}))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first letter gives a reason, the second does not. A `return` ends the function at once: once a test fails, the rest no longer matters. Steps 3 and 4 of the recipe ask for more than a reason: Exercise 1 tests all three.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_contestable():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Can someone object?

        The letters of the picture. `letters` holds the five letters; each is a dictionary with the keys `"reason"`, `"contact"` and `"days"`. Replace the `...` with the body of `contestable`: give back `False` when the letter has no reason or says nothing about where to object, and otherwise whether it gives the six weeks, 42 days, that Dutch law gives for an objection. The loop under it tries every letter, and the last line is `results`, so the check sees it. The check also tries other letters.
        """),
        L.task("Make `contestable` give `True` for a letter that someone can object to, and `False` for one they cannot."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contestable():
    box_contestable = L.code_box("contestable", lines=18)
    box_contestable
    return (box_contestable,)


@app.cell(hide_code=True)
def check_contestable(box_contestable):
    result_contestable = L.check("contestable", box_contestable.value)
    result_contestable
    return (result_contestable,)


@app.cell(hide_code=True)
def hints_contestable():
    L.hints(
        "contestable",
        [
            "Three tests, one after the other, as in the worked example: each test that fails ends the function with `False`.",
            "The worked example already does the reason. Which other field can be empty? And what does the last `return` compare?",
            "After the two `if` tests: `return letter[\"days\"] ___ 42`",
        ],
        why="Two tests as in the worked example, for `\"reason\"` and `\"contact\"`, then `return letter[\"days\"] >= 42`, give `[True, False, False, False, True]`: letter 1 gives exactly 42 days, which is enough.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fact():
    mo.md(r"""
    ### Exercise 2. A fact or a judgment?

    Six decisions about people, made up by the course. Put each one with what it does.
    """)
    return


@app.cell(hide_code=True)
def sort_fact_cell():
    sort_fact = L.sort("fact")
    sort_fact
    return (sort_fact,)


@app.cell(hide_code=True)
def check_fact(sort_fact):
    result_fact = L.check("fact", sort_fact)
    result_fact
    return (result_fact,)


@app.cell(hide_code=True)
def hints_fact():
    L.hints(
        "fact",
        [
            "Ask for each decision: could you settle it by looking something up, or does it depend on this person's situation?",
            "Three decisions compare a known value with a fixed one. The other three ask about intent, a burden or a child.",
            "Checks a fact: old enough, \\_\\_\\_ and \\_\\_\\_. Weighs circumstances: \\_\\_\\_, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Checks a fact:** old enough, paid and above the limit. **Weighs circumstances:** fraud, hardship and best for the child.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.md(r"""
    ### Exercise 3. What is missing?
    """)
    return


@app.cell(hide_code=True)
def choice_score_cell():
    choice_score = L.choice(
        "score",
        question="A letter says: *Your application was rejected because your risk score was above the threshold. You can object at the objections desk within six weeks.* What does someone most need, to object?",
        options={
            "phone": "A phone number, besides the desk",
            "right": "What made the score high",
            "time": "More time: six weeks is too short for anyone",
        },
        correct="right",
        why={
            "phone": "A phone number may help, but the letter already says where to object. What would the person write? See the *Watch out* box in *The idea*.",
            "right": "Without knowing what made the score high, the person cannot say what is wrong about it: the letter passes the tests, but gives nothing to argue with.",
            "time": "Six weeks is what Dutch law gives, and the letter gives it. The problem is what to write. See the *Watch out* box in *The idea*.",
        },
    )
    choice_score
    return (choice_score,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A person in the loop](/course/open/rai-human-oversight?lang=en) you saw that a person checking a system often adds little. Here you saw that what may be automated depends on what is decided, and that people need a way to contest the decision afterwards. In the next lesson, *How far can you predict a life?*, you ask how well a system can predict outcomes in people's lives at all.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_score, result_contestable, result_fact):
    L.wrap_up(result_contestable, result_fact, choice_score)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
