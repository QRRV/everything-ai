# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-design-with-values. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-design-with-values", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Principles and guidelines](/course/open/rai-principles?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_trustworthy():
    quick_trustworthy = L.choice(
        "quick-trustworthy",
        question="The EU's expert group described trustworthy AI. Which three things should it be?",
        options={
            "fast": "Fast, cheap and accurate",
            "open": "Open, explainable and free of every bias",
            "right": "Lawful, ethical and robust",
        },
        correct="right",
        why={
            "fast": "Those are things a buyer may want, not what the expert group asked for. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "open": "Transparency is one of the seven requirements, but not one of the three words. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "right": "Lawful, ethical and robust, with seven requirements such as human oversight, transparency and fairness.",
        },
    )
    quick_trustworthy
    return (quick_trustworthy,)


@app.cell(hide_code=True)
def quick_check_altai():
    quick_altai = L.choice(
        "quick-altai",
        question="What is ALTAI?",
        options={
            "right": "Questions to test a system against seven requirements",
            "law": "A European law that bans the riskiest uses of AI altogether",
            "oecd": "The group of countries that wrote the OECD AI Principles",
        },
        correct="right",
        why={
            "law": "ALTAI is not a law: nobody has to use it. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "right": "The Assessment List for Trustworthy AI turns the expert group's seven requirements into questions an organization answers about its own system.",
            "oecd": "ALTAI comes from the EU's expert group, not from the OECD. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
        },
    )
    quick_altai
    return (quick_altai,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A team wants its hiring tool to respect privacy. Good, but what does the engineer do on Monday morning? A value such as privacy is too broad to build or to test. Somewhere between the value and the code, someone has to decide what it means for this tool, and those decisions are where values are won or lost. This lesson turns a value into requirements you can check.

    **An everyday example.** A family wants a healthy kitchen. That becomes rules, such as vegetables every day and little sugar, and then things you can check: a vegetable on every plate, at most one soft drink a week.
    """)
    return


@app.cell(hide_code=True)
def predict_origin():
    guess_origin = L.guess(
        "origin",
        question="Participatory design lets the people who will use a system take part in designing it. Where and when do you think it started?",
        options={
            "valley": "Silicon Valley, 2000s",
            "right": "Scandinavia, 1970s",
            "japan": "Japan, 1950s",
        },
        reveal="**In Scandinavia in the 1970s.** Researchers worked with trade unions so that workers had a say in the computer systems that would change their jobs.",
    )
    guess_origin
    return (guess_origin,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Value-sensitive design**, from Batya Friedman and colleagues since the 1990s, takes the values of everyone a technology touches into account from the start, not only at the end. It asks three kinds of questions: which values are at stake, and whose (conceptual); what the people involved think and do (empirical); and how the design supports or hinders those values (technical).

        *Design for Values*, an approach from Delft in the same family, gives a way to make a value concrete: a *values hierarchy*. At the top is a value, such as privacy. Under it are norms, rules that say what the value asks of this system: keep only the data you need, only as long as you need it. At the bottom are requirements an engineer can check: a CV is deleted after at most 30 days. Each step down is a choice that can be argued about. A requirement that nobody can trace back to a value is hard to defend; a value without requirements is only a word, like the principles in [Principles and guidelines](/course/open/rai-principles?lang=en).

        *Participatory design* asks the people affected to take part in those choices: not only to be asked what they think, but to help decide.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **value-sensitive design** | a design method that takes the values of everyone a technology touches into account from the start and throughout, by asking which values are at stake, what the people involved think, and how the design affects those values |
        """),
        L.watch_out("A requirement can be met while the value is still lost. A tool that deletes every CV after 30 days can still learn too much from them before then: check the requirement, and now and then ask again what the value asks."),
        mo.Html(C.picture({"title": "From a value, through a norm, to a requirement",
                   "levels": ["value", "norm", "requirement"],
                   "columns": [["privacy", "keep only the data you need, only as long as you need it", "a CV is deleted after at most 30 days"],
                               ["fairness", "the tool is as accurate for every group of applicants", "the error rates of the groups differ by at most 0.05"],
                               ["transparency", "an applicant can learn why", "at most 0.05 of rejections come without a reason"]],
                   "alt": "Three values of a hiring tool, each with a norm and a requirement. Privacy: keep only the data you need, only as long as you need it; a CV is deleted after at most 30 days. Fairness: the tool is as accurate for every group of applicants; the error rates of the groups differ by at most 0.05. Transparency: an applicant can learn why; at most 0.05 of rejections come without a reason."})),
        mo.md(r"""
        A hiring tool made up by the course: three values, each with one norm and one requirement. A real hierarchy has more of both.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To design with a value:

    1. Name the value, and whose value it is.
    2. Write the norms: what the value asks of this system.
    3. Turn each norm into a requirement with a number or a yes or no.
    4. Measure the system and compare each measurement with its requirement.
    5. Trace each failed requirement back to its value, and decide with the people involved what to change.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A town hall's chatbot, made up by the course. The value *being reachable* gives the norm *citizens get an answer quickly*, and the requirement *an answer within 5 seconds*. The value *reliability* gives the norm *the chatbot does not tell citizens wrong things about their rights*, and the requirement *at most 0.02 of the answers wrong in a monthly test*. `limit` holds the most each requirement allows, `measured` what the team found. Run the cell.
    """)
    return


@app.cell
def example_chatbot():
    _limit = {"seconds": 5, "wrong answers": 0.02}
    _measured = {"seconds": 3.2, "wrong answers": 0.04}

    for _requirement in _limit:
        print(_requirement, _measured[_requirement] <= _limit[_requirement])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The chatbot is quick enough, but it gets twice as many answers wrong as the requirement allows. Step 5 of the recipe: that failure traces back to reliability, so that is what the team talks about next. Now the hiring tool of the picture: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_met():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which requirements are met?

        The hiring tool of the picture. `limit` holds the most each requirement allows, `measured` what the team found, and the loop goes through the requirements with `requirement`. Replace the `...` with one line, as in the worked example, but put the answer in `met` instead of printing it. The last line is `met`, so the check sees it. The check also tries other measurements.
        """),
        L.task("Put in `met`, under each requirement, True when the measured value is at most the limit, and False when it is not."),
    ])
    return


@app.cell(hide_code=True)
def exercise_met():
    box_met = L.code_box("met", lines=7)
    box_met
    return (box_met,)


@app.cell(hide_code=True)
def check_met(box_met):
    result_met = L.check("met", box_met.value)
    result_met
    return (result_met,)


@app.cell(hide_code=True)
def hints_met():
    L.hints(
        "met",
        [
            "Step 4 of the recipe: the comparison of the worked example, with its answer stored under the requirement.",
            "Which comparison says *at most*? And how do you put a value under a key in a dictionary?",
            "`met[requirement] = measured[requirement] ___ limit[requirement]`",
        ],
        why="`met[requirement] = measured[requirement] <= limit[requirement]` gives `{'days kept': False, 'error gap': True, 'no reason': False}`: the tool keeps CVs too long and rejects too many applicants without a reason.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_levels():
    mo.md(r"""
    ### Exercise 2. Value, norm or requirement?

    Six statements about a health app, made up by the course. Put each one at its level of the values hierarchy.
    """)
    return


@app.cell(hide_code=True)
def sort_levels_cell():
    sort_levels = L.sort("levels")
    sort_levels
    return (sort_levels,)


@app.cell(hide_code=True)
def check_levels(sort_levels):
    result_levels = L.check("levels", sort_levels)
    result_levels
    return (result_levels,)


@app.cell(hide_code=True)
def hints_levels():
    L.hints(
        "levels",
        [
            "Ask for each statement: is it one broad word, a rule for this app, or something you could test today?",
            "Two statements are single words. Of the four sentences, two could pass or fail a test as they stand.",
            "Value: autonomy and \\_\\_\\_. Norm: \\_\\_\\_ and no hurt. Requirement: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Value:** autonomy and safety. **Norm:** choose and no hurt. **Requirement:** off and a list.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_participation():
    mo.md(r"""
    ### Exercise 3. Taking part
    """)
    return


@app.cell(hide_code=True)
def choice_participation_cell():
    choice_participation = L.choice(
        "participation",
        question="A city builds an AI tool that plans its bus routes, and asks riders to fill in a survey. What would participatory design add?",
        options={
            "survey": "A longer survey, with more questions about the routes",
            "experts": "A panel of experts who decide on behalf of the riders",
            "right": "Riders take part in deciding what the tool optimizes",
        },
        correct="right",
        why={
            "survey": "A survey asks riders what they think; the city still decides alone. What does participatory design add to that? See *The idea*.",
            "experts": "Experts deciding for riders is the opposite: the riders still have no say. See *The idea*.",
            "right": "The riders help choose the values and the norms, such as short trips for most people or a bus near every home, not only answer questions afterwards.",
        },
    )
    choice_participation
    return (choice_participation,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Principles and guidelines](/course/open/rai-principles?lang=en) you saw that principles are short words that each document fills in its own way. A values hierarchy is how you fill one in for a single system, step by step, so that every requirement can be checked and traced back. In the next lesson, *Guidance ethics and DEDA*, you meet two Dutch methods that guide a team through these choices together.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_participation, result_levels, result_met):
    L.wrap_up(result_met, result_levels, choice_participation)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
