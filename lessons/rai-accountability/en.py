# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-accountability. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-accountability", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_allocative():
    quick_allocative = L.choice(
        "quick-allocative",
        question="A model gives fewer loans to people from one group. What kind of harm is that?",
        options={
            "right": "Allocative harm: it hands out less",
            "representational": "Representational harm: it shows a group badly",
            "none": "No harm at all, if accurate",
        },
        correct="right",
        why={
            "representational": "Representational harm is about how a group is shown, such as a stereotype. A loan is something handed out. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "A loan is something a model hands out, so a gap between groups is allocative harm, and you can measure it per group.",
            "none": "Accurate on average can still mean fewer loans for one group. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_allocative
    return (quick_allocative,)


@app.cell(hide_code=True)
def quick_check_stereotype():
    quick_stereotype = L.choice(
        "quick-stereotype",
        question="A search for pictures of a job shows fewer women than actually do that job. Which harm is that?",
        options={
            "allocative": "Allocative harm, because someone does not get the job",
            "real": "None: the images show the world as it is",
            "right": "Representational harm, a stereotype that sticks",
        },
        correct="right",
        why={
            "allocative": "Nobody is refused a job by the search itself; what goes wrong is how a group is shown. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "real": "The search shows fewer women than actually do the job, so it does not show the world as it is. See the CEO example in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "Showing a group less than it is there, or as a stereotype, is representational harm: you have to look for it.",
        },
    )
    quick_stereotype
    return (quick_stereotype,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Between 1985 and 1987, a radiation therapy machine called the Therac-25 gave at least six patients massive overdoses, and several of them died. Its software had bugs; its maker said an overdose was impossible; hospitals trusted the screen; operators had learned to skip past its error messages. Many people did their part, and nobody had arranged who answered for the whole. AI systems are built the same way, by many teams and companies, which makes the question *who is responsible?* hard to answer after something has gone wrong.

    **An everyday example.** In a shared kitchen at work, everyone uses the coffee machine and nobody descales it: each person thinks someone else will.
    """)
    return


@app.cell(hide_code=True)
def predict_months():
    guess_months = L.guess(
        "months",
        question="How long do you think it took, from the Therac-25's first known overdose, before regulators told hospitals to stop treating patients with it?",
        options={
            "week": "A few weeks",
            "months": "About two or three months",
            "right": "Over a year and a half",
        },
        reveal="**Over a year and a half**: from June 1985 to February 1987. The maker first said the machine could not have caused the overdoses. The FDA called it defective in May 1986, but it stayed in use while fixes were made, until a different bug caused another overdose in January 1987 (Leveson and Turner, 1993).",
    )
    guess_months
    return (guess_months,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Accountability** means that someone answers for what a system does: they can explain its decisions, can be corrected, and bear the consequences when it goes wrong. Three things make that hard.

        - *The problem of many hands.* When many people each build a part, everyone did their own part and nobody seems responsible for the whole. The philosopher Dennis Thompson named it in 1980 for governments; Helen Nissenbaum showed in 1996 how it works for computer systems.
        - *A moral crumple zone.* The crumple zone of a car takes the force of a crash. In the same way, the person closest to the machine takes the blame, even when the design of the system made the mistake likely (Madeleine Clare Elish, 2019).
        - A **responsibility gap**. A system that learns can do things its makers could not foresee and its users could not steer. Then nobody seems to deserve the blame (Andreas Matthias, 2004).

        The answer is to arrange accountability before something goes wrong: for every step a name, gaps and overlaps found and fixed, and a way for people to object.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **accountability** | being answerable for what an AI system does: someone can explain its decisions, can be corrected, and bears the consequences when things go wrong |
        | **responsibility gap** | the worry that when a learning system causes harm, nobody can fairly be blamed: the makers could not foresee it and the users could not control it |
        """),
        L.watch_out("Two names on a step is not twice as safe. When two teams both claim a step, each may count on the other: an overlap can leave a gap as wide as no name at all."),
        mo.Html(C.picture({"title": "Which team claims which step of a loan model", "teams": ["data", "model", "product", "service"],
                           "nobody": "nobody",
                           "alt": "Six steps of a made-up loan model against four teams. The data team claims collect data; the model team claims train model and test model; the product team claims test model and launch; the service desk claims nothing. Nobody claims set threshold or handle complaints."})),
        mo.md(r"""
        A bank made up by the course. A filled dot means that the team says it answers for that step.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To arrange accountability for a system:

    1. List the steps, from collecting data to handling complaints.
    2. Ask each team which steps it answers for.
    3. Count, for each step, how many teams claim it.
    4. Give every step with no name, or with more than one, a single owner.
    5. Write down how someone affected can object, and who answers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A hospital's system that suggests a dose of medicine, made up by the course. Three teams were asked which steps they answer for. `_count` starts at 0 for each of the three steps. The outer loop goes through the teams, the inner loop through the steps each one claims, and each claim adds 1 to the count of its step. Run the cell.
    """)
    return


@app.cell
def example_hospital():
    _claims = {"doctors": ["prescribe"], "pharmacy": ["prescribe"], "nurses": ["give medicine"]}
    _count = {"prescribe": 0, "check dose": 0, "give medicine": 0}

    for _team in _claims:
        for _step in _claims[_team]:
            _count[_step] = _count[_step] + 1

    for _step in _count:
        print(_step, _count[_step])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two teams claim *prescribe*, and nobody claims *check dose*, the step between prescribing and giving: its count stays 0. Gaps like that are easy to miss by eye, because a step nobody claims does not appear in anyone's answer. Now the bank of the picture, with six steps and four teams: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who claims what?

        The bank of the picture. `claims` holds, for each team, the steps it says it answers for, and `count` starts at 0 for every step. The outer loop goes through the teams with `team`, the inner loop through a team's steps with `step`. Replace the `...` in the inner loop with one line that adds 1 to the count of the step a team claims, as in the worked example. The last line is `count`, so the check sees it. The check also tries other answers from the teams.
        """),
        L.task("Make `count` say, for each step, how many teams claim it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count",
        [
            "Each time the inner loop comes to a step, one more team claims it: its count goes up by one.",
            "Where is the count of this step kept? And how do you put a new value there that is one more than the old one?",
            "`count[step] = ___ + 1`",
        ],
        why="`count[step] = count[step] + 1` gives `{'collect data': 1, 'train model': 1, 'test model': 2, 'set threshold': 0, 'launch': 1, 'handle complaints': 0}`: nobody claims the threshold or the complaints, and two teams claim the test.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_who():
    mo.md(r"""
    ### Exercise 2. Which idea fits?

    Six cases, made up by the course. Put each one with the idea that fits it.
    """)
    return


@app.cell(hide_code=True)
def sort_who_cell():
    sort_who = L.sort("who")
    sort_who
    return (sort_who,)


@app.cell(hide_code=True)
def check_who(sort_who):
    result_who = L.check("who", sort_who)
    result_who
    return (result_who,)


@app.cell(hide_code=True)
def hints_who():
    L.hints(
        "who",
        [
            "Ask for each case: is the blame spread over many, does it land on the person nearest the machine, or does nobody seem to deserve it?",
            "In two cases a single person is blamed. In two others the system did something nobody chose.",
            "Many hands: the credit score and \\_\\_\\_. Crumple zone: the safety driver and \\_\\_\\_. Responsibility gap: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Many hands:** the credit score and the chatbot. **A moral crumple zone:** the safety driver and the operator. **A responsibility gap:** it learned it, and nobody could know.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Exercise 3. When to arrange it
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when",
        question="A city will soon use an AI system to decide who gets help with their rent. When should it arrange who answers for what?",
        options={
            "complaint": "After the first complaint, once you know what went wrong",
            "right": "Before launch, with a name for every step",
            "court": "When a court or a journalist asks for it",
        },
        correct="right",
        why={
            "complaint": "By then, the many hands have each done their part and nobody owns the mistake, and someone has already been harmed. See *The idea*.",
            "right": "Before launch you can still give every step one owner and set up a way to object; afterwards, the blame tends to land on whoever is nearest.",
            "court": "Then it is too late to prevent the harm, and the blame lands where it is easiest, not where it belongs. See *The idea*.",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) you saw what a model can do to people. Here you asked who answers for it, and saw that this has to be arranged before the harm, not found afterwards. In the next lesson, *A person in the loop*, you look at one common answer, letting a person check the system's decisions, and at why it often works less well than it sounds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_count, result_who):
    L.wrap_up(result_count, result_who, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
