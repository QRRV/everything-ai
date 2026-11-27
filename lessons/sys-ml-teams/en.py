# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-ml-teams. Route: course/route/lessen/systemen.yaml. Checks, the plans of the four ways and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-ml-teams", "en")
    C = L.checks   # the roles, the plans of the four ways to organize the work, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_back():
    quick_back = L.choice(
        "quick-back",
        question="In the life cycle, where does a re-evaluation send a system that is still needed but no longer good enough?",
        options={
            "retire": "On to retirement",
            "right": "Back to design and development",
            "inception": "Back to inception, to start over with a new idea",
            "operation": "Straight back into operation, unchanged",
        },
        correct="right",
        why={
            "right": "Through design and development, verification and validation and deployment again: the system gets a new version. This lesson looks at who builds that new version, and how long it takes them.",
            "retire": "Retirement is for a system that is not needed anymore. Where does a system go that is still needed? See [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).",
            "inception": "The problem and the goal are still the same; only the system falls short. Which stage builds a new version? See [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).",
            "operation": "Running on unchanged would leave it below its goal. Which stage changes it? See [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).",
        },
    )
    quick_back
    return (quick_back,)


@app.cell(hide_code=True)
def quick_check_producer():
    quick_producer = L.choice(
        "quick-producer",
        question="A company builds the spam service and sells it to a city. Which roles does the company have, in the words of ISO/IEC 22989?",
        options={
            "customer": "AI customer and AI subject",
            "partner": "Only AI partner",
            "right": "AI producer and AI provider",
            "all": "AI producer, AI provider and relevant authorities",
        },
        correct="right",
        why={
            "right": "The company designs and builds the service (AI producer) and offers it (AI provider). Inside the company, the AI producer is not one person: this lesson looks at the people who build the service, and how they work together.",
            "customer": "The city uses the service, and the residents are affected by it. What does the company that builds and sells it do? See [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).",
            "partner": "An AI partner helps around the AI, as a labeling firm does. Which roles go with building a service and offering it? See [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).",
            "all": "Relevant authorities make the rules and see that they are kept, as the privacy regulator does. Is that the company that builds the service? See [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en).",
        },
    )
    quick_producer
    return (quick_producer,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2015, Uber began to build Michelangelo, its platform for machine learning. In September 2017, Jeremy Hermann and Mike Del Balso described on Uber's blog how Uber worked before it. Data scientists built models with a wide variety of tools, such as R and scikit-learn, while separate engineering teams built one-off systems to use these models in production. Uber had no established path to put a model into use, and what machine learning did for Uber was limited to what a few data scientists and engineers could build in a short time. A model is built by people, and how they work together decides how fast a new model gets into use, and whether it gets there at all. This lesson looks at the roles of the people who build and run a model, and at four ways to organize them.

    **An everyday example.** A new kitchen. Separate specialists, a plumber, an electrician and a tiler, each work fast at their own job, but the kitchen waits between them: the tiler can only start when the electrician is done, and the electrician has a full schedule. One handyman who does everything never waits for anyone, but takes longer on the wiring. A crew that works together waits less, but needs every trade for every kitchen. And a kitchen from a store, with ready-made parts, goes fastest, as long as you want what the store sells.
    """)
    return


@app.cell(hide_code=True)
def predict_models():
    guess_models = L.guess(
        "models",
        question="In 2024, Uber wrote on its engineering blog how many models it had in production. How many, do you think?",
        options={
            "fifty": "About 50",
            "hundreds": "About 500",
            "right": "More than 5,000",
            "half-million": "About half a million",
        },
        reveal="**More than 5,000.** In May 2024, Uber wrote that it had more than 5,000 models in production, serving 10 million predictions a second at the busiest moments, and that about 400 active ML projects ran on Michelangelo. In 2017, the post on Michelangelo said that dozens of teams used it. Uber's conclusion about the people behind it: the best structure has one central team for the ML platform, with data scientists and ML engineers inside each product team.",
    )
    guess_models
    return (guess_models,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The roles.** In [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en), the company was the AI producer. Inside the company, the work is split over roles. Dominik Kreuzberger, Niklas Kühl and Sebastian Hirschl (2023) list seven roles around MLOps, from the business stakeholder who sets the goal to the ML engineer/MLOps engineer. This lesson uses four roles that build a model and keep it running; the platform engineer is not one of the seven:

        | Role | What it does | In the spam service |
        |---|---|---|
        | **data scientist** | turns a question into an ML problem, collects data, trains and evaluates the model | collects and labels messages, and trains the spam model |
        | **ML engineer** | turns the model into reliable software: a tested service that answers fast enough | makes a service of the model that answers each message |
        | **MLOps engineer** | builds the path to production: pipelines, deployment and monitoring | puts the service into the mailbox, with alerts |
        | **platform engineer** | builds the shared tools that every team uses | builds the tools to train, deploy and watch any model of the company |

        The lines between the roles differ per company, and one person can have several roles: in a small company, one person does it all. The paper also names a data engineer, who builds the pipelines that bring the data in, and takes the ML engineer and the MLOps engineer together as one role.

        **Handoffs.** Each time the work passes from one person to another, there is a **handoff**. At a handoff, the work waits until the next person has time, and part of what the first person knew stays behind: why a feature was chosen, what the model does with odd messages. In 2019, Eric Colson of Stitch Fix wrote that the waits between pieces of work do more harm than the cost of coordinating people, and are often measured in days, weeks or even months.

        **Four ways.** A company can organize the same roles in different ways, and each way moves the place where the work gets stuck:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        | Way | Who does what | Where the work gets stuck |
        |---|---|---|
        | **teams by function** | a team of data scientists, a team of ML engineers and a team for operations; the model goes from team to team | in the queues between the teams, and in bugs that nobody can place, since each team sees one part |
        | **one team per product** | a data scientist, an ML engineer and an MLOps engineer in one team around one product | the waits are short, but every product needs every role, and specialists are scarce |
        | **one data scientist from start to finish** | one person does every step: an **end-to-end** data scientist | nobody waits, but the steps outside their field take longer, and few people can do all of it and want to |
        | **a data scientist with a platform** | one person does every step, with the tools of an **ML platform** that platform engineers build for every team | fast, as long as the platform can do what the model needs; and the platform first costs months of work |

        Chip Huyen (2022) names the cost of separate teams: the overhead of communication. Data scientists who do everything, Huyen writes, are hard to find, and not always happy doing it, unless good tools take the infrastructure off their hands. That is the job of a platform. Michelangelo is one: it covers six steps for every team at Uber, from managing data to monitoring the predictions.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **data scientist** | | someone who turns a question into an ML problem, and builds and evaluates the model | `"data scientist"` in `C.PLAN` |
        | **ML engineer** | M-L engineer | someone who turns a model into reliable software that answers fast enough, and keeps it running | `"ML engineer"` in `C.PLAN` |
        | **MLOps engineer** | M-L-ops engineer | someone who builds the path to production: pipelines, deployment and monitoring | `"MLOps engineer"` in `C.PLAN` |
        | **platform engineer** | | someone who builds the shared tools that every team uses | |
        | **ML platform** | M-L platform | the shared tools a company gives its teams to train, deploy and watch models | `C.PLATFORM_DAYS` |
        | **handoff** | | the moment the work passes to someone else: it waits there, and some of what the first person knew stays behind | two steps in a row with a different `who` |
        | **end-to-end** | | done by one person or team, from the data to a model in use | `C.PLANS["start to finish"]` |
        """),
        L.watch_out("Job titles mean different things at different companies: a data scientist at one company does what an ML engineer does at another, and an MLOps engineer may be called a platform engineer. Read what a job does, not what it is called."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Choosing how to organize the work on models:

    1. Write down the steps from the data to a model in use, and who does each step.
    2. Count the handoffs, and estimate how long the work waits at each: a day or two inside one team, often more between teams.
    3. Look for the steps someone does outside their own field: they take longer, and they go wrong more often.
    4. Count the models the company will build: a platform costs months of work before it saves anything, so it pays off only over many models.
    5. Check what the platform cannot do yet: work that falls outside it waits in the queue of the platform engineers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.PLAN` is the plan for a new version of the course's spam model at a company with teams by function, invented by the course: a list of steps `(step, who, days)`, in order. The cell prints each step and adds up the days of work. Run the cell.
    """)
    return


@app.cell
def example_plan():
    plan = C.PLAN
    work = 0
    for step, who, days in plan:
        print(step, "-", who, "-", days, "days")
        work = work + days
    print("days of work:", work)
    return plan, work


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The plan has 28 days of work. But the work passes from the data scientist to the ML engineer, and from the ML engineer to the MLOps engineer, each in another team. If the work waits 10 days in the queue of each team, the new version answers after 28 + 10 + 10 = 48 days, and for 20 of them nobody works on it. Exercise 1 counts the handoffs in a plan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_handoffs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Count the handoffs

        `handoffs(plan)` gets a plan as a list of steps `(step, who, days)`, in order, and gives back how many times the work passes to someone else. `count` counts the handoffs, and the loop takes each step apart into `step`, `who` and `days`. `previous` holds the `who` of the step before; it starts at `None`, since the first step has no step before it, and the condition checks that with `previous is not None`. Replace the `...` with the second half of the condition, which tells whether this step is a handoff.

        The check calls `handoffs(C.PLAN)`. Then an empty plan, a plan of one step, the plan of one data scientist from start to finish, `C.BACK_AND_FORTH`, in which the ML engineer sends the model back once, and a plan in which the work goes back and forth between two people.
        """),
        L.task("Count how many times the work passes to someone else."),
    ])
    return


@app.cell(hide_code=True)
def exercise_handoffs():
    box_handoffs = L.code_box("handoffs", lines=8)
    box_handoffs
    return (box_handoffs,)


@app.cell(hide_code=True)
def check_handoffs(box_handoffs):
    result_handoffs = L.check("handoffs", box_handoffs.value)
    result_handoffs
    return (result_handoffs,)


@app.cell(hide_code=True)
def hints_handoffs():
    L.hints(
        "handoffs",
        [
            "Step 2 of the recipe: a handoff is a step whose `who` is not the `who` of the step before. In the loop, `previous` holds that person.",
            "The condition compares the person of this step with the person of the step before. Which comparison says \"is not equal to\"?",
            "`        if previous is not None and ___:`",
        ],
        why="`who != previous` is true for each step whose person differs from the person of the step before. `C.PLAN` has 2 handoffs, and `C.BACK_AND_FORTH` has 4: the work goes to the ML engineer, back to the data scientist, to the ML engineer again, and on to the MLOps engineer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_payoff():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. When a platform pays off

        A platform costs work before it saves any: in the course's example, two platform engineers work about six months on it, 240 days of work. After that, a data scientist with the platform needs 3 days to package a new version and 1 to deploy it, where one data scientist without a platform needs 10 and 8: a new version takes 24 days of work instead of 38, a saving of 14 days on each model. The example leaves out the work of keeping the platform running.

        `models_to_pay_off(platform_days, saved)` gives back how many models it takes before the days saved add up to at least the days the platform cost. `models` counts the models, and `gained` adds up the days saved so far: after each model, `saved` more. A `while` loop goes on as long as its condition is true. Replace the `...` with the condition under which the platform has not paid off yet.

        The check calls `models_to_pay_off(240, 14)`. Then a platform of 100 days that saves 10 on each model, no platform at all, a platform that pays off with its first model, a platform of 50 days that saves 7, and the platform of 240 days compared with teams by function, where it saves 4 days of work on each model.
        """),
        L.task("Count the models until the days saved reach the days the platform cost."),
    ])
    return


@app.cell(hide_code=True)
def exercise_payoff():
    box_payoff = L.code_box("payoff", lines=8)
    box_payoff
    return (box_payoff,)


@app.cell(hide_code=True)
def check_payoff(box_payoff):
    result_payoff = L.check("payoff", box_payoff.value)
    result_payoff
    return (result_payoff,)


@app.cell(hide_code=True)
def hints_payoff():
    L.hints(
        "payoff",
        [
            "Step 4 of the recipe. The loop adds one model at a time and should stop as soon as `gained` reaches `platform_days`. The condition of a `while` says when to go on, not when to stop.",
            "Compare `gained` with the cost of the platform. Which comparison says \"still smaller than\"?",
            "`    while ___:`",
        ],
        why="`while gained < platform_days:` adds models until the days saved reach the cost of the platform. With 14 days saved on each model, 17 models save 238 days, just short of 240, so the platform pays off from the 18th model on.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### One new version, four ways

    The demo follows one new version of the spam model, day by day, at four companies that each organize the work in one of the four ways, with the plans of `C.PLANS`. Each bar has the days of work filled in, dark up to now and light for what is still to come, and the days of waiting as dashed boxes; the line is the current day. Under each bar is what happens now. The days are invented by the course: 28 days of work, and 10 days of waiting at each handoff between teams, or 2 inside one team.
    """)
    return


@app.cell(hide_code=True)
def demo_days():
    L.explore("day", "day", lambda _v: C.frame(int(_v), L.lang), start=1, values=list(range(1, 49)), role="",
              task="Go through the 48 days: in which way does the work wait longest, and where does the data scientist from start to finish lose time?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With teams by function, the new version takes 48 days, and for 20 of them nobody works on it. In one team per product, the waits shrink to 4 days, but each product team needs its own ML engineer and MLOps engineer, who work 5 and 3 days on this version. The data scientist from start to finish never waits, but spends 18 days packaging and deploying, where the specialists need 8. With a platform, the new version answers after 24 days, because platform engineers worked 240 days on the platform first; in exercise 2, it paid off from the 18th model on.

    Real companies mix the ways: Uber's conclusion of 2024 combines one central platform team with data scientists and ML engineers inside each product team. And the waits are real. In a survey of 403 business leaders at companies with at least 100 million dollars in revenue, published in December 2020 by Algorithmia, a company that sells MLOps tools, 64% said it takes their organization a month or longer to deploy a model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_role():
    mo.md(r"""
    ### Exercise 3. Whose work is it?

    Five product teams at a company each built their own way to put a model into use, with their own servers and their own alerts. The company wants one way that every team can use.
    """)
    return


@app.cell(hide_code=True)
def choice_role_cell():
    choice_role = L.choice(
        "role",
        question="Which role builds it?",
        options={
            "data": "A data scientist in each team, who knows the models best",
            "mle": "The ML engineer of the spam service, who knows serving",
            "authority": "The relevant authorities, who make the rules",
            "right": "A platform engineer, who builds tools for every team",
        },
        correct="right",
        why={
            "right": "Shared tools for every team are the work of platform engineers: an ML platform, such as Michelangelo at Uber. The teams stop building their own ways, and five ways to look after become one.",
            "data": "A data scientist knows the models of one team. Who builds tools that all five teams can use?",
            "mle": "The ML engineer of the spam service builds the service of one product. Whose work is a tool for every product?",
            "authority": "The relevant authorities make the rules around a system, outside the company. Who inside the company builds tools for all teams?",
        },
    )
    choice_role
    return (choice_role,)


@app.cell(hide_code=True)
def text_exercise_stuck():
    mo.md(r"""
    ### Exercise 4. Where it gets stuck

    With the platform, a data scientist puts a newly trained spam model into use in 4 days: 3 to package it and 1 to deploy it. Now the spam service must also run in the browser of the residents, and the platform only serves models on its own servers.
    """)
    return


@app.cell(hide_code=True)
def choice_stuck_cell():
    choice_stuck = L.choice(
        "stuck",
        question="Where does the work get stuck?",
        options={
            "right": "In the queue of the platform team, until the platform can do it",
            "nowhere": "Nowhere: with a platform, one person can do every step",
            "handoff": "Between the ML engineer and the MLOps engineer",
            "data": "In collecting data: the browser needs other messages, so the data scientist starts over",
        },
        correct="right",
        why={
            "right": "A platform is fast for what it can do. For anything new, the data scientist waits until the platform engineers build it, or builds it alone, and is then back to one data scientist from start to finish.",
            "nowhere": "The data scientist can do every step the platform supports. What happens with a step it does not support?",
            "handoff": "With a platform, the data scientist does all the steps. Who has to make the platform serve models in a browser?",
            "data": "The messages are the same; only the place where the model runs changes. Which part of the work does the platform not cover?",
        },
    )
    choice_stuck
    return (choice_stuck,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [From idea to retirement: the life cycle of an AI system](/course/open/sys-lifecycle?lang=en), the company that built the spam service was one party, the AI producer. This lesson looked inside it: the roles of the people who build and run a model, the handoffs between them, and four ways to organize them, each with its own place where the work gets stuck. In the next lesson, *Reliable, scalable, maintainable, adaptable*, you look at what such a team has to deliver: a system that keeps working, grows with its users, can be changed by others, and follows a changing world.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_role, choice_stuck, result_handoffs, result_payoff):
    L.wrap_up(result_handoffs, result_payoff, choice_role, choice_stuck)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
