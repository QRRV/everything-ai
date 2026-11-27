# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-lifecycle. Route: course/route/lessen/systemen.yaml. Checks, the life of the spam service and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-lifecycle", "en")
    C = L.checks   # the life of the course's spam service, the stages, the roles and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parts():
    quick_parts = L.choice(
        "quick-parts",
        question="In the course's spam service, which part notices that the answers look different from usual?",
        options={
            "verification": "Verification, which checks each message as it comes in",
            "right": "Monitoring, which watches the answers as they go out",
            "serving": "Serving, which always gives an answer",
            "collection": "Collection, which takes each message once",
        },
        correct="right",
        why={
            "right": "Monitoring looks at the answers over time, such as the share of spam or of answers without a score. Over the life of a system, monitoring is what tells the team that it is time to look again: this lesson follows a system through that life.",
            "verification": "Verification looks at one message at a time: is it usable? Which part looks at many answers together? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "serving": "Serving turns the probability of the model into an answer, one message at a time. Which part watches the answers that go out? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "collection": "Collection makes sure that each message is handled once. Which part looks at the answers over time? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_parts
    return (quick_parts,)


@app.cell(hide_code=True)
def quick_check_debt():
    quick_debt = L.choice(
        "quick-debt",
        question="A team takes a shortcut in its code to put a service into use a week earlier. What does technical debt say about the shortcut?",
        options={
            "right": "It saves time now and costs more later, each time someone works around it",
            "tests": "It costs nothing, as long as the tests still pass",
            "never": "It only matters for code that is never changed again",
            "data": "It is paid back by itself when the model is trained again on fresh data, with no extra work",
        },
        correct="right",
        why={
            "right": "A shortcut works like a loan: the interest comes later, every year the system runs and someone has to change the code around it. This lesson is about those years.",
            "tests": "Tests show what the code does now. What happens each time someone has to change the code around the shortcut? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "never": "Code that is never changed costs little. But for how long is a system changed and looked after? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "data": "Training again changes the model, not the code around it. Who pays for a shortcut in that code? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_debt
    return (quick_debt,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2008, Google started Flu Trends: from what people searched for, it estimated how much flu there was in the United States, faster than the CDC, the American health agency, which counts visits to doctors. In 2009, researchers at Google described the model in *Nature*. When it was built, its estimates matched years of the numbers of the CDC. In February 2013, Flu Trends made the news because its estimate was far off, and in August 2015 Google stopped publishing the estimates. The model was built in 2008 and updated in 2009; what went wrong happened in the years after, while the model hardly changed and its world did. This lesson follows an AI system through its whole life, from the first idea to the day it is switched off, and looks at who is involved along the way.

    **An everyday example.** A car is designed, tested and built in a few years, and then driven for ten or twenty. Along the way it gets a yearly inspection and repairs, and sometimes a part is replaced because the rules changed. One day the car is scrapped, and someone has to deal with its battery and its oil. Most of the life of a car comes after the factory, and so does much of what it costs. Many people take part along the way: the maker, the dealer, the owner, the garage, the inspectors, and the people who live along the roads it drives on.
    """)
    return


@app.cell(hide_code=True)
def predict_estimate():
    guess_estimate = L.guess(
        "estimate",
        question="In the flu season of 2012 to 2013, Google Flu Trends estimated the share of doctor visits in the United States that were for flu-like illness. How far was its estimate for the peak from what the CDC counted, do you think?",
        options={
            "close": "Within 5%",
            "fifth": "About 20% too high",
            "right": "About double",
            "ten": "About ten times too high",
        },
        reveal="**About double.** In February 2013, *Nature* reported that the estimate of Flu Trends for the peak of the season was almost double what the CDC counted; in 2014, David Lazer and others wrote in *Science* of more than double. From August 2011 to September 2013, its estimate was too high in 100 of 108 weeks. The model had been updated in 2009 and then ran almost unchanged until October 2013, while Google kept changing its own search engine: 86 changes in June and July 2012 alone, as its search blog reported.",
    )
    guess_estimate
    return (guess_estimate,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A life in stages.** An AI system is not done when the model is trained. It begins with an idea, is built and checked, goes into use, is watched for years, is changed now and then, and one day is switched off. These stages together are the **AI lifecycle**. In earlier lessons, a project ended with one measurement on the test set; in a real system, that measurement comes early: in the spam service, in month 5 of 36.

        **ISO/IEC 22989.** **ISO/IEC 22989**, an international standard from 2022, fixes the basic words and ideas of AI. It describes the life cycle of an AI system in eight stages, and ISO/IEC 5338, from 2023, builds on it with the work of each stage, based on the standards for the life cycle of systems and software. The picture shows the stages; the table follows the course's spam service through them, as it would run in the mailbox of a city.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        | Stage | What happens | The spam service |
        |---|---|---|
        | **inception** | the problem, what a mistake costs, a goal you can measure, and whether AI is needed at all | a real question in the spam folder is worse than spam in the inbox; at least 0.90 right |
        | **design and development** | data, features, the model and all the code around it | the eight parts of the service, and labeled messages to learn from |
        | **verification and validation** | verification: is it built as agreed? validation: does it do what the users need? | accuracy above the goal on test messages; staff of the city try it on a week of mail |
        | **deployment** | into the place where it is used | into the mailbox of the city, and staff learn to mark mistakes |
        | **operation and monitoring** | it runs, and its answers are watched | each month, staff check a sample of messages by hand |
        | **continuous validation** | for a system that keeps learning: each new version is checked before it answers | the model trained again on marks is checked on the test messages |
        | **re-evaluation** | a new look at the whole: still needed, still good enough? | after the alert: back to design; in month 35, not needed anymore: retirement |
        | **retirement** | switched off, users told, data deleted or kept as agreed | the stored messages and marks are deleted |

        **Not a straight line.** A re-evaluation can find that all is well, and then the system keeps running. Otherwise it has two ways out. When the system is still needed but no longer good enough, it goes back to design and development, and through verification and deployment again. When it is not needed anymore, or cannot be made good enough again, it goes to retirement. A system that keeps learning, as the spam service does from the marks of users, also gets continuous validation beside its operation: a model that changes by itself must be checked each time it changes.

        **Who is involved.** At each stage, others take part, and each party has interests of its own: an **AI stakeholder**. ISO/IEC 22989 gives the roles names:

        | Role | What it does | Around the spam service |
        |---|---|---|
        | **AI provider** | offers an AI product or service | the company |
        | **AI producer** | designs and builds the AI | the company |
        | **AI customer** | uses an AI product or service, often after buying it | the city |
        | **AI partner** | helps around the AI, for example with data, with fitting it into other systems, or with an audit | a labeling firm, which marks messages as spam or not |
        | **AI subject** | is affected by the system: the people whose data it uses or about whom it decides | the residents who mail the city |
        | **relevant authorities** | make the rules and see that they are kept | the privacy regulator |

        One party can have several roles: the company that builds the service and sells it is both AI producer and AI provider. And an **AI subject** often never sees the system: the residents only mail the city, but a real question in the spam folder hurts them first.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **AI lifecycle** | | the stages an AI system goes through, from the first idea through building, checking, use and monitoring to retirement; ISO/IEC 22989 writes life cycle | |
        | **ISO/IEC 22989** | I-S-O I-E-C twenty-two nine eighty-nine | an international standard of 2022 with the basic words and ideas of AI, among them the stages of the life cycle and the roles of the parties | |
        | **AI stakeholder** | | a party that is involved in an AI system or affected by it, in one of the roles of ISO/IEC 22989 | |
        | **AI subject** | | a person or group affected by an AI system: whose data it uses, or about whom it decides | |
        """),
        L.watch_out("The EU AI Act names roles too, with other words and legal meanings: the provider develops an AI system, or has it developed, and places it on the market or puts it into service under its own name or trademark; the deployer uses an AI system under its own authority. Keep the two sets apart: a party can be an AI customer in the standard and a deployer under the law, and the words do not always line up."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Following an AI system through its life:

    1. At inception, write down the problem, what each kind of mistake costs, and a goal you can measure; ask whether AI is needed at all.
    2. Name the parties and their roles: who offers it, who builds it, who uses it, who helps, who is affected, and who makes the rules.
    3. Before deployment, check both: that it is built as agreed (verification), and that it does what its users need (validation).
    4. In operation, measure against the goal regularly, and agree in advance what a drop below the goal sets off.
    5. At each re-evaluation, choose: keep it running, go back to design, or retire it; and plan the retirement: who is told, and what happens to the data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.HISTORY` is the life of the course's spam service in the mailbox of a city, invented by the course: a list of entries `(stage, months)`, in order. The cell prints each entry and adds up the months of the whole life. Run the cell.
    """)
    return


@app.cell
def example_history():
    history = C.HISTORY
    total = 0
    for stage, months in history:
        print(stage, months)
        total = total + months
    print("months in all:", total)
    return history, total


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The service lived 36 months. Several stages come twice: after the first period of operation, a re-evaluation sent the service back to design and development, through verification and validation and deployment, into a second period of operation. Exercise 1 adds up the months of each stage.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_months():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Where the months go

        `months_per_stage(history)` gets a list of entries `(stage, months)`, and gives back a dict from each stage to its months in all. The loop goes through the entries; when a `stage` comes for the first time, `totals[stage]` starts at 0. Replace the `...` with the new value of `totals[stage]`: what it held so far, plus the `months` of this entry.

        The check calls `months_per_stage(C.HISTORY)`. Then an empty list, a history of one entry, three entries of one stage, halves of a month, and another history.
        """),
        L.task("Add up the months of each stage, also when a stage comes back."),
    ])
    return


@app.cell(hide_code=True)
def exercise_months():
    box_months = L.code_box("months", lines=8)
    box_months
    return (box_months,)


@app.cell(hide_code=True)
def check_months(box_months):
    result_months = L.check("months", box_months.value)
    result_months
    return (result_months,)


@app.cell(hide_code=True)
def hints_months():
    L.hints(
        "months",
        [
            "`totals[stage]` holds the months of `stage` so far, and `months` holds the months of this entry. The new value of `totals[stage]` uses both.",
            "Add, do not replace: a stage that comes back, such as operation and monitoring, should get the months of both periods. Which value holds the months of this stage so far, and what do you add to it?",
            "`        totals[stage] = ___`",
        ],
        why="`totals[stage] = totals[stage] + months` adds the months of each entry to what its stage already had. Operation and monitoring gets 14 + 9 = 23 months, of 36 in all.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_below():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. When to look again

        `first_month_below(accuracies, goal)` gets the accuracy of each month of operation, in order, and the goal agreed on at inception. It gives back the number of the first month whose accuracy is below the goal, counted from 1, or `None` when no month is below it. `month` counts the months; in each round of the loop, `accuracy` is the accuracy of that month. Replace the `...` with the condition: `accuracy` is below `goal`. A month exactly at the goal is still good enough.

        The check calls `first_month_below(C.ACCURACY, 0.9)`: the 14 months of the first period of operation of the service. Then a month exactly at the goal, accuracies that never drop below it, an empty list, a first month below it, and a goal of 0.8.
        """),
        L.task("Give back the first month whose accuracy is below the goal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_below():
    box_below = L.code_box("below", lines=8)
    box_below
    return (box_below,)


@app.cell(hide_code=True)
def check_below(box_below):
    result_below = L.check("below", box_below.value)
    result_below
    return (result_below,)


@app.cell(hide_code=True)
def hints_below():
    L.hints(
        "below",
        [
            "Step 4 of the recipe. In each round, `accuracy` belongs to month `month`, and the condition compares it with `goal`.",
            "Below the goal means smaller than the goal, and a month exactly at the goal does not count. Which comparison says that?",
            "`        if ___:`",
        ],
        why="`if accuracy < goal:` stops at the first month below the goal and gives back its number. In the service, that is month 13 of the first period of operation, with an accuracy of 0.89.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The life of the service

    The demo goes through the 36 months of the service, one at a time. The picture has a row per stage: the months the service spent in it, dark up to now and light for what is still to come, with a line at the current month. Under the rows is the accuracy of each month of operation, with the goal as a dashed line, and under that who does something in this stage.
    """)
    return


@app.cell(hide_code=True)
def demo_life():
    L.explore("month", "month", lambda _v: C.frame(int(_v), L.lang), start=1, values=list(range(1, 37)), role="",
              task="Go through the 36 months: in which month does the service leave operation for the first time, and why?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The service spends 23 of its 36 months in operation and monitoring. In month 19, the accuracy drops below the goal for the first time, though feedback trained the model again in months 12 and 17. In month 21, the city and the company look again: the service is still needed, so it goes back to design and development, and version spam-2 answers from month 26. In month 35, a second re-evaluation ends in retirement: not because the service got worse, but because it was no longer needed. The city takes part in every stage except design and development; the residents never see the service, but their messages are what it decides about.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flu():
    mo.md(r"""
    ### Exercise 3. Google Flu Trends

    The model of Google Flu Trends was updated in 2009 and then ran almost unchanged until October 2013. Meanwhile, Google kept changing its search engine, and people changed what they searched for. From August 2011 to September 2013, the estimates were too high in 100 of 108 weeks, while the CDC published its own numbers every week.
    """)
    return


@app.cell(hide_code=True)
def choice_flu_cell():
    choice_flu = L.choice(
        "flu",
        question="Which stage of the life cycle fell short most?",
        options={
            "inception": "Inception: searches could never tell anything about flu, from the very start",
            "deploy": "Deployment: too few people visited the website",
            "right": "Re-evaluation: the model was rarely checked against a changing world",
            "retire": "Retirement: Google switched it off too early",
        },
        correct="right",
        why={
            "right": "The model fit the world of 2009. Search changed, and the model hardly did, while the numbers of the CDC could have shown every week that the estimates drifted. A regular re-evaluation against the numbers of the CDC would have caught it and sent the model back to design; in the end, only retirement was left.",
            "inception": "When it was built, the estimates matched years of the numbers of the CDC, so searches did say something about flu. What changed in the years after?",
            "deploy": "The estimates were published for anyone to see. Was the problem who saw them, or what they said?",
            "retire": "By 2015, the estimates had been too high for years. Which stage should have come long before retirement?",
        },
    )
    choice_flu
    return (choice_flu,)


@app.cell(hide_code=True)
def text_exercise_subject():
    mo.md(r"""
    ### Exercise 4. Who is it about?

    In the demo, residents mail their questions to the city. They never see the spam service and never chose it: the city bought it from the company.
    """)
    return


@app.cell(hide_code=True)
def choice_subject_cell():
    choice_subject = L.choice(
        "subject",
        question="Which role does ISO/IEC 22989 give the residents?",
        options={
            "none": "None: only the parties that build, sell or use a system have a role",
            "customer": "AI customer, since the mailbox is there for them",
            "partner": "AI partner, since their messages are the data",
            "right": "AI subject, since the service decides about their messages",
        },
        correct="right",
        why={
            "right": "An AI subject is affected by the system: by what it does with their data, or by what it decides about them. The residents never use the service, but a real question in the spam folder hurts them first. Their interests come up at inception, when the city and the company agree that a missed real question is the worst mistake.",
            "none": "The roles of the standard also cover parties that never touch the system. Whose questions end up in the spam folder when the service is wrong?",
            "customer": "The city is the AI customer: it bought the service and uses it. What do the residents do with the service themselves?",
            "partner": "An AI partner helps make or check the system, as the firm that labels messages does. Do the residents help make the service, or does the service decide about them?",
        },
    )
    choice_subject
    return (choice_subject,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en), the spam service was a set of parts at one moment. This lesson followed it through time: from the idea, through building, checking and years of use, back to design once, and on to retirement, with the parties around it in their roles. In the next lesson, *Who builds an ML system: roles and teams*, you look inside the company: which people build and run such a system, and how they work together.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flu, choice_subject, result_below, result_months):
    L.wrap_up(result_months, result_below, choice_flu, choice_subject)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
