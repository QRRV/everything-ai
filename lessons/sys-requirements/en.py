# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-requirements. Route: course/route/lessen/systemen.yaml. Checks, the busy day of the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-requirements", "en")
    C = L.checks   # the answer times, the outages, the busy day of the demo, and the pictures


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
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="In the course's spam service, how many of the 118 lines of code were the model?",
        options={
            "most": "About 100 of the 118",
            "half": "About half",
            "right": "5 of the 118",
            "all": "All 118: the settings count as model",
        },
        correct="right",
        why={
            "right": "5 lines: scikit-learn does the learning, and the rest collects, checks, serves and monitors. This lesson looks at what that rest has to deliver.",
            "most": "Most of the code was not the model at all. What did the other parts do, such as collecting and serving? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "half": "The model was much smaller than the rest. Which library did the learning for it? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "all": "The settings are one of the parts around the model, beside collecting, serving and monitoring. How many lines were the model itself? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def quick_check_production():
    quick_production = L.choice(
        "quick-production",
        question="What does production ask of a system that research often does not?",
        options={
            "score": "The best score on a fixed test set",
            "right": "Good answers for many users, quick and cheap enough",
            "biggest": "The biggest model there is, trained once on all data",
            "author": "Code that only its author can run",
        },
        correct="right",
        why={
            "right": "And on data that keeps changing. This lesson turns that into four requirements, each with a number you can check.",
            "score": "That is what research asks for. What does a system need when many users depend on it? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "biggest": "A bigger model is often slower and more expensive. What do the users of a system ask for? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "author": "Code that only one person can run is a risk in production. What does production ask for instead? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_production
    return (quick_production,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    On 1 August 2012, Knight Capital, a large trading firm in American stocks, put new code into use on the eight servers that sent its orders to the stock market. The new code reused a flag that had once switched on Power Peg, an old function that Knight had stopped using in 2003 but never removed. A technician copied the new code to seven of the eight servers, and nobody checked the work. On the eighth server, where the old code was still in place, the flag switched Power Peg back on, and when the market opened, that server sent a flood of orders, far more than its customers had asked for.

    The new code was not the problem: it worked on the seven servers that had it. What failed was the system around it: old code that nobody removed, and a deployment by hand (putting the new code into use) without a second pair of eyes. A system in production is judged on more than whether its answers are right. This lesson names four requirements, and shows how to turn each into a number you can check.

    **An everyday example.** A bridge has to carry traffic every day, even in a storm. It has to carry more traffic when the town grows. A new team of engineers has to be able to inspect and repair it. And it has to cope with heavier trucks than the ones it was built for. A bridge that is strong today, but that nobody can inspect, is not a good bridge.
    """)
    return


@app.cell(hide_code=True)
def predict_minutes():
    guess_minutes = L.guess(
        "minutes",
        question="The flood of orders cost Knight Capital more than 460 million dollars. How long did the flood last, do you think?",
        options={
            "right": "45 minutes",
            "days": "Two days",
            "weeks": "Two weeks",
            "half-year": "Half a year",
        },
        reveal="**45 minutes.** According to the SEC, the market regulator of the US, the eighth server sent more than 4 million orders in 45 minutes, while it tried to fill 212 orders of customers, and Knight lost more than 460 million dollars. In 2013, Knight agreed to pay the SEC a penalty of 12 million dollars.",
    )
    guess_minutes
    return (guess_minutes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Four requirements.** Chip Huyen (2022) names four requirements that most ML systems in production have to meet. A model with a good score on a test set meets none of them by itself:

        | Requirement | The question | A number you can check | In the spam service |
        |---|---|---|---|
        | **reliability** | does it keep doing the right thing at the right level, even when hardware, software or people make mistakes? | the share of messages that get a score in time | 99% of messages get a score within 1 minute |
        | **scalability** | does it keep up when the messages, the data or the system grow, and can it shrink again? | the messages a minute it keeps up with at the busiest moment | 1,400 messages a minute after a storm warning |
        | **maintainability** | can others understand it, change it and put the change into use? | the time from a change to its use, and how many people can make it | a new colleague changes the threshold and puts it into use in one day |
        | **adaptability** | does it follow a changing world, without stopping the service? | the days from a new kind of spam to a model that catches it | a new kind of spam caught within 7 days |

        Each requirement has gone wrong at a real company:

        - **reliability:** on 28 February 2017, someone at Amazon entered a command with an incorrect input and took more servers out of service than intended. S3, the storage service of Amazon, took about four hours to work fully again in one region of the US, partly because some of its parts had not been fully restarted for many years.
        - **scalability:** on 25 March 2025, OpenAI added a new way to make images to ChatGPT. Two days later, Sam Altman, the CEO of OpenAI, wrote "our GPUs are melting", and OpenAI limited for a while how many images people could make.
        - **maintainability:** Knight Capital, above: old code that was never removed, and a deployment by hand that nobody checked.
        - **adaptability:** in November 2021, Zillow stopped Zillow Offers, its business of buying and selling houses. Zillow's CEO wrote that the prices of houses had been far harder to forecast than Zillow had expected; Zillow wrote 304 million dollars off the value of its houses, and planned to let about a quarter of its staff go.

        **ML fails silently.** A server that is down gets noticed within minutes. A model that gives worse answers keeps answering, and nothing crashes; Huyen calls this failing silently. For an ML system, reliability is therefore also about the share of right answers, not only about answers on time.

        **From a wish to a number.** "Fast and reliable" is a wish nobody can check. Google's book *Site Reliability Engineering* (2016) turns it into numbers. **Availability** is the share of the time that a service works, or the share of requests that succeed: the minutes up divided by all minutes, or the requests that worked divided by all requests. More generally:

        - a **service level indicator** (SLI) is a number you measure, usually the good events divided by all events, such as the share of messages that got a score within 1 minute;
        - a **service level objective** (SLO) is a target for an SLI over a period, such as 99% over 30 days; the book's own example is "99.9% of Get RPC calls will complete in less than 100 ms", where an RPC call is a request from one program to another;
        - a **service level agreement** (SLA) is a promise in a contract, with consequences, such as money back, when it is broken.

        What the SLO leaves is the **error budget**: 100% minus the SLO. An SLO of 99.9% availability over 30 days allows 43.2 minutes of downtime. As long as budget is left, the team may put changes into use; when the budget is gone, reliability comes first. The picture shows how the budget shrinks with each nine.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **reliability** | | the system keeps doing the right thing at the right level, even when hardware, software or people make mistakes | |
        | **scalability** | | the system keeps up when the load, the data or the system grows, and can shrink again | `C.RUNS` |
        | **maintainability** | | others can understand the system, change it and put the change into use | |
        | **adaptability** | | the system follows a changing world, without stopping the service | |
        | **service level indicator** | S-L-I | a number you measure about a service, usually the good events divided by all events | `share_within(times, limit)` |
        | **service level objective** | S-L-O | a target for an SLI over a period, such as 99% over 30 days | `C.SLO_SHARE` |
        | **service level agreement** | S-L-A | a promise about a service in a contract, with consequences when it is broken | |
        | **error budget** | | what an SLO leaves: 100% minus the SLO, in minutes or in requests | `minutes_left(availability, days, outages)` |
        """),
        L.watch_out("An SLO of 100% is not a stricter version of 99.9%: it leaves no error budget at all. One minute of downtime breaks it, and every change becomes a risk the team cannot afford. Choose the SLO the users need, not the highest one you can think of."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Turning the requirements of a system into numbers:

    1. For each of the four requirements, ask what goes wrong for the users when the system fails it.
    2. Choose a number you can measure for each: an SLI, such as good answers divided by all answers.
    3. Set a target and a period: the SLO, such as 99% over 30 days.
    4. Work out the error budget: 100% minus the SLO, in minutes or in requests.
    5. Measure, take what went wrong off the budget, and when the budget is gone, put reliability before new changes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.OUTAGES` holds the outages of the spam service in one month of 30 days, in minutes, invented by the course. The cell adds them up and works out the availability: the minutes up divided by all minutes. Run the cell.
    """)
    return


@app.cell
def example_outages():
    outages = C.OUTAGES
    minutes = 30 * 24 * 60
    down = 0
    for m in outages:
        down = down + m
    print("minutes in 30 days:", minutes)
    print("minutes down:", down)
    print("availability:", (minutes - down) / minutes)
    return down, minutes, outages


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In 30 days, the spam service was down for 12 + 5 + 20 = 37 of its 43,200 minutes: an availability of 0.99914, or 99.914%. With an SLO of 99.9%, that is just enough. Exercise 2 works out how many minutes of the error budget are left.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Measure an SLI

        `share_within(times, limit)` gets the seconds that each message waited for its score, and a limit in seconds. It gives back the share of messages that got their score within the limit: a number from 0 to 1. The loop takes the messages one by one as `seconds`, and `good` counts those within the limit; a message exactly at the limit counts as good. Replace the `...` with the share of messages within the limit.

        The check calls `share_within(C.TIMES, 60)`, with the 20 messages of a busy morning. Then lists of other lengths, and other limits.
        """),
        L.task("Give back the share of messages within the limit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=7)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Step 2 of the recipe: an SLI is good events divided by all events. `good` holds the good events; all events are the messages in `times`.",
            "You need the number of messages in the list, not the list itself. Which function of Python gives how many items a list holds?",
            "`    return good / ___`",
        ],
        why="`good / len(times)` divides the messages within the limit by all messages. For `C.TIMES`, 18 of the 20 messages are within 60 seconds, a share of 0.9: below the SLO of 0.99.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What is left of the error budget

        `minutes_left(availability, days, outages)` gets an SLO for availability as a share, such as `0.999`, a period in `days`, and a list of `outages` in minutes. It gives back the minutes of the error budget that are left after the outages: a negative number when the service was down longer than the SLO allows. The loop already takes each outage, `minutes`, off `budget`. Replace the `...` with the budget at the start: the share of the time the service may be down, times all the minutes of the period, as in the worked example.

        The check calls `minutes_left(0.999, 30, C.OUTAGES)`, the month of the worked example. Then other SLOs, periods and outages.
        """),
        L.task("Start the budget as the share of the time that may go wrong, times all the minutes of the period."),
    ])
    return


@app.cell(hide_code=True)
def exercise_budget():
    box_budget = L.code_box("budget", lines=6)
    box_budget
    return (box_budget,)


@app.cell(hide_code=True)
def check_budget(box_budget):
    result_budget = L.check("budget", box_budget.value)
    result_budget
    return (result_budget,)


@app.cell(hide_code=True)
def hints_budget():
    L.hints(
        "budget",
        [
            "Step 4 of the recipe: the error budget is 100% minus the SLO. `budget` starts as that share times all the minutes of `days` days.",
            "With `availability` written as a share, 100% is the number 1. What share of the time is left for downtime?",
            "`    budget = (___) * days * 24 * 60`",
        ],
        why="`(1 - availability) * days * 24 * 60` is the error budget in minutes: 0.001 of the 43,200 minutes of 30 days is 43.2 minutes. The outages of `C.OUTAGES` take 37 of them, so 6.2 are left; a month with one outage of 50 minutes ends 6.8 minutes below zero.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### One busy day, more servers

    The demo follows the spam service through one busy day, now that it scores the mail of many cities. Each server scores 300 messages a minute. The upper chart shows the messages a minute: quiet at night, a peak in the morning, and at 17:00 a storm warning, after which 1,400 messages a minute come in for 20 minutes. The dashed line is what the servers can score; above it, messages wait in a queue. The lower chart shows how long a message waits, in the color of mistakes when it is more than 1 minute. The SLO is 99% of the messages within 1 minute. The numbers are invented by the course.
    """)
    return


@app.cell(hide_code=True)
def demo_servers():
    L.explore("servers", "servers", lambda _v: C.frame(int(_v), L.lang), start=2, values=list(C.SERVERS), role="",
              task="Find the fewest servers that meet the SLO. How many would the average minute of the day need?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The average minute of the day brings 244 messages, fewer than the 300 that 1 server scores. Yet the SLO needs 5 servers, because of 20 minutes of storm: scalability is about the busiest moment, not about the average. With 4 servers, the longest wait is 3.3 minutes, and 5.6% of the messages miss the limit.

    Running five servers all day means paying for 20 minutes of storm during the other 1,420 minutes too. Huyen points out that scaling also means scaling down: a system that adds servers when the load rises, and removes them after, pays for the storm only while it lasts. Later lessons on the cloud come back to this.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 3. Which requirement?

    Only one engineer knows how to train the spam model again: the steps are not written down, and the code has no tests. The engineer goes on leave for a month, and a small change to the threshold waits until the engineer is back. Meanwhile, every message still gets a right score within a second.
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="Which requirement does the spam service fail?",
        options={
            "reliability": "Reliability: the answers have become wrong",
            "scalability": "Scalability: too many messages for the servers",
            "adaptability": "Adaptability: the spam changed and the model did not follow",
            "right": "Maintainability: nobody else can change it",
        },
        correct="right",
        why={
            "right": "Nobody else can understand, change or train the system, so a small change waits a month. A maintainable system has its steps written down and tested, so a new colleague can make the change in a day.",
            "reliability": "Every message still gets a right score within a second. What is it that cannot happen while the engineer is away?",
            "scalability": "Nothing in the story is about the number of messages. Which kind of work waits for the engineer?",
            "adaptability": "The spam has not changed in the story. What makes the small change to the threshold wait a month?",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_slo():
    mo.md(r"""
    ### Exercise 4. Which goal is an SLO?

    The city writes down four goals for the spam service, and wants to know each month whether the service met them.
    """)
    return


@app.cell(hide_code=True)
def choice_slo_cell():
    choice_slo = L.choice(
        "slo",
        question="Which of the four goals is an SLO?",
        options={
            "vague": "The spam service must be fast and reliable for every resident",
            "right": "99% of messages get a score within 1 minute, over 30 days",
            "never": "The spam service is never down, not even for a minute",
            "test": "The model scores 95% on the test set before it goes into use",
        },
        correct="right",
        why={
            "right": "It names the number to measure (the share of messages within 1 minute), a target (99%) and a period (30 days), so after each month you know whether the service met it. It leaves an error budget of 1% of the messages.",
            "vague": "Fast and reliable are wishes: how fast, and measured how? Which goal names a number, a target and a period?",
            "never": "That is an SLO of 100%: no error budget at all, so one minute of downtime breaks it. Which goal leaves room for something to go wrong?",
            "test": "A score on a test set is measured once, before the service runs. In which goal do you measure the service itself, month after month?",
        },
    )
    choice_slo
    return (choice_slo,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en), production asked for good answers for many users, quick and cheap enough, on data that keeps changing. This lesson turned that into four requirements, and each requirement into a number with a target: an SLI, an SLO and the error budget it leaves. In the next lesson, *Framing the problem for production*, you turn a wish of the users into an ML problem, and see how that choice decides what you can do later.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_slo, choice_which, result_budget, result_share):
    L.wrap_up(result_share, result_budget, choice_which, choice_slo)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
