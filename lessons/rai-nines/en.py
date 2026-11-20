# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-nines. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-nines", "en")
    C = L.checks   # reliabilities from 90% to 99.999%, a day of 1,000,000 requests, and a task of 100 steps


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en) and [Probability: counting how often](/course/open/prob-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_risk():
    quick_risk = L.choice(
        "quick-risk",
        question="A made-up AI system has one hazard: in 1 of 100 answers it gives bad advice, with a severity of 50 harm points. By the classic risk equation, what is its risk per answer?",
        options={
            "right": "0.5 harm points: 0.01 times 50",
            "plus": "50.01 harm points: the chance plus the severity",
            "severity": "50 harm points: the harm when it goes wrong",
        },
        correct="right",
        why={
            "right": "Risk is the chance times the severity: 0.01 × 50 = 0.5 harm points per answer.",
            "plus": "A chance and a number of harm points are not added; the classic risk equation multiplies them. See [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en).",
            "severity": "That is the severity alone, as if the advice went wrong in every answer; it goes wrong in 1 of 100. See [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en).",
        },
    )
    quick_risk
    return (quick_risk,)


@app.cell(hide_code=True)
def quick_check_prob():
    quick_prob = L.choice(
        "quick-prob",
        question="In a test, the API of an AI model fails 6 of 2,000 requests. Counted this way, what is the probability that a request fails?",
        options={
            "complement": "0.997: the share of requests that do not fail",
            "right": "0.003: 6 failed requests divided by 2,000",
            "count": "6: the number of requests that failed",
        },
        correct="right",
        why={
            "right": "A probability counted from tries is how often it happened, divided by the number of tries: 6 / 2,000 = 0.003.",
            "complement": "That is the probability that a request does not fail, 1,994 of 2,000; the question is about the requests that fail. See [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "count": "That is how often it happened, not yet a probability, which lies between 0 and 1: divide by the number of tries. See [Probability: counting how often](/course/open/prob-basics?lang=en).",
        },
    )
    quick_prob
    return (quick_prob,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A provider of AI models promises that its service is available 99.9% of the time. A team builds an AI agent that files an expense report in 100 steps, and each step goes right 99% of the time. Both numbers sound almost perfect.

    But 99% also means 1 failure in 100. A service that answers 1,000,000 questions a day and fails on 1% of them fails 10,000 times, every day. How close to 100% is close enough depends on how often a system is used, on how many steps must all go right, and on what one failure costs. This lesson counts reliability in nines, reads what a provider's promise of nines does and does not cover, and multiplies reliability over the steps of a task.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A provider promises that the API of its AI model is available 99.9% of the time, counted in time. How long may the API be down in a year and still keep that promise?",
        options={
            "minutes": "About 5 minutes a year",
            "right": "About 9 hours a year",
            "days": "About 4 days a year",
        },
        reveal="**About 9 hours a year.** A year has 8,760 hours, and 0.1% of them is 8.76 hours, or 43.2 minutes in a month of 30 days. Three nines sound like always on, yet they allow more than a working day of downtime a year. About 5 minutes a year is 99.999%, five nines; about 4 days is 99%, two nines.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What a nine is

    Reliability is the chance that a system does not fail: one minus the chance of failure of [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en). Close to 100%, people count it in nines. In *Site Reliability Engineering* (online edition, 2017), the book in which engineers at Google describe how they run Google's services, a service aims for an availability such as 99.9%, 99.99% or 99.999%: "Each additional nine corresponds to an order of magnitude improvement toward 100% availability." An order of magnitude is a factor of 10.

    Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (2024) turns it into a measure, after a proposal by the mathematician Terence Tao: the **nines of reliability**. "Informally, a system has nines of reliability equal to the number of nines at the beginning of its decimal or percentage reliability." 99% has two nines, 99.9% three. Formally, with p the reliability as a decimal, the nines are k = −log10(1 − p). Here 1 − p is the chance of failure, and the base-10 logarithm of [Exponents and logarithms](/course/open/math-exp-log?lang=en) gives the power of 10 it is: at 99.9%, the chance of failure is 0.001, which is 10 to the power −3, so k = 3. The formula also gives the nines between the whole ones: 50% has about 0.3, and 99.5% about 2.3. Each extra nine makes the chance of failure ten times as small, the textbook says: ten times fewer failures, and ten times as long, on average, before the first one.

    ### More use, more failures

    "The more often we use a system, the more likely we are to encounter a failure." An autonomous vehicle, the textbook goes on, "is much more likely to make a mistake during a journey where it has to make 1000 decisions, than during a journey where it has to make only 10 decisions." At 1,000,000 requests a day:

    | reliability | nines | failed requests a day |
    |---|---|---|
    | 99% | 2 | 10,000 |
    | 99.9% | 3 | 1,000 |
    | 99.99% | 4 | 100 |
    | 99.999% | 5 | 10 |

    The textbook also gives the decision by which a mistake is likely: decision 100 at 99%, decision 1,000 at 99.9%. That number is 1/(1 − p): when every decision has the same chance of failing, independently of the others, the first mistake comes at that decision on average. The textbook does not spell out that condition.

    ### Counted in time or in requests

    The SRE book counts availability in two ways. In time, as "the proportion of system uptime": "a system with an availability target of 99.99% can be down for up to 52.56 minutes in a year and stay within its availability target". In requests, as "the request success rate": "a system that serves 2.5M requests in a day with a daily availability target of 99.99% can serve up to 250 errors and still hit its target for that given day." The book's availability table, which assumes no planned downtime, allows 3.65 days a year at 99%, 52.6 minutes at 99.99% and 5.26 minutes at 99.999%.

    ### What a provider promises

    A service level agreement, or SLA, is a provider's written promise about availability, with what customers get when the provider falls short. In the SLA for Bedrock, the service through which Amazon Web Services (AWS) offers AI models, AWS "will use commercially reasonable efforts" to keep Bedrock available (the version last updated on 4 October 2023). If a month falls below 99.9%, a customer can claim a service credit of 10% of that month's charges, to be set against future bills; below 99.0%, 25%; below 95.0%, 100%. "Service Credits will not entitle you to any refund or other payment from AWS." The SLA counts availability per 5-minute interval, as the share of requests that do not fail with an error, and "An ‘Error’ is any Request that returns a 500 error code": the code with which a server says that something went wrong on its side. Google's SLA for Vertex AI (last modified on 12 February 2026) counts in minutes: "‘Downtime’ means more than a 5% Error Rate", where an error is a response with code 500 or 503. Two promises of 99.9% need not count the same thing.

    ### An outage

    On 11 December 2024, from 3:16 PM to 7:38 PM Pacific time, "all OpenAI services experienced significant degradation or complete unavailability", OpenAI's post-mortem says: 4 hours and 22 minutes. The cause was a newly deployed service that collects telemetry, whose configuration overloaded the software that controls OpenAI's large clusters of servers. "The change was tested in a staging cluster, where no issues were observed." If all 262 minutes counted as downtime, that one afternoon alone would bring a month of 31 days down to about 99.41%, well under three nines; the post-mortem does not say how much of it was full downtime.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### The price of a nine

    A nine tends to cost more than the one before. In the SRE book: "an incremental improvement in reliability may cost 100x more than the previous increment." And past some point, users typically do not notice: "a user on a 99% reliable smartphone cannot tell the difference between 99.99% and 99.999% service reliability!" Google aims to make a service "reliable enough, but no more reliable than it needs to be". What the target leaves is an error budget: if a service aims to serve 99.999% of all queries in a quarter, "the service’s error budget is a failure rate of 0.001% for a given quarter." As long as some of the budget is left, new versions may be released; when it is used up, releases stop for a while.

    ### Nines are not risk

    The textbook warns: "The nines of reliability are only a measure of probability, not risk." They say how often a system fails, not what a failure costs: "A game playing AI will present a lot less risk than an autonomous vehicle even if both systems have three nines of reliability." For risk, the chance of failure is multiplied by the severity, as in [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en).

    ### A chain of steps

    An AI agent that books a trip searches, compares, fills in a form and pays: every step must go right. Two events are independent when one of them happening does not change the chance of the other. If the steps of a task are independent, their chances multiply: with n steps that each succeed with chance p, the whole task succeeds with chance p × p × … × p = pⁿ. Five steps at 90% succeed about 59% of the time. Long tasks lose their nines fast.

    In 2025, Toby Ord of the University of Oxford argued that the measurements of METR, a research organization that tests what AI agents can do, can be explained by a simple model: "a constant rate of failing during each minute a human would take to do the task". The success rate then falls exponentially with the length of a task, and each agent has a half-life: the length of task it completes half the time. A task three times as long is like three such tasks in a row: at 80% each, it succeeds 0.8³ = 0.512 of the time, about one half. So the length of task an agent completes 80% of the time is about a third of its half-life; at 90% about a seventh, at 99% about 1/70 and at 99.9% about 1/700: "each additional ‘nine’ of reliability beyond this divides the time horizon by 10", where the time horizon is the length of the tasks, in the time skilled people need for them, that a model completes with a given chance of success. In the version of METR's paper of July 2026, the 80% horizons are 4 to 6 times shorter than the 50% horizons, a bigger gap than the model's factor of about 3. Ord adds: "Note that I am not claiming AI agents have a precisely constant rate of failure per minute of time it would take a human to complete the task." Whether the model holds for other tasks, Ord writes, is unknown. And steps, like tries, are seldom fully independent: in [Robust, resilient, available, predictable](/course/open/rai-reliability?lang=en), the same task succeeded again and again, or failed again and again.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **nines of reliability** | the number of nines at the start of a reliability, or exactly k = −log10(1 − p): 99.9% has three |
        | reliability | the chance that a system does not fail, per request, per decision or per step |
        | availability | the share of the time, or of the requests, in which a service works |
        | order of magnitude | a factor of 10 |
        | SLA | service level agreement: a provider's written promise about availability, and what customers get when it falls short |
        | error budget | the failures that a reliability target still allows in a period |
        | independent | two events are independent when one of them happening does not change the chance of the other |
        | pⁿ | p multiplied by itself n times: the chance that n independent steps of reliability p all succeed |
        | half-life | in Ord's model, the length of task an agent completes half the time |
        | time horizon | METR's measure: the length of the tasks, in the time skilled people need, that a model completes with a given chance of success |
        | `np.log10` | NumPy's base-10 logarithm |
        """),
        L.watch_out("Nines say how often a system fails, not what a failure costs. And pⁿ holds only for independent steps: real steps often fail together."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write the reliability as a decimal p; the chance of failure is 1 − p.
    2. Count the nines: k = −log10(1 − p).
    3. Multiply the chance of failure by how often the system is used: failed requests a day, or time down a year.
    4. For a task of n steps that must all succeed, compute pⁿ, and say that it assumes independent steps.
    5. Read what a promise counts as a failure, and what the provider gives when it falls short.
    6. Weigh the failures by their harm before you call a system reliable enough.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes three nines, 99.9%, and computes how many of 1,000,000 requests fail, how many hours a year that is when availability is counted in time, and the decision of the first mistake, on average.
    """)
    return


@app.cell
def example_nines():
    p = 0.999                                # reliability: three nines
    print(round(1000000 * (1 - p)))          # failed requests in 1,000,000
    print(round(365 * 24 * (1 - p), 2))      # hours down in a year, counted in time
    print(round(1 / (1 - p)))                # the decision of the first mistake, on average
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Three nines: 1,000 failed requests in 1,000,000, 8.76 hours down in a year, and the first mistake at decision 1,000 on average. In Exercise 1 you count the nines themselves.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nines():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. nines: count the nines

        Write `nines(p)`. It gets a reliability `p` as a decimal, such as 0.999, and returns its nines of reliability by the formula of *What a nine is*. The box imports NumPy as `np`, so `np.log10` is ready; replace the `...`. The check calls `nines(0.999)`, then other reliabilities, also some between whole nines.
        """),
        L.task("The nines of a reliability, by the formula."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nines():
    box_nines = L.code_box("nines", lines=5)
    box_nines
    return (box_nines,)


@app.cell(hide_code=True)
def check_nines(box_nines):
    result_nines = L.check("nines", box_nines.value)
    result_nines
    return (result_nines,)


@app.cell(hide_code=True)
def hints_nines():
    L.hints(
        "nines",
        [
            "The formula k = −log10(1 − p) of *What a nine is*, written in Python.",
            "First the chance of failure, one minus `p`; then its base-10 logarithm, with NumPy's function from [Exponents and logarithms](/course/open/math-exp-log?lang=en); then a minus sign in front, so that −3 becomes 3.",
            "`    return ___(1 - p)`",
        ],
        why="`return -np.log10(1 - p)`. For 0.999, the chance of failure is 0.001, its base-10 logarithm −3, and the nines 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_nines():
    mo.md(r"""
    Three nines for 99.9%, five for 99.999%, and 0.3 for 50%: the formula counts how many times the chance of failure is divided by 10. Under the check, every row from 99% on has ten times fewer failures than the one above it, and one nine more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_success():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. task_success: a task of many steps

        An agent does a task of `steps` steps. Each step succeeds with chance `p`, independently of the others, and the task succeeds only if all of them do. Write `task_success(p, steps)`, which returns that chance. Compute it in a loop: a variable `chance` that starts at 1, and a `for` loop with `step` over `range(steps)`. Replace the `...`. The check calls `task_success(0.99, 100)`, then other chances and numbers of steps.
        """),
        L.task("The chance that every step of a task succeeds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_success():
    box_success = L.code_box("task_success", lines=5)
    box_success
    return (box_success,)


@app.cell(hide_code=True)
def check_success(box_success):
    result_success = L.check("task_success", box_success.value)
    result_success
    return (result_success,)


@app.cell(hide_code=True)
def hints_success():
    L.hints(
        "task_success",
        [
            "Every step must succeed. With independent steps, the chances multiply: one factor `p` for each step.",
            "Start `chance` at 1, the chance before any step. A `for` loop over `range(steps)` runs once for each step; in it, multiply `chance` by `p`. Return `chance` after the loop.",
            "`        chance = ___`",
        ],
        why="`chance = 1`, `for step in range(steps):` with `chance = chance * p` inside it, then `return chance`. For 100 steps at 0.99: 0.366.",
    )
    return


@app.cell(hide_code=True)
def text_after_success():
    mo.md(r"""
    A task of 100 steps at 99% succeeds only about 37% of the time: 0.99 to the power 100 is 0.366. Every step looks almost perfect, yet the task fails nearly two times in three. The chance falls below one half after 69 steps at 99%, and after 693 steps at 99.9%: one nine more, ten times as many steps. `p ** steps` gives the same in one line.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: from one nine to five

    The slider sets the reliability, from 90% to 99.999%. The picture shows the nines, the failed requests in 1,000,000, the time down in a year when availability is counted in time, and the chance that a task of 100 independent steps succeeds, with the curve of that chance over the number of steps.
    """)
    return


@app.cell(hide_code=True)
def demo_reliability():
    reliability = L.explore("reliability", "reliability, in %", lambda _v: C.demo_frame(_v, L.lang),
                            start=90, values=C.PERCENTS, role="",
                            task="Find the smallest reliability on the slider at which a task of 100 steps succeeds at least 9 times in 10.")
    reliability
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 99.9% on: there a task of 100 steps succeeds 90.5% of the time, and at 99% only 36.6%. Each nine more brings ten times fewer failures and ten times less downtime, and tasks about ten times as long then succeed as often.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_risk():
    mo.md(r"""
    ### Exercise 3. Three nines, twice

    A chess program and an AI assistant that suggests insulin doses to people with diabetes both have three nines of reliability, and both make 1,000 decisions a day.
    """)
    return


@app.cell(hide_code=True)
def choice_risk_cell():
    choice_risk = L.choice(
        "risk",
        question="Which of the two brings more risk?",
        options={
            "same": "The same risk: the same nines and the same number of decisions",
            "right": "The insulin assistant: its mistakes can do far more harm",
            "neither": "Neither: one mistake a day is too few to matter",
        },
        correct="right",
        why={
            "right": "Both make about one mistake a day, but a wrong move costs a game of chess, and a wrong dose of insulin can harm a person. Nines are \"only a measure of probability, not risk\"; risk weighs each failure by its severity.",
            "same": "The same nines and the same number of decisions give the same number of mistakes, about one a day; they say nothing about the harm of a mistake. See *Nines are not risk* in *The idea*.",
            "neither": "One mistake a day is 365 a year; whether that matters depends on what each mistake costs, and that differs greatly here. See *Nines are not risk* in *The idea*.",
        },
    )
    choice_risk
    return (choice_risk,)


@app.cell(hide_code=True)
def text_exercise_available():
    mo.md(r"""
    ### Exercise 4. Available, but wrong

    A team calls an AI model through Bedrock for a month. Every request gets an answer, and none returns a 500 error code, but 1 answer in 20 is wrong.
    """)
    return


@app.cell(hide_code=True)
def choice_available_cell():
    choice_available = L.choice(
        "available",
        question="Under Bedrock's SLA, what is the availability of that month?",
        options={
            "wrong": "95%: 19 of every 20 answers are right",
            "right": "100%: no request failed with an error",
            "minutes": "None: the SLA counts minutes of downtime, not answers",
        },
        correct="right",
        why={
            "right": "In Bedrock's SLA, an error is a request that returns a 500 error code. A wrong answer that arrives counts as available: the SLA is about whether the service answers, not whether the answer is right.",
            "wrong": "That is the share of right answers. Bedrock counts a request as failed only when it returns a 500 error code, and none did. See *What a provider promises* in *The idea*.",
            "minutes": "Vertex AI counts in minutes; Bedrock counts the share of requests that do not fail with an error, per 5-minute interval. See *What a provider promises* in *The idea*.",
        },
    )
    choice_available
    return (choice_available,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en) weighed the chance of failure by its harm. This lesson turned that chance around into reliability and counted it in nines: each nine is ten times fewer failures. Over many requests, the failures add up, as the counts of [Probability: counting how often](/course/open/prob-basics?lang=en); over many steps, the chances multiply. The next lesson, *Designing for safety*, gives principles from engineering that make failures rarer and less harmful, such as redundancy, separation of duties and fail-safes.

    **An everyday example.** A string of old Christmas lights goes dark when one bulb breaks. With 50 bulbs that each work 99% of the time, the whole string lights up only about 6 times in 10.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_available, choice_risk, result_nines, result_success):
    L.wrap_up(result_nines, result_success, choice_risk, choice_available)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
