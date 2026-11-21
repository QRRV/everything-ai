# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-time-horizon. Route: course/route/lessen/verantwoord.yaml. Checks, pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-time-horizon", "en")
    C = L.checks   # METR's curve, the doubling time of the trend, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Testing for dangerous capabilities](/course/open/rai-dangerous-capabilities?lang=en) and [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kimi():
    quick_kimi = L.choice(
        "quick-kimi",
        question="In July 2026, Kimi K3 completed a simulated attack on a small company network, a path of 32 steps, in one of its 10 attempts. What did the assessors conclude?",
        options={
            "fails": "That it fails 9 times in 10, so it cannot attack such networks",
            "right": "That it can attack small, weakly defended networks",
            "luck": "Nothing yet, as one success could be luck",
        },
        correct="right",
        why={
            "right": "One success in 10 was enough: the assessors wrote that Kimi K3 \"is capable of autonomously attacking small, weakly defended and vulnerable enterprise systems\". A test asks about one task at a time. This lesson puts many tasks of very different difficulty on one scale.",
            "fails": "A success shows that a capability is there; nine failures do not undo it. See [Testing for dangerous capabilities](/course/open/rai-dangerous-capabilities?lang=en).",
            "luck": "For the assessors, one success in 10 was enough to show the capability. See [Testing for dangerous capabilities](/course/open/rai-dangerous-capabilities?lang=en).",
        },
    )
    quick_kimi
    return (quick_kimi,)


@app.cell(hide_code=True)
def quick_check_boundary():
    quick_boundary = L.choice(
        "quick-boundary",
        question="A logistic regression gives a point the probability σ(z), with z the logit. Which probability does a point on the decision boundary get?",
        options={
            "zero": "0, because it lies on the line",
            "right": "0.5, since its logit is 0",
            "one": "1, since the model is most certain on the line",
        },
        correct="right",
        why={
            "right": "On the decision boundary the logit is 0, and σ(0) = 0.5. In this lesson, a curve of the same kind gives the chance that an AI agent completes a task, and the length of task where it gives 0.5 is the measure.",
            "zero": "The sigmoid never gives exactly 0: on the decision boundary the logit is 0, and σ(0) = 0.5. See [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "one": "On the line, the model is least certain: the logit is 0 and σ(0) = 0.5. See [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
        },
    )
    quick_boundary
    return (quick_boundary,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A test for a dangerous capability asks about one task at a time: can this model do that? To follow how fast AI agents get better overall, you need one scale for tasks as different as fixing a small bug and training a model. METR, a research organization that evaluates AI models, chose time as that scale: how long the tasks take skilled people. On that scale you can say how long a task a model can do, and how fast that length grows. That matters for safety: the longer the tasks an agent can finish on its own, the more of the work of AI research itself it could take over, and METR uses the measure to follow that.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In METR's data of September 2026, o3, a model by OpenAI released in April 2025, has a time horizon of about 2 hours: on METR's software tasks that take skilled people about 2 hours, it is predicted to succeed half the time. On tasks of which length does the same measurement predict that o3 succeeds 80% of the time?",
        options={
            "right": "About 30 minutes",
            "most": "About 1 hour and 40 minutes",
            "hour": "About 1 hour, half as long",
        },
        reveal="**About 30 minutes.** METR gives o3 a 50% time horizon of 119.7 minutes and an 80% time horizon of 30.0 minutes: four times shorter. In their paper, Thomas Kwa and colleagues at METR found a similar gap: \"models’ 80% time horizons are 4-6x shorter\". Asking for a little more reliability costs a lot of task length; this lesson shows why.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Tasks measured in human time

    Thomas Kwa and colleagues at METR (*Measuring AI Ability to Complete Long Tasks*, first version of 18 March 2025; the fourth version, of 10 July 2026, is called *Measuring AI Ability to Complete Long Software Tasks*) tested AI agents: AI models with tools that work on a task in many steps. They used 170 tasks about software, machine learning and cybersecurity, each with a clear way to score it: 97 of HCAST, from 1 minute to 30 hours, 7 of RE-Bench, of 8 hours each, and 66 short ones of 1 to 30 seconds. Skilled people, with "an average of about 5 years of relevant experience", did the same tasks: "over 800 baselines totaling 2,529 hours". The length of a task is the geometric mean of the times of the people who succeeded: multiply the $n$ times and take the $n$-th root. For 20 and 80 minutes, that is $\sqrt{20 \times 80} = 40$ minutes, the middle on a log scale, where the ordinary mean gives 50. For 21 tasks, the researchers estimated the time themselves. Each agent then did each task 8 times.

    Their measure is the "50%-task-completion time horizon, the time humans typically take to complete tasks that AI models can complete with 50% success rate". A **time horizon** is such a length of task, in the time skilled people need, at which an AI agent succeeds with a given chance; without a chance, it means 50%. It is not how long an agent can work on its own. METR's page on time horizons asks: "Does “time horizon” mean the length of time that current AI agents can act autonomously?" Its answer starts with "No." and goes on: "It’s a measure of the difficulty of a task, rather than the time an AI spends to complete the task." On the tasks they complete, agents are typically several times faster than people, METR adds.

    ### One curve per agent

    Kwa and colleagues found that "Model success rates are negatively correlated with how much time it takes a human to complete the task." For each agent, they fit a curve, as in logistic regression with one feature: the log of the length of the task.
    """)
    return


@app.cell(hide_code=True)
def text_idea_curve():
    mo.vstack([
        L.formula(r"p_{\text{success}} = \sigma\big((\log_2 [[param:h]] - \log_2 [[data:t]]) \cdot [[param:\beta]]\big)"),
        mo.md(r"""
        Here $t$ is the length of the task, $h$ the agent's time horizon, and $\beta$ how steeply its chance falls; $h$ and $\beta$ are learned from the agent's results. At $t = h$ the logit is 0 and the chance is $\sigma(0) = 0.5$: the time horizon is where the curve crosses one half, as the decision boundary is where a logistic regression gives 0.5. The paper does not say which base its log has; METR's code fits the curve "as a function of log2(human_minutes)". With base 2, every doubling of the length of a task lowers the logit by $\beta$, so the odds of success fall by the same factor at each doubling: $e^{\beta}$, about 1.8 for $\beta = 0.6$.

        The same curve gives other levels. METR finds "the task duration where the fitted curve intersects with 50% (or 80%) success probability". And 50% is a point on the curve, not the chance on each task. For GPT-5, a model by OpenAI, on tasks of 90 minutes to 3 hours, METR's page says that it "succeeds 100% of the time for around one-third of the tasks, fails 100% of the time for around one-third of the tasks, and sometimes succeeds and sometimes fails on the remaining third of tasks."

        ### Doubling about every seven months

        Kwa and colleagues then plotted the time horizon of each model against its release date, and fitted a straight line through the log of the horizons: in the fourth version, "time horizon has doubled every 207 days", from 2019 to 2025 (the first version had 212 days). On a log scale, growth by the same factor in equal steps is a straight line, and the doubling time is the time in which the line climbs one doubling. "GPT-2 has a 50% time horizon of only 2 seconds, while o3 has a 110-minute time horizon". They trust the line more than its points: "we are more confident in the slope of the time horizon trend than in the time horizon of any particular model". A doubling every 207 days is about 3.4 times longer tasks per year, by the course's own count.

        Where does that lead? "Naively extrapolating the trend in horizon length implies that AI will reach a time horizon of >1 month (167 work hours) between mid-2028 and mid-2031". They add that this depends on whether the trend holds on real work and whether it changes. And on why a month matters: "We cannot rule out that a system capable of 1-month tasks, even at 50% reliability, could be transformative for our society, potentially including the ability for catastrophic harm."
        """),
    ])
    return


@app.cell(hide_code=True)
def text_idea_later():
    mo.md(r"""
    ### Later measurements

    METR kept measuring. In *Time Horizon 1.1* (29 January 2026), with 228 tasks instead of 170, the doubling time over all the years was 196 days, the same as the old tasks gave by then, and from 2023 on: "The post-2023 doubling-time is 131 days under TH1.1, compared to 165 days under TH1". On 3 March 2026, METR corrected a mistake in how it fitted the curves, which had affected its measurements. Its page on time horizons, last updated on 8 September 2026, gives the horizons of the models it measured under the CC BY 4.0 license, so anyone may use them; o3 has 119.7 minutes there, against 110 in the paper. The page warns: "Measurements above 16 hrs are unreliable with our current task suite". In Exercise 2 you fit METR's data yourself.

    On 20 November 2026, METR reported on a harder suite of tasks together with the old one: "time horizons are doubling roughly every 3.6 months", and "Current public frontier models have an estimated time horizon of up to 27 hours on the verifiable subset of our TH2 tasks." Within a year, the old suite, with tasks up to 30 hours, had become too easy for the best models. With two warnings: "We are increasingly concerned that our easy-to-run, containerized evals could overestimate real-world capabilities", because similar environments are now common in training, and "Human baselines are directly used for only 9% of our estimates".

    ### What a time horizon does not say

    Kwa and colleagues: "time horizon is always measured relative to a domain, task distribution, and human baseliners’ level of skill and context". On a small set of fixes to METR's own code, contractors took "5-18x longer" than the people who maintain the code. METR's advice is to read a 2-hour task "as what someone with low or no prior context (like a new hire or freelance contractor) could complete in 2 hours". Other kinds of tasks give other horizons. In a post of 14 July 2025, Thomas Kwa and Vincent Cheng found: "Visual computer use (OSWorld, WebArena) time horizons are 40-100x shorter than the software and reasoning cluster but are increasing at similar rates." And: "Self-driving (Tesla FSD) is improving more slowly at around 0.6 doublings / year." Many short tasks do not make a long one either, METR's page says: "solving 1000 separate 1-hour math problems isn’t a 1000-hour task".

    ### Where they disagree

    On how fast: 207 days in the paper, 196 or 131 days in *Time Horizon 1.1*, about 3.6 months in November 2026. The tasks, the years and the method differ between these numbers, so they show both a faster trend and a moving measure; METR's blog of March 2025 kept a wide margin: "at around 1-4 doublings per year". On which level to report: the International AI Safety Report 2026 uses the 80% horizon, "AI systems can complete well-specified software engineering tasks that take human experts 30 minutes around 80% of the time, and the duration of these tasks has been doubling every seven months", while METR leads with 50%; numbers from the two cannot be compared as they are. And on what comes next: METR's page fits a steady doubling to the trend over time, because an S-shaped curve that levels off "is a poor fit because we haven’t seen any evidence of the exponential growth in time horizon slowing down", while its own report of November 2026 warns that its tests could overestimate what models can do in the real world, so a steady trend on the tests need not hold for real work. Kwa and colleagues call their own forecast naive.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **time horizon** | the length of the tasks, in the time skilled people need for them, at which an AI agent succeeds with a given chance; without a chance, 50% |
        | AI agent | an AI model with tools that works on a task in many steps |
        | human baseline | a skilled person doing a task, timed, to find out how long the task takes |
        | geometric mean | the $n$-th root of the product of $n$ numbers: the middle on a log scale |
        | $t$, $h$ | the length of a task and an agent's time horizon, both in minutes for skilled people |
        | $\beta$ | how much an agent's logit falls per doubling of the length of a task: how steep its curve is |
        | doubling time | the time in which something that grows by the same factor in equal steps doubles |
        """),
        L.watch_out("A time horizon is not how long a model can work on its own: it is how long the tasks take people. It holds for one chance of success, for METR's tasks, mostly software, and for people without prior context on them. The made-up cases of the exercises, the levels above 80% in the demo and the comparisons of numbers between sources are the course's own."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a time horizon:

    1. Ask for which chance of success it is given: 50%, 80% or another.
    2. Ask which tasks were used, and which people set their lengths.
    3. Read it as how hard the tasks are in human time, not as how long the model works.
    4. For one agent, the chance at another length is $\sigma((\log_2 h - \log_2 t) \cdot \beta)$.
    5. For the trend, fit a straight line through $\log_2$ of the horizons against the release dates; one over its slope is the doubling time.
    6. Give a date and a version with every number.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        Step 5 with two models. If the horizon grows from one model's $h_a$ to a later model's $h_b$ in $d$ days, the number of doublings is $\log_2(h_b / h_a)$, and each took:
        """),
        L.formula(r"\text{doubling time} = \frac{[[data:d]]}{\log_2([[data:h_b]] / [[data:h_a]])}"),
        mo.md(r"""
        The cell has METR's leading models from 2023 on, from the data of September 2026; like METR's own fit, it leaves out the one model above 16 hours. It works it out for GPT-4 to GPT-5 and for o1 to o3, all four by OpenAI.
        """),
    ])
    return


@app.cell
def example_doubling():
    # METR, Time Horizon 1.1, data of 8 September 2026 (CC BY 4.0):
    # the models it marks as the best when they came out
    models = ["GPT-4", "GPT-4 (2023-11)", "GPT-4o", "Anthropic (2024-06)", "o1-preview",
              "Anthropic (2024-10)", "o1", "Anthropic (2025-02)", "o3", "GPT-5", "Gemini 3 Pro",
              "Anthropic (2025-11)", "GPT-5.2", "Anthropic (2026-02)"]
    days = [0, 237, 426, 464, 548, 588, 632, 713, 764, 877, 980, 986, 1003, 1059]   # days after 14 March 2023
    # the 50% horizon of each, in minutes
    minutes = [4.0, 4.0, 7.0, 11.4, 20.3, 20.5, 38.8, 60.4, 119.7, 203.0, 224.3, 293.0, 352.2, 718.8]

    for a, b in [(0, 9), (6, 8)]:   # GPT-4 to GPT-5, and o1 to o3
        doublings = np.log2(minutes[b] / minutes[a])
        each = round((days[b] - days[a]) / doublings)
        print(models[a], "to", models[b], round(doublings, 2), "doublings,", each, "days each")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two pairs, two answers: 155 days a doubling from GPT-4 to GPT-5, 81 days from o1 to o3. Kwa and colleagues noted that "o3 lies above the long-run trend", so a pair that ends with o3 runs fast. A straight line through all the models takes every one into account, and Exercise 2 fits it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_success():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. success: the chance at a length of task

        `success(horizon, minutes, beta)` gives back the chance that an agent with this 50% time horizon, in minutes, and this `beta` completes a task that takes skilled people `minutes` minutes, by METR's curve. Work out the logit `z`: the log base 2 of `horizon` minus the log base 2 of `minutes`, times `beta`, with `np.log2`. Return its sigmoid, with `np.exp` as in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).

        The check calls `success(120, 30, 0.6)`: a horizon of 2 hours, a task of half an hour, and a `beta` of 0.6. Then seven other cases.
        """),
        L.task("The chance that an agent completes a task of a given length."),
    ])
    return


@app.cell(hide_code=True)
def exercise_success():
    box_success = L.code_box("success", lines=6)
    box_success
    return (box_success,)


@app.cell(hide_code=True)
def check_success(box_success):
    result_success = L.check("success", box_success.value)
    result_success
    return (result_success,)


@app.cell(hide_code=True)
def hints_success():
    L.hints(
        "success",
        [
            "Do it by hand for `success(120, 30, 0.6)`: how many doublings is a task of 30 minutes below a horizon of 120? Each doubling below the horizon adds `beta` to the logit. Which chance does the sigmoid give for that logit?",
            "Write `z = (... - ...) * beta`. What do you take of each length, so that one doubling is one step, and which comes first, so that a task shorter than the horizon gets a positive logit? Then `return 1 / (1 + np.exp(-z))`.",
            "`    z = (np.log2(horizon) - ___) * beta`",
        ],
        why="`z = (np.log2(horizon) - np.log2(minutes)) * beta` and `return 1 / (1 + np.exp(-z))`. By hand: 30 minutes is two doublings below 120, so the logit is 2 × 0.6 = 1.2, and the chance is 1 / (1 + e^(−1.2)), about 0.7685.",
    )
    return


@app.cell(hide_code=True)
def text_after_success():
    mo.md(r"""
    A task as long as the horizon gets 0.5; two doublings shorter, 0.77; two doublings longer, 0.23; four doublings shorter, 7.5 minutes, 0.92. The curve falls by the same step in the logit at each doubling, and a larger `beta` makes it steeper: with a `beta` of 1.0 and a horizon of 60 minutes, a task of 15 minutes gets 0.88 and one of 240 minutes 0.12.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: more than half the time

    METR leads with the 50% time horizon, and also gives the 80% one. In its data of September 2026, o3 has a 50% horizon of 119.7 minutes and an 80% horizon of 30.0 minutes; from the two follows its $\beta$, about 0.69, by the course's own count: 30 minutes is almost exactly two doublings below 119.7, and at 80% the logit is $\ln(0.8/0.2) = \ln 4 \approx 1.39$, so $\beta \approx 1.39 / 2$. Choose the chance of success you want, and see the length of task at which o3's curve gives it.
    """)
    return


@app.cell(hide_code=True)
def demo_reliable():
    explore_reliable = L.explore(
        "reliable", "chance of success (%)",
        lambda _v: C.demo_frame(_v, L.lang),
        start=C.P_START, values=C.P_VALUES, role="hyper",
        task="Find the lowest chance of success at which o3's horizon is below 10 minutes.",
    )
    explore_reliable
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 93% on, o3's horizon is below 10 minutes, and at 99% it is just over a minute: 98 times shorter than at 50%. METR does not report such levels: measuring a 99% horizon well "would require many more tasks", very short ones, and the results would depend more on how they are measured. The levels above 80% in the demo are the course's own, read from the same curve. A task that matters often needs more than a coin flip, and there the horizon is much shorter.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_doubling():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. doubling_days: a line through every model

        `doubling_days(days, minutes)` gives back the doubling time, in days, of the straight line through the log base 2 of the horizons `minutes` against the release dates `days`. Fit the line with `np.polyfit`, which gives back the `slope` first and then the `start`, where the line crosses day 0; with the horizons counted in doublings, the slope is the doublings per day. Return the days per doubling.

        The check calls `doubling_days(days, minutes)`, with METR's 14 models of the worked example. Then five other cases.
        """),
        L.task("The doubling time of the time horizon, from every model at once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_doubling():
    box_doubling = L.code_box("doubling", lines=6)
    box_doubling
    return (box_doubling,)


@app.cell(hide_code=True)
def check_doubling(box_doubling):
    result_doubling = L.check("doubling", box_doubling.value)
    result_doubling
    return (result_doubling,)


@app.cell(hide_code=True)
def hints_doubling():
    L.hints(
        "doubling",
        [
            "Do it by hand for `doubling_days([0, 10, 20], [1, 2, 4])`: count the horizons in doublings above 1, so 0, 1 and 2. How many doublings per day is that, and how many days per doubling?",
            "`np.polyfit(x, y, 1)` fits a straight line through `y` against `x`. Which of the two lists is `x` here, and what do you do to the other one first, so that one step up is one doubling? The slope is then the doublings per day; return one over it.",
            "`    slope, start = np.polyfit(___, 1)`",
        ],
        why="`slope, start = np.polyfit(days, np.log2(minutes), 1)` and `return 1 / slope`. By hand: 0, 1 and 2 doublings on days 0, 10 and 20 is a slope of 0.1 doubling per day, so one doubling takes 10 days.",
    )
    return


@app.cell(hide_code=True)
def text_after_doubling():
    mo.md(r"""
    METR's 14 leading models from 2023 on give a doubling every 128.7 days, close to the 131 days of METR's post of January 2026; METR's own file, after the correction of March 2026, gives 128.744 days from the unrounded numbers. With the three older models from 2019 to 2022 as well, METR's file gives 187.8 days. For two models, the line goes through both, and gives the 155 days of the worked example. Each new model moves the line a little, and a point estimate such as 128.7 has a wide margin: METR's file gives 104 to 158 days.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Exercise 3. A headline

    A news site writes: new AI model can now work on its own for 12 hours. The article is based on METR's report of a 50% time horizon of about 12 hours for the model.
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Which reading fits METR's measure best?",
        options={
            "run": "It can keep working unattended for 12 hours before it stops",
            "right": "It succeeds about half the time on METR's software tasks of 12 hours",
            "any": "It can do any job of 12 hours, in any field, at half the speed of a person",
            "share": "It passes half of all the tasks METR has, whatever their length",
        },
        correct="right",
        why={
            "right": "A time horizon is \"a measure of the difficulty of a task, rather than the time an AI spends to complete the task\", on METR's tasks, mostly software, timed by skilled people. On shorter tasks the model does better, on longer ones worse. And on METR's old suite, measurements above 16 hours were unreliable, METR wrote.",
            "run": "METR's page asks exactly this, and answers \"No.\" A time horizon is how long the tasks take people, and agents are typically several times faster than people on the tasks they complete. See *Tasks measured in human time*.",
            "any": "The horizon holds for METR's tasks, mostly software, machine learning and cybersecurity; for visual computer use, horizons were 40 to 100 times shorter. And it says nothing about speed. See *What a time horizon does not say*.",
            "share": "The horizon is a length of task on a fitted curve, not a share of all tasks: shorter tasks succeed more often, longer ones less often. See *One curve per agent*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_exercise_forecast():
    mo.md(r"""
    ### Exercise 4. A plan for 2030

    A manager reads that time horizons double about every seven months, and that the best models were at about 2 hours in 2025. The manager concludes: by 2030, an AI will do any job in our company that takes a person a month.
    """)
    return


@app.cell(hide_code=True)
def choice_forecast_cell():
    choice_forecast = L.choice(
        "forecast",
        question="Which reply fits the sources of this lesson best?",
        options={
            "law": "Yes, the trend has held without a break since 2019, so it is sure to go on",
            "slowed": "No, METR found that the trend slowed down after 2025",
            "month": "Probably not, since a month of work is 720 hours, beyond any forecast",
            "right": "Perhaps for tasks like METR's, if the trend holds, but not any job",
        },
        correct="right",
        why={
            "right": "Kwa and colleagues call their forecast naive: a month of software work between mid-2028 and mid-2031, at a 50% chance, if the trend goes on and holds for real work. METR's tasks are well specified and scored by a program; METR's page adds that \"Most jobs are not composed of well-specified, algorithmic tasks.\" And in November 2026, METR worried that its tests could overestimate.",
            "law": "A trend is not a law: Kwa and colleagues call the forecast naive, and add that it depends on whether the trend holds on real work and whether it changes. See *Doubling about every seven months*.",
            "slowed": "METR found the opposite: in November 2026, it reported a doubling about every 3.6 months. See *Later measurements*.",
            "month": "Here a month of work is 167 work hours, not 30 days of 24 hours, and the forecast of Kwa and colleagues reaches it between mid-2028 and mid-2031. See *Doubling about every seven months*.",
        },
    )
    choice_forecast
    return (choice_forecast,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Testing for dangerous capabilities](/course/open/rai-dangerous-capabilities?lang=en) asked whether a model can do one dangerous thing; the time horizon puts many tasks on one scale and shows how fast models move along it. Its curve is the one of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), with one feature: the time horizon is where it crosses one half, as the decision boundary was. And as in [Log scales and power laws](/course/open/math-log-scales?lang=en), growth by the same factor is a straight line on a log scale, so the doubling time is one over the slope of a line. This was the last lesson of the module alignment.

    **An everyday example.** A climbing gym grades its routes by how hard they are for experienced climbers. A beginner's level is the grade at which half of the routes are climbed: easier routes nearly always, harder ones rarely. That level says nothing about how long the beginner can climb in one go, and the grade at which nine routes in ten are climbed is lower. Grades also belong to one gym: another gym, or rock outside, grades differently.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_forecast, choice_headline, result_doubling, result_success):
    L.wrap_up(result_success, result_doubling, choice_headline, choice_forecast)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
