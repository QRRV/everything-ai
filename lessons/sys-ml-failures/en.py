# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-ml-failures. Route: course/route/lessen/systemen.yaml. Checks, the model, the days of 2011 and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-ml-failures", "en")
    C = L.checks   # the course's model of the project, its answers for every hour of 2011, the requests and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="An alert rule says: the p99 above 300 ms for 3 minutes in a row. The p99 of the last five minutes is 120, 950, 130, 320 and 330 ms. Does the rule go off at the last minute?",
        options={
            "spike": "Yes: the minute of 950 ms lies far above 300",
            "most": "Yes: three of the five minutes lie above 300, as many as the rule asks for",
            "right": "No: one of the last 3 minutes, at 130 ms, lies under 300",
            "five": "No: a rule always looks at 5 minutes, not 3",
        },
        correct="right",
        why={
            "right": "The rule looks at the last 3 minutes: 130, 320 and 330 ms. One of them lies under 300, so the rule stays quiet; if the next minute lies above 300 too, it goes off. Rules like this watch the service. In this lesson, other numbers watch the model.",
            "spike": "One minute of 950 ms is a spike. Does the rule go off on one minute, or on minutes in a row? See [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).",
            "most": "The three minutes above 300 are not in a row: 130 lies between them. Which minutes does the rule look at? See [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).",
            "five": "The number of minutes is part of the rule, and this rule says 3. Which 3 minutes does it look at? See [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="A service logs every request as a line of JSON with the fields `time`, `status`, `ms`, `temp` and `bikes`. Which question can a program answer from that log alone?",
        options={
            "rented": "How many bikes were really rented at 16:00",
            "right": "How many answers above 5,000 bikes the service gave at 16:00",
            "warm": "How warm it really was at 16:00, whatever the app sent",
            "who": "Which person sent the slowest request of the day, and from where",
        },
        correct="right",
        why={
            "right": "Each line holds the answer under `bikes`, so a program can count the answers above 5,000 per minute. What was really rented, and how warm it really was, are not in the log: the first is known only later, and `temp` is what the app sent, right or wrong. That difference is what this lesson is about.",
            "rented": "The log holds what the service answered, not what happened in the hour. Which field holds the answer? See [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).",
            "warm": "`temp` is what the app sent, which can be wrong, such as degrees instead of the data's scale. Which question needs only the fields of the log? See [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).",
            "who": "The fields hold no person and no place. Which question needs only `time`, `status`, `ms`, `temp` and `bikes`? See [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A program that breaks usually says so. A server that runs out of memory stops, a missing file gives an error, a backup makes every answer slow, and the numbers of [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en) jump: errors, traffic, latency. In a talk at the conference OpML in 2020, Daniel Papasian and Todd Underwood of Google went through a decade of outages of one large ML pipeline, and found that most were not about ML at all: they came from the many machines and steps around it.

    An ML system can also fail without a sound. It keeps answering, with status 200 and as fast as ever, only worse: an input from upstream breaks, a day comes that looks like nothing in training, or the world moves away from the data the model learned from. No error message tells you; only the model's own numbers do.

    **An everyday example.** A fuel gauge that sticks at half full gives no warning. The car drives on, the needle looks normal, and nothing beeps, until the tank is empty. Only a second number shows it: the kilometers driven since the last fill-up.
    """)
    return


@app.cell(hide_code=True)
def predict_heat():
    guess_heat = L.guess(
        "heat",
        question="On Saturday 23 July 2011, a heat wave hit Washington: 38.5 degrees in the afternoon, by the data, and 3,285 bikes were rented. The model of the project learned from 2011 that warmer days bring more riders. How many bikes did it predict for the whole day?",
        options={
            "half": "About 1,600: in such heat, half as many",
            "close": "About 3,300: close to what was rented",
            "right": "About 6,800: more than twice as many",
            "none": "None: it stops with an error, since 38.5 degrees lies outside training",
        },
        reveal="**About 6,800: more than twice as many.** The model answered 6,824 bikes for the whole day, where 3,285 were rented: 147.5 bikes off per hour, almost six times a usual day. In 2011, warmer days brought more riders, up to a point; at 38.5 degrees, many people stayed out of the sun. The model saw too few such hours to learn that: only 21 of the 6,912 hours of training were 37 degrees or warmer, and 23 July is even one of the training days. And 38.5 degrees lies inside what training saw, so nothing stopped the requests: the service answered every hour with status 200, as fast as ever.",
    )
    guess_heat
    return (guess_heat,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Two ways to fail.** A *software failure* breaks the service or slows it down: a crash, a missing file, a full memory, a server that is too busy. The service answers with an error, late, or not at all, and the numbers about the service show it: errors, traffic, the p99. An ML system can also fail in its own way. The picture shows both, for the same hour.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Silent failure.** A **silent failure** is a failure in which the system keeps answering, with status 200 and as fast as ever, while the answers get worse. Nothing about the service changes, so a software monitor sees nothing. The causes come in four kinds:

        - *An input breaks or changes upstream*, in a step before the model. On 10 March 2011, the humidity sensor sent 0 for 22 hours, while it rained. An app sends degrees, a step starts sending km/h, or a step switches local time to UTC. The request is well formed; only its meaning changed.
        - *An edge case*: a day that looks like nothing in training.
        - *The world moves away from the training data*, such as far more riders a year later. That is the next lesson.
        - *The model changes its own data*, such as a recommender that only shows what is already popular. That is the lesson after.

        **Edge case.** An **edge case** is an input at the edge of what the model knows: rare in the training data, or missing from it, such as a heat wave, a hurricane or a holiday. The model still answers, and its answer can be far off: it carries what it learned on usual days over to a day that is not usual. In 2011, warmer days brought more riders, so the model expects even more at 38.5 degrees. Only 21 of the 6,912 hours of training were 37 degrees or warmer, and only 5 of the 292 training days were holidays.

        **Two checks from the side of the model.** Two numbers of the model catch many silent failures. The first compares every request with training: does each input lie within the range training saw? It works at once, on the first request, and catches a unit that changed: 31.2 degrees lies far above the 0.96 of training. It lets the heat of 23 July through, since 38.5 degrees lies inside the range. The second waits for the truth: once the hour is over and the rentals are counted, how far off was the model, compared with a usual day? It catches whatever makes the answers worse, but only afterwards. You write both, in Exercises 1 and 2.

        **At scale.** Libraries such as TensorFlow Data Validation keep a schema of the training data, with the type of each feature and the values or range it may take, and check every new batch against it: your first check, for every feature at once. Tools for model monitoring compare how far off a model is with earlier weeks, once the truth comes in. And teams write down each failure, with its cause and what would have caught it sooner, in a write-up called a *postmortem*.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **silent failure** | | a failure in which a system keeps answering, fast and with status 200, while its answers get worse, so that only the model's own numbers show it | |
        | **edge case** | | an input that is rare or missing in the training data, such as a heat wave or a holiday, where a model can be far off | |
        | upstream | | the steps before the model that make its inputs, such as a sensor, a weather service or an app | |
        """),
        L.watch_out("Training data can hold the same failure. 10 March 2011 is one of the training days, so the humidity in training already starts at 0, and a check against the range of training lets the broken sensor through."),
        L.watch_out("How far off the model was is known only once the truth is in: here an hour later, when the rentals are counted. For many models the truth comes much later, such as whether a loan is paid back, or never, for a loan that was refused."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Finding out how an ML system can fail:

    1. List what the system needs: the server and its code, the model file, every input and where it comes from, and the world the model learned from.
    2. Watch the service with its own numbers, such as errors, traffic and the p99: they catch the loud failures.
    3. Check every request against training: does each input lie within the range training saw?
    4. Once the truth is in, compute how far off the model was per day, compared with a usual day.
    5. Look at the worst days one by one: a broken input, an edge case, or a world that changed?
    6. Write each failure down, with its cause and the check that would have caught it sooner.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.SENSOR` is the request the service got for 8:00 on 10 March 2011: a dict with the hour and the ten features. `C.answer(request)` gives what the service answers, the model's bikes, whole and never below 0, and `C.rented(day, hour)` the bikes rented, with `C.SENSOR_DAY` for 10 March. The cell prints the request, the answer and the bikes rented, then the answer with the humidity of other rainy hours, 0.88. Run the cell.
    """)
    return


@app.cell
def example_sensor():
    print(dict(C.SENSOR))
    print("answer", C.answer(C.SENSOR), "rented", C.rented(C.SENSOR_DAY, 8))
    _fixed = dict(C.SENSOR)
    _fixed["hum"] = 0.88                 # the humidity of other rainy hours
    print("answer with hum 0.88", C.answer(_fixed))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At 8:00 on 10 March, it rained (`weathersit` 3), the humidity sensor sent 0, and the model answered 226 bikes, where 44 were rented. With a humidity of 0.88, as in other rainy hours, it answers 190: still too many, since rain keeps more people at home than the model expects, but 36 fewer. Over the whole day, the model answered 2,137 bikes; with 0.88, 1,795; 623 were rented. The service answered every hour with status 200, as fast as ever. In Exercise 1, you write a check that looks at every request before the model does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Outside what training saw

        `unseen(request, low, high)` gets a `request`, a dict with the hour and the ten features, and two dicts with a number for each of `temp`, `atemp`, `hum` and `windspeed`: `low`, the lowest value in the training hours, and `high`, the highest. The loop goes over each `name` in `low`. The function gives back the `name` of the first input outside what training saw, or `None` when every input lies within training. Replace the `...` so that the condition holds when the value of the input lies under its lowest value in training or above its highest.

        The check calls `unseen(C.NEW_APP, C.LOW, C.HIGH)`: the request of the new app for 16:00 on 14 July, with the temperature in degrees. Then a usual morning, a hot hour of 2012, an hour of the heat of 23 July, the wind in km/h, a sensor with no reading that sends −1, a morning colder than any hour of training, a morning without wind, and a rainy morning with a humidity of 1.
        """),
        L.task("Make the condition hold when the input lies under its lowest value in training or above its highest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unseen():
    box_unseen = L.code_box("unseen", lines=5)
    box_unseen
    return (box_unseen,)


@app.cell(hide_code=True)
def check_unseen(box_unseen):
    result_unseen = L.check("unseen", box_unseen.value)
    result_unseen
    return (result_unseen,)


@app.cell(hide_code=True)
def hints_unseen():
    L.hints(
        "unseen",
        [
            "Step 3 of the recipe: `request[name]` is the value of the input in the request, `low[name]` its lowest value in training and `high[name]` its highest. The first half, `request[name] < low[name]`, catches a value under the range. Which comparison catches a value above it?",
            "A value lies outside when either half holds, so `or` joins them. The second half compares `request[name]` with `high[name]`. With which sign?",
            "`        if request[name] < low[name] or ___:`",
        ],
        why="`if request[name] < low[name] or request[name] > high[name]:`: the value lies under the lowest value of training or above the highest. The new app's 31.2 lies far above the 0.96 of training, so `unseen` gives `'temp'`; a usual morning gives `None`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_off():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How far off, per day

        Once an hour is over, the bikes rented are known, and the model's answer can be held next to them. `off(predicted, rented)` gets two lists with a number for each hour of a day: `predicted`, the service's answers, and `rented`, the bikes rented. The loop goes over every hour `h`, with `range` up to `len(rented)`, and `total` adds up how far off the model was; the function divides `total` by the number of hours. Replace the `...` so that `total` grows by how far off the model was in hour `h`, up or down.

        The check calls `off([120, 80, 30], [100, 95, 30])`: 20 too many, 15 too few, and exactly right. Then the days of 14 July, 23 July and 10 March, and two short lists.
        """),
        L.task("Make `total` grow by how far off the model was in hour `h`, whether too high or too low."),
    ])
    return


@app.cell(hide_code=True)
def exercise_off():
    box_off = L.code_box("off", lines=5)
    box_off
    return (box_off,)


@app.cell(hide_code=True)
def check_off(box_off):
    result_off = L.check("off", box_off.value)
    result_off
    return (result_off,)


@app.cell(hide_code=True)
def hints_off():
    L.hints(
        "off",
        [
            "Step 4 of the recipe: in hour `h`, the model answered `predicted[h]` and `rented[h]` bikes were rented. Their difference is positive when the model was too high and negative when it was too low. The line adds a number to `total`; which number?",
            "Too high and too low should both count: `abs` turns a negative number into a positive one, as `abs(-15)` gives 15. Which difference goes inside it?",
            "`        total = total + ___`",
        ],
        why="`total = total + abs(predicted[h] - rented[h])`: how far off the model was in hour `h`, too high or too low. For the short lists: (20 + 15 + 0) / 3 = 11.7 bikes per hour. Without `abs`, 20 too many and 15 too few largely cancel out, leaving 1.7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Exercise 3. Which check sees it first?

    On Thursday 14 July 2011, a step upstream starts to send `windspeed` in km/h, where the data and training have km/h divided by 67. At 8:00, the request holds 24.0 instead of 0.36. The service answers every request with status 200, as fast as ever.
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first",
        question="Which check sees it first?",
        options={
            "tail": "The p99 of the latency of each minute",
            "errors": "The share of answers with status 400 or 500 in each minute",
            "off": "Your `off` on each day",
            "right": "Your `unseen` on each request",
        },
        correct="right",
        why={
            "right": "`unseen` gives `'windspeed'` on the very first request: 24.0 lies far above the 0.85 of training. `off` sees it too, but only once the hours are over: the model answers 17,202 bikes for the whole day, where 5,084 were rented, 505 off per hour instead of 26. The p99 and the errors watch the service, and the service is fine.",
            "tail": "The model answers a request with 24.0 just as fast as one with 0.36. Would a slow minute show it?",
            "errors": "24.0 is a well-formed number, answered with status 200. Would it count as an error?",
            "off": "`off` needs the bikes rented, which are known only once each hour is over. Which check looks at the request itself, before the model answers?",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_exercise_failures():
    mo.md(r"""
    ### Exercise 4. Loud or silent?

    Eight failures that could happen to the service of the project. Put each with what sees it: a software monitor, with the numbers about the service, or only the model's own numbers.
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
            "Ask for each failure: does the server still answer every request, fast and with status 200?",
            "Four failures break the service or slow it down. In the other four, the requests are well formed and the answers come fast: only the number of bikes in the answer is wrong.",
            "Monitor: out of memory, a missing package, \\_\\_\\_ and \\_\\_\\_. Model: a broken sensor, degrees, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**A software monitor:** out of memory, a missing package, a backup and a missing file, since the service breaks or slows down. **Only the model's numbers:** a broken sensor, degrees, a heat wave and another clock, since every request gets a fast answer with status 200, only a wrong one.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Six days of 2011

    Choose a day. The picture shows the bikes rented in each hour, and what the service answered: the model of the project, in whole bikes, never below 0.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_day = mo.ui.dropdown(
        options={
            "Thursday 14 July: a usual day": "usual",
            "Thursday 10 March: the broken sensor": "sensor",
            "Saturday 23 July: the heat wave": "heat",
            "Saturday 27 August: Hurricane Irene": "irene",
            "Monday 4 July: Independence Day": "july4",
            "Friday 25 November: after Thanksgiving": "thanksgiving",
        },
        value="Thursday 14 July: a usual day",
        label="Day",
    )
    mo.vstack([
        demo_day,
        L.task("Start with the usual day, then go through the other five. On which days does the model answer too much, and on which too little? Which of them would your `unseen` have caught?"),
    ])
    return (demo_day,)


@app.cell(hide_code=True)
def demo_picture(demo_day):
    mo.Html(C.demo_picture(demo_day.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    On 14 July, the model follows the rush hours, 25.7 bikes off per hour. On 10 March, it answers far too much: rain, and the broken sensor on top. On 23 July, the heat kept people at home, and the model answers more than twice the rentals. On 27 August, Hurricane Irene reached Washington: the model answers more than twice the rentals again, and the data stops at 17:00. On 4 July, Independence Day, it answers too little: more people rode than the model expects for a holiday, and it saw only 5 holidays in training. On 25 November, the day after Thanksgiving, it answers too much: a working day in the data like any other, but many people were off. Your `unseen` catches none of the six days, since every input lies inside training; only how far off the model was shows them.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Watching a running model: metrics, logs and alerts](/course/open/sys-monitoring?lang=en), the numbers about the service caught a pause and a slowdown, and the mean of the answers showed a new app that sent degrees; in this lesson, such failures have a name, and two checks from the side of the model. The edge cases of this lesson are single days. In the next lesson, *When the world changes: drift*, the world itself moves away from 2011: in 2012, far more people ride, and the model falls behind.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_failures, result_off, result_unseen):
    L.wrap_up(result_unseen, result_off, choice_first, result_failures)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
