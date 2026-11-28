# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-monitoring. Route: course/route/lessen/systemen.yaml. Checks, the log of the day, the rules and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-monitoring", "en")
    C = L.checks   # the log of one day of the service, the p99 of every minute, the rules and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) and [Your model behind an API](/course/open/sys-serving-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_quantile():
    quick_quantile = L.choice(
        "quick-quantile",
        question="For 200 trips to school, `np.quantile(times, 0.9)` gives 41. What does that say?",
        options={
            "mean": "The trips took 41 minutes on average",
            "right": "Nine in ten trips took 41 minutes or less",
            "longest": "The longest trip took 41 minutes, and the rest less",
            "ninety": "90 of the trips took 41 minutes",
        },
        correct="right",
        why={
            "right": "The 0.9 quantile is the value below which 9 in 10 values lie: 180 of the 200 trips took 41 minutes or less, and 20 took longer. In this lesson, the same kind of number describes the slowest requests of a service.",
            "mean": "A mean says where the middle is, pulled up by a few long trips. What does a quantile say instead? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "longest": "The longest trip is what `np.max` gives. Which share of the trips lies below the 0.9 quantile? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "ninety": "0.9 is a share of the trips, not a count. Which share of the 200 trips lies below 41? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
        },
    )
    quick_quantile
    return (quick_quantile,)


@app.cell(hide_code=True)
def quick_check_500():
    quick_500 = L.choice(
        "quick-500",
        question="A server answers a request for a prediction with status 500. What does that tell you?",
        options={
            "bad-request": "The request was wrong, for instance a missing field",
            "right": "The server broke on the request, which may well have been fine",
            "bikes": "The model expects 500 bikes for that hour",
            "slow": "The answer took 500 milliseconds, longer than allowed, and the client gave up",
        },
        correct="right",
        why={
            "right": "500 is *Internal Server Error*: the server broke while it handled the request, for instance because its code stopped with an error or its memory ran out. In this lesson, the share of answers with 500 is one of the numbers a service watches.",
            "bad-request": "A wrong request, such as one with a missing field, gets a code from 400 on. Which side does a code from 500 on blame? See [Your model behind an API](/course/open/sys-serving-api?lang=en).",
            "bikes": "The status code says how the request went, not what the model said. What does 500 say? See [Your model behind an API](/course/open/sys-serving-api?lang=en).",
            "slow": "A status code is not a time. What does 500 say about the server? See [Your model behind an API](/course/open/sys-serving-api?lang=en).",
        },
    )
    quick_500
    return (quick_500,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model in production runs day and night, and nobody reads every request. In [Your model behind an API](/course/open/sys-serving-api?lang=en), the service of the project answered your own requests on your own computer. In production, the same service answers tens of thousands of requests a day, and some go wrong in ways nobody sees: a server that stops for two seconds, a backup that makes every answer slow, a new version of an app that sends a temperature in degrees. Without numbers, you hear about it from the people who use the app, if you hear about it at all.

    In their book *Site Reliability Engineering*, engineers at Google name four golden signals that every service should watch: latency, traffic, errors and saturation, which is how full the server is. Numbers like these, measured again and again while the service runs, are called metrics. A model needs more than that. It can answer fast, with status 200, and still be wrong, so its own numbers need watching too: what it answers, and what goes in.

    **An everyday example.** A bus company says that its buses run 2 minutes late on average. That sounds fine. But if 1 bus in 100 is 40 minutes late, someone who takes the bus twice a day meets such a bus about every seven weeks, and remembers it. The average says nothing about that bus; only a number about the slowest buses does.
    """)
    return


@app.cell(hide_code=True)
def predict_tail():
    guess_tail = L.guess(
        "tail",
        question="On 14 July 2011, the service of the project answered 59,912 requests, in 28.0 ms on average. Sorted by latency, where do the slowest 1 in 100 requests start, do you think?",
        options={
            "close": "About 30 ms, just above the mean",
            "half": "About 45 ms, one and a half times the mean",
            "right": "About 160 ms, almost six times the mean",
            "slowest": "About 1,600 ms, as long as the slowest request",
        },
        reveal="**About 160 ms, almost six times the mean.** Sorted by latency, the slowest 1 in 100 requests start at 159 ms: 599 requests took longer. Half of all requests took less than 22.6 ms, and almost 19 in 20 less than 40 ms. The mean of 28.0 ms lies close to that fast majority and hides the slow ones: a server that stopped for two seconds at 09:12, a backup that made every answer slow from 14:00 to 14:25, and now and then a single slow request. In this lesson, you find them in the log, and decide when someone has to look.",
    )
    guess_tail
    return (guess_tail,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Model monitoring.** A running model is watched with two kinds of numbers. Numbers about the *service* say whether it answers, and how fast: the four golden signals. **Latency** is the time between a request and its answer, here in milliseconds (ms). *Traffic* is the number of requests per minute. *Errors* are the answers with a status code from 500 on, where the server broke. *Saturation* says how full the server is: its memory and its processor. Numbers about the *model* say whether its answers are any good: what it answers, hour by hour; what goes in, next to what went in during training; and, once the truth is known, such as the bikes really rented in that hour, how far off it was. Watching both is **model monitoring**.

        **Structured logging.** Every number starts from a log. A line such as *Request for 9:00 took 30 ms, all fine* is easy for a person to read, but a program has to guess where each number is. With **structured logging**, a service writes one line per request as a JSON object, with fixed fields:

        ```
        {"time": "09:00:02.418", "path": "/predict", "status": 200, "ms": 29.7, "temp": 0.68, "bikes": 223}
        ```

        `json.loads` turns such a line into a dict again, so a program can count the requests of a minute, take their latencies, or group the answers by hour. The line holds what went in and what came out, so the log watches the model as well as the server.

        **The day of this lesson.** The log of this lesson covers one day of the service of the project, Thursday 14 July 2011, made by the course: every time someone opens the bike-share app, the app asks the service how busy the hour will be. The hours and the weather come from the bike data, and the answers from the course's model; the requests and their latencies are made up, with three things that went wrong.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Tail latency.** The picture shows the 59,912 requests of the day. Most took 15 to 40 ms, and the mean is 28.0 ms. But a thin tail runs far to the right: requests that took ten or fifty times as long as the median. The mean hides them, since a few slow requests among thousands of fast ones hardly move it. So a service describes its latency with quantiles, as in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en). The *p50* is the median: half of the requests were faster. The *p99*, the 99th percentile, is the latency below which 99 in 100 requests lie. The latency of the slowest requests, above the p99, is the **tail latency**. It matters more than it seems. A page that needs 100 requests to a service is only as fast as the slowest of them: with 1 in 100 requests slow, 63 in 100 such pages wait for at least one slow request.

        **Alert rule.** Nobody watches a chart all day. An **alert rule** says when a person has to look: a number, a limit, and how long the number has to stay over it, such as *the p99 above 300 ms for 5 minutes in a row*. The time matters as much as the limit. A rule that goes off on every spike, such as the pause of two seconds at 09:12, calls someone for nothing; after a few of those, people stop looking, which is called *alert fatigue*. A rule that waits too long, or a limit set too high, lets a real problem run for an hour before anyone knows. You write the condition in Exercise 2, and tune the rule on the day in the demo.

        **At scale.** Prometheus collects numbers from many servers, such as how many requests fell in each range of latency, and estimates the p99 per minute from those counts with `histogram_quantile`. An alerting rule in Prometheus has a `for` clause, such as `for: 5m`: the minutes of Exercise 2. Grafana draws the charts, and OpenTelemetry is a common way for a service to send its logs and numbers to tools such as these. Tools for model monitoring add the numbers of the model: what goes in, what comes out, and how far off it was.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **model monitoring** | | watching a model in use with numbers about the service and about the model, and an alert when something goes wrong | |
        | **structured logging** | | writing one line per request with fixed fields, such as a JSON object, so that a program can read the log | `json.dumps(...)`, `json.loads(line)` |
        | **tail latency** | | the latency of the slowest requests, such as those above the 99th percentile | |
        | p50, p99 | p fifty, p ninety-nine | the latency below which half of the requests lie, or 99 in 100 | |
        | **alert rule** | | a number, a limit, and how long the number has to stay over it before a person is called | |
        """),
        L.watch_out("Numbers about the service do not show a model that is wrong. A request with a temperature in degrees gets status 200, as fast as ever, and an answer of thousands of bikes. Watch what goes in and what comes out too."),
        L.watch_out("A p99 over a few requests lies close to the slowest of them: with 40 requests in a minute, one slow request decides it. One more reason why an alert should wait for a number that stays high."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Watching a running model:

    1. Log every request as one structured line: the time, the path, the status, the latency, the inputs and the answer.
    2. Per minute, compute the numbers about the service: the requests, the share of errors, and the p50 and p99 of the latency; and keep an eye on the memory and the processor of the server.
    3. Per hour, compute the numbers about the model: the mean answer, the inputs next to those of training, and, once the truth is known, how far off it was.
    4. Draw each number over time, next to its usual values.
    5. For each number that needs a person, write an alert rule: a limit, and how long the number has to stay over it.
    6. Try every rule on a past day: does it catch the real problems, and how often does it go off for nothing?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.at(time)` gives the log lines of one minute as a list of texts: `C.at("09:00")` holds the requests from 09:00:00 to 09:00:59. The cell prints the first line of 09:00, then, for 09:00 and 09:12, how many requests came in, the mean and the median of their latency, and the slowest. Run the cell.
    """)
    return


@app.cell
def example_log():
    import json
    import numpy as np

    print(C.at("09:00")[0])
    for _time in ("09:00", "09:12"):
        _ms = [json.loads(_line)["ms"] for _line in C.at(_time)]
        print(_time, len(_ms), "requests, mean", round(float(np.mean(_ms)), 1),
              "median", round(float(np.median(_ms)), 1), "slowest", max(_ms))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the quiet minute of 09:00, 49 requests came in, and the mean of 24.6 ms lies close to the median of 22.2 ms. At 09:12, the server stopped for two seconds, and one request waited 1,579.6 ms. That one request pulls the mean up to 70.4 ms, as one long trip pulls up a mean, while the median hardly moves: 23.5 ms. Neither says how slow the slow requests were. In Exercise 1, you write the number that does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tail():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The slowest 1 in 100

        `tail(lines)` gets `lines`, the log lines of one minute, as a list of texts. Each `line` is one JSON object: `json.loads` turns it into a dict, and `ms` collects the latency of every request, from the field `"ms"`. `p50` is the median, from `np.median`, and `float` makes it a plain Python number; `round` rounds both numbers to whole milliseconds in the dict the function gives back. Replace the `...` so that `p99` is the latency below which 99 in 100 requests of the minute lie.

        The check calls `tail(C.at("09:00"))`, then the minutes of 09:12, 14:05, 03:00 and 17:30.
        """),
        L.task("Make `p99` the latency below which 99 in 100 requests of the minute lie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tail():
    box_tail = L.code_box("tail", lines=5)
    box_tail
    return (box_tail,)


@app.cell(hide_code=True)
def check_tail(box_tail):
    result_tail = L.check("tail", box_tail.value)
    result_tail
    return (result_tail,)


@app.cell(hide_code=True)
def hints_tail():
    L.hints(
        "tail",
        [
            "Step 2 of the recipe: the p99 is a quantile of `ms`, as in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en). `np.quantile` takes the values and the share of them that should lie below, and `float` makes the result a plain Python number. The result goes in `p99`.",
            "A quantile takes the share as a number from 0 to 1, not as a percentage: the median is the 0.5 quantile. Which number is the share 99 in 100?",
            "`    p99 = float(np.quantile(ms, ___))`",
        ],
        why="`p99 = float(np.quantile(ms, 0.99))`: the latency below which 99 in 100 requests of the minute lie. At 09:00, that is 55 ms, two and a half times the median of 22 ms; at 09:12, one request of 1,579.6 ms pulls it up to 1,064 ms.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_alert():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Not on one spike

        `alert(p99s, limit, minutes)` gets `p99s`, the p99 of each minute in order, a `limit` in milliseconds, and a number of `minutes`. The loop goes over every minute `i` from `minutes - 1` on, with `range` up to `len(p99s)`, and `last` holds the p99 of the last `minutes` minutes, up to and including minute `i`. The function gives back the first minute at which the alert goes off, or `None` when it never does. Replace the `...` so that the alert goes off only when every minute in `last` is above `limit`.

        The check calls `alert([120, 950, 130, 340, 360, 380, 410, 390], 300, 5)`: one spike of 950 in minute 1, and from minute 3 on, a number that stays above 300. Then `C.P99`, the p99 of every minute of the day, with three rules, and a few short lists.
        """),
        L.task("Make the alert go off only when every minute in `last` is above `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_alert():
    box_alert = L.code_box("alert", lines=6)
    box_alert
    return (box_alert,)


@app.cell(hide_code=True)
def check_alert(box_alert):
    result_alert = L.check("alert", box_alert.value)
    result_alert
    return (result_alert,)


@app.cell(hide_code=True)
def hints_alert():
    L.hints(
        "alert",
        [
            "Step 5 of the recipe: the alert goes off only when the number stayed above the limit in every one of the last minutes, so one minute of `last` at or under `limit` keeps it quiet. Which single number of `last` tells you whether all of them are above `limit`?",
            "If the smallest number of `last` is above `limit`, so is every other number of `last`. Which Python function gives the smallest number of a list?",
            "`        if ___ > limit:`",
        ],
        why="`if min(last) > limit:`: the smallest p99 of the last minutes is above the limit only when every one of them is. The spike of 950 between minutes of 120 and 130 sets nothing off; at minute 7, the last 5 minutes all lie above 300, and the alert goes off.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.md(r"""
    ### Exercise 3. Fast, and wrong

    At 16:00, a new version of the bike-share app goes live, for one user in five. Nobody notices: the app still gets its answers, fast. Yet something is wrong with what the new version sends.
    """)
    return


@app.cell(hide_code=True)
def choice_model_cell():
    choice_model = L.choice(
        "model",
        question="Which number in the log shows that something is wrong?",
        options={
            "tail": "The p99 of the latency of each minute",
            "errors": "The share of answers with status 400 or 500 in each minute",
            "median": "The median of `bikes` per hour, next to other days",
            "right": "The mean of `bikes` per hour, next to other days",
        },
        correct="right",
        why={
            "right": "The new version sends temp in degrees: 31.2 at 16:00, where the data has 0.76. To the service, such a request is well formed: it answers with status 200, as fast as ever, with over 5,000 bikes. From 16:00, the mean of `bikes` per hour jumps to 1,311, and to 1,537 at 17:00, where the old version gets 289 and 537. The median stays at 289 and 537, since four in five requests still come from the old version: here the mean is the number to watch, since one answer in five is ten times too high or more, and that pulls the mean far up. The p99 and the errors watch the service, and the service is fine.",
            "tail": "The new version is as fast as the old one. Would a slow minute show it?",
            "errors": "A request with a temperature in degrees gets status 200: the number is well formed. Would it count as an error?",
            "median": "The median stays at 289 at 16:00 and 537 at 17:00, the answers of the old version: four in five requests still come from it, and the median is the middle answer. Which number moves when one answer in five is far too high?",
        },
    )
    choice_model
    return (choice_model,)


@app.cell(hide_code=True)
def text_exercise_watch():
    mo.md(r"""
    ### Exercise 4. The service or the model?

    Eight numbers a team could watch for the service of the project. Put each with the question it answers: does the service answer, and how fast, or are the answers of the model any good?
    """)
    return


@app.cell(hide_code=True)
def sort_watch_cell():
    sort_watch = L.sort("watch")
    sort_watch
    return (sort_watch,)


@app.cell(hide_code=True)
def check_watch(sort_watch):
    result_watch = L.check("watch", sort_watch)
    result_watch
    return (result_watch,)


@app.cell(hide_code=True)
def hints_watch():
    L.hints(
        "watch",
        [
            "Ask for each number: would it change if the model gave other answers, or other numbers went into it, while the server stayed just as fast and healthy?",
            "Four numbers are about the server: how many requests come in, how many break, how slow they are, and how full the server is. Which four are about what the model gets and answers?",
            "Service: traffic, errors, \\_\\_\\_ and \\_\\_\\_. Model: answers, zeros, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**The service:** traffic, errors, tail latency and memory, since they say whether answers come, and how fast. **The model:** answers, zeros, inputs and how far off, since they say whether the answers are any good. The new version of the app at 16:00 shows only in the second group.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Tuning the alert

    Choose a limit for the p99 and a number of minutes in a row. The picture shows the p99 of every minute of 14 July, the limit, and a dot at the top for each alert: one during the slowdown from 14:00 to 14:25, or one for nothing.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_limit = mo.ui.slider(start=100, stop=600, step=50, value=300, label="Limit in ms", show_value=True)
    demo_minutes = mo.ui.slider(start=1, stop=10, step=1, value=1, label="Minutes in a row", show_value=True)
    mo.vstack([
        demo_limit,
        demo_minutes,
        L.task("Keep the limit at 300 ms and raise the minutes from 1: from how many minutes on is the spike of 09:12 no longer an alert? Then try 150 ms with 1 minute, and 600 ms with 10 minutes. What goes wrong in each?"),
    ])
    return demo_limit, demo_minutes


@app.cell(hide_code=True)
def demo_picture(demo_limit, demo_minutes):
    mo.Html(C.demo_picture(demo_limit.value, demo_minutes.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    At 300 ms with 1 minute, the rule goes off twice: at 09:12, for a pause of two seconds that was over before anyone could look, and at 14:00. From 2 minutes on, only the slowdown sets it off: at 14:01 with 2 minutes, and a minute later for each minute more. At 150 ms with 1 minute, it goes off 58 times, 57 of them for nothing, spread over the whole day: in a minute of a few dozen requests, one slow request is enough. This is how alert fatigue starts. At 600 ms with 10 minutes, it never goes off, though the requests were slow for 25 minutes. The rule of Exercise 2 (300 ms for 5 minutes) goes off once, at 14:04.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Your model behind an API](/course/open/sys-serving-api?lang=en), the service answered every request with a status code; in this lesson, it writes a structured line for each, and the log becomes numbers. The median and the quantiles of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) turned the latencies of each minute into a p50 and a p99, and an alert rule decided when a person has to look. *Before and after the model: the serving path* advised a log of the inputs a service gets; the new version of the app at 16:00 shows why. In the next lesson, *How ML systems fail*, such quiet failures get a name, next to the crashes that numbers about the service do show.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_model, result_alert, result_tail, result_watch):
    L.wrap_up(result_tail, result_alert, choice_model, result_watch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
