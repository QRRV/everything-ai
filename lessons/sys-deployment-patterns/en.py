# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-deployment-patterns. Route: course/route/lessen/systemen.yaml. Checks, the simulated day of the app, the stream of station 31 and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-deployment-patterns", "en")
    C = L.checks   # the simulated day of the bike-share app, the stream of one station, and the pictures


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
def quick_check_serving():
    quick_serving = L.choice(
        "quick-serving",
        question="In the course's spam service, which part makes sure that every message gets an answer, even when the model cannot help?",
        options={
            "features": "Features, which turns the words of a message into counts",
            "monitoring": "Monitoring, which watches the answers and raises an alert",
            "right": "Serving, which gives every message an answer: spam, ok or no score",
            "feedback": "Feedback, which keeps every message whose answer a user corrects later",
        },
        correct="right",
        why={
            "right": "Serving turns the probability of the model into an answer, and answers even when the model cannot help. This lesson asks when the model does its part of that work: at the moment of each request, or in advance for everyone.",
            "features": "Features turns words into counts for the model. Which part turns what the model says into an answer, and answers even without the model? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "monitoring": "Monitoring watches the answers once they are given. Which part gives them? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "feedback": "Feedback collects corrections to train on later. Which part answers each message as it comes in? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_serving
    return (quick_serving,)


@app.cell(hide_code=True)
def quick_check_production():
    quick_production = L.choice(
        "quick-production",
        question="What does production ask of a model that research does not?",
        options={
            "score": "The highest score on a test set that stays the same",
            "right": "Quick answers for many users, on data that keeps changing",
            "bigger": "A bigger model, since more users means more data",
            "notebook": "Results written down in a notebook, so that others can repeat them",
        },
        correct="right",
        why={
            "right": "In production, every single answer counts, for many users at once, and the data keeps coming. How soon an answer must come is the first question of this lesson: it decides when the model does its work.",
            "score": "A fixed test set is what research aims at. What does a model in use have to do for real users, every day? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "bigger": "More users mostly means more requests at once. What does each of them need from the model? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "notebook": "Writing the results down ends a research project. When is a model in production done? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_production
    return (quick_production,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model that is trained and measured does nothing yet: it has to make predictions where someone uses them. *When* it makes them is the first choice, and it decides much of the rest. A bike-share app can work out a suggestion for every member during the night, so that the app shows it at once: a station with a free bike, at the time the member usually rides. Or the app can ask the model the moment a member opens it, so that the suggestion fits that moment. The first is cheap, and nobody waits, but some of the work may be for nobody. The second is always fresh, but then the model has to answer within a fraction of a second, every time, while the member waits.

    **An everyday example.** A bakery bakes its bread at 4 in the morning for the whole day: by the evening, some is left over, and none of it is fresh. A pizzeria makes each pizza when it is ordered: always hot, but the customer waits, and on a busy evening the kitchen has to keep up. A sushi bar with a belt puts a new plate on as soon as one is taken: nobody orders, and the kitchen acts on what happens. And a bread maker at home bakes in your own kitchen, without a shop: small, but always at hand.
    """)
    return


@app.cell(hide_code=True)
def predict_used():
    guess_used = L.guess(
        "used",
        question="The bike-share app has 10,000 members. At 2 at night, a batch makes a suggestion for each of them, which the app shows as soon as it opens. On a simulated day, how many of the 10,000 suggestions do members see, do you think?",
        options={
            "most": "Most of them: every member has a suggestion",
            "half": "About half",
            "quarter": "Between 2,000 and 3,000",
            "right": "Fewer than 2,000",
        },
        reveal="**Fewer than 2,000.** On the simulated day, 1,316 members open the app and see their suggestion. The other 8,684, 86.8% of the work of the night, were made for nobody. And the 60 members who join during the day find no suggestion at all: the batch did not know them yet. The day is a simulation the course made, with members who ride every day, every week or rarely; a real app has its own numbers, but the pattern is common.",
    )
    guess_used
    return (guess_used,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Deployment.** **Deployment** is putting a model to use, in a place where people or programs get its predictions. One question shapes it more than any other: *when* does the model make a prediction? There are three answers, and a fourth pattern answers *where*: on the user's own phone.

        **Batch.** With **batch inference**, the model makes a prediction for everyone at a fixed time, such as every night, and the predictions are stored; the app only looks one up. Nobody waits for the model, a large batch is cheap per prediction, and a slow model matters less, as long as each run is done in time. The price: predictions for whoever does not come, predictions that are hours old by the time someone sees them, and nothing for whoever is new since the last run.

        **Online.** With **online inference**, the model makes a prediction the moment a request comes in, from the input of that moment. The prediction is never old and fits the request, even for a new user. The price: someone waits, so the model has to answer quickly, at the busiest hour too, and there is one model call for every request.

        **Streaming.** With **streaming inference**, the input is a stream: events that keep coming in, such as rentals, positions or payments. After every event, the model or a rule makes a prediction, often from what happened in a window of time, such as the last 15 minutes, and something happens when it matters, without anyone asking. It reacts within seconds to what is going on now, but it has to keep up with every event, and it only knows what came in so far.

        **On the device.** The model can also run on the user's own phone or laptop. It then works without a network, answers without a trip to a server, and the data stays with the user. The price: the model has to be small enough for the phone, and every phone runs its own copy, which is harder to update. On the phone, the model can still work in each of the three ways: on a request, after every event, or in a batch at night.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Made against used.** For a batch, two numbers say a lot: how many predictions are made against how many are used, and how old a prediction is when it is used. Online makes as many predictions as there are requests, each zero hours old; the question there is how quickly each one comes. Many systems combine the patterns: a batch for what can wait, and online for what cannot.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **deployment** | | putting a model to use, in a place where people or programs get its predictions | |
        | **batch inference** | | making predictions for everyone in advance, at a fixed time such as every night, and storing them | |
        | **online inference** | | making a prediction the moment a request comes in, from the input of that moment, while someone waits | |
        | **streaming inference** | | making a prediction after every event of a stream, such as every rental at a station, without anyone asking | |
        """),
        L.watch_out("*Streaming* here means a stream of events coming in. A chatbot that shows its words while it writes them is said to stream its answer too: there, the answer goes out in pieces, but it is still one online request."),
        L.watch_out("*On the device* here means the user's own phone or laptop. In code for neural networks, *device* means something else: the chip a model runs on, such as the processor or a graphics card."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Choosing when a model makes its predictions:

    1. Find out when the input is known: the evening before, only at the moment of the request, or as a stream of events.
    2. Ask how soon the answer is needed, and who waits for it: a person in the app, a program, or nobody.
    3. For a batch, count how many predictions get used, and how old they are by then.
    4. Choose: everything known in advance and the answer may wait, batch; someone waits for an answer to new input, online; events keep coming and something must happen on them, streaming; no network, or data that must stay with the user, on the device.
    5. Combine where it pays: a batch for what can wait, online or streaming for what cannot.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small day: the batch of 2:00 makes a suggestion for members a to e. During the day, the app is opened by b at 8:00, d at 8:30, f at 12:00 and b again at 18:00; f joined that morning. The cell goes through the opens and keeps those of members the batch knows: for each, it adds up how many hours the suggestion is old. Run the cell.
    """)
    return


@app.cell
def example_small_day():
    batch = ["a", "b", "c", "d", "e"]
    opened = ["b", "d", "f", "b"]
    hours = [8.0, 8.5, 12.0, 18.0]
    seen = set(opened)
    print("members who opened the app:", len(seen))
    total = 0
    count = 0
    for i in range(len(opened)):
        if opened[i] in batch:
            total = total + hours[i] - 2
            count = count + 1
    print("opens with a suggestion:", count)
    print("mean age in hours:", total / count)
    print("model calls online:", len(opened))
    return batch, count, hours, i, opened, seen, total


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Three members opened the app, but only b and d had a suggestion: 2 of the 5 were seen, and the suggestions of a, c and e were made for nobody. b opened the app twice and saw the same suggestion twice. f is new since the night and got none. The suggestions were 6, 6.5 and 16 hours old when seen, 9.5 on average. Online, the model would have made 4 predictions, each at the moment of the open, including one for f. Exercise 1 counts the suggestions nobody saw.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unused():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Suggestions for nobody

        `unused(batch, opened)` gets the members the batch made a suggestion for, and the member of every time the app was opened, with repeats and with new members. It gives back how many suggestions nobody saw. `seen = set(opened)` is a set of everyone who opened the app, the loop goes through the members of `batch` with `user`, and `used` counts those who opened the app, so that `len(batch) - used` is what nobody saw. Replace the `...` with two lines that count them.

        The check calls `unused(C.BATCH, C.OPENED)`: the 10,000 members of the night and the 2,473 opens of the simulated day. Then the worked example, which should give 3, a day nobody opened the app, a day only new members did, a batch of nobody, and a day of only repeats.
        """),
        L.task("Count the members of the batch who opened the app, and give back how many did not."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unused():
    box_unused = L.code_box("unused", lines=7)
    box_unused
    return (box_unused,)


@app.cell(hide_code=True)
def check_unused(box_unused):
    result_unused = L.check("unused", box_unused.value)
    result_unused
    return (result_unused,)


@app.cell(hide_code=True)
def hints_unused():
    L.hints(
        "unused",
        [
            "Step 3 of the recipe. For each `user` of the batch, the loop asks one question: did this member open the app? If so, `used` goes up by one, as `count` did in the worked example.",
            "Python's `in` asks whether a value is among the values of a set: `\"b\" in set([\"b\", \"d\", \"b\"])` is `True`, and the set holds `\"b\"` once. Which set above holds everyone who opened the app?",
            "`        if user ___:`",
        ],
        why="`if user in seen:` asks of every member of the batch whether that member opened the app, and `used = used + 1` counts it once, however often the app was opened. New members are not in `batch`, so they are never counted. A set answers `in` at once, even for 10,000 members.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_recent():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The last quarter of an hour

        The same company has a stream: every rental at a station comes in as an event, with the minute of the day it happened, such as 480 for 8:00. After every rental, a rule counts the rentals of the last 15 minutes; from 10 on, a truck goes there with bikes. `recent(times, now, minutes)` gets the minutes of rentals, the minute `now` and the length of the window in minutes. It gives back how many rentals fall in the window: after `now - minutes`, and at `now` or before. The list may go on past `now`, as when you replay a day afterwards. The loop goes through the rentals with `t`. Replace the `...` with two lines that count those in the window in `count`.

        The check calls `recent(C.TIMES, 1050, 15)`: the 83 rentals of a simulated day at station 31, all of them, with 17:30 as `now`. Then rentals every 5 minutes, a window with no rentals in it, a window longer than the day so far, no rentals at all, and rentals after `now`.
        """),
        L.task("Count the rentals after `now - minutes`, and at `now` or before."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recent():
    box_recent = L.code_box("recent", lines=6)
    box_recent
    return (box_recent,)


@app.cell(hide_code=True)
def check_recent(box_recent):
    result_recent = L.check("recent", box_recent.value)
    result_recent
    return (result_recent,)


@app.cell(hide_code=True)
def hints_recent():
    L.hints(
        "recent",
        [
            "Step 1 of the recipe: the input is a stream. The condition has two parts joined by `and`, each comparing `t` with a number: one keeps out the rentals from more than `minutes` minutes before `now`, the other the rentals that have not happened yet.",
            "A rental at exactly `now` is the one that just came in, so it counts; a rental after `now` has not happened yet at that moment.",
            "`        if t > now - minutes and t ___ now:`",
        ],
        why="`if t > now - minutes and t <= now:` keeps the rentals of the last 15 minutes, with the one of this very minute: at 17:30, that is 17 rentals, from 17:16 on.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_patterns():
    mo.md(r"""
    ### Exercise 3. Which pattern?

    Eight uses of a model at the bike-share company. Put each under the pattern that fits it best; where the model has to run on the phone itself, choose On the device.
    """)
    return


@app.cell(hide_code=True)
def sort_patterns_cell():
    sort_patterns = L.sort("patterns")
    sort_patterns
    return (sort_patterns,)


@app.cell(hide_code=True)
def check_patterns(sort_patterns):
    result_patterns = L.check("patterns", sort_patterns)
    result_patterns
    return (result_patterns,)


@app.cell(hide_code=True)
def hints_patterns():
    L.hints(
        "patterns",
        [
            "Step 4 of the recipe. Ask for each use: is everything known in advance, does someone wait for the answer, do events keep coming in, or may there be no network?",
            "Two predictions can be made the evening before, and two depend on what a rider just typed. Which two react to events, without anyone asking?",
            "Batch: trucks and \\_\\_\\_. Online: price and \\_\\_\\_. Streaming: a bike on the move and \\_\\_\\_. On the device: voice and \\_\\_\\_.",
        ],
        why="**Batch:** the trucks at night and the Monday email, since everything is known in advance and nobody waits. **Online:** the price of a ride and the help desk, since someone waits for an answer to what was just typed. **Streaming:** a bike on the move and a station almost empty, since events keep coming and something must happen on them. **On the device:** voice, since the recording must stay on the phone, and a fall, since there may be no network.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hourly():
    mo.md(r"""
    ### Exercise 4. Every hour

    Members complain that the suggestion of the night is out of date by the evening. The team runs the batch every hour instead, from 2 at night on.
    """)
    return


@app.cell(hide_code=True)
def choice_hourly_cell():
    choice_hourly = L.choice(
        "hourly",
        question="What changes?",
        options={
            "all": "Every member, new ones too, now sees a suggestion of at most an hour old",
            "right": "Fresher suggestions, 24 times the work, and new members still wait for a run",
            "cheaper": "About the same work: the batch only runs for members who open the app",
            "online": "The same as online: an hour is so short that nobody can notice the difference",
        },
        correct="right",
        why={
            "right": "Each run makes a suggestion for every member, so 24 runs make 24 times as many, while members open the app as often as before: nearly all the extra suggestions are for nobody. A suggestion is fresher when it is seen. And a member who joins at 14:20 is in no batch until the run of 15:00. The demo shows the numbers.",
            "all": "A member who joins during the day is in no batch until the next run. When do new members open the app?",
            "cheaper": "A batch does not know who will open the app: it makes a suggestion for every member, at every run. How many does it make in 24 runs?",
            "online": "Online makes a suggestion at the moment of the request, even for a member who joined a minute ago. What does the hourly batch give that member?",
        },
    )
    choice_hourly
    return (choice_hourly,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### How often the batch runs

    The demo runs the batch of the app 1, 2, 4, 8 or 24 times a day, on the same simulated day, from 2 at night on at equal steps. For each choice, the picture shows how many suggestions the model makes, how many of them a member sees, and how old a suggestion is when it is seen. Beside it is online, with one model call for each time the app is opened.
    """)
    return


@app.cell(hide_code=True)
def demo_runs():
    L.explore("runs", "runs per day", lambda _v: C.frame(int(_v), L.lang), start=1, values=C.RUNS, role="",
              task="Run the batch more often: what happens to the work, to the age of a suggestion, and to the new members?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Once a day, the batch makes 10,000 suggestions; members see 1,316 of them, 11.8 hours old on average. Every hour, it makes 240,491, about 24 times as many, and members see 2,379 of them: 1.0%. A suggestion is then half an hour old on average, but 60 opens still find none, since every new member opens the app right after joining, before the next run. Online makes 2,473 calls, each at the moment of the request, new members included: almost a hundred times fewer than the hourly batch, but each one while a member waits.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en), serving was the part that gives every message an answer. This lesson asked *when* the model does its work: in advance for everyone, at each request, after every event, or on the user's own phone. In the next lesson, *Your model behind an API*, you put a model online: a small server that waits for requests and answers each one. Models on phones, laptops and sensors come back in *At the edge: devices, sensors and the cloud*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hourly, result_patterns, result_recent, result_unused):
    L.wrap_up(result_unused, result_recent, result_patterns, choice_hourly)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
