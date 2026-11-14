# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-labels. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-labels", "en")
    C = L.checks   # a web shop the course simulates: 300 orders, their returns, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Classification: predicting a class](/course/open/ml-classification?lang=en) and one about [Where does this data come from?](/course/open/data-provenance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_classes():
    quick_classes = L.choice(
        "quick-classes",
        question="A web shop's model says for every order: it will come back, or it will not. What kind of task is that?",
        options={
            "regression": "Regression: it predicts a number, 1 or 0",
            "right": "Classification: the label is one of two classes",
            "neither": "Neither, because nobody knows the answer yet",
        },
        correct="right",
        why={
            "regression": "Writing the classes as 1 and 0 does not make them numbers to predict: a value in between, such as 0.4 of a return, makes no sense. See *The idea* in [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "neither": "Nobody knows the answer when the model predicts; that holds for every prediction. The answer comes later, as the label. See *The idea* in [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "right": "Two classes, returned and kept. Today is about where such a label comes from, and when.",
        },
    )
    quick_classes
    return (quick_classes,)


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="You want to train on a table of orders that you found online, without a license. What do you do?",
        options={
            "online": "Use it as you like: it is online",
            "name": "Use it, and name where you found it",
            "right": "Ask the maker first: without a license you may not just share it or build on it",
        },
        correct="right",
        why={
            "online": "Free to download is not free to use: the license says what you may do. See *The idea* in [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "name": "Naming the maker is what CC BY asks, but this table has no license at all, so nothing allows you to use it yet. See *The idea* in [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "Right, and orders are data about people, so consent matters too. Where the data comes from belongs in its datasheet, and today you add one more line: where the labels come from.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A web shop wants a model that predicts which orders will come back, so that it can help customers choose the right size. The model learns from earlier orders with a label: returned or kept. But nobody types in that label. It appears when a parcel comes back, up to 30 days after the order. Until then, what is the label of an order that has not come back?

    **An everyday example.** A restaurant asks every guest to rate the meal afterward. A guest who has not answered yet did not give a bad rating: the rating is unknown. And the guests who answer first are often the ones with a complaint. Count the silent ones as satisfied, or count only the first answers, and you get the restaurant wrong both times.
    """)
    return


@app.cell(hide_code=True)
def predict_rate_cell():
    guess_rate = L.guess(
        "rate-guess",
        question="On day 60 the shop has 95 orders and 15 returns so far. 49 of the orders are younger than 30 days, so some of them can still come back. In the end about one in five orders comes back. What does 15 / 95 give?",
        options={
            "right": "Too little: 16%, because some young orders will still come back",
            "fair": "About right: 16% is close to one in five",
            "much": "Too much, because returns come in first",
        },
        reveal="**Too little: 16%.** 15 / 95 counts the young orders that have not come back as kept, but some of them will still come back. The 46 orders that are at least 30 days old give 9 / 46 = 20%.",
    )
    guess_rate
    return (guess_rate,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Where a label comes from.** Every label in a dataset was made somehow, in one of three ways.

        - A person judges: someone reads each message and marks it spam or not, or a doctor marks what a scan shows. That is an **annotation label**. It says what that person thinks is true, and it costs time and money; the next lesson is about doing it well.
        - A measurement: a sensor records that a machine failed, or a lab test gives a result. The label is as good as the instrument.
        - Behavior: what people do gives the label for free: an order comes back or not, a reader clicks or not, a loan is paid back or not. That is a **natural label**. It is plentiful, but it can take a while to arrive, and it says what people did, which is not always what you want to know: a click says that a headline drew someone in, not that the story is good.

        **The wait.** A click arrives within seconds, a return within 30 days, and whether a loan is paid back after years. Until the wait is over, the label is unknown, not 0. In Python, `None`, the word for nothing, says that: no label yet.

        **The feedback loop.** The model predicts, the shop acts on it, customers react, and their reactions come back as labels for the next model. That way back is a **feedback loop**, and its length is how long a label takes to arrive: here 30 days. The loop also feeds on itself: the shop only sees returns of what it sold, and a news app only sees clicks on what it showed, so a model learns only about what earlier choices let through.

        **The same wait for every example.** On a given day, give an order a label only when its 30 days are over. Both shortcuts get the rate wrong. Counting the young orders that have not come back as kept gives too little, because some of them will still come back. Counting the young orders that already came back, but not the young ones that have not come back yet, gives too much, because among the newest orders only the returns are known yet.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **annotation label** | a label a person gives after judging the example |
        | **natural label** | a label that comes for free from what people do, such as a return or a click |
        | **feedback loop** | the way from a model's predictions, through what people do, back into its training data; its length is how long a label takes to arrive |
        """),
        L.watch_out("An order that has not come back yet is not a kept order. Until its 30 days are over, its label is unknown: `None`, not 0."),
        mo.md(r"""
        The course simulated a web shop: 300 orders on days 0 to 179, of which 61, about one in five, come back within 30 days. With the slider you choose which day it is today. Left, every order placed so far, at its day, in the row of its label: returned (filled), kept (open) or not known yet (gray); the shaded band is the last 30 days. Right, the return rate three ways: *unknown = 0* counts the orders not known yet as kept, *wait 30 days* waits 30 days for every order, and *early too* counts the early returns of young orders but not the young orders that have not come back yet. The dashed line is the share of all 300 orders that will come back, 20%, which only the simulation knows. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_today_cell():
    def _frame(today):
        said = C.today_said(today, L.lang)
        alt = ("Left: the orders of a web shop by the day they were placed, in three rows: returned, kept and not known yet, "
               "with the last 30 days shaded. Right: three bars with the return rate when unknown orders count as kept, "
               "after 30 days, and with the early returns of young orders too, beside a dashed line at 20%. " + said)
        return (C.today_picture(today, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_today = L.explore("today", "today", _frame, start=60, values=C.TODAYS, role="",
                              task="Go from day 30 to day 180. Which of the three rates stays near the dashed line, and when are the other two furthest from it?")
    explore_today
    return


@app.cell(hide_code=True)
def text_after_today():
    mo.md(r"""
    On day 30 not a single order is 30 days old: there is no label yet, and the two other rates are furthest from the line, 13% and 100%. From day 60 on, waiting 30 days gives 20% or 21% every time, close to the 20% that really come back. Counting unknown orders as kept gives too little, 16% on day 60, and counting the early returns too gives too much, 29% on day 60. The longer the shop has been selling, the smaller the gap, because the last 30 days become a smaller part of all orders: on day 180 they are 20% (unknown = 0), 21% (wait 30 days) and 22% (early too). But a model that should learn from the newest orders sees the biggest gap. With 300 orders the rates also jump by a point or two by chance.

    ### Recipe

    To label examples whose label comes from behavior:

    1. Write down where every label comes from, a person, a measurement or behavior, and what it really says.
    2. For a label from behavior, find the wait: how long until the label is final.
    3. Give an example a label only when its wait is over; until then its label is `None`.
    4. Count and train only on examples with a final label.
    5. Write the source of the labels and the wait in the datasheet, next to the data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Eight orders on day 60: the day each was placed, and whether it has come back so far. The cell prints, for every order, its day, its age in days, whether it is at least 30 days old, and whether it came back. Run the cell.
    """)
    return


@app.cell
def example_ages():
    today = 60
    ordered = [5, 12, 20, 30, 38, 44, 52, 57]
    returned = [False, True, False, False, True, False, True, False]

    for i in range(len(ordered)):
        age = today - ordered[i]                           # how many days ago the order was placed
        print(ordered[i], age, age >= 30, returned[i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first four orders are 55, 48, 40 and 30 days old: their returns are all in, so their labels are 0, 1, 0 and 0. The other four are 22, 16, 8 and 3 days old. Two of them came back already, but the two others can still come back, so all four wait.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A label only after 30 days

        `label(ordered, returned, today)` gets the day an order was placed, whether it has come back so far, and today. It should give `None` while the order is younger than 30 days, and after that 1 if it came back and 0 if not. Replace the `...`. The check calls it for the eight orders of the worked example on day 60, such as `label(30, False, 60)`, and for five more orders on other days.
        """),
        L.task("Let `label` give `None`, 1 or 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label", lines=8)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label):
    result_label = L.check("label", box_label.value)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label",
        [
            "Two questions, in this order: is the order old enough for its label to be known? Only then: did it come back?",
            "`age` is how many days ago the order was placed. While it is below 30, the function can give `None` at once with `return`; after that, `returned` decides between 1 and 0.",
            "`if age < ___:` with `return None` one step in, then `if returned:` with `return ___` one step in, and at the end `return ___`.",
        ],
        why="An order younger than 30 days can still come back, so its label is not known yet, also when it already came back: otherwise the newest orders would only get labels when they are returns. After 30 days every return is in, and the label is final.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The return rate of the known labels

        `return_rate(labels)` gets a list of labels, each 1, 0 or `None`, and should give the share of returned orders among the orders whose label is known. Replace the `...`. The check calls `return_rate([0, 1, 0, 0, None, None, None, None])`, the labels of the eight orders on day 60, which should give 0.25, and four more lists.
        """),
        L.task("Let `return_rate` count only the labels that are known."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rate():
    box_rate = L.code_box("return-rate", lines=10)
    box_rate
    return (box_rate,)


@app.cell(hide_code=True)
def check_rate(box_rate):
    result_rate = L.check("return-rate", box_rate.value)
    result_rate
    return (result_rate,)


@app.cell(hide_code=True)
def hints_rate():
    L.hints(
        "return-rate",
        [
            "A share is a count divided by a count. Which labels count above the line, and which below it?",
            "Above the line: the labels that are 1. Below it: every label that is known, 1 or 0. A `None` counts in neither. Compare a label with `==`.",
            "`if value == 1:` adds 1 to `returned` and to `known`, and `if value == ___:` adds 1 to `known` only.",
        ],
        why="Of the four known labels, one is 1, so the return rate is 1 / 4 = 0.25. Counting the four `None` as kept would give 1 / 8 = 0.125: too little, because some of those orders will still come back.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clicks():
    mo.md(r"""
    ### Exercise 3. What a click says
    """)
    return


@app.cell(hide_code=True)
def choice_clicks_cell():
    choice_clicks = L.choice(
        "what-clicks-say",
        question="A news app wants a model that picks good stories. It labels a story 1 when a reader clicks it. What do these labels teach the model?",
        options={
            "true": "Which stories are true and well written",
            "right": "Which headlines get clicked, whether or not the story is good",
            "liked": "Which stories readers liked after reading them",
        },
        correct="right",
        why={
            "true": "A click comes before the reading: it says that the headline drew someone in, not that the story is true. A natural label says what people did. Whether a story is true, a person has to judge: an annotation label.",
            "liked": "A click comes before the reading, so it cannot say whether the reader liked the story. Reading to the end or a rating would come closer, and those labels arrive later.",
            "right": "Right: a natural label says what people did. And the loop feeds on itself: the app shows stories that get clicks, so it only learns about clicks on what it showed.",
        },
    )
    choice_clicks
    return (choice_clicks,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The label of an order is one of two classes, as in [Classification: predicting a class](/course/open/ml-classification?lang=en), and the datasheet of [Where does this data come from?](/course/open/data-provenance?lang=en) gets one more line: where the labels come from and how long they take to arrive. The next lesson, *Labeling well: guidelines and agreement*, is about labels from people: how two people can label the same examples and agree.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_clicks, result_label, result_rate):
    L.wrap_up(result_label, result_rate, choice_clicks)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
