# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-cloud-costs. Route: course/route/lessen/systemen.yaml. Checks, the prices, the stops of the spot run and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-cloud-costs", "en")
    C = L.checks   # the prices, the stops of the spot run, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en) and [Saving and loading a model](/course/open/torch-checkpoints?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_job():
    quick_job = L.choice(
        "quick-job",
        question="A job takes 30 hours on a GPU you know. A rented GPU is 3 times as fast and costs 6 dollars an hour. What does the job cost on it?",
        options={
            "per-hour": "180 dollars: 30 hours at 6 dollars an hour",
            "right": "60 dollars: 10 hours at 6 dollars an hour",
            "times": "540 dollars: 30 hours, times 3 for the speed, times 6 dollars",
            "divide": "2 dollars: 6 dollars divided by 3",
        },
        correct="right",
        why={
            "right": "On a GPU 3 times as fast, the job takes 30 / 3 = 10 hours, and 10 hours at 6 dollars cost 60 dollars. This lesson adds that the same GPU has more than one price per hour, depending on how you rent it.",
            "per-hour": "The price is per hour, but on a faster GPU the job takes fewer hours. How many? See [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).",
            "times": "A faster GPU makes the job shorter, not longer. What do you divide by the speed? See [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).",
            "divide": "The job still takes hours, each at 6 dollars. How many hours does it take on the faster GPU? See [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).",
        },
    )
    quick_job
    return (quick_job,)


@app.cell(hide_code=True)
def quick_check_checkpoint():
    quick_checkpoint = L.choice(
        "quick-checkpoint",
        question="A training run stops after epoch 6 of 10. What must the checkpoint hold to go on from epoch 7 as if nothing happened?",
        options={
            "right": "The state_dict of the model and of the optimizer, and the epoch",
            "model": "Only the state_dict of the model, since it holds every parameter",
            "class": "The code of the class of the model, and the loss of every epoch so far for the curve",
            "data": "The whole dataset, in the order of the epochs",
        },
        correct="right",
        why={
            "right": "The parameters alone are not enough: the optimizer keeps numbers of its own for every parameter, and the loop needs to know where it was. With all three, a run that stops goes on from the last checkpoint, and that is what makes cheap machines that can stop at any moment usable.",
            "model": "The parameters are the larger part, but the optimizer keeps numbers of its own, and the loop needs to know where it was. What else goes in? See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "class": "The code stays in your own files, and the curve is not needed to train on. What does training need to go on? See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "data": "The data stay where they are. What changes during training, and must be kept? See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
        },
    )
    quick_checkpoint
    return (quick_checkpoint,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Training a model can keep a GPU busy for days, and those GPU hours are most of the bill. In [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en), you worked out what a whole job costs at one price per hour. But the same GPU in the same data center has more than one price: it depends on how you rent it. Pay per hour whenever you like, and you pay the most. Promise to rent it for a year, and the hour gets much cheaper, even the hours you do not use. Take a GPU that nobody uses right now, and the hour costs a fraction, but the cloud can take it back at any moment. For a model behind an API, there is a fourth way: no server of your own, and a price per request. Choosing well can make the same work three times cheaper; choosing badly can make it never finish.

    **An everyday example.** A hotel room by the night at the desk costs the most. A flat rented for a year costs less per night, but you pay for the nights you are away too. A room that is empty tonight can go cheap, on the condition that you leave when a guest who pays full price arrives. And a taxi costs per ride: cheaper than a car of your own when you ride a few times a month, dearer when you ride every day.
    """)
    return


@app.cell(hide_code=True)
def predict_spot():
    guess_spot = L.guess(
        "spot",
        question="A training run takes 100 hours on a GPU that costs 4 dollars an hour on demand. As a spot instance, the same GPU costs 1.20 dollars an hour, but the cloud takes it back about once every 10 hours. You save a checkpoint every hour, which takes 3 minutes, and starting again on a new machine takes 12 minutes. What does the run cost as spot, do you think?",
        options={
            "same": "About 400 dollars, the same as on demand, since the lost work eats up the discount",
            "price": "About 120 dollars: 100 hours at 1.20 dollars",
            "right": "About 135 dollars: some more than 100 hours at 1.20 dollars",
            "more": "More than 400 dollars, since every stop starts the run over",
        },
        reveal="**About 135 dollars.** The run pays for about 113 hours: 100 hours of training, 5 hours to save 100 checkpoints of 3 minutes, and about 11 stops that each throw away half an hour of work on average and take 12 minutes to start again. At 1.20 dollars an hour, that is 135.48 dollars, a third of the 400 dollars on demand. Without checkpoints, every stop would throw away everything since the start, and the run would hardly get anywhere. The demo below shows how often to save.",
    )
    guess_spot
    return (guess_spot,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **On demand, reserved and spot.** A machine you rent in the cloud is called an **instance**. An **on-demand instance** you start and stop whenever you like, and you pay for each hour or second it runs: the highest price per hour, without promises. A **reserved instance** you promise to pay for during one or three years, used or not; in return, the hour is much cheaper, at AWS up to 72% off for three years. A **spot instance** runs on capacity that is free at the moment, at a fraction of the price: at AWS up to 90% off, at Google Cloud 60 to 91% off. When the cloud needs the capacity back, it takes the machine back, with a warning of two minutes at AWS and of 30 seconds at Google Cloud. These discounts are from the providers' pages on 28 November 2026. In this lesson, an example GPU costs 4 dollars an hour on demand, 2.40 dollars reserved (40% off) and 1.20 dollars as spot (70% off).
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Which way when.** Reserved is paid for every hour of the month, all 730 of them, used or not: 1,752 dollars at 2.40 dollars an hour. On demand costs 4 dollars for each hour used. Up to 438 hours a month, on demand is cheaper; from there on, reserved: with 40% off, from 60% of the hours. Spot is the cheapest per hour, but only for work that can stop and go on later, such as training that saves checkpoints, or a large job in parts that can each start again. A service that must answer at every moment never runs on spot alone.

        **Checkpoints on spot.** When the cloud takes a spot machine back, everything since the last checkpoint is lost: the training goes on from that checkpoint, on a new machine. A checkpoint holds, as in [Saving and loading a model](/course/open/torch-checkpoints?lang=en), the state_dict of the model and of the optimizer, and the epoch. Saving more often throws away less at each stop, but saving takes time too, and that time is paid for as well.

        **Serverless inference.** With **serverless inference**, you have no server of your own: the provider starts your model when requests arrive, and you pay per request or per second of compute. Without requests, you pay nothing; the first request after a quiet period waits while the model loads, from seconds for a small model to minutes for a large one. A fixed server costs the same every hour, busy or idle. With few requests, serverless is cheaper; with many, a fixed server is, and somewhere in between both cost the same per day.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **instance** | | a machine you rent in the cloud | |
        | **on-demand instance** | | an instance you start and stop when you like, paid for each hour or second it runs, at the highest price | |
        | **reserved instance** | | an instance you promise to pay for during one or three years, used or not, at a lower price per hour | |
        | **spot instance** | | an instance on capacity that is free at the moment, at a fraction of the price, which the cloud can take back at any moment | |
        | **serverless inference** | | running a model without a server of your own, paid per request or per second | |
        """),
        L.watch_out("A reserved instance is paid for every hour, even an hour in which it stands idle. Reserve only what runs nearly all the time, and only for as long as you are sure: a project that stops after three months still pays for the rest of the year."),
        L.watch_out("A spot machine that is taken back takes its own disk with it, and a checkpoint on that disk goes too. Save checkpoints to storage that stays when the machine goes."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Choosing how to pay for compute:

    1. Work out how many hours a month the GPU really runs.
    2. Reserve when that is more than the share at which reserved and on demand cost the same (60% of the hours at 40% off), for at least as long as the promise; otherwise, pay on demand.
    3. Put work that can stop and go on on spot instances: training that saves checkpoints often enough, to storage that stays, and large jobs in parts that can each start again.
    4. For a model behind an API, compare a day: the requests times the price per request for serverless, against 24 hours of a server. Below the number of requests at which both cost the same, serverless is cheaper; above it, a fixed server.
    5. Write the date by every price, and look again later: prices and discounts change often.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A team uses the example GPU. From how many hours a month is reserved cheaper than on demand? In the cell, `reserved` is what a month of reserved costs: all 730 hours at 2.40 dollars. The loop counts `hours` up as long as that many hours on demand cost no more than a month of reserved. Run the cell.
    """)
    return


@app.cell
def example_reserved():
    on_demand = 4.00              # dollars per GPU hour, for each hour you use
    reserved = 2.40 * 730         # 40% off, but for all 730 hours of a month
    hours = 0
    while hours * on_demand <= reserved:
        hours = hours + 1
    print("reserved per month:", reserved, "dollars")
    print("reserved is cheaper from", hours, "hours a month")
    return hours, on_demand, reserved


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At 438 hours, on demand costs 438 × 4 = 1,752 dollars, exactly as much as reserved, so the loop goes on; at 439 hours, on demand costs 1,756 dollars, and reserved is cheaper. 439 of 730 hours is 60%: with 40% off, reserved pays off when the GPU runs more than 60% of the time. Exercise 2 finds such a point for serverless and a fixed server; exercise 1 first counts what a spot run pays for.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hours_paid():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What a spot run pays for

        `hours_paid(hours, every, stops)` gets the hours of training a run needs, the hours of training between two checkpoints, and a list of stops: for each time the cloud took the machine back, how many hours of training were done at that moment. It gives back how many hours the run pays for, leaving out the minutes to save and to start again. `paid` begins at `hours`. For each `stop`, the `while` loop walks `saved` up in steps of `every`, as long as the next checkpoint came no later than the stop: `saved` ends at the hours of the last checkpoint before the stop. Replace the `...` with the hours of work the stop throws away.

        `C.STOPS` holds the ten stops of a run of 100 hours: 9.5, 17.25, 26, 35.5, 43.75, 52, 61.5, 69.25, 78 and 87.5. The check calls `hours_paid(100, 2, C.STOPS)`: a checkpoint every 2 hours. Then a checkpoint every 4 hours and every hour, a run without stops, stops exactly at a checkpoint, a checkpoint every half hour, and a stop before the first checkpoint.
        """),
        L.task("Add up the hours that the stops throw away."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours_paid():
    box_hours_paid = L.code_box("hours_paid", lines=9)
    box_hours_paid
    return (box_hours_paid,)


@app.cell(hide_code=True)
def check_hours_paid(box_hours_paid):
    result_hours_paid = L.check("hours_paid", box_hours_paid.value)
    result_hours_paid
    return (result_hours_paid,)


@app.cell(hide_code=True)
def hints_hours_paid():
    L.hints(
        "hours_paid",
        [
            "The idea, under *Checkpoints on spot*: a stop throws away everything since the last checkpoint. At each `stop`, `paid` has to grow by those hours. Where did the last checkpoint fall, and where the stop?",
            "In `C.STOPS`, the first stop came at 9.5 hours. With a checkpoint every 2 hours, the last checkpoint before it was at 8 hours: 1.5 hours of work are lost. Which two names hold 9.5 and 8 in the loop?",
            "`        paid = paid + stop ___`",
        ],
        why="`paid = paid + stop - saved` adds the hours between the last checkpoint and the stop. With a checkpoint every 2 hours, the ten stops of `C.STOPS` throw away 1.5 + 1.25 + 0 + 1.5 + 1.75 + 0 + 1.5 + 1.25 + 0 + 1.5 = 10.25 hours: 110.25 paid hours. A stop at 26 hours loses nothing, since the checkpoint of 26 hours had just been saved.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_break_even():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. When serverless costs more

        A small model answers requests behind an API. On serverless, a request costs 0.00035 dollars, about 1.5 seconds of a GPU; a fixed server with a GPU costs 0.75 dollars an hour, 18 dollars a day. `break_even(per_request, per_day)` gives back the smallest number of requests a day at which serverless costs more than the server. `n` counts up from 0, and the `while` loop goes on as long as `n` requests do not cost more than the server. Replace the `...` with that question.

        The check calls `break_even(0.00035, 18.0)`. Then a request of 0.5 dollars and a server of 12 dollars, where 24 requests cost exactly as much as the server; a request of 0.0003 dollars and a server of 20 dollars; a request dearer than the whole server; and a server that costs nothing.
        """),
        L.task("Find the first number of requests at which serverless costs more."),
    ])
    return


@app.cell(hide_code=True)
def exercise_break_even():
    box_break_even = L.code_box("break_even", lines=6)
    box_break_even
    return (box_break_even,)


@app.cell(hide_code=True)
def check_break_even(box_break_even):
    result_break_even = L.check("break_even", box_break_even.value)
    result_break_even
    return (result_break_even,)


@app.cell(hide_code=True)
def hints_break_even():
    L.hints(
        "break_even",
        [
            "In the worked example, the loop goes on as long as `hours` on demand cost no more than a month of reserved. Here, `n` requests cost `n * per_request` dollars on serverless. With what does the loop compare that, and how?",
            "No more than is `<=` in Python, as in the worked example. With 24 requests of 0.5 dollars, serverless costs 12 dollars, exactly as much as a server of 12 dollars: does the loop go on there?",
            "`    while n * per_request ___:`",
        ],
        why="`while n * per_request <= per_day:` goes on as long as serverless costs no more than the server, and stops at the first `n` at which it costs more. With 0.00035 dollars a request and 18 dollars a day: 18 / 0.00035 = 51,428.57, so serverless costs more from 51,429 requests a day.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.md(r"""
    ### Exercise 3. Which way fits?

    Eight jobs, each with the way of paying that fits it best: on demand, reserved, spot or serverless. Put each job in its column.
    """)
    return


@app.cell(hide_code=True)
def sort_fits_cell():
    sort_fits = L.sort("fits")
    sort_fits
    return (sort_fits,)


@app.cell(hide_code=True)
def check_fits(sort_fits):
    result_fits = L.check("fits", sort_fits)
    result_fits
    return (result_fits,)


@app.cell(hide_code=True)
def hints_fits():
    L.hints(
        "fits",
        [
            "The recipe. Ask for each job: how long does it run, may it stop halfway, and does it run every hour or now and then?",
            "Two jobs go in each column.",
            "On demand: the test before a deadline and \\_\\_\\_. Reserved: the service for three years and \\_\\_\\_. Spot: the run of three days and \\_\\_\\_. Serverless: the demo and \\_\\_\\_.",
        ],
        why="**On demand:** the test before a deadline, which must not stop, and the week to try an idea, for which nobody can promise a year. **Reserved:** the service for three years and the team that trains every day, since those GPUs run nearly every hour. **Spot:** the run with a checkpoint every hour and the photos in parts, which simply go on after a stop. **Serverless:** the demo and the form, with so few requests that a server would mostly stand idle.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rhythm():
    mo.md(r"""
    ### Exercise 4. A checkpoint too rarely

    A run on spot instances saves a checkpoint every 12 hours. On average, the cloud takes the machine back once every 6 hours.
    """)
    return


@app.cell(hide_code=True)
def choice_rhythm_cell():
    choice_rhythm = L.choice(
        "rhythm",
        question="What happens?",
        options={
            "discount": "It finishes at 70% off, since only the price per hour counts",
            "right": "It hardly gets anywhere: most stops come before the next checkpoint",
            "double": "It takes about twice as long, since half of the work is lost",
            "wait": "The cloud waits until the next checkpoint is saved before it takes the machine back",
        },
        correct="right",
        why={
            "right": "With a stop every 6 hours on average, 12 hours without a stop are rare: most stops come before the first checkpoint, and each time the run starts over from the last one, often the very start. Save much more often than the stops come: in the demo, with a stop every 10 hours, about every hour. A warning of two minutes or 30 seconds is enough to save a small checkpoint, rarely a large one.",
            "discount": "The price per hour is lower, but the work since the last checkpoint is lost at every stop. How many stops come before the first checkpoint?",
            "double": "That would hold if each stop lost a little. With 12 hours between checkpoints and a stop every 6 hours, how far does the run usually get before a stop?",
            "wait": "The cloud gives a warning of two minutes at most, and then takes the machine, saved or not. What happens to the hours since the last checkpoint?",
        },
    )
    choice_rhythm
    return (choice_rhythm,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### How often to save?

    A spot run of 100 hours at 1.20 dollars an hour, in which the cloud takes the machine back on average once every 10 paid hours. Saving a checkpoint takes 3 minutes, starting again 12 minutes, and each stop throws away half the time between two checkpoints on average. Choose how many hours of training go between two checkpoints, and see what the run pays for.
    """)
    return


@app.cell(hide_code=True)
def demo_every():
    L.explore("every", "hours between checkpoints", lambda _v: C.frame(_v, L.lang), start=4,
              values=C.DEMO_VALUES, role="", task="Go through the six values: at which one does the run pay for the fewest hours, and why do both ends cost more?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With a checkpoint every quarter of an hour, saving takes 20 hours, far more than the 1.6 hours the stops throw away; every 8 hours, the stops throw away 69.4 hours. In between lies the lowest point: a checkpoint every hour, 112.9 paid hours and 135.48 dollars, about a third of the 400 dollars on demand. The best interval is shorter when saving is quick or the stops come often, and longer when saving takes long or the stops come rarely.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en), a job had one price per hour; now the same hour has three prices, and serverless has a price per request. A spot instance only pays off with the checkpoints of [Saving and loading a model](/course/open/torch-checkpoints?lang=en): the state_dict of the model and of the optimizer, and the epoch, saved often, to storage that stays. In the next lesson, *MLOps: how far along is a team?*, the steps of this module come together in one loop that runs by itself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rhythm, result_break_even, result_fits, result_hours_paid):
    L.wrap_up(result_hours_paid, result_break_even, result_fits, choice_rhythm)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
