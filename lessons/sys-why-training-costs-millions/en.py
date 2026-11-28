# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-why-training-costs-millions. Route: course/route/lessen/systemen.yaml. Checks, the published GPU hours, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-why-training-costs-millions", "en")
    C = L.checks   # the GPU hours the makers published, the runs of the exercises, the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_time():
    quick_time = L.choice(
        "quick-time",
        question="A training step needs 6 trillion FLOPs, and your GPU reaches 3 trillion FLOP/s on it. How long does one step take?",
        options={
            "half": "0.5 seconds",
            "right": "About 2 seconds",
            "big": "18 seconds",
            "depends": "That depends on the size of the model, so FLOPs and FLOP/s alone cannot say",
        },
        correct="right",
        why={
            "right": "Time is the work divided by the rate: 6 trillion FLOPs at 3 trillion per second take 2 seconds. In this lesson, the same division runs over months, on thousands of GPUs.",
            "half": "0.5 is the rate divided by the work. Which of the two goes on top when you want a time? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
            "big": "18 is the work times the rate, and that is not a time. What do you do with the work and the rate to get seconds? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
            "depends": "The size of the model is already in the 6 trillion FLOPs. What do the work and the rate give together? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
        },
    )
    quick_time
    return (quick_time,)


@app.cell(hide_code=True)
def quick_check_many():
    quick_many = L.choice(
        "quick-many",
        question="Your network writes 1 text at 50 characters per second. With 100 texts at once, it writes 2,000 characters per second in all. What went up?",
        options={
            "each": "The speed of each single text, from 50 to 2,000 characters per second",
            "flops": "The FLOPs of one character",
            "right": "The throughput: the work of all texts together per second",
            "rate": "The most FLOP/s the GPU can reach",
        },
        correct="right",
        why={
            "right": "Throughput counts the characters of all texts together: 2,000 per second, while each text gets 20, slower than alone. Thousands of GPUs work in the same way: together they do more per second, and no single GPU gets faster.",
            "each": "2,000 per second is all 100 texts together, so each text gets 20. Which word counts the work of everything together per second? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
            "flops": "A character takes as many FLOPs as before; only more of them run at once. What went up? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
            "rate": "The most a GPU can do does not change with what you run on it. What went up? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
        },
    )
    quick_many
    return (quick_many,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your bigram network trained in seconds on your own computer. A large language model trains in the same way, with billions of parameters and trillions of tokens, and the work grows with both. Meta reports 30,840,000 GPU hours for the training of Llama 3.1 405B, and every one of those hours is paid for: as rent, or as GPUs bought, power and people. Headlines turn such numbers into one price, and the price depends on what is counted. In this lesson, you work out the bill yourself, from the numbers the makers published, and you see what it leaves out.

    **An everyday example.** A painter gives a quote for a house: 3 painters for 4 days of 8 hours make 96 hours of work, at 45 euros an hour, 4,320 euros. Send 6 painters, and they finish in 2 days, for the same 4,320 euros. The quote leaves out the paint, and the visit to look at the walls first.
    """)
    return


@app.cell(hide_code=True)
def predict_years():
    guess_years = L.guess(
        "years",
        question="Meta reports 30,840,000 H100 GPU hours for the training of Llama 3.1 405B. If one GPU had done all that work on its own, how long would it have taken, do you think?",
        options={
            "month": "About a month, since a GPU does trillions of FLOPs every second",
            "years": "About 35 years",
            "centuries": "About 350 years",
            "right": "About 3,500 years",
        },
        reveal="**About 3,500 years.** 30,840,000 hours are 1,285,000 days, which is 3,518 years. That is the reason a large model trains on thousands of GPUs at once: Meta trained the 405B model on up to 16,384 H100 GPUs, which brings the time down to months. The work stays the same, and so does the bill: every one of the 30,840,000 GPU hours is paid for.",
    )
    guess_years
    return (guess_years,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A GPU hour.** A **GPU hour** is one GPU used for one hour. A run on 8 GPUs for 10 hours uses 80 GPU hours, and so does a run on 80 GPUs for 1 hour. The makers of large models report their training in GPU hours:

        | Model | Maker, year | GPU | GPU hours |
        |---|---|---|---|
        | Llama 2 7B | Meta, 2023 | A100 | 184,320 |
        | Llama 2 70B | Meta, 2023 | A100 | 1,720,320 |
        | DeepSeek-V3 | DeepSeek, 2024 | H800 | 2,788,000 |
        | Llama 3.1 405B | Meta, 2024 | H100 | 30,840,000 |

        The H100 and the H800 are newer than the A100, and an hour of a newer GPU does more work.

        **The bill.** The **training cost** is the GPU hours times the price of one GPU hour. DeepSeek counted the training of V3 in three parts, and priced it at 2 dollars per H800 GPU hour, a price to rent one:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Read the picture from the top: three parts make 2,788,000 GPU hours, and at 2 dollars an hour that is 5,576,000 dollars. DeepSeek writes itself what this leaves out: the earlier research, and the trial runs to choose the architecture, the method and the data.

        **The time.** The GPU hours divided by the number of GPUs give the hours the run takes, and divided by 24 the days. DeepSeek's cluster has 2,048 H800 GPUs, and the pretraining took 2,664,000 GPU hours; DeepSeek writes that it took less than two months. More GPUs make a run shorter, not cheaper: the GPU hours stay the same.

        **What else costs money.** Runs stop. Meta trained the same 405B model on up to 16,384 H100 GPUs, and in a stretch of 54 days of pretraining the run stopped 466 times, 419 of them unplanned, about half of those from failing GPUs and their memory. The cluster still spent more than 90 percent of the time training. Before the final run come the tests and the trial runs, and people build the model. And a rental price is not what an owner pays: an owner buys the GPUs, and pays for the power and the building. A later lesson adds up what owning computers costs. All numbers here were read on 28 November 2026.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **GPU hour** | | one GPU used for one hour; the usual unit for the cost of training | `gpus * hours` |
        | **training cost** | | the GPU hours of a training times the price of one GPU hour; the research, the trial runs and the people around it come on top | `gpu_hours * price` |
        | `( )` | "parentheses" | Python works out what is between them first | `12 / (2 * 3)` |
        """),
        L.watch_out("A cost in a headline is often the final run alone. DeepSeek's 5,576,000 dollars says so itself, and a lab pays for many more runs than the one that worked."),
        L.watch_out("Compare GPU hours on the same GPU only. An hour of an H100 does more work than an hour of an A100, so fewer GPU hours on a newer GPU need not mean less work."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The bill of a training:

    1. Find the GPU hours of each run, or its GPUs and its time: GPU hours are GPUs times hours.
    2. Add up every run: the tests, the trial runs, the runs that stopped, and the final run.
    3. Multiply by the price of one GPU hour: the rent, or what an hour of a GPU costs you when you own it.
    4. For the time, divide the GPU hours by the number of GPUs, and by 24 for days.
    5. Say what is left out: the research, the people, the data, and the GPUs themselves when you bought them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.DEEPSEEK` is a dict with the GPU hours of the three parts of the training of DeepSeek-V3, and `C.PRICE` is the 2 dollars per GPU hour DeepSeek assumed. The cell adds the parts up, then multiplies by the price. Run the cell.
    """)
    return


@app.cell
def example_bill():
    _hours = 0
    for _part in C.DEEPSEEK:
        print(_part, C.DEEPSEEK[_part])
        _hours = _hours + C.DEEPSEEK[_part]
    print("GPU hours", _hours)
    print("dollars", _hours * C.PRICE)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    2,788,000 GPU hours at 2 dollars make 5,576,000 dollars, the number DeepSeek reported, and the pretraining is 96 percent of it. The result is a float, `5576000.0`, since `C.PRICE` is the float `2.0`. Exercise 1 does the same for all the runs of a project.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The bill of a project

        `cost(runs, price)` gets a list of runs, each a dict with its number of GPUs under `"gpus"` and its number of days under `"days"`, and the price of one GPU hour. The loop walks every `run`, and `total = total + hours * price` adds the cost of the run to `total`. Replace the `...` with the GPU hours of the run: its GPUs times its hours, where a day has 24 hours.

        `C.RUNS` holds the five runs of a made-up project: a small test on 8 GPUs for 2 days, two trial runs on 64 GPUs for 5 days, a run on 512 GPUs that stopped after 9 days, and the final run on 512 GPUs for 30 days. The check calls `cost(C.RUNS, 2.0)`. Then one GPU for one day, the same runs at 1 dollar, no runs at all, and runs of its own.
        """),
        L.task("Count the GPU hours of each run, so that cost gives the bill of all the runs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=7)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "`run[\"gpus\"]` is the number of GPUs of the run, and `run[\"days\"]` its number of days. `hours` should hold the GPU hours: every GPU, for every hour of the run. How many hours does a day have, and what do you multiply?",
            "8 GPUs for 2 days: `8 * 2 * 24` gives `384` GPU hours.",
            "`        hours = run[\"gpus\"] * ___`",
        ],
        why="`hours = run[\"gpus\"] * run[\"days\"] * 24` counts every GPU for every hour of the run. The five runs of `C.RUNS` make 494,976 GPU hours: 989,952 dollars at 2 dollars.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_days():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How many days?

        `days(gpu_hours, gpus)` gets the GPU hours of a run and the number of GPUs it runs on, and gives back the days the run takes. Write after `return` one line that divides the GPU hours by the GPUs and by 24. Python divides and multiplies from left to right: `12 / 2 * 3` gives `18.0`, and `12 / (2 * 3)` gives `2.0`.

        The check calls `days(2664000, 2048)`: the pretraining of DeepSeek-V3 on its cluster. Then one trillion tokens of the same training, Llama 3.1 405B on 16,384 GPUs, and small numbers of its own.
        """),
        L.task("Give back the days a run of gpu_hours takes on gpus GPUs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_days():
    box_days = L.code_box("days", lines=3)
    box_days
    return (box_days,)


@app.cell(hide_code=True)
def check_days(box_days):
    result_days = L.check("days", box_days.value)
    result_days
    return (result_days,)


@app.cell(hide_code=True)
def hints_days():
    L.hints(
        "days",
        [
            "`gpu_hours` divided by the GPUs gives the hours of the run, and divided by 24 more, the days. Parentheses make Python multiply before it divides. What goes between them?",
            "`2664000 / 2048` gives `1300.78125` hours of the whole cluster, and `1300.78125 / 24` gives about `54.2` days.",
            "`    return gpu_hours / (___)`",
        ],
        why="`return gpu_hours / (gpus * 24)` divides by the GPUs and by the 24 hours of a day at once. 2,664,000 GPU hours on 2,048 GPUs take 54.2 days, and on 16,384 GPUs, eight times as many, 6.8 days.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bill():
    mo.md(r"""
    ### Exercise 3. What is in the 5,576,000 dollars?

    Seven parts of making DeepSeek-V3. Put each one under *In the 5,576,000 dollars* when DeepSeek counted it, or under *Not in it*.
    """)
    return


@app.cell(hide_code=True)
def sort_bill_cell():
    sort_bill = L.sort("bill")
    sort_bill
    return (sort_bill,)


@app.cell(hide_code=True)
def check_bill(sort_bill):
    result_bill = L.check("bill", sort_bill)
    result_bill
    return (result_bill,)


@app.cell(hide_code=True)
def hints_bill():
    L.hints(
        "bill",
        [
            "Ask of each part: is it one of the parts of the training of V3 itself, as DeepSeek counted them?",
            "Three parts are in, and four are not.",
            "In: pretraining, \\_\\_\\_ and \\_\\_\\_. Not in it: trial runs, \\_\\_\\_, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**In:** pretraining, longer texts and post-training. **Not in it:** trial runs, earlier research, the people, and the GPUs themselves.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_same():
    mo.md(r"""
    ### Exercise 4. Two labs

    Lab A rents 1,024 GPUs for 40 days. Lab B rents 2,048 of the same GPUs for 20 days, at the same price per GPU hour.
    """)
    return


@app.cell(hide_code=True)
def choice_same_cell():
    choice_same = L.choice(
        "same",
        question="Which lab pays more?",
        options={
            "long": "Lab A, since its run takes twice as long",
            "right": "Neither: both pay for 983,040 GPU hours",
            "many": "Lab B, since it rents twice as many GPUs",
            "order": "Lab B, since a cluster twice as large costs more per GPU, as any larger order would",
        },
        correct="right",
        why={
            "right": "The bill is GPUs times hours times the price: 1,024 GPUs for 960 hours and 2,048 GPUs for 480 hours both make 983,040 GPU hours. Lab B has its model after 20 days instead of 40: more GPUs buy time, not a smaller bill. On a real cluster, B loses a little more to GPUs that wait for each other.",
            "long": "A runs twice as long, on half as many GPUs. What do 1,024 times 960 hours and 2,048 times 480 hours give?",
            "many": "B rents twice as many GPUs, for half as long. What do the GPU hours of the two labs come to?",
            "order": "The price per GPU hour is the same for both, as the question says. What else could make B pay more?",
        },
    )
    choice_same
    return (choice_same,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Seven trainings

    The GPU hours that Meta and DeepSeek published for seven models, from the fewest to the most. For the chosen model, you see the cost at 2 dollars per GPU hour, how long one GPU would take, and how long DeepSeek's cluster of 2,048 GPUs would take. Choose a model.
    """)
    return


@app.cell(hide_code=True)
def demo_model():
    L.explore("model", "model", lambda _v: C.frame(_v, L.lang), start=5, values=C.DEMO_VALUES, role="",
              task="Go from the smallest model to the largest: how do the cost and the time grow?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Llama 3.1 405B took 30,840,000 GPU hours, 11 times the 2,788,000 of DeepSeek-V3: at 2 dollars, 61,680,000 dollars against 5,576,000. On DeepSeek's 2,048 GPUs, the 405B model would have taken 627 days; Meta used up to 16,384, eight times as many. Llama 2 7B, the smallest, took 184,320 GPU hours: 21 years on one GPU, and 4 days on 2,048. The bars compare hours of different GPUs: the three Llama 2 bars are hours of the older A100. And each number is what its maker counted: Meta counts the pretraining of Llama 2 and the training of each Llama 3.1 model, and DeepSeek counts three parts and leaves the research out. Nobody counts the people.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en), time was the work divided by the rate of one computer; here the work is so large that thousands of GPUs share it, and every GPU hour has a price. The next lesson is *Moore's law and the growth of compute*: why each new generation of GPUs does more work per hour, and how fast the compute of the largest trainings grew.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same, result_bill, result_cost, result_days):
    L.wrap_up(result_cost, result_days, result_bill, choice_same)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
