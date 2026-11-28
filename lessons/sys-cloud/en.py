# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-cloud. Route: course/route/lessen/systemen.yaml. Checks, the eight made-up offers, the prices of the guess and the demo, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-cloud", "en")
    C = L.checks   # the eight made-up GPU offers, the EU countries, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en) and [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gpu():
    quick_gpu = L.choice(
        "quick-gpu",
        question="Why does a large network train many times faster on a GPU than on a CPU?",
        options={
            "cores": "Each of its cores is many times faster than a core of a CPU",
            "right": "Its thousands of simple cores do the same calculation on many numbers at once",
            "flops": "A GPU needs fewer FLOPs for each weight of the network",
            "memory": "A GPU keeps the whole dataset in its memory, so it never has to read from the disk",
        },
        correct="right",
        why={
            "right": "The numbers of a matrix product can be worked out independently of each other, so thousands of simple cores do them at the same time. A large model needs such a GPU, often with more memory than yours, or several of them: this lesson is about renting one.",
            "cores": "A core of a GPU is simpler and slower than a core of a CPU. What does a GPU have many more of? See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "flops": "A weight costs the same FLOPs on every chip: 2 per example in a forward pass. What does a GPU do with them that a CPU cannot? See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "memory": "Every mini-batch is still moved to the GPU with `.to(device)`. What makes the calculations on it so quick? See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
        },
    )
    quick_gpu
    return (quick_gpu,)


@app.cell(hide_code=True)
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="In the course's spam service, 5 of the 118 lines of code were the model. What did most of the other lines do?",
        options={
            "tests": "They tested the model on a fixed test set, again and again",
            "learn": "They did the learning, which scikit-learn could not do by itself",
            "right": "They took messages in, gave answers, watched them and trained again",
            "comments": "They were comments that explain the five lines of the model to a new reader of the code",
        },
        correct="right",
        why={
            "right": "An ML system is the model with everything that makes it useful: taking data in, serving answers, monitoring and training again. All those parts run on computers that must keep running. This lesson is about renting those computers, or a platform that runs the parts for you.",
            "tests": "Tests on a fixed test set belong to research. What does a service do with every message that comes in? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "learn": "scikit-learn does the learning itself, in the five lines of the model. What is left for the rest of the code? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "comments": "The 118 lines are code that runs. What does a service need around a model, so that every message gets an answer? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every network in this course so far trained on your own computer. That works for small networks, but a larger model needs a GPU with more memory than yours, or eight of them for a week, and a service with many users needs computers that keep running, day and night. Buying them costs a lot of money before the first step of training, and most of the time they would stand still. The other way is to rent: someone else's computer, for as long as you need it, paid by the hour. Renting raises three questions: how much do you leave to the provider, what does the whole job cost, and where are your data while it runs?

    **An everyday example.** You can cook in your own kitchen. You can rent a kitchen for a day, with its stove and pans, and bring your own ingredients and recipe. You can give your recipe to a caterer, who cooks it in the caterer's kitchen and brings the meal. Or you go to a restaurant and choose from the menu. The further down that list, the less you look after yourself, and the less you decide.
    """)
    return


@app.cell(hide_code=True)
def predict_job():
    guess_job = L.guess(
        "job",
        question="A team has a training job that takes 30 hours on an NVIDIA A100 with 40 GB of memory, rented at 1.99 dollars an hour. An H100 costs 4.59 dollars an hour, and on this job it is 2.5 times as fast. Which GPU costs less for the whole job, do you think?",
        options={
            "a100": "The A100, since its hour costs less than half as much",
            "same": "Both cost the same, since the speed and the price even out",
            "right": "The H100, though its hour costs more",
            "memory": "The A100, since 40 GB is all the job needs anyway",
        },
        reveal="**The H100.** On the A100, the job costs 30 × 1.99 = 59.70 dollars. On the H100, it takes 30 / 2.5 = 12 hours, and costs 12 × 4.59 = 55.08 dollars. An hour on the H100 costs 2.3 times as much, but it does the work 2.5 times as fast. The prices are those of the GPU cloud Lambda on 28 November 2026; the 2.5 is an example, since how much faster an H100 is depends on the job.",
    )
    guess_job
    return (guess_job,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The cloud.** **Cloud computing**, or **the cloud** for short, is renting computers, storage and software over the internet from a provider, instead of owning them. The US standards institute NIST described it in 2011 with five properties: you take what you need yourself, without asking anyone; over the network; from a large pool of computers that many customers share, in the provider's data centers; more or less of it within minutes; and you pay for what you use, measured, such as GPU hours or gigabytes.

        **Three layers.** NIST also named three ways to rent, by how much the provider looks after. With **IaaS**, infrastructure as a service, you rent bare computers, storage and networks, and install everything on them yourself, from the operating system to your model. With **PaaS**, platform as a service, the provider runs the computers and the software under your code; you bring the code or the model. With **SaaS**, software as a service, you use finished software, such as a chat assistant, and look after nothing but your account.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Free GPUs to start with.** **Colab**, from Google, runs notebooks in the browser on a borrowed GPU, free in small amounts. Its own FAQ says the resources are "not guaranteed and not unlimited": the limits are not published and change over time, a notebook runs for at most 12 hours, and the virtual machine is deleted after a while, with every file on it. Kaggle notebooks work in a similar way, with a GPU quota per week that your account settings show; Kaggle says it can be more than 30 hours, depending on how busy it is. Both are good for learning and trying things; neither is for a run that must finish, or for a service.

        **Renting a GPU.** The large clouds, such as AWS, Google Cloud and Microsoft Azure, rent out everything, GPUs included. A **GPU cloud** rents mostly GPUs, often for less. On 28 November 2026, the GPU cloud Lambda asked 4.59 dollars per GPU hour for an H100 with 80 GB, and 1.99 dollars for an A100 with 40 GB; AWS asked 55.04 dollars an hour for a machine with eight H100s, 6.88 dollars per GPU hour. Prices change quickly: in June 2025, AWS lowered the price of that machine by 44%. So always write down the date with a price.

        **The cost of a job.** What counts is not the price of an hour, but what the whole job costs: the hours the job takes on a GPU you know, divided by how many times as fast the rented GPU is, times its price per hour. And memory comes first: a model that does not fit in the memory of the GPU does not run at all, however cheap the hour.

        **Where, and under whose law.** The last question is where your data are while the job runs, and which law applies to them. Those are two questions. One is the country of the data center. The other is the country of the company: under the US CLOUD Act of 2018, a US court can order a US company to hand over data it holds, even when the data are in a data center abroad. Data in the Frankfurt data center of a US company are in the EU, and still within reach of US law. **Sovereign compute** is compute that a country or region controls itself, in data centers on its own ground and under its own laws.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **cloud** | | computers, storage and software rented over the internet from a provider, paid for what you use; in full, cloud computing | |
        | **IaaS** | eye-as | infrastructure as a service: bare computers, storage and networks, on which you install the rest | |
        | **PaaS** | pass | platform as a service: the provider runs the computers and the software, and you bring code or a model | |
        | **SaaS** | sass | software as a service: finished software that you use | |
        | **Colab** | | Google's notebooks in the browser, free in small amounts, on a borrowed GPU, with limits that change | |
        | **GPU cloud** | | a provider that rents mostly GPUs, by the hour | |
        | **sovereign compute** | | compute that a country or region controls itself: on its own ground, under its own laws | |
        """),
        L.watch_out("*Free* at Colab and Kaggle means free in small amounts, as long as there is room. A session can stop before your run is done, and its files go with it, so save what you need somewhere of your own along the way."),
        L.watch_out("A data center in the EU does not make a company European. The country of the data center and the country of the company are two questions, and a rule about data may ask both."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Renting compute for a job:

    1. Write down what the job needs: memory on the GPU, and how many hours it takes on a GPU you know.
    2. Choose the layer: IaaS when you want to install and control everything, PaaS when you only want to bring code or a model, SaaS when finished software does the job.
    3. Keep only the offers that may hold your data: ask about the country of the data center, and about the country of the company.
    4. For every offer with enough memory: the hours, divided by how many times as fast the GPU is, times the price per hour.
    5. Choose the lowest cost for the job, and write down the date of the prices. For learning and trying things, start with Colab or Kaggle, and save your work along the way.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A job takes 50 hours on the A100 with 40 GB and needs 30 GB of memory. One offer: an H100 with 80 GB, 2.5 times as fast as that A100, at 4.20 dollars an hour, in a data center in Germany, from a company in the United States. The cell asks the questions of the recipe about this offer. Run the cell.
    """)
    return


@app.cell
def example_one_offer():
    eu = {"DE", "FR", "NL"}     # three of the 27 countries of the EU: enough here
    country = "DE"              # where the data center is
    home = "US"                 # where the company is from
    print("data center in the EU:", country in eu)
    print("company from the EU:", home in eu)
    memory = 80                 # GB on the rented GPU
    print("enough memory for 30 GB:", memory >= 30)
    hours = 50                  # hours on the A100 with 40 GB
    speed = 2.5                 # how many times as fast as that A100
    price = 4.2                 # dollars per hour
    print("hours on this GPU:", hours / speed)
    print("dollars for the job:", round(hours / speed * price, 2))
    return country, eu, home, hours, memory, price, speed


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The data center is in the EU, the company is not: a rule that wants both in the EU leaves this offer out, however good its price. The memory fits. On this GPU, the job takes 50 / 2.5 = 20 hours, and costs 20 × 4.20 = 84 dollars. On an A100 with 40 GB at 1.99 dollars, it would take the 50 hours and cost 99.50 dollars. Exercise 1 asks the two questions about the data for eight offers at once, and exercise 2 works out the cost of a job on each of them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_allowed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The hospital's rule

        A hospital wants to train a model on scans of its patients, and has a rule: the data stay in a data center in the EU, at a company from the EU. `allowed(offers, eu)` gets a list of offers and the set of EU countries, and gives back the names of the offers the rule allows. Every `offer` is a dictionary with, among others, `"name"`, `"country"`, the country of the data center, and `"home"`, the country of the company, each as a two-letter code, such as `"NL"`. `names` collects the names, the loop goes through the offers with `offer`, and `names.append(offer["name"])` adds one. Replace the `...` with the condition of the rule.

        The check calls `allowed(C.OFFERS, C.EU)`: eight made-up offers A to H, with prices like those of November 2026, and the 27 countries of the EU. Then no offers, one offer, three offers with at most one of the two countries in the EU, the eight offers with only France as the EU, and a made-up offer from a German company in Ireland.
        """),
        mo.md(C.offers_table(L.lang)),
        L.task("Keep the offers whose data center and company are both in `eu`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_allowed():
    box_allowed = L.code_box("allowed", lines=7)
    box_allowed
    return (box_allowed,)


@app.cell(hide_code=True)
def check_allowed(box_allowed):
    result_allowed = L.check("allowed", box_allowed.value)
    result_allowed
    return (result_allowed,)


@app.cell(hide_code=True)
def hints_allowed():
    L.hints(
        "allowed",
        [
            "Step 3 of the recipe. Each `offer` has two countries: `offer[\"country\"]` for the data center and `offer[\"home\"]` for the company. The rule asks a question about each of them, with `eu`.",
            "Python's `in` asks whether a value is in a set: `\"DE\" in {\"DE\", \"FR\"}` is `True`. The worked example asks the two questions on two lines; `and` joins two questions that must both be yes into one.",
            "`        if offer[\"country\"] in eu and ___:`",
        ],
        why="`if offer[\"country\"] in eu and offer[\"home\"] in eu:` asks both questions, and keeps an offer only when both answers are yes: D in the Netherlands, E in Finland and F in France. B has its data center in Germany, but belongs to a US company; G belongs to a French company, but keeps the data in Canada.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cheapest():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The cheapest job

        `cheapest(offers, hours, memory)` gets the offers, the hours a job takes on the A100 with 40 GB, and the memory the job needs, in GB. It gives back the name of the offer on which the whole job costs least. Besides `"name"`, every `offer` has `"memory"`, the GB of its GPU, `"speed"`, how many times as fast as the A100 with 40 GB it is on this kind of job, and `"price"`, in dollars per hour. The loop skips offers with too little memory; `best` and `best_cost` keep the cheapest offer so far, and `None` at the start means none yet. Replace the `...` with what the job costs on `offer`, in dollars, as `cost`.

        The check calls `cheapest(C.OFFERS, 30, 40)`: the job of the guess, on the eight offers. Then two offers at a time, three made-up GPUs that are each cheapest in another way, no offers at all, and only an offer with too little memory.
        """),
        L.task("Work out what the whole job costs on `offer`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cheapest():
    box_cheapest = L.code_box("cheapest", lines=11)
    box_cheapest
    return (box_cheapest,)


@app.cell(hide_code=True)
def check_cheapest(box_cheapest):
    result_cheapest = L.check("cheapest", box_cheapest.value)
    result_cheapest
    return (result_cheapest,)


@app.cell(hide_code=True)
def hints_cheapest():
    L.hints(
        "cheapest",
        [
            "Step 4 of the recipe: first how many hours the job takes on `offer`, from `hours` and its speed, then that times the price of an hour there, `offer[\"price\"]`. The result goes in `cost`.",
            "A GPU that is 2.5 times as fast does a job of 30 hours in 12, as in the guess. Does the speed multiply the hours, or divide them?",
            "`            cost = hours ___ * offer[\"price\"]`",
        ],
        why="`cost = hours / offer[\"speed\"] * offer[\"price\"]` gives the hours on that GPU, times the price of each: on G, an H100 at 4.10 dollars, 30 / 2.5 = 12 hours and 49.20 dollars. Of the offers with enough memory, C has the cheapest hour, 1.99 dollars, but needs all 30 hours: 59.70 dollars.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 3. Who looks after what?

    Eight ways to run software. Put each in the column that fits it: your own computers, or IaaS, PaaS or SaaS.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Step 2 of the recipe, and the picture of the idea. Ask for each: does anyone rent anything out, and if so, does the team install the software itself, bring only code or a model, or use what is finished?",
            "Two are bought, not rented. Of the rest, two are bare building blocks, two run the team's own code or model, and two are ready to use.",
            "Your own: PC and \\_\\_\\_. IaaS: virtual machine and \\_\\_\\_. PaaS: training service and \\_\\_\\_. SaaS: chat assistant and \\_\\_\\_.",
        ],
        why="**Your own:** your PC and the hospital's server room, since nothing is rented. **IaaS:** the virtual machine and the storage, since the team installs and fills them itself. **PaaS:** the training service and the model service, since the team brings code or a model and the provider runs it. **SaaS:** the chat assistant and the transcription, since the company only uses finished software.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_colab():
    mo.md(r"""
    ### Exercise 4. A night in Colab

    In the evening, you start a training run of 14 hours in the free version of Colab, and save the model only at the end, in the files of the notebook.
    """)
    return


@app.cell(hide_code=True)
def choice_colab_cell():
    choice_colab = L.choice(
        "colab",
        question="What can happen?",
        options={
            "done": "The run finishes, since Colab never stops a run that is going",
            "right": "The session can stop before the end, and its files go with it",
            "pause": "Colab pauses the run at night, and goes on with it in the morning when there is room again",
            "slow": "After 12 hours the run gets a slower GPU, but it does finish",
        },
        correct="right",
        why={
            "right": "Colab says that a notebook runs for at most 12 hours, depending on what is free and on how much you have already used, and that its virtual machine is deleted, with every file on it. A run of 14 hours does not fit in one session: save the model somewhere of your own along the way, such as every hour, or train in shorter pieces. The paid Colab Pro+ runs longer, up to 24 hours, and is not guaranteed either.",
            "done": "Colab's FAQ says the resources are not guaranteed, with a longest time per notebook. How long can a free session run?",
            "pause": "A Colab session does not wait for you: when it ends, its virtual machine is gone. What happens to the files in it?",
            "slow": "Colab gives no slower GPU to finish on. What happens when a session reaches its longest time?",
        },
    )
    choice_colab
    return (choice_colab,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### How fast must the H100 be?

    The guess compared an A100 with 40 GB at 1.99 dollars an hour with an H100 at 4.59 dollars, taking the H100 to be 2.5 times as fast. How much faster an H100 really is depends on the job: on a small network, the A100 may keep up. The demo lets you choose the speed of the H100, from as fast as the A100 to 3.5 times as fast, and shows what the job of 30 hours costs on each GPU, with Lambda's A100 with 80 GB beside them, at 3.19 dollars an hour.
    """)
    return


@app.cell(hide_code=True)
def demo_speed():
    L.explore("speed", "how many times as fast the H100 is", lambda _v: C.frame(_v, L.lang), start=1.0, values=C.SPEEDS,
              role="", task="Make the H100 faster: from what speed on does it cost less for the job than the A100 with 40 GB?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    As fast as the A100, the H100 costs 137.70 dollars for the job: 2.3 times as much. At 2.25 times as fast, it costs 61.20 dollars, still more than the 59.70 dollars of the A100. From 2.31 times as fast on, the H100 is cheaper for the job: 55.08 dollars at 2.5 times, and 39.34 dollars at 3.5 times. The 2.31 is the ratio of the two prices per hour: 4.59 / 1.99 = 2.31. The A100 with 80 GB costs 95.70 dollars at every speed: you pay for memory that this job does not need.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en), the same lines ran a model on any computer, with a GPU or without one: on a rented GPU they do the same. [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en) showed how much more than the model a system is; the cloud rents each part of it, from bare computers to finished software. In the next lesson, *A box that runs anywhere: containers*, you pack a program with everything it needs, so that it runs the same on your computer and on a rented one. *What compute costs in the cloud* comes back to the price, with cheaper hours on a GPU that the cloud can take back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_colab, result_allowed, result_cheapest, result_kinds):
    L.wrap_up(result_allowed, result_cheapest, result_kinds, choice_colab)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
