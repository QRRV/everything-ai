# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-moores-law. Route: course/route/lessen/systemen.yaml. Checks, the chips and trainings, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-moores-law", "en")
    C = L.checks   # the transistors of fifteen chips, the compute of four trainings, the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Log scales and power laws](/course/open/math-log-scales?lang=en) and [Why training costs millions](/course/open/sys-why-training-costs-millions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="A chart has a log scale up, with marks at 1, 10, 100 and 1,000, evenly spaced. A straight line on it rises one mark every 5 years. What does that say?",
        options={
            "add": "The value grows by 10 every 5 years",
            "right": "The value grows 10 times every 5 years",
            "double": "The value doubles every 5 years",
            "slow": "The growth slows down, since each mark is worth more than the one below",
        },
        correct="right",
        why={
            "right": "Each mark is ten times the one below, so a straight line on a log scale is the same factor in the same time. In this lesson, the transistors on a chip lie on such a line, for fifty years.",
            "add": "Growing by 10 every 5 years is a straight line on an ordinary scale. What is each mark worth, compared with the one below? See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "double": "Doubling would need each mark to be twice the one below. What is each mark worth here? See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "slow": "The line climbs one mark every 5 years, and each mark is the same factor up. Does that factor change? See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def quick_check_gpus():
    quick_gpus = L.choice(
        "quick-gpus",
        question="A training needs 1,000,000 GPU hours, at 2 dollars per GPU hour. You run it on twice as many of the same GPUs, at the same price. What happens to the bill?",
        options={
            "double": "It doubles to 4,000,000 dollars",
            "right": "It stays 2,000,000 dollars, and the run takes half as long",
            "half": "It halves to 1,000,000 dollars, since the run takes half as long",
            "hours": "It doubles, since twice as many GPUs use twice the GPU hours",
        },
        correct="right",
        why={
            "right": "The bill is the GPU hours times the price, and the work needs 1,000,000 GPU hours however many GPUs share it. In this lesson, you see why each new generation of GPUs does more work per hour, so the same work needs fewer GPU hours.",
            "double": "Twice as many GPUs for half as long make the same GPU hours. What does the bill count? See [Why training costs millions](/course/open/sys-why-training-costs-millions?lang=en).",
            "half": "Half as long, on twice as many GPUs: the GPU hours stay 1,000,000. What does the bill count? See [Why training costs millions](/course/open/sys-why-training-costs-millions?lang=en).",
            "hours": "Each GPU runs half as long, so twice as many GPUs use the same 1,000,000 GPU hours. What do GPUs times hours give? See [Why training costs millions](/course/open/sys-why-training-costs-millions?lang=en).",
        },
    )
    quick_gpus
    return (quick_gpus,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 1971, Intel's first processor, the 4004, held 2,300 transistors. NVIDIA's H100 of 2022 holds 80 billion: 35 million times as many, 51 years later. That growth is the reason an hour of a new GPU does more work than an hour of an old one, and the reason trainings that no computer could run twenty years ago now take months. The compute of the largest trainings grew faster still. In this lesson, you measure such growth with one number, the doubling time, and you see where it slowed down and where it ran into a wall.

    **An everyday example.** Fold a sheet of paper of 0.1 millimeter in half, and it is twice as thick. After 10 folds, it is about 10 centimeters thick; after 20 folds, it would be about 105 meters, if you could still fold it. Each fold adds as much as all the folds before it together: that is what doubling does.
    """)
    return


@app.cell(hide_code=True)
def predict_times():
    guess_times = L.guess(
        "times",
        question="OpenAI trained GPT-3 in 2020 with 3.14 × 10²³ FLOPs. Meta trained Llama 3.1 405B in 2024. How many times as much compute did that training take, do you think?",
        options={
            "chip": "About 4 times: as fast as the transistors on a chip grow",
            "twelve": "About 12 times",
            "right": "About 120 times",
            "thousand": "About 1,000 times",
        },
        reveal="**About 120 times.** Meta reports 3.8 × 10²⁵ FLOPs, 121 times the 3.14 × 10²³ of GPT-3: about 7 doublings in 4 years, one every 7 months. The transistors on a chip double about every two years, so in 4 years they give 4 times as many. The rest came from more GPUs working for longer: Meta trained the 405B model on up to 16,384 H100 GPUs.",
    )
    guess_times
    return (guess_times,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Moore's law.** In 1965, Gordon Moore, who later cofounded Intel, saw that the number of parts on a chip had doubled every year, and predicted that it would go on. In 1975, Moore changed that to every two years. **Moore's law** is that prediction: the number of transistors on a chip doubles about every two years. It is not a law of nature, like gravity. It is a trend that chip makers kept up for fifty years, by making transistors ever smaller.

        On a log scale, a doubling every two years is a straight line. Fifteen chips, from the Intel 4004 of 1971 to the NVIDIA B200 of 2024, lie close to one:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The best straight line through the fifteen chips rises one doubling every 2.04 years. The B200 puts two chips in one package, with 208 billion transistors together.

        **Doubling time.** The **doubling time** of a growth is the time it takes to double. From a start to an end, count how many doublings fit in, and divide the years by that number. The number of doublings is the log2 of how many times as large the end is. From the 4004 in 1971 to the Pentium 4 in 2000, the transistors grew 18,261 times, and the log2 of 18,261 is 14.16: 14.16 doublings in 29 years, one every 2.05 years.

        **Dennard scaling.** In 1974, Robert Dennard and colleagues at IBM described why smaller transistors were also better: a smaller transistor needs less voltage and switches faster. A chip full of smaller transistors runs faster on the same power for each square millimeter. That is **Dennard scaling**, and around 2005 it stopped. Transistors became so small that they leak current even when they are off, and a higher clock speed made chips too hot. In November 2004, Intel brought out a Pentium 4 at 3.8 GHz and canceled the one at 4 GHz. Since then, the transistors kept growing in number, but they went into more cores rather than a higher clock speed, and later into GPUs, which compute with thousands of cores at once.

        **The memory wall.** A GPU computes on numbers it reads from memory. The bandwidth of that memory, how many bytes per second it can deliver, grew more slowly than the computing. Amir Gholami and others measured over the 20 years up to 2024: the peak FLOP/s of server hardware grew 3.0 times every two years, the bandwidth of memory 1.6 times, and the bandwidth of the connections between chips 1.4 times. So each new GPU can read fewer bytes for every FLOP it does, and many computations spend more time waiting for data than computing. That is the **memory wall**. A later lesson shows how memory is built, and why data close by is fast.

        **The compute of training.** The largest trainings grew much faster than the chips, since they also use more chips, for longer. Epoch AI, a research group, keeps a database of the training compute of notable models: from 2010 to May 2024, their compute grew 4.1 times a year, and that of the ten largest at each moment 5.3 times a year. Its page of trends, updated on 5 November 2026, puts the largest trainings of language models at 5 times a year since 2020: one doubling every 5.2 months. [Why training costs millions](/course/open/sys-why-training-costs-millions?lang=en) showed what each GPU hour of those trainings costs.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Moore's law** | | the prediction, from 1975, that the number of transistors on a chip doubles about every two years; a trend, not a law of nature | |
        | **Dennard scaling** | | smaller transistors need less voltage, so a chip gets faster on the same power for each square millimeter; it stopped around 2005 | |
        | **memory wall** | | the computing of chips grows faster than the bandwidth of their memory, so the computing waits for data more and more | |
        | **doubling time** | | the time a growth takes to double; after `t` years, a doubling time `d` gives `2 ** (t / d)` times as much | `2 ** (t / d)` |
        | `3.8e25` | "3.8 e 25" | how Python writes 3.8 × 10²⁵: 38 followed by 24 zeros | `3.8e25` |
        """),
        L.watch_out("More transistors do not make a chip faster on their own. Since Dennard scaling stopped, the extra transistors go into more cores, and a program only gets faster when its work is split over them."),
        L.watch_out("A straight line on a log scale invites you to draw it on. Moore's law held for fifty years because chip makers worked to keep it up, and a trend does not say for sure what the next ten years bring."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The doubling time of a growth:

    1. Take the value at the start, the value at the end, and the years between them.
    2. Divide the end by the start: how many times as large has it become?
    3. Take the log2 of that: the number of doublings.
    4. Divide the years by the number of doublings: the years per doubling.
    5. With many points, fit a straight line through the years and the log2 of the values: its slope is the doublings per year, and one over the slope the years per doubling.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    With one doubling every 2 years, a number becomes `2 ** (years / 2)` times as large in `years` years. The cell starts from the 2,300 transistors of the 4004 and predicts the count every ten years, and in 2022, the year of the H100. Run the cell.
    """)
    return


@app.cell
def example_moore():
    _start = 2300   # the transistors of the Intel 4004, 1971
    for _years in [0, 10, 20, 30, 40, 51]:
        print(1971 + _years, round(_start * 2 ** (_years / 2)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Every ten years, the count grows 32 times: five doublings. The prediction for 2022 is 109 billion, and the H100 of that year has 80 billion, about three quarters of it, after more than 25 doublings. On a log scale, that gap is small. Exercise 1 turns this around: from two counts and the years between them, it finds the doubling time.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_time():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The doubling time

        `doubling_time(start, end, years)` gets a value at the start, a value at the end and the years between them, and gives back the years per doubling. Replace the `...` with the years divided by the number of doublings: the log2 of how many times as large `end` is as `start`. NumPy is there as `np`.

        The check calls `doubling_time(2300, 80000000000, 51)`: the transistors of the 4004 and of the H100, 51 years apart. Then the training compute from GPT-3 to Llama 3.1 405B in 4 years, and small numbers of its own.
        """),
        L.task("Give back the years per doubling from start to end."),
    ])
    return


@app.cell(hide_code=True)
def exercise_time():
    box_time = L.code_box("time", lines=3)
    box_time
    return (box_time,)


@app.cell(hide_code=True)
def check_time(box_time):
    result_time = L.check("time", box_time.value)
    result_time
    return (result_time,)


@app.cell(hide_code=True)
def hints_time():
    L.hints(
        "time",
        [
            "`end / start` says how many times as large the end is, and `np.log2` of a number counts the doublings in it. `years` divided by the number of doublings gives the years per doubling. What goes after the `/`?",
            "`np.log2(1024)` gives `10.0`: 1,024 is ten doublings of 1. From 1 to 1,024 in 20 years is one doubling every 2 years.",
            "`    return years / ___`",
        ],
        why="`return years / np.log2(end / start)` divides the years by the number of doublings. From the 4004 to the H100: 34.8 million times as many transistors, 25.05 doublings in 51 years, one every 2.04 years.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A line through all the points

        Two chips give a doubling time, and fifteen chips give fourteen slightly different ones, one for each pair next to each other. A straight line through all of them gives one. On a scale of log2, its slope is the doublings per year, and one over the slope is the years per doubling.

        `np.polyfit(x, y, 1)` fits the best straight line through points and gives back two numbers: first the slope, then the bias term. `fit_doubling(years, counts)` gets two NumPy arrays. Replace the `...` so that `slope` holds the slope of the straight line through `years` and the log2 of `counts`; the function then gives back `1 / slope`.

        The check calls `fit_doubling(C.YEARS, C.COUNTS)`: the fifteen chips of the picture. Then `C.TRAIN_YEARS` and `C.TRAIN_FLOP`, the training compute of BERT-Large, T5-11B, GPT-3 and Llama 3.1 405B, and small lists of its own.
        """),
        L.task("Make slope the slope of the line through the years and the log2 of the counts."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit", lines=4)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit",
        [
            "`np.polyfit` gets what goes across, what goes up, and `1` for a straight line, and `[0]` takes the slope out of what it gives back. Across go the `years`. Up goes the log2 of every count, and `np.log2` works on a whole array at once. What goes up, so that `slope` counts doublings?",
            "Counts 1, 2 and 4 in the years 0, 1 and 2 have the log2 0, 1 and 2: a line with slope 1, one doubling per year.",
            "`    slope = np.polyfit(years, ___, 1)[0]`",
        ],
        why="`slope = np.polyfit(years, np.log2(counts), 1)[0]` fits the line through the years and the doublings. The fifteen chips rise 0.491 doublings a year: one every 2.04 years. The four trainings rise 2.44 doublings a year: one every 4.9 months.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_laws():
    mo.md(r"""
    ### Exercise 3. Three ideas

    Eight facts from this lesson. Put each one under the idea it belongs to: *Moore's law*, *Dennard scaling and its end*, or *The memory wall*.
    """)
    return


@app.cell(hide_code=True)
def sort_laws_cell():
    sort_laws = L.sort("laws")
    sort_laws
    return (sort_laws,)


@app.cell(hide_code=True)
def check_laws(sort_laws):
    result_laws = L.check("laws", sort_laws)
    result_laws
    return (result_laws,)


@app.cell(hide_code=True)
def hints_laws():
    L.hints(
        "laws",
        [
            "Moore's law counts transistors. Dennard scaling is about power and clock speed. The memory wall is about computing against memory.",
            "Two facts go under Moore's law, three under Dennard scaling and its end, and three under the memory wall.",
            "Moore's law: 1965 and 1975, \\_\\_\\_. Dennard scaling and its end: smaller, same power, \\_\\_\\_, \\_\\_\\_. The memory wall: 3.0 against 1.6, \\_\\_\\_, \\_\\_\\_.",
        ],
        why="**Moore's law:** 1965 and 1975, 2,300 and 80 billion. **Dennard scaling and its end:** smaller, same power, 3.8 GHz, more cores. **The memory wall:** 3.0 against 1.6, waiting for data, fewer bytes per FLOP.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_wall():
    mo.md(r"""
    ### Exercise 4. Twenty years of the memory wall

    Over 20 years, the peak FLOP/s of a server grew 3.0 times every two years, and the bandwidth of its memory 1.6 times every two years.
    """)
    return


@app.cell(hide_code=True)
def choice_wall_cell():
    choice_wall = L.choice(
        "wall",
        question="After those 20 years, how many times as many FLOPs can a server do for each byte it reads from memory?",
        options={
            "ratio": "About 1.9 times: 3.0 divided by 1.6",
            "fourteen": "About 14 times",
            "right": "About 540 times",
            "flops": "About 59,000 times, since 3.0 to the power 10 is 59,049 and memory does not count",
        },
        correct="right",
        why={
            "right": "20 years are ten periods of two years. The FLOP/s grow 3.0¹⁰ = 59,049 times, the bandwidth 1.6¹⁰ = 110 times, and 59,049 divided by 110 is about 540. Each byte from memory must feed 540 times as many FLOPs, so a GPU of today waits for its memory far more often than one of 20 years ago.",
            "ratio": "1.9 is the gap after one period of two years. How many periods fit in 20 years, and what does each one do to the gap?",
            "fourteen": "The growth multiplies; it does not add. What do ten periods of 3.0 and ten of 1.6 give?",
            "flops": "59,049 is the growth of the FLOP/s alone, and memory grew too, 1.6 times every two years. How much of the 59,049 is left for each byte?",
        },
    )
    choice_wall
    return (choice_wall,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Five kinds of growth

    How many times each kind of growth has multiplied after a number of years, on a log scale: the compute of the largest trainings at 5.3 times a year, peak FLOP/s at 3.0 times in two years, transistors at 2 times, the bandwidth of memory at 1.6 times, and the bandwidth between chips at 1.4 times. Choose a number of years.
    """)
    return


@app.cell(hide_code=True)
def demo_years():
    L.explore("years", "years", lambda _v: C.frame(_v, L.lang), start=4, values=C.DEMO_VALUES, role="",
              task="Go from 0 to 14 years: which bar runs away, and what happens to the FLOPs per byte read?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    After 14 years, the largest trainings have grown 13.8 billion times, peak FLOP/s 2,187 times, and the transistors 128 times. The bandwidth of memory has grown 27 times, so each byte read must feed 81 times as many FLOPs. The trainings outgrow the chips by using more of them, and the gap between the top bar and the transistors is paid for in GPUs, in power and in money. The rates are averages over the years in which they were measured, and the bars carry them on: no one knows whether they hold for another 14 years.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Log scales and power laws](/course/open/math-log-scales?lang=en), growth by the same factor per step was a straight line on a log scale; here that line is fifty years of chips, and its slope is a doubling time. In [Why training costs millions](/course/open/sys-why-training-costs-millions?lang=en), a training cost GPU hours: Moore's law is the reason an hour of a newer GPU does more work, and the memory wall the reason it often waits. The next lesson is *Close and fast, far and slow: the memory hierarchy*: how memory is built, and why data close by is fast.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_wall, result_fit, result_laws, result_time):
    L.wrap_up(result_time, result_fit, result_laws, choice_wall)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
