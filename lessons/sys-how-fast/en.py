# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-how-fast. Route: course/route/lessen/systemen.yaml. Checks, the course's bigram writer, the measurements and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-how-fast", "en")
    C = L.checks   # the course's bigram writer, the measurements and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en) and [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="A network has 50,000 weights. About how many FLOPs does one training step cost for one example?",
        options={
            "forward": "About 100,000: 2 per weight, as in the forward pass",
            "right": "About 300,000: 6 per weight",
            "one": "About 50,000: 1 per weight",
            "pairs": "About 2.5 billion: each weight with each other",
        },
        correct="right",
        why={
            "right": "The forward pass costs 2 FLOPs per weight, and the backward pass uses every weight twice more: 6 · 50,000 = 300,000. This lesson turns FLOPs into time: how many of them a computer does in one second.",
            "forward": "2 FLOPs per weight is the forward pass alone. What does the backward pass add? See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "one": "In the forward pass alone, each weight takes a multiplication and an addition, and the backward pass uses it again. How many FLOPs per weight does that give? See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "pairs": "A weight is used once per example in the forward pass, not once for every other weight. How many FLOPs per weight does a training step cost? See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`W` is the 65 × 65 table of your bigram network, and `x` holds the numbers of 1,000 characters. What is `W[x]`?",
        options={
            "one": "One row of 65 numbers: the row of the last character",
            "table": "A 65 × 65 tensor: the table itself",
            "right": "A 1,000 × 65 tensor: the row of each character",
            "pairs": "A 1,000 × 1,000 tensor of all pairs",
        },
        correct="right",
        why={
            "right": "`W[x]` picks the row of `W` for every number in `x`: the same as `torch.eye(65)[x] @ W`, a one-hot row per character times the table. In this lesson, that product is what the network calculates for every character it writes.",
            "one": "`x` holds 1,000 numbers, and `W[x]` picks a row for each of them. How many rows does that give? See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
            "table": "Indexing with `x` picks rows of `W`: one for every number in `x`. How many numbers does `x` hold? See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
            "pairs": "`W[x]` picks rows of `W`, and every row has 65 numbers, one for each next character. How many numbers does each row of the result have? See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model that gives good answers slowly is of little use: nobody waits a minute for every word of an answer, and a training run that would take a year never ends. In [Too many slopes to measure one by one](/course/open/nn-why-backprop?lang=en), you counted the work of a calculation in FLOPs, and that count is the same on every computer. How long the work takes is not: that depends on how many FLOPs the computer does per second, and on how much of its time goes to work that has nothing to do with FLOPs. This lesson measures both on your own computer, for a matrix product and for the bigram network of your project: how many FLOPs per second, how many characters per second, and where the time of one character goes.

    **An everyday example.** A trip of 30 km is just as far on the highway as through the city. On the highway, at 100 km/h, it takes 18 minutes. Through the city, with traffic lights, crossings and parking, the same 30 km can take an hour, in a car that could easily drive 100 km/h the whole way. The distance is the work, the speed is the rate, and the time is the work divided by the rate you actually reach.
    """)
    return


@app.cell(hide_code=True)
def predict_speed():
    guess_speed = L.guess(
        "speed",
        question="One character of your bigram network costs a one-hot row times the 65 × 65 table: 2 · 65 · 65 = 8,450 FLOPs. The course's computer does about 450 billion FLOPs per second on a large matrix product. How many characters per second does the network write on that computer, do you think?",
        options={
            "flops": "About 50 million: 450 billion divided by 8,450",
            "billions": "Billions: a product is what a computer does fastest",
            "right": "Tens of thousands",
            "few": "A few hundred: Python is slow",
        },
        reveal="**Tens of thousands.** On the course's computer, one character takes 20.5 microseconds: about 49,000 characters per second, about a thousand times fewer than the 53 million the FLOPs alone would allow. The product itself takes 1.9 of those microseconds, and the draw of the next character 14.5. A product as small as one row times a 65 × 65 table reaches only a few billion FLOPs per second, and most of the time goes to work that has nothing to do with FLOPs.",
    )
    guess_speed
    return (guess_speed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Work and rate.** FLOPs count the work of a calculation. A product of two n × n matrices gives n · n numbers, and each of them takes n multiplications and n additions, so the product costs $2n^3$ FLOPs. **FLOP/s**, FLOPs per second, is the rate: how many of them a computer does in one second. The time of the work is the work divided by the rate. Watch the spelling: FLOPs, with a small s, is a number of operations; FLOP/s, also written FLOPS, is a speed.

        **Measuring FLOP/s.** To measure a rate, you time work whose FLOPs you know. `time.perf_counter()`, from Python's module `time`, reads a clock in seconds that is made for measuring short times: read it before and after the work, and the difference is the time the work took. One measurement can be unlucky, since the computer also does other things now and then, so you measure a few times and keep the fastest: that one had the fewest interruptions. The first run of a calculation is slower, since PyTorch prepares it then, so it does not count. Short times are easiest to read in **microseconds**, µs, millionths of a second: one second is $10^6$ µs.

        **Small is slow.** The picture shows the FLOP/s of an n × n product on the course's computer. At 8 × 8, a product costs 1,024 FLOPs and takes 1.1 µs: about 0.9 billion FLOP/s. At 1,024 × 1,024, it costs 2.1 billion FLOPs and takes 4,800 µs: about 450 billion FLOP/s, 480 times as fast. Every product has a fixed cost on top of its FLOPs: Python calls PyTorch, PyTorch checks the shapes, picks a way to calculate and makes room for the result. On the course's computer that takes about a microsecond, whatever the size. For a small product, the fixed cost is nearly all of the time; for a large one it disappears next to the FLOPs, and all the cores can be kept busy.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Tokens per second.** For a model that writes, the rate that counts is **tokens per second**: how many tokens it writes in one second. For your bigram network, a token is one character, and every token takes one step: the one-hot row of the last character times the table, softmax, and a draw of the next character, as in [Sampling: let the model write](/course/open/lm-sampling?lang=en). Only the product does real work with the table; the other parts mostly have their fixed cost.

        **Throughput.** One step can also give many texts their next character at once: put the last characters of all texts in `x`, and `torch.eye(65)[x] @ W` gives all their rows in one product. **Throughput** is the work done per second by everything together: with 100 texts at once, the tokens per second of all 100 texts. The fixed cost of each part is then paid once per step, and not once per text.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **FLOP/s** | flop per second | the number of floating-point operations a computer does in one second: how fast it calculates; also written FLOPS | `flops / seconds` |
        | **tokens per second** | | how many tokens a model writes in one second | `n / seconds` |
        | **throughput** | | the work done per second by everything together, such as the tokens per second of all texts at once | |
        | `time.perf_counter()` | time dot perf counter | a clock in seconds, made for measuring short times: only the difference between two readings means something | `time.perf_counter()` |
        | **microsecond**, µs | | a millionth of a second | `seconds * 10 ** 6` |
        """),
        L.watch_out("The FLOPs of a model say how much it calculates, not how long it takes. Two models with the same FLOPs can differ many times in speed, when one does a few large products and the other many small ones."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Measuring how fast something runs:

    1. Count the work: FLOPs for a calculation, tokens for a model that writes.
    2. Run it once first, so that PyTorch has prepared everything.
    3. Read `time.perf_counter()` before and after, and take the difference: the seconds.
    4. Repeat a few times, and keep the fastest.
    5. Divide the work by the seconds: FLOP/s, or tokens per second.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell times a product of two 512 × 512 matrices five times, keeps the fastest, and works out the FLOP/s. `torch.ones((size, size))` makes a matrix full of ones: which numbers a matrix holds does not change how long a product takes. Run the cell.
    """)
    return


@app.cell
def example_measure():
    import time

    import torch

    size = 512
    a = torch.ones((size, size))
    b = torch.ones((size, size))
    flops = 2 * size ** 3
    a @ b   # the first product of a size also prepares it
    fastest = None
    for _ in range(5):
        before = time.perf_counter()
        a @ b
        seconds = time.perf_counter() - before
        if fastest is None or seconds < fastest:
            fastest = seconds
    print("FLOPs:", flops)
    print("fastest time in µs:", round(fastest * 10 ** 6, 1))
    print("billions of FLOP/s:", round(flops / fastest / 10 ** 9, 1))
    return a, b, before, fastest, flops, seconds, size, time, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The product costs 268,435,456 FLOPs. On the course's computer, the fastest of the five takes about 650 µs, which gives about 413 billion FLOP/s. Your computer gives other numbers: fewer cores give fewer FLOP/s, a newer processor more. Run the cell again, and the time changes a little with every run; keeping the fastest gives a number you can compare.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flop():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. FLOP/s from measured times

        `flop_per_second(sizes, micros)` gets a list of matrix sizes and a list of times: `micros[i]` is the time in microseconds of one product of two `sizes[i]` × `sizes[i]` matrices. It gives back the list `rates`, with the FLOP/s of each size. The loop `for i in range(len(sizes))` goes through the sizes; `n` is the size, and `seconds` the time in seconds. Replace the `...` with a line that adds the FLOP/s of this size to `rates`.

        The check calls `flop_per_second` with the sizes `[16, 512]` and the times `[1.3, 650.0]` of the course's computer. Then four other cases: all eight sizes of the picture with their times, a 2 × 2 product of 1 µs, a 1,000 × 1,000 product of 2,000 µs, and two empty lists.
        """),
        L.task("Add the FLOP/s of each size to `rates`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flop():
    box_flop = L.code_box("flop-rate", lines=8)
    box_flop
    return (box_flop,)


@app.cell(hide_code=True)
def check_flop(box_flop):
    result_flop = L.check("flop-rate", box_flop.value)
    result_flop
    return (result_flop,)


@app.cell(hide_code=True)
def hints_flop():
    L.hints(
        "flop-rate",
        [
            "Step 5 of the recipe: the FLOP/s of a size is its FLOPs divided by its seconds. The line calls `rates.append` with the FLOPs of a product of two `n` × `n` matrices, divided by `seconds`.",
            "A product of two n × n matrices gives n · n numbers, and each of them takes n multiplications and n additions. For n = 16 that is 8,192 FLOPs, and with 1.3 µs, 0.0000013 seconds, about 6.3 billion FLOP/s.",
            "`        rates.append(___ / seconds)`",
        ],
        why="`rates.append(2 * n ** 3 / seconds)` divides the work of one product by its time. `n ** 3` counts the n · n numbers of n steps each, and the 2 the multiplication and the addition of every step.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_tokens():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Tokens per second

        `tokens_per_second(write, n)` gets a function `write` that writes `n` tokens, and measures how many tokens per second it writes. The first line, `import time`, makes the clock available, and the `return` line divides `n` by `seconds`. Replace the `...` with the clock around `write(n)`: read `time.perf_counter()` before it, call `write(n)`, and make `seconds` the time that passed.

        The check calls `tokens_per_second` with a stand-in for a writer: `slow_writer(0.05)` takes 0.05 seconds, whatever `n` is, so 1,000 tokens give 20,000 tokens per second. A clock never gives exactly the same time twice, so the check accepts an answer within 30% of that. Then two other cases: 500 tokens in 0.1 seconds, and 30 tokens in 0.05 seconds.
        """),
        L.task("Time one call of `write(n)`, and make `seconds` its time."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tokens():
    box_tokens = L.code_box("tokens-rate", lines=8)
    box_tokens
    return (box_tokens,)


@app.cell(hide_code=True)
def check_tokens(box_tokens):
    result_tokens = L.check("tokens-rate", box_tokens.value)
    result_tokens
    return (result_tokens,)


@app.cell(hide_code=True)
def hints_tokens():
    L.hints(
        "tokens-rate",
        [
            "Step 3 of the recipe, in three lines: a reading of `time.perf_counter()` with a name of its own, then `write(n)`, then `seconds`: a new reading of `time.perf_counter()` minus the first one.",
            "In the worked example, the first reading was called `before`. With `slow_writer(0.05)`, the two readings lie about 0.05 seconds apart, so 1,000 tokens give about 20,000 tokens per second.",
            "`    seconds = time.perf_counter() - ___`",
        ],
        why="`start = time.perf_counter()` before `write(n)`, and `seconds = time.perf_counter() - start` after it: the clock runs around exactly one call of `write(n)`, and nothing else.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Where the time of a character goes

    The demo writes 2,000 characters on your computer with the course's bigram table, counted on the plays of Shakespeare, the table your network learned in the project. It times each part of every step: the one-hot row, the product with the table, softmax, and the draw of the next character. Then it writes 1, 10, 100 and 1,000 texts at once, 200 steps each, and measures how long one step takes. Press the button; measuring takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    measure_button = mo.ui.run_button(label="Measure on my computer")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(measure_button):
    if not measure_button.value:
        _out = mo.md("*Press the button to measure on your computer.*")
    else:
        _m = C.demo()
        _out = mo.vstack([
            mo.Html(C.parts_picture(_m["parts"], L.lang)),
            mo.md(C.parts_say(_m["parts"], L.lang)),
            L.explore("texts", "texts at once", lambda _v: C.texts_frame(_m["texts"], int(_v), L.lang), start=1,
                      values=list(C.TEXTS), role="",
                      task="Go from 1 text to 1,000: how many times as many characters per second do 1,000 texts at once give on your computer?"),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    On the course's computer, one character takes 20.5 µs: about 49,000 characters per second. The product takes 1.9 µs of that, less than a tenth, and the draw 14.5 µs, more than two thirds; the one-hot row takes 2.5 µs and softmax 1.3 µs, and the last 0.3 µs go to the loop around the parts and to reading the clock. A step with 100 texts takes 180 µs, about 9 times as long as a step with one, and gives 100 characters: about 560,000 characters per second, 11 times the throughput of one text. With 1,000 texts, a step takes 1,300 µs, and the throughput is about 770,000 characters per second, 16 times that of one text.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_part_faster():
    mo.md(r"""
    ### Exercise 3. A faster product

    Someone finds a way to make the product of the bigram network ten times as fast, and changes nothing else.
    """)
    return


@app.cell(hide_code=True)
def choice_part_faster_cell():
    choice_part_faster = L.choice(
        "part-faster",
        question="On the course's computer, how much less time does one character take?",
        options={
            "ten": "Ten times less: 2.1 µs instead of 20.5",
            "half": "About half: the product is the main work",
            "none": "None: FLOPs take no time",
            "right": "About 8% less: 1.7 of the 20.5 µs",
        },
        correct="right",
        why={
            "right": "The product takes 1.9 of the 20.5 µs, and ten times as fast it takes 0.19: 1.7 µs less, about 8%. Speeding up the part that does the FLOPs helps little when most of the time goes elsewhere: the draw alone takes 14.5 µs.",
            "ten": "Only the product gets ten times as fast, and the product takes 1.9 of the 20.5 µs. What happens to the other 18.6 µs?",
            "half": "The product does nearly all the FLOPs, but how much of the time does it take on the course's computer?",
            "none": "The product does take time: 1.9 µs on the course's computer. How much of that goes away when it is ten times as fast?",
        },
    )
    choice_part_faster
    return (choice_part_faster,)


@app.cell(hide_code=True)
def text_exercise_many_texts():
    mo.md(r"""
    ### Exercise 4. Many people at once

    A service writes texts for many people with one model. Instead of a step for each person, it gives 100 people their next token in one step.
    """)
    return


@app.cell(hide_code=True)
def choice_many_texts_cell():
    choice_many_texts = L.choice(
        "many-texts",
        question="What happens to a step, compared with a step for one person?",
        options={
            "right": "It takes longer, but far less than 100 times as long",
            "same": "It takes 100 times as long, so the throughput stays the same",
            "shared": "It takes as long as for one person: the work is shared",
            "fewer": "It takes so long that fewer tokens come out per second",
        },
        correct="right",
        why={
            "right": "On the course's computer, a step with 100 texts takes 180 µs instead of 20.5: about 9 times as long, for 100 times the characters. The fixed cost of each part is paid once per step, so the throughput rises 11 times. Services that write for many people at once keep their computers busy this way.",
            "same": "The 100 one-hot rows go through one product, one softmax and one draw together. Is the fixed cost of each part then paid once, or 100 times?",
            "shared": "A step with 100 texts does 100 times the FLOPs of a step with one. Can it take no longer at all?",
            "fewer": "The texts share each step: one product for all their rows. In the demo, did the characters per second go down or up with more texts?",
        },
    )
    choice_many_texts
    return (choice_many_texts,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Too many slopes to measure one by one](/course/open/nn-why-backprop?lang=en) you counted FLOPs, and in [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en) a GPU paid off for large products only. This lesson put a clock next to the counting: FLOP/s for a matrix product, tokens per second for the bigram network of [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en), and the throughput of many texts at once. The domain *ML systems, hardware and MLOps* goes much further: the chips that train models, serving many people at once, and what training and running a model costs. With *Show extra lessons* on, *Spam or not: naive Bayes* comes next, where counted words sort messages into spam and ok. In the next lesson, *What a paper, a preprint and peer review are*, you start reading research yourself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_many_texts, choice_part_faster, result_flop, result_tokens):
    L.wrap_up(result_flop, result_tokens, choice_part_faster, choice_many_texts)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
