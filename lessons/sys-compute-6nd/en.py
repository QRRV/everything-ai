# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-compute-6nd. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-compute-6nd", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en) and [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_six():
    quick_six = L.choice(
        "quick-six",
        question="Reading a token costs a GPT about 2 FLOPs per parameter, and training on it about 6. Where do the other 4 come from?",
        options={
            "right": "From the backward pass, which costs about twice the forward pass",
            "adam": "From AdamW, which does 4 FLOPs per parameter for every token",
            "twice": "From reading every token three times",
        },
        correct="right",
        why={
            "right": "For every parameter, the backward pass does two multiplications per token: one for the gradient of that parameter, one to pass the gradient back to the layer before. The step of the optimizer comes once per mini-batch, not per token.",
            "adam": "AdamW updates each parameter once per step, for a whole mini-batch of tokens: a few FLOPs per parameter per step, which hardly count. See [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en).",
            "twice": "Training reads every token once, forward, and then goes back once, backward. See [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en).",
        },
    )
    quick_six
    return (quick_six,)


@app.cell(hide_code=True)
def quick_check_peak():
    quick_peak = L.choice(
        "quick-peak",
        question="By its specifications, a GPU does 100 TFLOP/s. A training step takes $6 \\times 10^{12}$ FLOPs. How long does the step take at least?",
        options={
            "ms": "0.6 milliseconds",
            "right": "0.06 seconds",
            "s": "6 seconds",
        },
        correct="right",
        why={
            "ms": "100 TFLOP/s is $10^{14}$ FLOPs a second: $6 \\times 10^{12} / 10^{14} = 0.06$ seconds. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "s": "A TFLOP is $10^{12}$ FLOPs, so 100 TFLOP/s is $10^{14}$ a second, not $10^{12}$. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "right": "$6 \\times 10^{12} / 10^{14} = 0.06$ seconds, at the full peak. Real code takes longer: it reaches only part of the peak.",
        },
    )
    quick_peak
    return (quick_peak,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before a training run, you want to know how long it takes and what it costs: an hour on your laptop, a night on a rented GPU, or more than you can pay. Big labs plan runs of months on thousands of GPUs this way, and you can plan your own runs the same way. One rule of thumb gives the answer before you run anything, from three numbers: the size of the model, the number of tokens, and the speed of your GPUs.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt3():
    guess_gpt3 = L.guess(
        "gpt3-one-gpu",
        question="GPT-3 has 175 billion parameters and trained on 300 billion tokens. Training on one token costs about 6 FLOPs per parameter. How long would one A100 GPU need for it, at its full peak of 312 TFLOP/s in BF16?",
        options={
            "months": "About 3 months",
            "years": "About 3 years",
            "right": "About 30 years",
        },
        reveal="**About 30 years.** $6 \\times 175 \\times 10^9 \\times 300 \\times 10^9 \\approx 3.2 \\times 10^{23}$ FLOPs, at $312 \\times 10^{12}$ a second: $10^9$ seconds, 32 years. That is at the full peak, which no run reaches. The GPT-3 paper reports $3.14 \\times 10^{23}$ FLOPs for its training, done on a large cluster of GPUs.",
    )
    guess_gpt3
    return (guess_gpt3,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: the whole run, 6ND.** Training on one token costs about $6N$ FLOPs for a model of $N$ parameters. A run that trains on $D$ tokens then costs
        """),
        L.formula(r"\text{FLOPs} \approx 6\,N\,D"),
        mo.md(r"""
        $D$ counts every token the model trains on: 20 epochs over a book of 1 million tokens is $D = 20$ million. The rule leaves out the part of attention that grows with the window: comparing queries with keys, and mixing the values. For training, that part adds about $n/(12d)$ of $6N$, for windows of $n$ tokens and vectors of $d$ numbers: about 5.6% for your GPT of the book, with $n = 256$ and $d = 384$, and 1.4% for GPT-3, with $n = 2{,}048$ and $d = 12{,}288$.

        The picture: what training on one token costs, in units of $N$ FLOPs: the forward pass, $2N$, the backward pass, $4N$, and the thin strip at the end, attention.
        """),
        mo.Html(C.token_picture({"title": "Training on one token", "x": "FLOPs, in units of N", "forward": "forward", "backward": "backward",
                                 "row": "{name}: +{pct}%", "book": "your GPT of the book", "gpt3": "GPT-3", "lang": L.lang,
                                 "alt": "Two bars, each 6N long: the forward pass, 2N, and the backward pass, 4N. Attention adds 5.6% on top for your GPT of the book, and 1.4% for GPT-3."})),
        mo.md(r"""
        **Step 2: from FLOPs to time.** The specifications of a GPU give its peak, for each number format: an A100 does 312 TFLOP/s in BF16 and 19.5 TFLOP/s in FP32 on its ordinary cores; an H100 SXM does 989 TFLOP/s in BF16. A real run reaches only part of the peak: large runs about 30 to 50%, small models often far less. The time is the FLOPs divided by what all GPUs together really do in a second:
        """),
        L.formula(r"\text{time} = \frac{6\,N\,D}{\text{peak} \times \text{part} \times \text{GPUs}}"),
        mo.md(r"""
        **Step 3: GPU hours and the bill.** Time multiplied by the number of GPUs gives **GPU hours**, and GPU hours multiplied by the price of one GPU hour give the bill. Twice as many GPUs make the run about twice as short, for about the same GPU hours.

        **Step 4: backwards.** Turned around, the same rule tells what fits in a budget, $D = \text{FLOPs} / (6N)$: for the same budget, a model twice as big trains on half as many tokens. And from the GPU hours a lab reports, it gives the part of the peak their run reached.
        """),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **6ND rule** | training a model of $N$ parameters on $D$ tokens costs about $6ND$ FLOPs | `6 * n_params * n_tokens` |
        | **GPU hour** | one GPU used for one hour; 8 GPUs for 10 hours is 80 GPU hours | `hours * n_gpus` |
        | $D$ | the number of tokens a model trains on, epochs included | `n_tokens` |
        """),
        L.watch_out("Use the peak of the number format you train in, and without sparsity: some specifications give a peak twice as high that counts only for matrices with two zeros in every four numbers."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To estimate a training run:

    1. Count the parameters $N$, and the tokens $D$ the model trains on: tokens per step times steps, or tokens in the data times epochs.
    2. Multiply: $6ND$ FLOPs.
    3. Divide by the peak of your GPU in your number format, by the part of it you expect to reach, and by the number of GPUs: that is the time in seconds.
    4. Multiply the time in hours by the number of GPUs for the GPU hours, and those by the price of one GPU hour for the bill.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    What is new is working backwards from a report. Llama 2 70B has 70 billion parameters and trained on 2 trillion tokens; its paper reports 1,720,320 GPU hours on A100 GPUs. How much of their peak did those GPUs reach?
    """)
    return


@app.cell
def example_llama():
    _n = 70e9                  # parameters of Llama 2 70B
    _d = 2e12                  # tokens it trained on
    _peak = 312e12             # an A100 in BF16, FLOPs per second
    _reported = 1_720_320      # GPU hours, in the Llama 2 paper

    _flops = 6 * _n * _d
    _at_peak = _flops / _peak / 3600
    print("FLOPs:", _flops)
    print("GPU hours at the full peak:", round(_at_peak))
    print("part of the peak reached:", round(_at_peak / _reported, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At the full peak, $8.4 \times 10^{23}$ FLOPs would take 747,863 GPU hours; the run took 1,720,320. Over the whole run, the GPUs reached about 43% of their peak. The rest went to things such as operations that wait for memory, GPUs waiting for each other, and everything else a run does besides multiplying matrices. The next lesson measures that part on your own machine.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How long?

        Write `train_hours(n_params, n_tokens, peak, utilization, n_gpus)`: the time in hours of training a model of `n_params` parameters on `n_tokens` tokens, on `n_gpus` GPUs with a peak of `peak` FLOPs a second each, of which they reach the part `utilization`, such as `0.4`. The check calls `train_hours(1e9, 20e9, 989e12, 0.4, 8)`, a model of a billion parameters on 8 H100 GPUs, which should give about 10.53, and tries other runs.
        """),
        L.task("Write `train_hours(n_params, n_tokens, peak, utilization, n_gpus)`: the time of the run in hours, from the 6ND rule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("train-hours", lines=6)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("train-hours", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "train-hours",
        [
            "Steps 2 and 3 of the recipe: the FLOPs of the whole run, divided by what all GPUs together really do in a second. That gives seconds; you want hours.",
        ],
        why="$6 \\times 10^9 \\times 20 \\times 10^9 = 1.2 \\times 10^{20}$ FLOPs, at $989 \\times 10^{12} \\times 0.4 \\times 8 \\approx 3.2 \\times 10^{15}$ a second: 37,917 seconds, 10.53 hours.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### From a book to 70 billion parameters

    The cell below runs your `train_hours` on four runs, all on A100 GPUs at 40% of their peak in BF16: your GPT of the book, 10.8 million parameters on 20 epochs over a book of about a million tokens, on one GPU; a model of 124 million parameters, the size of GPT-2 small, on 10 billion tokens on 8 GPUs; and models of 7 and 70 billion parameters on 2 trillion tokens, the sizes and token counts of Llama 2, on 1,024 and 2,048 GPUs. It works once exercise 1 is right.
    """)
    return


@app.cell(hide_code=True)
def demo_runs(box_hours, result_hours):
    if result_hours.state != "correct":
        _out = mo.md("*Finish exercise 1 first: this cell uses your `train_hours`.*")
    else:
        _rows = C.demo(box_hours.value)
        if _rows == "error":
            _out = mo.md("*Your `train_hours` stopped with an error on one of these runs.*")
        else:
            _names = {"book": "your GPT, the book, 1 GPU", "small": "124 million, 8 GPUs", "7b": "7 billion, 1,024 GPUs", "70b": "70 billion, 2,048 GPUs"}
            _gh = lambda _g: C.number(_g, 3 if _g < 1 else 0, L.lang)
            _table = "\n".join(f"| {_names[_k]} | {C.duration(_h, L.lang)} | {_gh(_g)} |" for _k, _h, _g in _rows)
            _out = mo.md(f"""
    | run | time | GPU hours |
    |---|---|---|
    {_table}

    From seconds to more than a month, for the same rule. For the model of 70 billion parameters, your function estimates about 1.9 million GPU hours; the Llama 2 paper reports 1.7 million, as its GPUs reached a little more than 40%. A model as small as your GPT of the book reaches far less than 40%: its matrices are too small to keep a large GPU busy, and many of its operations wait for memory. The next lesson measures how much it does reach.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.md(r"""
    ### Exercise 2. A budget
    """)
    return


@app.cell(hide_code=True)
def choice_budget_cell():
    choice_budget = L.choice(
        "budget",
        question="You may use 64 H100 GPUs, at 989 TFLOP/s in BF16, for one week, and you expect to reach 40% of their peak. You train a model of 7 billion parameters. On about how many tokens can it train?",
        options={
            "hours": "About 100 million",
            "right": "About 365 billion",
            "forward": "About 1.1 trillion",
        },
        correct="right",
        why={
            "hours": "That is what you get when you count the week in hours, while the peak is in FLOPs a second. A week has 604,800 seconds.",
            "forward": "That is what you get with 2 FLOPs per parameter per token, the forward pass alone. Training costs 6.",
            "right": "$64 \\times 604{,}800 \\times 989 \\times 10^{12} \\times 0.4 \\approx 1.5 \\times 10^{22}$ FLOPs, divided by $6 \\times 7 \\times 10^9$: about $3.6 \\times 10^{11}$ tokens.",
        },
    )
    choice_budget
    return (choice_budget,)


@app.cell(hide_code=True)
def text_exercise_optimistic():
    mo.md(r"""
    ### Exercise 3. Longer than planned
    """)
    return


@app.cell(hide_code=True)
def explain_optimistic():
    explain_optimistic_cell = L.explain(
        "too-optimistic",
        question="With the 6ND rule and 40% of the peak, you estimated a run at 10 days. It takes 14. Name two reasons why the estimate can come out too short, and how you could have found out before the run.",
        model_answer="**The run reached less of the peak than 40%:** operations that wait for memory, small matrices, GPUs waiting for each other or for data, evaluations and checkpoints along the way. **The run does more FLOPs than 6ND:** attention on long windows comes on top, and recomputing parts of the forward pass, to save memory, adds up to a third. Restarts after a crash or a bad stretch of training cost time too. **To find out before:** run a few hundred steps on the real hardware, measure the tokens per second, and divide the tokens you need by it; the next lesson does that.",
        points=[
            "you name a lower part of the peak than planned, with a reason such as waiting for memory, for other GPUs or for data",
            "you name a second reason, such as attention on long windows, recomputation or restarts",
            "you say to measure a short run on the real hardware first",
        ],
    )
    explain_optimistic_cell
    return (explain_optimistic_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The 6 FLOPs per parameter per token come from [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en), the peak of a GPU from [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en), and BF16 from [Mixed-precision training](/course/open/sys-mixed-precision?lang=en). Together with [Where the memory goes](/course/open/sys-memory-accounting?lang=en), you can now tell before a run whether it fits and how long it takes. The next lesson, *Profiling: tokens per second and MFU*, measures the part of the peak your own runs reach.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_budget, explain_optimistic_cell, result_hours):
    L.wrap_up(result_hours, choice_budget, explain_optimistic_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
