# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-profiling-mfu. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-profiling-mfu", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en) and [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_time():
    quick_time = L.choice(
        "quick-time",
        question="The 6ND rule gives the FLOPs of a training run. What else do you need to turn them into a time?",
        options={
            "layers": "The number of layers of the model",
            "lr": "The learning rate",
            "right": "How many FLOPs a second your GPUs really do",
        },
        correct="right",
        why={
            "layers": "The layers are already in $N$, the number of parameters. See [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).",
            "lr": "The learning rate changes what the model learns, not what a step costs. See [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).",
            "right": "The peak times the part of it a run reaches, times the number of GPUs. This lesson measures that part.",
        },
    )
    quick_time
    return (quick_time,)


@app.cell(hide_code=True)
def quick_check_bound():
    quick_bound = L.choice(
        "quick-bound",
        question="An operation does 2 FLOPs for every byte it reads. It runs on a GPU whose ridge point lies at 80 FLOPs per byte. What limits its speed?",
        options={
            "right": "The memory: it waits for data",
            "compute": "The calculations: it waits for the cores",
            "neither": "Neither: 2 FLOPs per byte is too little to matter",
        },
        correct="right",
        why={
            "right": "Far left of the ridge point, the cores wait for the bytes. Many small operations in a network are like this.",
            "compute": "Compute-bound operations lie right of the ridge point, at more than 80 FLOPs per byte. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "neither": "Every operation is limited by something: here the bandwidth, as it reads many bytes for few FLOPs. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
        },
    )
    quick_bound
    return (quick_bound,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The 6ND rule gives a time if you know what part of the peak your run reaches. That part you measure: run a few steps, count the tokens per second, and compare the model's FLOPs per second with the peak. When it comes out low, a profiler shows where the time goes, operation by operation, and so what to change first. A run that goes from 15% to 45% of the peak takes a third of the time.
    """)
    return


@app.cell(hide_code=True)
def predict_async():
    guess_async = L.guess(
        "async-clock",
        question="On a GPU, you time one training step: `time.perf_counter()` before it and after it, and nothing else. What does the difference show?",
        options={
            "right": "Less than the step takes, often much less",
            "same": "The time of the step",
            "more": "More than the step takes: the clock also counts sending the work to the GPU",
        },
        reveal="**Less than the step takes, often much less.** PyTorch puts the work of each operation in the GPU's queue and goes on with the next line of Python at once. The GPU works through the queue at its own pace, so the second `time.perf_counter()` can run while the GPU is still busy. `torch.cuda.synchronize()` waits until the queue is empty.",
    )
    guess_async
    return (guess_async,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: tokens per second.** Run a number of steps, and divide the tokens they trained on by the seconds they took: the **throughput** of the run. Three things make the measurement true. Skip the first steps: in them, PyTorch asks for memory and picks how to compute each operation, so they are slower than the rest. On a GPU, Python puts the work in a queue and runs ahead; the queue holds only so much, and when it is full, Python waits for room. So on a GPU, wait for the device before you read the clock, with `torch.cuda.synchronize()`, both before you start the clock and before you stop it. And time several steps, not one: steps differ a little from each other.

        **Step 2: MFU.** Tokens per second times the FLOPs per token, $6N$ plus attention, gives the FLOPs per second the model needs. Divided by the peak, that is the **MFU**, the model FLOPs utilization:
        """),
        L.formula(r"\text{MFU} = \frac{\text{tokens/s} \times \text{FLOPs per token}}{\text{peak}}"),
        mo.md(r"""
        The MFU counts only the FLOPs the model needs. Some runs compute more: they throw away activations and compute them again in the backward pass, to save memory. The **HFU**, the hardware FLOPs utilization, counts those FLOPs too, so it is never lower than the MFU. For the peak, take the specifications of your GPU in your number format; on your own machine, the demo takes the fastest matrix multiplication of your device instead, as in [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en), which comes out a little lower than the specifications.

        **Step 3: the profiler.** `torch.profiler.profile` records every operation PyTorch runs, with its time; `key_averages()` adds them up per operation. An operation's *self* time leaves out the operations it calls, so that no time counts twice. Most names start with `aten::`, PyTorch's library of operations: `aten::mm`, `aten::addmm` and `aten::bmm` are matrix multiplications, and most of the others are small operations that read and write every number once, which are memory-bound. On an NVIDIA GPU, Nsight Systems shows the same on a timeline, with the CPU and the GPU side by side.

        **Step 4: what to change.** If the matrix multiplications take most of the time, the GPU works on what it is good at. If small operations take most of it, a bigger batch or longer windows give each operation more work, mixed precision halves the bytes, and fusing operations, so that the data is read once for several of them, removes trips to memory: the next lesson does that. If the GPU waits for the CPU, look for steps that wait on purpose. Measure again after each change.

        The picture: three training steps on a GPU, as Python and the GPU see them.
        """),
        mo.Html(C.queue_picture({"title": "Python runs ahead of the GPU", "time": "time", "python": "Python", "gpu": "GPU", "waits": "sync: waits",
                                 "early": "too early", "after": "after sync", "lang": L.lang,
                                 "alt": "Two rows over time. Above, Python puts steps 1, 2 and 3 in the queue in a moment each, then waits in a sync until the GPU is done. Below, the GPU works through steps 1, 2 and 3 one after the other, for far longer. A clock read right after step 3 stops too early; read after the sync, it stops when the GPU is done."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **throughput** | how much work a run finishes per second, such as tokens per second | |
        | **MFU** | model FLOPs utilization: the FLOPs per second the model needs, as a part of the peak | |
        | **HFU** | hardware FLOPs utilization: the same, with the FLOPs computed twice counted too | |
        | **profiler** | a tool that measures where a program spends its time, operation by operation | `torch.profiler` |
        """),
        L.watch_out("`loss.item()` and `print(loss)` also wait for the GPU: they need the number. A loop that reads the loss every step waits every step, and Python can no longer run ahead. Read it every hundred steps instead."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out how fast a run is, and why:

    1. Run a few steps untimed, wait for the device, and start the clock.
    2. Run a number of steps, wait for the device, and stop the clock: tokens divided by seconds.
    3. Multiply by the FLOPs per token and divide by the peak: the MFU.
    4. Profile a few steps, and see which operations take the time.
    5. Change the biggest one, and measure again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The profiler is new: ten training steps of a small MLP on the CPU, recorded, and every operation that took more than 5% of the time, with how often it ran.
    """)
    return


@app.cell
def example_profile():
    import torch
    from torch.profiler import profile

    torch.manual_seed(0)
    _hidden, _out = torch.nn.Linear(256, 1024), torch.nn.Linear(1024, 10)
    _opt = torch.optim.AdamW(list(_hidden.parameters()) + list(_out.parameters()))
    _x, _y = torch.randn(512, 256), torch.randint(0, 10, (512,))

    with profile() as _prof:
        for _ in range(10):
            _loss = torch.nn.functional.cross_entropy(_out(torch.relu(_hidden(_x))), _y)
            _opt.zero_grad()
            _loss.backward()
            _opt.step()

    _total = sum([_e.self_cpu_time_total for _e in _prof.key_averages()])
    for _e in _prof.key_averages():
        _share = _e.self_cpu_time_total / _total
        if _share > 0.05:
            print(_e.key, round(100 * _share), "%,", _e.count, "times")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The matrix multiplications take the biggest shares: `aten::addmm` in the forward pass, a multiplication with the bias added, 20 times for 2 layers in 10 steps, and `aten::mm` in the backward pass. The rest is spread over small operations: the ReLU, `aten::clamp_min`, and its backward pass, `aten::threshold_backward`; `aten::copy_`, which copies tensors; and AdamW's updates, such as `aten::sqrt`, which go over every parameter several times per step. `Optimizer.step#AdamW.step` is not an operation but a label: the time AdamW's `step` spends in Python itself, outside the operations it calls. How the shares split depends on your machine: with more CPU cores, the multiplications go faster, and the small operations weigh more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tps():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Tokens per second

        Write `tokens_per_second(run, n_tokens, skip, steps)`. `run.step()` starts one training step of `n_tokens` tokens, `run.sync()` waits until the device is done, and `run.clock()` gives the time in seconds. Run `skip` steps untimed, then time `steps` steps, and give back the tokens per second. The check plays a GPU: `run.step()` returns at once while the device works on, the first steps are slow, and the others differ a little from each other. Its clock is a clock of its own, so the check gives the same answer on any machine, and its queue never fills, unlike a real one. The check calls `tokens_per_second(run, 2048, 5, 10)`, which should give about 409,600, and tries other runs.
        """),
        L.task("Write `tokens_per_second(run, n_tokens, skip, steps)`: skip the first steps, and time the others with the device done at the start and at the end."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tps():
    box_tps = L.code_box("tokens-per-second", lines=10)
    box_tps
    return (box_tps,)


@app.cell(hide_code=True)
def check_tps(box_tps):
    result_tps = L.check("tokens-per-second", box_tps.value)
    result_tps
    return (result_tps,)


@app.cell(hide_code=True)
def hints_tps():
    L.hints(
        "tokens-per-second",
        [
            "Steps 1 and 2 of the recipe: two loops, with a `run.sync()` after each, and the clock read right after each `run.sync()`.",
        ],
        why="After the skipped steps, `run.sync()` makes sure they are done before the clock starts; after the timed steps, it makes sure they are done before the clock stops. The 10 timed steps take 0.05 seconds together: $2048 \\times 10 / 0.05 = 409{,}600$ tokens per second.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your GPT, measured and profiled

    The button runs your `tokens_per_second` on your `GPT` from [Assemble your GPT](/course/open/gpt-assemble?lang=en), training with AdamW on random tokens, on your GPU if you have one, else on the CPU. Where a module of yours is missing, the course's version stands in. On an NVIDIA GPU it trains the GPT of the book, mini-batches of 32 windows of 256 tokens; on a CPU or a Mac a smaller one, 4 blocks of $d = 128$ with mini-batches of 8 windows of 128 tokens. Then it measures the fastest matrix multiplication of your device as its peak, and profiles two steps. It takes a few seconds on a CPU. It works once exercise 1 is right.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Measure and profile my GPT")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(box_tps, measure_button, result_tps):
    if result_tps.state != "correct":
        _out = mo.md("*Finish exercise 1 first: this cell uses your `tokens_per_second`.*")
    elif not measure_button.value:
        _out = mo.md("*Press the button to measure.*")
    else:
        _m = C.demo(box_tps.value)
        if _m == "error":
            _out = mo.md("*The measurement stopped with an error: does your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en) still work?*")
        else:
            _n = lambda _x, _d=0: C.number(_x, _d, L.lang)
            _where = {"cuda": "your GPU", "mps": "your Mac's GPU", "cpu": "your CPU"}[_m["device"]]
            _stand = (" The course's version stood in for: " + ", ".join(f"`{_s}`" for _s in _m["stand_ins"]) + ".") if _m["stand_ins"] else ""
            _rows = "\n".join(f"| `{_k}` | {_n(100 * _s)}% | {_c} |" for _k, _s, _c in _m["top"])
            _out = mo.md(f"""
    On {_where}, your GPT of {_n(_m['params'])} parameters trained at **{_n(_m['tps'])} tokens per second**. At {_n(_m['per_token'] / 1e6, 1)} million FLOPs per token, that is {_n(_m['flops'] / 1e9, 1)} billion FLOP/s. The fastest matrix multiplication reached {_n(_m['peak'] / 1e9, 1)} billion FLOP/s: an MFU of **{_n(100 * _m['mfu'])}%**.{_stand}

    The operations that took the most time in two steps:

    | operation | share of the time | times |
    |---|---|---|
    {_rows}

    Measure twice before you trust one number: other programs, the temperature of the chip and the caches all play a part.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_hfu():
    mo.md(r"""
    ### Exercise 2. MFU and HFU
    """)
    return


@app.cell(hide_code=True)
def choice_hfu_cell():
    choice_hfu = L.choice(
        "hfu",
        question="To save memory, a run throws away the activations of every block in the forward pass, and computes the whole forward pass again during the backward pass. Its MFU is 30%. About what is its HFU?",
        options={
            "less": "About 22%",
            "same": "30%, the same",
            "right": "About 40%",
        },
        correct="right",
        why={
            "less": "The HFU counts more FLOPs than the MFU, never fewer: the work computed again comes on top.",
            "same": "The MFU leaves the second forward pass out, the HFU counts it: the HFU comes out higher.",
            "right": "Per token, the hardware computes $6N + 2N = 8N$ FLOPs while the model needs $6N$: the HFU is $30\\% \\times 8/6 = 40\\%$.",
        },
    )
    choice_hfu
    return (choice_hfu,)


@app.cell(hide_code=True)
def text_exercise_low():
    mo.md(r"""
    ### Exercise 3. A low MFU
    """)
    return


@app.cell(hide_code=True)
def explain_low():
    explain_low_cell = L.explain(
        "low-mfu",
        question="Your GPT trains on a GPU at an MFU of 8%. The profiler says the matrix multiplications take a fifth of the time. What do you try, and how do you know whether it helped?",
        model_answer="**What the profile says:** four fifths of the time go to other operations, most of them small ones that wait for memory, or to waiting. **What to try:** a bigger batch or longer windows, so that each multiplication has more work and the small operations weigh less; mixed precision, which halves the bytes of most operations; fusing small operations, so that the data is read once for several of them; and looking for waits, such as `loss.item()` every step or a data loader that cannot keep up. **How you know:** change one thing at a time, measure the tokens per second and the MFU again the same way, and profile again to see which operations now take the time.",
        points=[
            "you say that most of the time goes to small, memory-bound operations or to waiting",
            "you name at least two changes, such as a bigger batch, mixed precision, fusing operations or removing waits",
            "you measure again after each change, one change at a time",
        ],
    )
    explain_low_cell
    return (explain_low_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The estimate of [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en) now has a measured part of the peak, and the profiler shows the memory-bound operations of [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en) in your own GPT. The next lesson, *Write your own GPU kernel*, fuses such small operations into one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hfu, explain_low_cell, result_tps):
    L.wrap_up(result_tps, choice_hfu, explain_low_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
