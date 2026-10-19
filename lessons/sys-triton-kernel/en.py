# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-triton-kernel. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-triton-kernel", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en) and [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_small():
    quick_small = L.choice(
        "quick-small",
        question="In a profile, additions and ReLUs run thousands of times and take half the time of a step, although they do almost no FLOPs. What makes them slow?",
        options={
            "flops": "They do more FLOPs than the profiler shows",
            "right": "They read and write every number for about one FLOP each",
            "count": "The profiler counts their time twice",
        },
        correct="right",
        why={
            "flops": "An addition does one FLOP per number, and the profiler times the operation as it runs. See [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en).",
            "right": "They are memory-bound: the cores wait for the numbers to come from memory and go back.",
            "count": "The profiler counts the time of each operation once, by itself. See [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en).",
        },
    )
    quick_small
    return (quick_small,)


@app.cell(hide_code=True)
def quick_check_max():
    quick_max = L.choice(
        "quick-max",
        question="A stable softmax takes the largest number of a row off every number before it takes the powers of $e$. Why?",
        options={
            "sum": "So that the result adds up to 1",
            "fast": "So that it computes faster",
            "right": "So that the powers cannot overflow, while the result stays the same",
        },
        correct="right",
        why={
            "sum": "Dividing by the sum makes it add up to 1, with or without the shift. See [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).",
            "fast": "The shift is one subtraction more, not less work. See [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).",
            "right": "The largest power becomes $e^0 = 1$, and the factor $e^{-m}$ drops out when you divide by the sum.",
        },
    )
    quick_max
    return (quick_max,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The profiler showed small operations that wait for memory. Written as PyTorch operations, a softmax is five of them: the largest number of each row, a subtraction, the powers of $e$, their sum, and a division. Each one reads its input from the GPU's memory and writes its output back. A single program that does all five on a row, while the row stays on the chip, reads it once and writes it once. PyTorch's own `torch.softmax` is such a program. For a combination nobody has written one for, you write your own **GPU kernel**: in Triton, in Python.
    """)
    return


@app.cell(hide_code=True)
def predict_fused():
    guess_fused = L.guess(
        "fused-speed",
        question="Over a large matrix, the five operations of a softmax read the whole matrix 5 times and write it 3 times. One fused kernel reads it once and writes it once, and does the same FLOPs. On a GPU, where softmax waits for memory, how much faster can the fused kernel be?",
        options={
            "same": "Hardly faster: the FLOPs are the same",
            "right": "Up to about 4 times as fast",
            "hundred": "Up to about 100 times as fast",
        },
        reveal="**Up to about 4 times as fast.** A memory-bound operation takes about as long as its bytes take to move: 8 trips of the matrix against 2. It cannot get faster than reading the input once and writing the output once. The demo below measures it on your machine.",
    )
    guess_fused
    return (guess_fused,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: a kernel and its grid.** A GPU runs a kernel as many copies at once, each on its own piece of the data. In CUDA, NVIDIA's way to program its GPUs, the copies are threads, grouped in **thread blocks**, and a launch starts a grid of blocks. In Triton you write what one block does, on a whole block of numbers at once, and Triton divides that work over the threads. Each running copy is a program: `tl.program_id(0)` says which one it is, and the launch `kernel[grid](...)` says how many there are.

        **Step 2: pointers, places and a mask.** A tensor comes into a kernel as a pointer: the place in memory of its first number, as `data_ptr()` gives it. `x_ptr + offsets` are the places of a block of numbers, `tl.load` reads them and `tl.store` writes them. A block's size is a power of two that Triton knows when it compiles, a `tl.constexpr`: the def line says so with `BLOCK: tl.constexpr`. Where a block reaches past the end of the data, the mask turns places off: `tl.load(..., mask=mask, other=v)` gives `v` there, and `tl.store(..., mask=mask)` writes nothing there.

        **Step 3: fusion.** Between `tl.load` and `tl.store`, the numbers stay on the chip, and every operation in between costs no trip to memory. On the chip, `tl.max(x, axis=0)`, `tl.exp(x)` and `tl.sum(x, axis=0)` work on a block as `max`, `exp` and `sum` do on a NumPy array. Several operations in one kernel is **operator fusion**. A fused softmax gives each row its own program: it loads the row, takes its largest number off, takes the powers of $e$, divides them by their sum, and stores the row. Two trips instead of eight.

        **Step 4: without an NVIDIA GPU.** With `TRITON_INTERPRET=1`, Triton runs a kernel in Python on the CPU: right, but slow. This lesson always runs the worked example and the checks that way, on CPU tensors, so that every check gives the same answer on any computer with Triton. To measure a kernel's speed on a GPU, time it as in the last lesson, and divide the bytes it reads and writes by the time: GB a second, to compare with the bandwidth of the GPU's memory.

        The picture: the trips to memory of a softmax, as five operations and as one kernel.
        """),
        mo.Html(C.traffic_picture({"title": "Trips to memory", "memory": "GPU memory", "five": "5 reads, 3 writes",
                                   "fused": "one kernel", "one": "1 read, 1 write", "lang": L.lang,
                                   "alt": "Above, five boxes for max, subtract, exp, sum and divide, each with an arrow from the GPU's memory, and three with an arrow back: 5 reads and 3 writes. Below, one box for the fused kernel, with one arrow from memory and one back."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **GPU kernel** | a small function that runs on the GPU in many copies at once, each on its own piece of the data | `@triton.jit` |
        | **thread block** | in CUDA, a group of threads that run together; in Triton, one program, which works on a block of data | `tl.program_id(0)` |
        | **operator fusion** | several operations in one kernel, so that the data comes from memory once | |
        | **CUDA** | NVIDIA's platform for programming its GPUs; in PyTorch, an NVIDIA GPU is the device `"cuda"` | |
        | **Triton** | a language with a compiler, used from Python, for writing GPU kernels | `import triton` |
        """),
        L.watch_out("A kernel can read and write any place in memory, even past the end of a tensor: nothing stops it. Without the mask, a kernel reads numbers that are not its own, or damages them, and may crash the program."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To write a kernel that works on rows:

    1. Give each row a program: `row = tl.program_id(0)`, and a launch with one program per row.
    2. Make the places of the row, `row * n_cols + tl.arange(0, BLOCK)`, and a mask for the places past its end.
    3. Load the row once, with the mask, and an `other` that changes nothing in what follows.
    4. Compute on the chip, and store once, with the mask.
    5. Check the result against PyTorch, then measure its speed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A Triton kernel is new: an addition and a ReLU fused, `relu(x + y)`, over vectors of 1,000 numbers, in blocks of 256. Without Triton, on macOS or Windows without WSL2, this cell stops at `import triton`; read along.
    """)
    return


@app.cell
def example_add_relu():
    import torch
    import triton
    import triton.language as tl


    @triton.jit
    def add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        start = tl.program_id(0) * BLOCK
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, tl.maximum(x + y, 0.0), mask=mask)


    torch.manual_seed(0)
    _x, _y = torch.randn(1000), torch.randn(1000)
    _out = torch.empty_like(_x)
    add_relu_kernel[(triton.cdiv(1000, 256),)](_x, _y, _out, 1000, BLOCK=256)
    print("programs:", triton.cdiv(1000, 256))
    print("same as PyTorch:", torch.equal(_out, torch.relu(_x + _y)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `torch.empty_like(_x)` makes a tensor of the same shape for the result, without filling in its numbers: the kernel writes them. `triton.cdiv(1000, 256)` divides and rounds up: 4 programs. Program 3 starts at place 768, and its block reaches to 1,023; the mask turns off the 24 places from 1,000 on. Each program loads its piece of `x` and of `y`, adds them and takes the ReLU on the chip, and stores the result: `x + y` never goes to memory, where PyTorch's `torch.relu(x + y)` writes it and reads it back. `torch.equal` says whether two tensors are exactly the same: the result is the same as PyTorch's, to the last bit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kernel():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A fused softmax

        Complete `softmax_kernel`: the program of one row of `x`, with `n_cols` numbers, that writes the softmax of that row to the same row of `out`. The function `softmax` under it launches one program per row, with `BLOCK` the power of two from `triton.next_power_of_2(n_cols)`. The check launches your kernel with its own copy of `softmax`, which puts a fence around the tensors, to see whether your kernel stays inside them. It calls `softmax(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))`, which should give about `[[0.09, 0.2447, 0.6652], [0.3333, 0.3333, 0.3333]]`, and tries rows of other lengths and numbers of a thousand.
        """),
        L.task("Complete `softmax_kernel`: load the row once with the mask, compute its softmax stably, and store it once with the mask."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kernel():
    box_kernel = L.code_box("softmax-kernel", lines=22)
    box_kernel
    return (box_kernel,)


@app.cell(hide_code=True)
def check_kernel(box_kernel):
    result_kernel = L.check("softmax-kernel", box_kernel.value)
    result_kernel
    return (result_kernel,)


@app.cell(hide_code=True)
def hints_kernel():
    L.hints(
        "softmax-kernel",
        [
            "Steps 2 to 4 of the recipe, with the worked example as the form. For `other`: which number, at the places past the end of the row, changes neither the largest number nor the sum of the powers?",
        ],
        why="`float(\"-inf\")` past the end never wins the max, and its power, $e^{-\\infty} = 0$, adds nothing to the sum. Taking the max off keeps `tl.exp` from overflowing, and `tl.max` and `tl.sum` with `axis=0` work along the block.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Three softmaxes, timed

    The button times the softmax over the rows of a matrix of 1,024 columns, more than the caches of your chip hold: 8,192 rows, 32 MB, on a CPU, and 65,536 rows, 256 MB, on an NVIDIA GPU: as the five PyTorch operations, as PyTorch's own `torch.softmax`, and as your kernel, compiled for your GPU if you have an NVIDIA GPU. On a CPU, Triton's interpreter runs your kernel in Python, so it gets a smaller matrix, 64 × 1,024, and its time says nothing about a GPU. The GB a second count one read and one write of the matrix. It takes a few seconds; it works once exercise 1 is right.
    """)
    return


@app.cell(hide_code=True)
def time_button_cell():
    time_button = mo.ui.run_button(label="Time the three softmaxes")
    time_button
    return (time_button,)


@app.cell(hide_code=True)
def demo_time(box_kernel, result_kernel, time_button):
    if result_kernel.state != "correct":
        _out = mo.md("*Finish exercise 1 first: this cell uses your kernel.*")
    elif not time_button.value:
        _out = mo.md("*Press the button to time.*")
    else:
        _m = C.demo(box_kernel.value)
        if _m == "error":
            _out = mo.md("*Your kernel stopped with an error on this machine.*")
        else:
            _n = lambda _x, _d=1: C.number(_x, _d, L.lang)
            _names = {"unfused": "five PyTorch operations", "torch": "`torch.softmax`", "yours": "your kernel"}
            _rows = "\n".join(f"| {_names[_k]} | {_n(_s * 1000, 2)} ms | {_n(_g / _s, 1 if _g / _s >= 1 else 3)} GB/s |" for _k, _s, _g in _m["rows"])
            _speed = _m["rows"][0][1] / _m["rows"][1][1]
            _where = "your GPU" if _m["device"] == "cuda" else "your CPU"
            _note = ("" if _m["device"] == "cuda" else
                     f" Your kernel ran in the interpreter on {_m['small'][0]} × {C.number(_m['small'][1], 0, L.lang)} numbers: its time shows how slow Python is, not your kernel on a GPU.")
            _out = mo.md(f"""
    On {_where}:

    | softmax | time | GB/s |
    |---|---|---|
    {_rows}

    The fused `torch.softmax` is {_n(_speed)} times as fast as the five operations.{_note} Your kernel gives the same as `torch.softmax`, {C.agreement(_m['error'], L.lang)}.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_grid():
    mo.md(r"""
    ### Exercise 2. Programs and blocks
    """)
    return


@app.cell(hide_code=True)
def choice_grid_cell():
    choice_grid = L.choice(
        "grid",
        question="Your `softmax` runs on a matrix of 512 rows of 3,000 numbers. How many programs does the launch start, and how many places of each block does the mask turn off?",
        options={
            "cols": "3,000 programs, and no places",
            "right": "512 programs, and 1,096 places",
            "exact": "512 programs, and no places: the block is 3,000",
        },
        correct="right",
        why={
            "cols": "The launch starts one program per row: `(n_rows,)`, 512. Each program works on a whole row.",
            "exact": "A block's size has to be a power of two: `triton.next_power_of_2(3000)` is 4,096.",
            "right": "One program per row, and a block of 4,096 places, of which $4{,}096 - 3{,}000 = 1{,}096$ lie past the end of the row.",
        },
    )
    choice_grid
    return (choice_grid,)


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. Why fused is faster
    """)
    return


@app.cell(hide_code=True)
def explain_why_fused():
    explain_fused_cell = L.explain(
        "why-fused",
        question="Your fused softmax does the same FLOPs as the five PyTorch operations, yet on a GPU it is faster. Why? And why can it not get much faster than about four times?",
        model_answer="**Why faster:** softmax is memory-bound: its time goes to moving the numbers between the GPU's memory and the chip, not to computing. The five operations read the matrix 5 times and write it 3 times; the fused kernel reads it once and writes it once, and keeps it on the chip in between: a quarter of the bytes. **Why not more:** every softmax has to read its input once and write its output once. At that point the kernel moves as many bytes a second as the memory allows, and only a faster memory makes it faster.",
        points=[
            "you say that softmax is memory-bound: its time goes to moving numbers, not to FLOPs",
            "you compare the trips to memory: 8 for the five operations, 2 for the fused kernel",
            "you say that reading once and writing once is the least any softmax needs, so the bandwidth sets the limit",
        ],
    )
    explain_fused_cell
    return (explain_fused_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The memory-bound operations of [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en), found with the profiler of [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en), are now fused into one kernel, with the stable softmax of [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en) inside it. The next lesson, *RMSNorm*, starts on the parts of a modern GPT.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_grid, explain_fused_cell, result_kernel):
    L.wrap_up(result_kernel, choice_grid, explain_fused_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
