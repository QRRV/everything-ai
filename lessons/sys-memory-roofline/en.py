# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-memory-roofline. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-memory-roofline", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Also from outside the course:** two words. A **thread** is one of the many copies of the same small program that a GPU runs at the same time, each on its own numbers. A **cache** is a small, fast memory that keeps a copy of recently used data, so that it does not have to come from the slower memory again. If either is new, read [Thread (computing)](https://en.wikipedia.org/wiki/Thread_(computing)) or [Cache (computing)](https://en.wikipedia.org/wiki/Cache_(computing)) first.
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en) and [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_device():
    quick_device = L.choice(
        "quick-device",
        question="Your model is on the GPU, and a batch is still on the CPU. What happens when you call `model(batch)`?",
        options={
            "copy": "PyTorch copies the batch to the GPU for you",
            "right": "PyTorch stops with an error",
            "cpu": "The model runs on the CPU this time",
        },
        correct="right",
        why={
            "copy": "PyTorch never moves a tensor by itself: you move the batch with `.to(device)`. See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "cpu": "The weights stay where they are, on the GPU, and they cannot meet a batch on the CPU. See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "right": "The weights and the batch must be on the same device. Moving data between the two is slow, which is part of this lesson.",
        },
    )
    quick_device
    return (quick_device,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="A GPT has 2 million parameters. About how many FLOPs does a forward pass over 1,000 tokens take?",
        options={
            "params": "About 2 million",
            "right": "About 4 billion",
            "train": "About 12 billion",
        },
        correct="right",
        why={
            "params": "That is the number of parameters. Each token costs about 2 FLOPs per parameter. See [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en).",
            "train": "That is a training step, forward and backward, about 6 FLOPs per parameter per token. See [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en).",
            "right": "About 2 FLOPs per parameter per token: $2 \\times 2{,}000{,}000 \\times 1{,}000 = 4 \\times 10^9$.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The specifications of a GPU give two speeds. An NVIDIA RTX 4090 does 82.6 trillion FLOPs a second on float32 numbers, FP32 for short, with its ordinary cores: 82.6 TFLOP/s. It reads its memory at 1.008 trillion bytes a second, 1.008 TB/s. To keep its cores busy, it has to do about 82 FLOPs for every byte it reads. Many operations in a neural network do far fewer: they spend most of their time waiting for data. Whether an operation waits for memory or for calculations decides what makes it faster: fewer bytes, or fewer FLOPs.
    """)
    return


@app.cell(hide_code=True)
def predict_add():
    guess_add = L.guess(
        "add-vs-matmul",
        question="On an RTX 4090, you add two arrays of 100 million float32 numbers, and you multiply two 4096 × 4096 matrices of float32, which takes about 1,400 times as many FLOPs. How long does the addition take, compared with the multiplication, by the GPU's specifications?",
        options={
            "thousandth": "About a thousandth as long",
            "tenth": "About a tenth as long",
            "right": "About as long",
        },
        reveal="**About as long: 1.2 ms against 1.7 ms.** The addition does $10^8$ FLOPs, but it moves 1.2 GB: two arrays read and one written, 4 bytes per number. At 1.008 TB/s that takes 1.2 ms. The multiplication does $2 \\cdot 4096^3 \\approx 1.4 \\times 10^{11}$ FLOPs, 1.7 ms at 82.6 TFLOP/s, and moves only 0.2 GB. These are estimates from the specifications; real code reaches somewhat less than the peak.",
    )
    guess_add
    return (guess_add,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: the memory hierarchy.** A GPU keeps data at several levels, from small and fast to large and slow:

        | Level | Size on an RTX 4090 | What it holds |
        |---|---|---|
        | registers | up to about 1 KB per thread | the numbers a calculation works on right now |
        | shared memory and L1 cache | 128 KB per streaming multiprocessor, the group of cores that shares it | data that a group of threads reuses |
        | L2 cache | 72 MB for the whole chip | recently used data |
        | GPU memory | 24 GB, read at about 1 TB/s | the weights, the activations, the batch |
        | host memory | the computer's own RAM, over the PCIe connection at tens of GB/s | what is not on the GPU yet |

        This lesson counts the traffic between GPU memory and the chip: every number an operation reads from GPU memory or writes back.

        **Step 2: arithmetic intensity.** The **arithmetic intensity** of an operation is its number of FLOPs divided by the number of bytes it moves. Adding two float32 arrays does 1 FLOP per element and moves 12 bytes: an intensity of $1/12$. Multiplying two $n \times n$ matrices does $2n^3$ FLOPs and moves $3 n^2 \cdot 4$ bytes, if each number travels once: an intensity of $n/6$, about 683 for $n = 4096$.

        **Step 3: the roofline.** An operation cannot calculate faster than the peak, and it cannot calculate faster than its data arrives: intensity times bandwidth. So the speed it can reach is
        """),
        L.formula(r"\text{FLOP/s} = \min(\text{peak},\ \text{intensity} \times \text{bandwidth})"),
        mo.md(r"""
        On log-log axes that is a slanted roof that turns flat: the **roofline**. Where the two meet, the **ridge point**, lies at peak divided by bandwidth: about 82 FLOPs per byte for the RTX 4090. An operation to the left of it is **memory-bound**, to the right **compute-bound**. Its time by this estimate is the slower of the two parts, if the GPU calculates while data moves:
        """),
        L.formula(r"t = \max\left(\frac{\text{FLOPs}}{\text{peak}},\ \frac{\text{bytes}}{\text{bandwidth}}\right)"),
        mo.md(r"""
        **Step 4: your GPT.** Element-wise operations, such as ReLU and the addition of a residual connection, and also LayerNorm and softmax, do a few FLOPs per number they read: memory-bound. A linear layer reads its weights once per batch and uses them for every token in it, so its intensity grows with the number of tokens. It does not grow forever: the inputs and outputs of every token move too, so for a $d \times d$ layer the intensity levels off at about $d/4$. A big layer with a big batch is compute-bound. Your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en), with $d = 64$, stays far left of the ridge point of an RTX 4090 whatever the batch: $d/4 = 16$. With $d = 384$, as in the GPT of 10.8 million parameters, $d/4 = 96$ lies just right of it. And when a model generates one token at a time, even the linear layers of a big model are memory-bound.

        The picture: the roofline of the RTX 4090 at FP32 from its specifications, with the two operations of the guess.
        """),
        mo.Html(C.roof_picture({"title": "RTX 4090, FP32", "x": "FLOPs per byte", "y": "FLOP/s", "add": "add", "matmul": "matmul 4096", "ridge": "ridge 82", "lang": L.lang,
                                "alt": "Log-log axes: FLOP/s against FLOPs per byte. The roof rises along a slanted line and turns flat at 82.6 trillion FLOP/s, at the ridge point of 82 FLOPs per byte. The addition lies far left on the slanted part, at 1/12 FLOP per byte and 84 billion FLOP/s; the matrix multiplication lies on the flat part, at 683 FLOPs per byte."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **FLOP/s** | FLOPs per second: how fast a chip calculates, where FLOPs count how much there is to calculate |
        | **memory hierarchy** | the levels where a computer keeps data, from small and fast, such as registers, to large and slow, such as GPU memory |
        | **bandwidth** | how many bytes a second can move between two levels, such as from GPU memory to the chip |
        | **arithmetic intensity** | the FLOPs of an operation divided by the bytes it moves |
        | **roofline** | the highest FLOP/s an operation can reach at each intensity: the bandwidth times the intensity, at most the peak |
        | **ridge point** | the intensity where the two roofs meet: peak divided by bandwidth |
        | **memory-bound**, **compute-bound** | limited by moving data, or by calculating |
        """),
        L.watch_out("The roofline is an ideal. It assumes that calculating and moving data overlap completely, that each number travels between GPU memory and the chip only once, and that the operation reaches the peak and the full bandwidth. Real code reaches less, and small operations far less: starting each operation on a GPU also takes a few microseconds."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Counting bytes is new: the roofline estimate of the 4096 × 4096 matrix multiplication on the RTX 4090.
    """)
    return


@app.cell
def example_roofline():
    peak, bandwidth = 82.6e12, 1008e9  # RTX 4090, FP32, from its specifications
    n = 4096
    flops = 2 * n**3  # n * n results, each n multiplications and n additions
    bytes_moved = 3 * n**2 * 4  # A and B read once, C written once, 4 bytes per float32
    print("intensity:", flops / bytes_moved, " ridge point:", peak / bandwidth)
    print("compute:", flops / peak, "s  memory:", bytes_moved / bandwidth, "s")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The intensity, 683, lies far right of the ridge point, 82: the calculations take 1.7 ms and the data 0.2 ms, so the multiplication is compute-bound, and the estimate is 1.7 ms. With $n = 64$ the intensity is about 11, left of the ridge: a small matrix multiplication is memory-bound too.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_time():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The roofline estimate

        Write `roofline_time(flops, bytes_moved, peak_flops, bandwidth)`: the time an operation takes by the roofline estimate, in seconds, and what limits it, `"compute"` or `"memory"`, as `[time, limit]`. The check calls `roofline_time(1e8, 1.2e9, 82.6e12, 1008e9)`, the addition of the guess, which should give about `[0.00119, "memory"]`, and tries other operations and GPUs.
        """),
        L.task("Write `roofline_time(flops, bytes_moved, peak_flops, bandwidth)`: give back the time by the roofline estimate and `\"compute\"` or `\"memory\"`, whichever limits it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_time():
    box_time = L.code_box("roofline-time", lines=8)
    box_time
    return (box_time,)


@app.cell(hide_code=True)
def check_time(box_time):
    result_time = L.check("roofline-time", box_time.value)
    result_time
    return (result_time,)


@app.cell(hide_code=True)
def hints_time():
    L.hints(
        "roofline-time",
        [
            "Time is work divided by speed, once for the FLOPs and once for the bytes. The operation is done when the slower of the two is done.",
        ],
        why="```python\ndef roofline_time(flops, bytes_moved, peak_flops, bandwidth):\n    t_compute = flops / peak_flops\n    t_memory = bytes_moved / bandwidth\n    if t_compute >= t_memory:\n        return [t_compute, \"compute\"]\n    return [t_memory, \"memory\"]\n```\n\nThe GPU calculates while it moves data, so the two times overlap and the larger one sets the pace. Which of the two is larger says which limit the operation hits.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your own machine

    The button measures your machine: matrix multiplications from 64 × 64 to 2048 × 2048, and the addition of two arrays of 32 million numbers, on your GPU if you have one, else on the CPU. It takes a few seconds. The picture shows each measurement at its intensity, with a roof drawn from your fastest multiplication and the bandwidth of your addition. On a CPU the numbers are lower, and the shape is the same.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Measure my machine")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(measure_button):
    if not measure_button.value:
        _out = mo.md("*Press the button to measure your machine.*")
    else:
        _m = C.measure(lambda _i, _n: mo.output.replace(mo.md(f"*Measuring {_i + 1} of {_n}*")))
        _peak = max(_f for _n, _x, _f in _m["matmul"])
        _bw = _m["add"][2]
        _points = [(_x, _f, f"{_n}" if _n in (64, 2048) else "") for _n, _x, _f in _m["matmul"]] + [(_m["add"][0], _m["add"][1], "add")]
        _where = {"cuda": "your GPU", "mps": "your Mac's GPU", "cpu": "your CPU"}[_m["device"]]
        _ridge = C.number(_peak / _bw, 0 if _peak / _bw >= 10 else 1, L.lang)
        _out = mo.vstack([
            mo.Html(C.roof_picture({"title": f"Measured on {_where}", "x": "FLOPs per byte", "y": "FLOP/s", "ridge": f"ridge {_ridge}", "lang": L.lang,
                                    "alt": f"Log-log axes with your measurements: matrix multiplications from 64 to 2048, and the addition. Your fastest multiplication reached {C.number(_peak / 1e9, 0, L.lang)} billion FLOP/s; the addition moved {C.number(_bw / 1e9, 0, L.lang)} GB a second."},
                                   points=_points, peak=_peak, bandwidth=_bw)),
            mo.md(f"On {_where}, the fastest matrix multiplication reached {C.number(_peak / 1e9, 0, L.lang)} billion FLOP/s, and the addition moved {C.number(_bw / 1e9, 0, L.lang)} GB a second: a ridge point at about {_ridge} FLOPs per byte. Points below the roof lose time elsewhere: starting each operation takes time, which weighs most for small ones, and caches, other programs and the temperature of the chip all play a part. Measure twice before you trust one number."),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_bound():
    mo.md(r"""
    ### Exercise 2. Which part waits for memory?
    """)
    return


@app.cell(hide_code=True)
def choice_bound_cell():
    choice_bound = L.choice(
        "memory-bound",
        question="A GPT with the sizes of the smallest GPT-2, $d = 768$, MLPs that take every vector to 3,072 numbers and back, and a vocabulary of 50,257 tokens, trains on an RTX 4090 with batches of 8 windows of 1,024 tokens. Which part is most likely memory-bound?",
        options={
            "mlp": "The matrix multiplications of the MLPs",
            "right": "LayerNorm",
            "head": "The output layer, from $d$ to all tokens of the vocabulary",
        },
        correct="right",
        why={
            "mlp": "Their weights are read once per batch and used for 8,192 tokens: about 290 FLOPs per byte, right of the ridge point, so compute-bound.",
            "head": "That is a big matrix multiplication too, with its weights used for every token of the batch: about 350 FLOPs per byte, compute-bound.",
            "right": "LayerNorm reads every number, does a handful of FLOPs on it, and writes it back: an intensity of about one FLOP per byte, far left of the ridge point.",
        },
    )
    choice_bound
    return (choice_bound,)


@app.cell(hide_code=True)
def text_exercise_batch():
    mo.md(r"""
    ### Exercise 3. Bigger batches
    """)
    return


@app.cell(hide_code=True)
def explain_batch():
    explain_batch_cell = L.explain(
        "batch",
        question="A linear layer with a $d \\times d$ weight matrix runs on a batch of $B$ tokens. Why does the time per token drop when $B$ grows, and why does it stop dropping after a point? Use the arithmetic intensity.",
        model_answer="**FLOPs:** about $2 B d^2$, growing with $B$. **Bytes:** the weights, $d^2$ numbers, once per batch, plus the inputs and outputs, $2 B d$ numbers, at 4 bytes each. For small $B$ the weights dominate the bytes, so the intensity is about $2 B d^2 / (4 d^2) = B/2$: it grows with the batch. The layer is memory-bound, its time is mostly reading the weights, and that time is shared by more tokens as $B$ grows. For large $B$ the inputs and outputs dominate, and the intensity levels off at about $2 B d^2 / (8 B d) = d/4$. If $d/4$ lies past the ridge point, the layer becomes compute-bound, each extra token adds its own FLOPs, and the time per token stays about the same; if not, as with $d = 64$ on an RTX 4090, it stays memory-bound and the time per token levels off anyway, because each token then brings its own bytes.",
        points=[
            "you say that the weights are read once per batch, so the bytes grow much more slowly than the FLOPs",
            "you say that the intensity grows with the batch, so a memory-bound layer gets faster per token",
            "you say that the intensity levels off, at about $d/4$, so the time per token stops dropping: compute-bound if that lies past the ridge point, still memory-bound if not",
        ],
    )
    explain_batch_cell
    return (explain_batch_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The FLOPs come from [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en), the GPU from [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en). The next lesson, *Fewer bits: FP32, FP16 and BF16*, halves the bytes of every number: for a memory-bound operation, that is up to twice the speed.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bound, explain_batch_cell, result_time):
    L.wrap_up(result_time, choice_bound, explain_batch_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
