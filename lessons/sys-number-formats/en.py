# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-number-formats. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-number-formats", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Also from outside the course:** binary numbers: a **bit** is a 0 or a 1, $n$ bits make $2^n$ different patterns, and binary digits after the point count halves, quarters, eighths and so on, so 1.1 in binary is 1.5. If that is rusty, read [Binary number](https://en.wikipedia.org/wiki/Binary_number) first.
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en) and [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_overflow():
    quick_overflow = L.choice(
        "quick-overflow",
        question="A float stores a number as a few significant digits times a power of 2. What happens to a result that needs a bigger power of 2 than the format has room for?",
        options={
            "wrap": "It wraps around to a negative number",
            "right": "It becomes `inf`",
            "largest": "It is rounded to the largest number the format can store",
        },
        correct="right",
        why={
            "wrap": "Whole numbers in some languages wrap around; floats do not. See [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).",
            "largest": "A float has no room for it, and becomes `inf`: an overflow. See [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).",
            "right": "That is an overflow. With fewer bits for the power, it happens much sooner, as this lesson shows.",
        },
    )
    quick_overflow
    return (quick_overflow,)


@app.cell(hide_code=True)
def quick_check_halve():
    quick_halve = L.choice(
        "quick-halve",
        question="An operation is memory-bound. You halve the bytes of every number it reads and writes. By the roofline estimate, what happens to its time?",
        options={
            "same": "It stays the same: the FLOPs do not change",
            "right": "It about halves",
            "quarter": "It drops to about a quarter",
        },
        correct="right",
        why={
            "same": "A memory-bound operation waits for its bytes, not for its FLOPs. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "quarter": "Its time is its bytes divided by the bandwidth: half the bytes, half the time. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "right": "Its time is bytes divided by bandwidth, as long as it stays memory-bound. That is the first reason to use fewer bits.",
        },
    )
    quick_halve
    return (quick_halve,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    PyTorch stores every float in 32 bits unless you ask for less. Half the bits means half the memory, half the bytes to move, and on modern GPUs a different unit to calculate with: the **tensor cores**, which multiply small blocks of matrices in 16-bit formats far faster. An NVIDIA A100 does 19.5 trillion FP32 operations a second on its ordinary cores, and 312 trillion in BF16 on its tensor cores. That is why large models do most of their training calculations in 16 bits. But fewer bits leave less room: for big and small numbers, or for digits. Which bits a format gives up decides what goes wrong.
    """)
    return


@app.cell(hide_code=True)
def predict_store():
    guess_store = L.guess(
        "seventy-thousand",
        question="You store the number 70,000 in two 16-bit formats, FP16 and BF16. What comes out?",
        options={
            "both": "70,000 in both",
            "right": "`inf` in FP16, close to 70,000 in BF16",
            "swap": "Close to 70,000 in FP16, `inf` in BF16",
        },
        reveal="**`inf` in FP16, 70,144 in BF16.** FP16 has 5 bits for the power of 2, and its largest number is 65,504: 70,000 overflows. BF16 has 8 bits for the power, as many as FP32, so 70,000 fits, but only 7 bits for the digits: near 70,000 its numbers lie 512 apart, and 70,000 is rounded to the nearest one, 70,144.",
    )
    guess_store
    return (guess_store,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: three parts.** A floating-point number is a **sign**, a power of 2 and a number of significant digits, in binary: $\pm\, 1.b_1 b_2 \ldots b_m \times 2^{e}$. The **exponent** bits store $e$ and decide the **range**: how big and how small a number can get. The **mantissa** bits store the $m$ binary digits after the point, and decide the **numerical precision**, how fine the steps are: between two powers of 2, $2^e$ and $2^{e+1}$, $m$ mantissa bits make $2^m$ equal steps. That is not the precision of a classifier. Every number is rounded to the nearest number the format can store.

        | Format | Bits | Sign, exponent, mantissa | Largest number | Step near 1 |
        |---|---|---|---|---|
        | **FP32** | 32 | 1, 8, 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-7}$ |
        | **TF32** | 19, inside tensor cores | 1, 8, 10 | $3.4 \times 10^{38}$ | $9.8 \times 10^{-4}$ |
        | **BF16** | 16 | 1, 8, 7 | $3.4 \times 10^{38}$ | $7.8 \times 10^{-3}$ |
        | **FP16** | 16 | 1, 5, 10 | 65,504 | $9.8 \times 10^{-4}$ |

        **Step 2: two ways to halve.** **FP16** keeps more digits, about 3 significant ones, but has only 5 exponent bits: above 65,504 a number overflows, below about $6 \times 10^{-5}$ it keeps fewer digits, and below about $3 \times 10^{-8}$ it becomes 0. Small gradients can vanish that way, which training in FP16 has to work around. **BF16**, brain floating point, keeps the 8 exponent bits of FP32 and gives up digits instead: it has the range of FP32 and about 2 significant digits. It rarely overflows, and it is the usual 16-bit format to train in today, on GPUs that support it.

        **Step 3: TF32.** The tensor cores of NVIDIA GPUs since 2020 can also multiply FP32 matrices with only 10 mantissa bits: **TF32**. The tensors stay FP32 in memory; only the multiplication inside the tensor core rounds to fewer digits. PyTorch allows it with `torch.set_float32_matmul_precision("high")`.

        **Step 4: whole numbers.** **INT8** and **INT4** store whole numbers, 256 and 16 different values, with one scale factor for a group of numbers. They hold the weights of a trained model in a quarter or an eighth of the memory of FP32; turning a model into them is quantization, in a later lesson.

        **Step 5: the size of a step.** Near a number $x$, the steps of a format are
        """),
        L.formula(r"\text{spacing}(x) = 2^{\lfloor \log_2 |x| \rfloor - m}"),
        mo.md(r"""
        with $m$ mantissa bits, for every $x$ that is not 0, not too big and not so small that it keeps fewer digits. Add a number smaller than half a step to $x$, and $x$ does not change at all: it rounds back to itself. Going down from a power of 2, the limit is half that, because just below $2^e$ the steps are those of the power below, half as big.

        The picture: the bits of each format, sign, exponent and mantissa.
        """),
        mo.Html(C.bits_picture({"title": "Where the bits go", "bits": "bits", "sign": "sign", "exp": "exponent", "man": "mantissa", "note_TF32": ", inside tensor cores",
                                "alt": "Four rows of boxes, one per bit. FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits. TF32: 1, 8 and 10, 19 bits, inside tensor cores. BF16: 1, 8 and 7, 16 bits. FP16: 1, 5 and 10, 16 bits."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **exponent**, **mantissa** | the bits of a float that store its power of 2, and its significant digits |
        | **range**, **numerical precision** | how big and how small a format's numbers can get, and how fine its steps are; not the precision of a classifier |
        | **FP32**, **FP16**, **BF16** | floating-point formats of 32 and 16 bits; BF16 has the range of FP32 with fewer digits |
        | **TF32** | the format in which tensor cores multiply FP32 matrices, with 10 mantissa bits |
        | **INT8**, **INT4** | whole-number formats of 8 and 4 bits, with a scale factor, for storing weights |
        | **tensor core** | a unit in an NVIDIA GPU that multiplies small blocks of matrices at once, fastest in 16 bits or fewer |
        | `torch.float16`, `torch.bfloat16` | the dtypes of FP16 and BF16 in PyTorch; `x.to(torch.bfloat16)` converts |
        """),
        L.watch_out("Older GPUs, and the CPU of many laptops, have no fast BF16: there it can even be slower than FP32. Check what your hardware supports before you switch."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Number formats in PyTorch are new: three numbers stored in FP32, FP16 and BF16.
    """)
    return


@app.cell
def example_formats():
    import torch

    x = torch.tensor([70000.0, 1e-8, 1.0 + 1 / 256])
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(dtype, x.to(dtype).tolist())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    FP32 keeps all three; only $10^{-8}$ is rounded, in its last digits. FP16 turns 70,000 into `inf` and $10^{-8}$ into 0: too big and too small for 5 exponent bits. BF16 keeps both, rounded, but turns $1 + 1/256$ into 1: its steps near 1 are $1/128$, so $1 + 1/256$ lies exactly halfway between 1 and $1 + 1/128$. A tie like that goes to the one of the two whose last mantissa bit is 0, here 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spacing():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How big is a step?

        Write `spacing(x, mantissa_bits)`: the step between the numbers a format with `mantissa_bits` mantissa bits can store, in the power of 2 that `abs(x)` lies in, as in step 5, for any `x` that is not 0. `math.log2` and `math.floor` are in Python's `math` module. The check calls `spacing(70000.0, 7)`, which should give 512, the steps of BF16 in the guess, and tries other numbers and formats.
        """),
        L.task("Write `spacing(x, mantissa_bits)`: the step between the numbers of a format with that many mantissa bits, in the power of 2 that `abs(x)` lies in."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spacing():
    box_spacing = L.code_box("spacing", lines=6)
    box_spacing
    return (box_spacing,)


@app.cell(hide_code=True)
def check_spacing(box_spacing):
    result_spacing = L.check("spacing", box_spacing.value)
    result_spacing
    return (result_spacing,)


@app.cell(hide_code=True)
def hints_spacing():
    L.hints(
        "spacing",
        [
            "First the power of 2 that `x` lies in: `math.floor(math.log2(abs(x)))`. That power, $2^p$ to $2^{p+1}$, is split into `2 ** mantissa_bits` equal steps.",
        ],
        why="```python\nimport math\n\n\ndef spacing(x, mantissa_bits):\n    power = math.floor(math.log2(abs(x)))\n    return 2.0 ** (power - mantissa_bits)\n```\n\nBetween $2^p$ and $2^{p+1}$ lies a stretch of length $2^p$, and the mantissa splits it into $2^m$ steps of $2^{p-m}$. The sign does not matter, hence `abs`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_format():
    mo.md(r"""
    ### Exercise 2. Which format?
    """)
    return


@app.cell(hide_code=True)
def choice_format_cell():
    choice_format = L.choice(
        "format",
        question="You train a large model on a recent GPU that supports BF16, and you want the matrix multiplications in fewer bits than FP32. Some activations reach a few hundred thousand, and some gradients are as small as $10^{-9}$. In which format do you do them?",
        options={
            "fp16": "FP16, because it keeps more digits",
            "right": "BF16, because it has the range of FP32",
            "int8": "INT8, because it needs the least memory",
        },
        correct="right",
        why={
            "fp16": "Its extra digits do not help when a few hundred thousand overflows past 65,504 and $10^{-9}$ becomes 0.",
            "int8": "INT8 has 256 whole numbers and one scale factor: stretched to reach a few hundred thousand, its steps are thousands wide, and $10^{-9}$ becomes 0. It is for storing the weights of a trained model, not for training.",
            "right": "With 8 exponent bits, both a few hundred thousand and $10^{-9}$ fit. Its steps are coarse, which the next lesson handles by keeping some numbers in FP32.",
        },
    )
    choice_format
    return (choice_format,)


@app.cell(hide_code=True)
def text_exercise_lost():
    mo.md(r"""
    ### Exercise 3. A lost update
    """)
    return


@app.cell(hide_code=True)
def explain_lost():
    explain_lost_cell = L.explain(
        "lost-update",
        question="A weight of 1.0 is stored in BF16. Each training step adds 0.001 to it, 1,000 steps in a row. What is the weight after those steps, and why? What would you change?",
        model_answer="**Still 1.0.** From 1 to 2, the steps of BF16 are $2^{0-7} = 1/128 \\approx 0.0078$. The update, 0.001, is smaller than half a step, 0.0039, so $1.0 + 0.001$ rounds back to 1.0, every step again. The weight should have reached 2.0, and it never moves. **Change:** keep a copy of the weights in FP32, whose steps near 1 are $1.2 \\times 10^{-7}$, and apply the updates to that copy; use BF16 only for the calculations. That is mixed-precision training, the next lesson.",
        points=[
            "you say that the weight stays 1.0",
            "you compare the update with the step of BF16 near 1, about 0.0078, and its half, about 0.004",
            "you propose keeping the weights in a format with finer steps, such as FP32, and applying the updates to them",
        ],
    )
    explain_lost_cell
    return (explain_lost_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Overflow, underflow and rounding come from [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en), the bytes that fewer bits save from [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en). The next lesson, *Mixed-precision training*, calculates in BF16 where that is safe and keeps the weights in FP32, so updates like the one in Exercise 3 are not lost.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_format, explain_lost_cell, result_spacing):
    L.wrap_up(result_spacing, choice_format, explain_lost_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
