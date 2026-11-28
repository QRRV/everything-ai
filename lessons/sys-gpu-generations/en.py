# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-gpu-generations. Route: course/route/lessen/systemen.yaml. Checks, the five GPUs, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-gpu-generations", "en")
    C = L.checks   # the published numbers of five GPUs, the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Close and fast, far and slow: the memory hierarchy](/course/open/sys-memory-hierarchy?lang=en) and [Moore's law and the growth of compute](/course/open/sys-moores-law?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bandwidth():
    quick_bandwidth = L.choice(
        "quick-bandwidth",
        question="What does the bandwidth of a memory, such as the HBM next to a GPU, tell you?",
        options={
            "hold": "How many bytes it can hold",
            "right": "How many bytes per second it can deliver",
            "first": "How long the first byte takes to arrive after a request",
            "flops": "How many FLOPs per second the GPU can do with the numbers that are in it",
        },
        correct="right",
        why={
            "right": "Reading takes the bytes divided by the bandwidth. In this lesson, you see how the bandwidth of GPU memory grew from one generation to the next, next to the FLOP/s.",
            "hold": "That is the size of the memory. What does the bandwidth count? See [Close and fast, far and slow: the memory hierarchy](/course/open/sys-memory-hierarchy?lang=en).",
            "first": "A bandwidth says how fast the bytes come once they flow, not how long the first one takes. What does it count? See [Close and fast, far and slow: the memory hierarchy](/course/open/sys-memory-hierarchy?lang=en).",
            "flops": "FLOP/s count the computing of the cores, not the memory. What does the bandwidth of a memory count? See [Close and fast, far and slow: the memory hierarchy](/course/open/sys-memory-hierarchy?lang=en).",
        },
    )
    quick_bandwidth
    return (quick_bandwidth,)


@app.cell(hide_code=True)
def quick_check_doubling():
    quick_doubling = L.choice(
        "quick-doubling",
        question="A number grows 8 times in 6 years, at the same pace every year. What is its doubling time?",
        options={
            "right": "2 years, since 8 times is three doublings",
            "divide": "0.75 years: 6 years divided by 8",
            "whole": "6 years, the whole period",
            "half": "3 years, since after half of the 6 years the number has doubled",
        },
        correct="right",
        why={
            "right": "8 is 2 × 2 × 2: three doublings in 6 years, one every 2 years. In this lesson, you compare how fast the FLOP/s and the bandwidth of GPUs doubled.",
            "divide": "Dividing the years by the growth does not count doublings. How many doublings make 8 times? See [Moore's law and the growth of compute](/course/open/sys-moores-law?lang=en).",
            "whole": "In 6 years the number grew 8 times, so it doubled more than once. How many doublings make 8 times? See [Moore's law and the growth of compute](/course/open/sys-moores-law?lang=en).",
            "half": "Growth at the same pace doubles in equal steps; after 3 years, the number has grown 2.8 times. How many doublings make 8 times? See [Moore's law and the growth of compute](/course/open/sys-moores-law?lang=en).",
        },
    )
    quick_doubling
    return (quick_doubling,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every two years or so, NVIDIA brings out a new GPU for servers, and AMD answers with one of its own. Each new GPU computes faster, holds more and reads its memory faster than the one before. But these numbers do not grow at the same pace, and the one that lags behind decides how fast many jobs run. In this lesson, you read the numbers of five GPUs from 2017 to 2024 as their makers published them, you compute what each step gained, and you work out how many GPUs a model needs.

    **An everyday example.** A printing press gets twice as fast every two years. The truck that brings its paper gets only a little faster. After a few years, the press stands still half of the time, waiting for paper.
    """)
    return


@app.cell(hide_code=True)
def predict_bandwidth():
    guess_bandwidth = L.guess(
        "bandwidth",
        question="From the V100 of 2017 to the B200 of 2024, NVIDIA's FLOP/s for numbers of 16 bits grew 18 times. How many times did the bandwidth of the GPU's own memory grow, do you think?",
        options={
            "pace": "About 18 times as well, at the same pace",
            "right": "About 9 times",
            "more": "About 40 times, since memory is simpler to make than computing cores",
            "none": "Hardly at all: about 1.5 times",
        },
        reveal="**About 9 times.** From 0.9 TB/s to 8 TB/s is 8.9 times: half the growth of the FLOP/s. In the time a V100 reads one byte, it can do 139 FLOPs; a B200 can do 281. Each new GPU computes faster, and waits longer for its numbers.",
    )
    guess_bandwidth
    return (guess_bandwidth,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **GPU generation.** NVIDIA designs its GPUs anew every two years or so, and names each design after a scientist. The GPUs of one design, from one maker, form a **GPU generation**: Volta with the V100 in 2017, Ampere with the A100 in 2020, Hopper with the H100 in 2022 and Blackwell with the B200 in 2024. AMD's MI300X of 2023 is of its CDNA 3 generation. The makers publish the numbers of each GPU in a datasheet:

        | GPU | Generation | Year | FLOP/s at 16 bits | VRAM | Bandwidth | Links |
        |---|---|---|---|---|---|---|
        | V100 | NVIDIA Volta | 2017 | 125 trillion | 32 GB HBM2 | 0.9 TB/s | 300 GB/s |
        | A100 | NVIDIA Ampere | 2020 | 312 trillion | 80 GB HBM2e | 2.04 TB/s | 600 GB/s |
        | H100 | NVIDIA Hopper | 2022 | 989 trillion | 80 GB HBM3 | 3.35 TB/s | 900 GB/s |
        | MI300X | AMD CDNA 3 | 2023 | 1,307 trillion | 192 GB HBM3 | 5.3 TB/s | 896 GB/s |
        | B200 | NVIDIA Blackwell | 2024 | 2,250 trillion | 180 GB HBM3e | 8 TB/s | 1,800 GB/s |

        The links are the connections from one GPU to the others in the same server: NVLink at NVIDIA, Infinity Fabric at AMD. The B200 is two chips in one package, joined so that they work as one GPU.

        **VRAM.** The memory of a GPU itself is its **VRAM**, for video RAM: the name comes from the time when GPUs only drew pictures. On a server GPU, the VRAM is the HBM next to the chip. The parameters of a model have to fit in it, or be spread over more GPUs. A byte is 8 bits, so a parameter of 16 bits takes 2 bytes: a model with 70 billion of them takes 140 GB, too much for one H100 of 80 GB, and it fits on one MI300X or B200. Running the model takes extra room, for the numbers it computes along the way.

        **Fewer bits.** Each generation added FLOP/s for numbers with fewer bits. The V100 computes fastest on numbers of 16 bits; the H100 does twice as many FLOP/s on numbers of 8 bits as on 16, and the B200 does 9,000 trillion on numbers of 4 bits, 72 times the V100 at 16 bits. Datasheets like to print the number for the fewest bits, and often one twice as large, marked "with sparsity": it counts on matrices in which half of the numbers are zero, in a fixed pattern. This lesson compares every GPU at 16 bits, without sparsity. What fewer bits cost comes in a later lesson, *How many bits per number: from FP32 to INT4*.

        **Computing against memory.** Divide the FLOP/s by the bandwidth, and you get the FLOPs a GPU can do in the time it reads one byte from its VRAM. A computation that does fewer FLOPs than that for each byte it reads waits for its memory. The picture sets every GPU against the V100: the dark dot is its FLOP/s, the open dot its bandwidth.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        From one generation to the next, the gap between the two dots grows: the memory wall of [Moore's law and the growth of compute](/course/open/sys-moores-law?lang=en), in the numbers of five GPUs.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **GPU generation** | | the GPUs of one design from one maker, such as NVIDIA's Hopper or Blackwell; each new one brings more FLOP/s, more and faster memory, and faster links | |
        | **VRAM** | "V-RAM" | the memory of a GPU itself; on a server GPU, the HBM next to the chip | |
        | FLOPs per byte | | the FLOP/s divided by the bandwidth: the FLOPs a GPU can do in the time it reads one byte | `flops / bandwidth` |
        """),
        L.watch_out("A datasheet gives the peak: the most the GPU can do, on the work that suits it best. A real job reaches a part of it, often well under half."),
        L.watch_out("Compare like with like: the same bits, and both numbers with or both without sparsity. NVIDIA prints 1,979 trillion FLOP/s for the H100 at 16 bits with sparsity; without, it is 989 trillion."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compare two GPUs, and to see whether a model fits:

    1. Take the same kind of number from both datasheets: the same bits, both with or both without sparsity.
    2. Divide the new number by the old one: how many times it grew.
    3. Do that for the FLOP/s and for the bandwidth. When the FLOP/s grow faster, a job that reads a lot and computes little gains less than the FLOP/s promise.
    4. To see whether a model fits, count its bytes: the parameters times the bytes of each. Divide by the VRAM of one GPU, and round up.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The FLOPs per byte of four NVIDIA GPUs: the FLOP/s at 16 bits divided by the bandwidth, both per second. Run the cell.
    """)
    return


@app.cell
def example_per_byte():
    _gpus = {"V100": (125e12, 0.9e12), "A100": (312e12, 2.039e12),
             "H100": (989.4e12, 3.35e12), "B200": (2250e12, 8e12)}
    for _name in _gpus:
        _flops, _bandwidth = _gpus[_name]          # FLOP/s at 16 bits, bytes per second
        print(_name, round(_flops / _bandwidth), "FLOPs per byte")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the time a V100 reads one byte, it can do 139 FLOPs; a B200 can do 281, about twice as many. The H100 has the most: from the H100 to the B200, the bandwidth grew a little faster than the FLOP/s at 16 bits. Exercise 1 computes the growth of every number at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gain():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What each step gained

        `gains(old, new)` gets the numbers of two GPUs, each a dict like `C.GPUS["V100"]`, and gives back a dict with how many times each number grew from `old` to `new`. Replace the `...` so that every round of the loop puts the number of `new` divided by the number of `old` for `key` in `out[key]`.

        The check calls `gains(C.GPUS["V100"], C.GPUS["B200"])`. `C.GPUS["V100"]` is `{"flops": 125, "bandwidth": 0.9, "vram": 32, "link": 300}`: trillions of FLOP/s at 16 bits, TB/s, GB and GB/s. Then other GPUs and other dicts.
        """),
        L.task("Make every round of the loop put the growth of `key` in `out`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gain():
    box_gain = L.code_box("gain", lines=6)
    box_gain
    return (box_gain,)


@app.cell(hide_code=True)
def check_gain(box_gain):
    result_gain = L.check("gain", box_gain.value)
    result_gain
    return (result_gain,)


@app.cell(hide_code=True)
def hints_gain():
    L.hints(
        "gain",
        [
            "In the loop, `key` is one key of `old`, such as `\"flops\"`, and `old[key]` is its old number. `out[key]` gets the number of `new` for the same key, divided by `old[key]`. What goes before the `/`?",
            "With `{\"a\": 2}` as `old` and `{\"a\": 6}` as `new`, the number of `\"a\"` grew 6 / 2 = 3 times.",
            "`        out[key] = ___ / old[key]`",
        ],
        why="`out[key] = new[key] / old[key]` divides the new number by the old one, for every key. From the V100 to the B200: the FLOP/s ×18, the bandwidth ×8.9, the VRAM ×5.6 and the links ×6.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Does the model fit?

        `gpus_needed(params, bits, vram_gb)` gets the number of parameters of a model, the bits of each parameter and the VRAM of one GPU in GB, and gives back how many of those GPUs it takes to hold all parameters. `math.floor(gb / vram_gb)` puts the GPUs that are filled completely in `count`, and the `if` adds one more for what is left. Replace the `...` so that `gb` is the gigabytes of all parameters: `params` times the bytes of one parameter, divided by `1e9`.

        The check calls `gpus_needed(70_000_000_000, 16, 80)`: 70 billion parameters of 16 bits, on H100s of 80 GB. Then other models and other GPUs; the picture shows two models on all five GPUs.
        """),
        L.task("Make `gb` the gigabytes of all parameters of the model."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit", lines=10)
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
            "`gb` is `params` times the bytes of one parameter, divided by `1e9`. A parameter has `bits` bits, and a byte is 8 bits: how many bytes is that?",
            "A parameter of 16 bits takes 2 bytes, so 70 billion of them take 140,000,000,000 bytes: 140 GB.",
            "`    gb = params * ___ / 1e9`",
        ],
        why="`gb = params * bits / 8 / 1e9` turns the bits of each parameter into bytes, and the bytes into gigabytes. 70 billion parameters of 16 bits are 140 GB: two H100s of 80 GB, or one MI300X or B200.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_helps():
    mo.md(r"""
    ### Exercise 3. What helps which job

    Eight jobs for a GPU. Put each one under the number that makes it faster or possible: *More FLOP/s*, *More bandwidth*, *More VRAM* or *Faster links*.
    """)
    return


@app.cell(hide_code=True)
def sort_helps_cell():
    sort_helps = L.sort("helps")
    sort_helps
    return (sort_helps,)


@app.cell(hide_code=True)
def check_helps(sort_helps):
    result_helps = L.check("helps", sort_helps)
    result_helps
    return (result_helps,)


@app.cell(hide_code=True)
def hints_helps():
    L.hints(
        "helps",
        [
            "Ask for each job what the GPU waits for: its cores, when every number it reads is used many times; its memory, when every number is used once; room for all the numbers; or the other GPUs.",
            "Two jobs go under each of the four.",
            "More FLOP/s: training on large batches, \\_\\_\\_. More bandwidth: one user, one token at a time, \\_\\_\\_. More VRAM: a model of 140 GB on one GPU, \\_\\_\\_. Faster links: training on 1,000 GPUs, \\_\\_\\_.",
        ],
        why="**More FLOP/s:** training on large batches, 256 users at once. **More bandwidth:** one user, one token at a time, the sum of a huge array. **More VRAM:** a model of 140 GB on one GPU, training a larger model on a single GPU. **Faster links:** training on 1,000 GPUs, a model split over eight GPUs.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Five GPUs

    Every GPU against the V100 of 2017, on a log scale: its FLOP/s at 16 bits, the bandwidth of its VRAM, its VRAM and its links. Choose a year.
    """)
    return


@app.cell(hide_code=True)
def demo_year():
    L.explore("year", "year", lambda _v: C.frame(_v, L.lang), start=2022, values=C.DEMO_VALUES, role="",
              task="Go from 2017 to 2024: which number grows fastest, and which lags behind?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From the V100 to the B200, the FLOP/s at 16 bits doubled about every 1.7 years, the bandwidth every 2.2 years and the VRAM every 2.8 years; the links grew 6 times in the seven years. So each new GPU waits longer for its numbers, unless the job uses every number it reads many times. The MI300X shows that a maker can choose: fewer FLOP/s than the B200, and more VRAM than any of the five.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_token():
    mo.md(r"""
    ### Exercise 4. One token at a time

    A model of 16 GB writes an answer for one user, one token at a time. For every token, the GPU reads all 16 GB from its VRAM, and does about one FLOP for each byte. On a V100, reading the 16 GB takes 17.8 milliseconds, and the FLOPs take 0.13 milliseconds.
    """)
    return


@app.cell(hide_code=True)
def choice_token_cell():
    choice_token = L.choice(
        "token",
        question="From the V100 to the B200, the FLOP/s grew 18 times and the bandwidth 8.9 times. About how much faster does the B200 write this answer?",
        options={
            "flops": "About 18 times, like the FLOP/s",
            "right": "About 9 times, like the bandwidth: reading the 16 GB takes most of the time",
            "both": "About 160 times: the gains in FLOP/s and bandwidth multiplied",
            "vram": "Not faster at all: the 16 GB fit in the VRAM of both GPUs, and the VRAM decides how fast an answer comes",
        },
        correct="right",
        why={
            "right": "Each token waits for 16 GB to be read: 17.8 milliseconds on a V100, 2 on a B200, 8.9 times faster. The FLOPs take a fraction of a millisecond on both, so their gain of 18 times hardly shows.",
            "flops": "The FLOPs take 0.13 of the 17.8 milliseconds on a V100. What takes the rest of the time, and how much did that grow?",
            "both": "The two do not add up to one gain: each token waits mostly for one of them. Which of the two takes most of the 17.8 milliseconds?",
            "vram": "The VRAM says whether the model fits, not how fast it is read. What sets the time of reading 16 GB?",
        },
    )
    choice_token
    return (choice_token,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Close and fast, far and slow: the memory hierarchy](/course/open/sys-memory-hierarchy?lang=en), reading took the bytes divided by the bandwidth; here you see that the bandwidth of GPUs grows more slowly than their FLOP/s, so more and more jobs wait for their bytes. In [Moore's law and the growth of compute](/course/open/sys-moores-law?lang=en), the memory wall was a trend over 20 years; here it is in the numbers of five GPUs, from 139 FLOPs per byte for the V100 to 281 for the B200. The next lesson is *How a computer stores a number: integers and floats*: what a number of 16, 8 or 4 bits can hold.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_token, result_fit, result_gain, result_helps):
    L.wrap_up(result_gain, result_fit, result_helps, choice_token)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
