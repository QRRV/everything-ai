# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-memory-hierarchy. Route: course/route/lessen/systemen.yaml. Checks, the H100, the measurements, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-memory-hierarchy", "en")
    C = L.checks   # the memory of an H100, the sums measured on the course's computer, the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Calculating without loops](/course/open/np-vectorization?lang=en) and [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vector():
    quick_vector = L.choice(
        "quick-vector",
        question="`prices` is a NumPy array of a million numbers. What does `prices * 1.1` give?",
        options={
            "right": "A new array of a million numbers, each 1.1 times its price",
            "repeat": "The million numbers repeated, as `*` does with a list",
            "one": "One number: 1.1 times the sum of all prices",
            "error": "An error, since `*` works on one number at a time and needs a loop around it",
        },
        correct="right",
        why={
            "right": "A calculation on an array works on every number at once, in one fast loop inside NumPy. In this lesson, you see one more reason that such a loop is fast: the numbers of an array lie next to each other in memory.",
            "repeat": "Repeating is what `*` does with a list and a whole number. What does a calculation on an array do with each of its numbers? See [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "one": "Nothing is added up here. What happens to each number of the array? See [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "error": "NumPy runs the loop for you, inside the calculation. What does a calculation on an array give? See [Calculating without loops](/course/open/np-vectorization?lang=en).",
        },
    )
    quick_vector
    return (quick_vector,)


@app.cell(hide_code=True)
def quick_check_fastest():
    quick_fastest = L.choice(
        "quick-fastest",
        question="You time the same calculation five times with `time.perf_counter()`: 3.1, 2.4, 2.5, 2.4 and 5.0 milliseconds. Which time does the course keep?",
        options={
            "first": "3.1 milliseconds, the first run",
            "right": "2.4 milliseconds, the fastest of the five",
            "mean": "3.08 milliseconds, the mean of all five",
            "slow": "5.0 milliseconds, the slowest, so that the time is never too low",
        },
        correct="right",
        why={
            "right": "The work takes at least its fastest time; the slower runs also waited for other things, and the first run did some work only once. In this lesson, you keep the fastest of a few sums to measure how fast your computer reads numbers.",
            "first": "The first run often takes longer, since some things happen only once. Which run says the most about the work itself? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
            "mean": "The mean takes the 5.0 along, a run that waited for something else. Which time says the most about the work itself? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
            "slow": "The slowest run waited the longest for other things. Which time says the most about the work itself? See [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en).",
        },
    )
    quick_fastest
    return (quick_fastest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A GPU can only compute with numbers that are in the chip itself, in a few hundred kilobytes right next to its cores. Everything else sits further away: in a larger memory on the chip, in the GPU's own memory beside the chip, in the RAM of the computer, on its SSD. Each step further away holds more and delivers its bytes more slowly. So how fast a computation runs depends not only on how many FLOPs it does, but on where its numbers come from. In this lesson, you see how the memory of a GPU and of your own computer is built, you work out how long it takes to read data from each place, and you measure on your own computer how much faster a sum is when its numbers fit close by.

    **An everyday example.** A cook keeps salt and oil on the counter, flour in the cupboard, and sacks of potatoes in a storeroom at the back. The counter is small and within reach; the storeroom holds much more, and every trip there takes minutes. So a good cook fetches a whole sack at once, and keeps what is needed again close at hand.
    """)
    return


@app.cell(hide_code=True)
def predict_hbm():
    guess_hbm = L.guess(
        "hbm",
        question="An NVIDIA H100 has 80 GB of memory beside its chip, which it can read at 3.35 terabytes per second. How long does it take to read all 80 GB once, do you think?",
        options={
            "micro": "About 24 microseconds",
            "right": "About 24 milliseconds",
            "second": "About 2.4 seconds",
            "minute": "About a minute, since 80 GB is a lot of data",
        },
        reveal="**About 24 milliseconds.** 80 GB divided by 3,350 GB per second is 0.024 seconds: the H100 can read all of its memory about 42 times a second. The same 80 GB from the RAM of the computer, over the PCIe connection at 64 GB per second, take 1.25 seconds, 52 times as long; from a fast SSD at 7.45 GB per second, about 11 seconds.",
    )
    guess_hbm
    return (guess_hbm,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Memory hierarchy.** A computer keeps its numbers in layers. Close to the cores, memory is small and fast; each layer further away is larger and slower. That ladder is the **memory hierarchy**. On an NVIDIA H100, it looks like this:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        An H100 has 132 groups of cores. Each group keeps the numbers it computes with right now in its registers, 256 KB, and numbers it needs again soon in its L1 cache, another 256 KB. All groups share an L2 cache of 50 MB. Next to the chip are 80 GB of HBM3, and the data of the computer comes in over a PCIe Gen5 connection. NVIDIA's own advice for fast code is to move the data "as close as possible" to the cores.

        **Bandwidth.** The **bandwidth** of a memory, or of a connection, is how many bytes per second it can deliver. Reading takes the bytes divided by the bandwidth: a file of 1 GB takes 1 / 7.45 = 0.13 seconds from an SSD that reads 7.45 GB per second. The HBM3 of the H100 delivers 3.35 TB per second, the PCIe connection 64 GB per second in each direction, and one module of DDR5 memory in a computer 38.4 GB per second. NVIDIA publishes no single number for the registers and caches; they deliver more still.

        **Cache.** A **cache** is a small, fast memory that keeps copies of data that was used recently. When the processor needs a number that is not in the cache, the cache fetches it from further away, together with its neighbors: a block of 64 bytes on most processors, eight float64 numbers. The numbers of a NumPy array lie next to each other in memory, so a sum that runs through them in order finds most of them in the cache already. Numbers read in a random order each need a trip of their own.

        **HBM.** **HBM**, short for high-bandwidth memory, is memory chips stacked on top of each other, placed right next to the GPU chip and joined to it by thousands of short wires. That makes it much faster than the RAM of a computer, which sits on separate modules on the motherboard. The H100 has five stacks, 80 GB together.

        **Your own computer.** The processor of your computer has the same ladder: registers, two or three levels of cache, then the RAM, then the SSD. On the course's computer, a sum of 512 KB of numbers reads 34 GB per second, and a sum of 512 MB only 8.2: the large array does not fit in the caches and comes from RAM. Very small arrays are slow too, for the reason you met in [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en): every call has a fixed cost.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **memory hierarchy** | | the layers of memory, from small and fast close to the cores to large and slow further away: registers, caches, HBM or RAM, SSD | |
        | **bandwidth** | | how many bytes per second a memory or a connection delivers; reading takes the bytes divided by the bandwidth | `nbytes / bandwidth` |
        | **cache** | "cash" | a small, fast memory that keeps copies of recently used data, together with its neighbors | |
        | **HBM** | "H B M" | high-bandwidth memory: stacks of memory chips right next to a GPU chip | |
        | `a.nbytes` | "a dot n bytes" | the bytes an array takes: its number of values times the bytes of each | `np.zeros(1000).nbytes` gives `8000` |
        """),
        L.watch_out("A bandwidth says how fast the bytes come once they flow, not how long the first byte takes to arrive. Many small reads, each from somewhere else, are slower than one large read of the same bytes."),
        L.watch_out("GB and TB count in thousands here: 1 TB is 1,000 GB, and 1 GB is 1,000,000,000 bytes, which Python writes as `1e9`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The time to read data, and the bandwidth you reach:

    1. Count the bytes: `a.nbytes` for an array, or the number of values times the bytes of each.
    2. Find the bandwidth of the place the bytes come from: the HBM, PCIe, the RAM or the SSD.
    3. Divide the bytes by the bandwidth: the seconds to read them once.
    4. To measure a bandwidth yourself, run the work once first, read `time.perf_counter()` before and after it, and divide the bytes by the seconds.
    5. Divide by `1e9` for gigabytes per second, and compare small arrays with large ones: where the rate drops, the data no longer fits in the cache.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An array of a million float64 numbers: how many bytes it takes, and how long the HBM3 of an H100 takes to read it once. Run the cell.
    """)
    return


@app.cell
def example_bytes():
    import numpy as np

    _a = np.random.default_rng(0).random(1_000_000)   # a million float64 numbers
    print(_a.size, "numbers,", _a.nbytes, "bytes")
    _seconds = _a.nbytes / 3.35e12                     # read once from the HBM3 of an H100
    print(round(_seconds * 1e6, 2), "microseconds")
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A float64 number takes 8 bytes, so a million of them take 8,000,000. From the HBM, that is 2.39 microseconds; from an SSD at 7.45 GB per second, it would take 1,074 microseconds, 450 times as long. Exercise 1 works out such times for several places at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Read times

        `read_seconds(nbytes, bandwidths)` gets a number of bytes and a dict with the bandwidth of each place, in bytes per second, and gives back a dict with the seconds it takes to read the bytes once from each place. Replace the `...` so that every round of the loop puts `nbytes` divided by the bandwidth of `name` in `seconds[name]`.

        The check calls `read_seconds(16_000_000_000, C.LEVELS)`: a model of 16 GB, and `C.LEVELS` is `{"HBM": 3.35e12, "PCIe": 64e9, "SSD": 7.45e9}`. Then other bytes and other dicts, one of them empty.
        """),
        L.task("Make every round of the loop put the read time of `name` in `seconds`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read", lines=6)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read",
        [
            "In the loop, `name` is one key of `bandwidths`, such as `\"HBM\"`, and a dict gives the value of a key between square brackets. `seconds[name]` gets `nbytes` divided by that bandwidth. What goes after the `/`?",
            "With 1,000 bytes and `{\"a\": 10}`, the place `\"a\"` takes 1,000 / 10 = 100 seconds.",
            "`        seconds[name] = nbytes / ___`",
        ],
        why="`seconds[name] = nbytes / bandwidths[name]` divides the bytes by the bandwidth of each place. A model of 16 GB takes 4.8 ms from the HBM, 0.25 seconds over PCIe and 2.1 seconds from the SSD.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Measure a bandwidth

        `gb_per_second(values)` sums an array once first with `values.sum()`, puts the clock `time.perf_counter()` in `start`, sums once more, and keeps the time of that sum in `seconds`. Replace the `...` with the gigabytes per second the sum read: the bytes of `values` divided by `seconds`, and by `1e9`.

        The check calls your function with stand-ins for an array: `slow_array(800000000, 0.1)` holds 800,000,000 bytes, and each of its sums takes 0.1 seconds. Then the picture calls it on real arrays of 8 KB to 128 MB, on your own computer.
        """),
        L.task("Give back the gigabytes per second of one sum of `values`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rate():
    box_rate = L.code_box("rate", lines=10)
    box_rate
    return (box_rate,)


@app.cell(hide_code=True)
def check_rate(box_rate):
    result_rate = L.check("rate", box_rate.value)
    result_rate
    return (result_rate,)


@app.cell(hide_code=True)
def hints_rate():
    L.hints(
        "rate",
        [
            "An array knows how many bytes it takes: the table of new words has the attribute. `seconds` is the time of one sum, and a gigabyte is `1e9` bytes. Which bytes go before `/ seconds`?",
            "An array of a million float64 numbers takes 8,000,000 bytes. Read in 0.001 seconds, that is 8,000,000 / 0.001 / 1e9 = 8 GB per second.",
            "`    return ___ / seconds / 1e9`",
        ],
        why="`return values.nbytes / seconds / 1e9` divides the bytes of the array by the time of one sum. The stand-in reads 800,000,000 bytes in 0.1 seconds: 8 GB per second.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Exercise 3. Where the numbers sit

    Eight places from this lesson. Put each one where it sits: *On the GPU chip*, *Next to the chip* or *Outside the GPU*.
    """)
    return


@app.cell(hide_code=True)
def sort_where_cell():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "The registers and the caches are part of the chip itself. HBM is stacked beside it. What sits on the motherboard, or connects to it, is outside the GPU.",
            "Three go on the chip, two next to it, and three outside the GPU.",
            "On the GPU chip: registers, L1 cache, \\_\\_\\_. Next to the chip: HBM3, \\_\\_\\_. Outside the GPU: RAM, \\_\\_\\_, \\_\\_\\_.",
        ],
        why="**On the GPU chip:** registers, L1 cache, L2 cache. **Next to the chip:** HBM3, a model during training. **Outside the GPU:** RAM, PCIe Gen5, SSD.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Sums of every size

    On the course's computer, the fastest of many sums of float64 numbers with NumPy, for arrays of 1 KB to 512 MB: how many gigabytes per second each sum reads. Choose a size.
    """)
    return


@app.cell(hide_code=True)
def demo_size():
    L.explore("size", "kilobytes", lambda _v: C.frame(_v, L.lang), start=512, values=C.DEMO_VALUES, role="",
              task="Go from 1 to 512,000 kilobytes: where does the rate peak, and where does it drop?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 1 KB to 512 KB the rate climbs, since the fixed cost of a call weighs less and less. Up to 32 MB, the numbers come from the caches of the processor, at 26 to 34 GB per second. From 128 MB on, they come from RAM, at about 8 GB per second: the same additions, at a quarter of the speed. A GPU has the same ladder, with every layer larger and faster: its HBM3 alone delivers 3.35 TB per second.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ### Exercise 4. In order or shuffled

    On the course's computer, `a` holds 16 million float64 numbers, 128 MB. `order` holds the positions 0, 1, 2 and so on, and `shuffled` the same positions in a random order, made before the clock starts. `a[order].sum()` takes 67 milliseconds, and `a[shuffled].sum()` takes 229 milliseconds.
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="The same numbers and the same additions. Why does the shuffled order take more than three times as long?",
        options={
            "shuffle": "The shuffling takes the extra time",
            "right": "In order, each block the cache fetches is used whole; shuffled, most of each block goes unused",
            "bytes": "Shuffled numbers take more bytes in memory, so there is more to read",
            "adds": "Adding numbers in a random order takes more FLOPs than adding them in order, since each needs its own step",
        },
        correct="right",
        why={
            "right": "The cache fetches a whole block of neighbors for every number it reads from RAM. In order, the next numbers are already in that block. Shuffled, almost every number needs a block of its own from RAM, and most of each block is never used.",
            "shuffle": "`shuffled` was made before the clock started; both lines only pick the numbers and add them. What differs is where the next number lies: is it close to the last one?",
            "bytes": "Both lines read the same 16 million float64 numbers, 8 bytes each. What differs is the order of reading: what does the cache fetch with each number?",
            "adds": "Both lines add the same 16 million numbers, so they do the same FLOPs. What differs is the order of reading: what does the cache fetch with each number?",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Calculating without loops](/course/open/np-vectorization?lang=en), NumPy worked on a whole array at once; here you see that such an array lies in one row in memory, which is just what a cache needs. In [How fast is your model? FLOPs, FLOP/s and tokens per second](/course/open/sys-how-fast?lang=en), a computation took its FLOPs divided by the FLOP/s it reached; here a read takes its bytes divided by the bandwidth, and often that is the longer of the two times. The next lesson is *GPU generations: from Volta to Blackwell*: how the GPUs of NVIDIA grew, in FLOP/s and in memory.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_rate, result_read, result_where):
    L.wrap_up(result_read, result_rate, result_where, choice_order)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
