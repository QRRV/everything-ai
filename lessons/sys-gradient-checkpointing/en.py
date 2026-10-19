# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-gradient-checkpointing. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-gradient-checkpointing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where the memory goes](/course/open/sys-memory-accounting?lang=en) and [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_batch():
    quick_batch = L.choice(
        "quick-batch",
        question="Which part of the memory of a training run grows with the batch size?",
        options={
            "weights": "The weights and their gradients",
            "right": "The activations",
            "state": "The optimizer state",
        },
        correct="right",
        why={
            "weights": "The weights and their gradients have one number per parameter, whatever the batch. See [Where the memory goes](/course/open/sys-memory-accounting?lang=en).",
            "state": "The optimizer keeps a fixed number of values per parameter, whatever the batch. See [Where the memory goes](/course/open/sys-memory-accounting?lang=en).",
            "right": "The forward pass keeps tensors for every example in the batch, and every layer. This lesson trades some of them for compute.",
        },
    )
    quick_batch
    return (quick_batch,)


@app.cell(hide_code=True)
def quick_check_input():
    quick_input = L.choice(
        "quick-input",
        question="A linear layer works out $\\mathbf{Y} = \\mathbf{X}\\mathbf{W}^\\top$. What does the backward pass need from the forward pass, to work out the gradient of $\\mathbf{W}$?",
        options={
            "right": "The input $\\mathbf{X}$",
            "output": "The output $\\mathbf{Y}$",
            "nothing": "Nothing: $\\mathbf{W}$ is enough",
        },
        correct="right",
        why={
            "output": "The gradient of $\\mathbf{W}$ is $\\mathbf{G}^\\top\\mathbf{X}$, with $\\mathbf{G}$ the gradient of $\\mathbf{Y}$: it needs the input. See [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en).",
            "nothing": "The gradient of $\\mathbf{W}$ depends on what came in: $\\mathbf{G}^\\top\\mathbf{X}$, with $\\mathbf{G}$ the gradient of $\\mathbf{Y}$. See [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en).",
            "right": "$\\mathbf{G}^\\top\\mathbf{X}$, with $\\mathbf{G}$ the gradient of $\\mathbf{Y}$. So the forward pass keeps every input until the backward pass: those are the activations.",
        },
    )
    quick_input
    return (quick_input,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    With long texts and large batches, the activations take more memory than the weights, the gradients and the optimizer state together. When a run does not fit, you can make the batch smaller, but then each step sees fewer examples. **Gradient checkpointing** keeps the batch: it throws most activations away in the forward pass, and works them out again in the backward pass when they are needed. Less memory, for some extra compute.
    """)
    return


@app.cell(hide_code=True)
def predict_compute():
    guess_compute = L.guess(
        "extra-flops",
        question="A training step is a forward pass and a backward pass, and the backward pass takes about twice the FLOPs of the forward pass. You checkpoint every block of a deep network. About how many more FLOPs does a step take?",
        options={
            "few": "About 5% more",
            "right": "About a third more",
            "double": "About twice as many",
        },
        reveal="**About a third more:** every layer's forward pass runs once more, during the backward pass. A step took 1 + 2 = 3 forward passes' worth; now it takes 4. How much memory that saves depends on how long the segments are, as this lesson shows.",
    )
    guess_compute
    return (guess_compute,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Segments.** Split the $n$ layers into segments of $k$ layers. In the forward pass, keep only the input of every segment, and throw the activations inside a segment away. In the backward pass, when the gradient reaches a segment, run its forward pass again from its kept input, this time keeping its activations, and go back through it. Then throw them away, and go on to the segment before.

        **Memory.** Count in units of one layer's activations. Without checkpoints, the network's input and the output of every layer are kept: $n + 1$. With segments of $k$, the forward pass keeps $\lceil n / k \rceil$ segment inputs, and during the backward pass one segment's $k$ are added:
        """),
        L.formula(r"[[hidden:\text{memory}(k)]] = \lceil n / k \rceil + k"),
        mo.md(r"""
        Short segments keep many inputs; long ones recompute much at once. The sum is smallest near $k = \sqrt{n}$: about $2\sqrt{n}$ instead of $n + 1$. For 16 layers that is 8 instead of 17; for 100 layers, 20 instead of 101. Segments of 1 layer, or one segment of all $n$, save nothing in this count: $n + 1$ again.

        **Compute.** Whatever $k$ is, every layer's forward pass runs one extra time: about a third more FLOPs per step. That is the price.

        **In PyTorch.** `checkpoint(fn, *inputs, use_reentrant=False)`, from `torch.utils.checkpoint`, runs `fn` on the inputs without keeping what happens inside it, and runs it again in the backward pass. `use_reentrant=False` asks for the way of checkpointing that PyTorch recommends: it also works when the input needs no gradient, as the data that goes into the first segment. A checkpoint here is not a model saved to disk, as the checkpoints of a training run are: it is a kept segment input, a point the backward pass computes again from.

        The picture: for 16 layers, the activation memory for every segment length $k$: the kept segment inputs, filled, and the recomputed segment on top, light. The dashed line is the 17 units without checkpoints.
        """),
        mo.Html(C.segments_picture({"title": "16 layers in segments of k", "x": "segment length", "y": "units of one layer",
                                    "plain": "without checkpoints", "lang": L.lang,
                                    "alt": "Five stacked bars, for k = 1, 2, 4, 8 and 16, with heights 17, 10, 8, 10 and 17: the kept inputs, 16, 8, 4, 2 and 1, with the recomputed segment, 1, 2, 4, 8 and 16, on top. A dashed line at 17 marks the memory without checkpoints, level with the bars for k = 1 and 16; the lowest bar is k = 4."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **gradient checkpointing** | | keeping only some activations in the forward pass, and working the others out again in the backward pass | `checkpoint(fn, x, use_reentrant=False)` |
        | **segment** | | layers run under one checkpoint: only their input is kept | |
        | $\lceil n / k \rceil$ | ceiling of n over k | $n / k$ rounded up: the number of segments | `math.ceil(n / k)` |
        """),
        L.watch_out("Checkpointing a segment with randomness inside, such as dropout, must give the same random numbers when it runs again, or the recomputed activations differ from the first ones. PyTorch's checkpoint saves and restores the random state for that; your own code that recomputes things must do the same. And very short segments add an overhead of their own, on top of the extra FLOPs."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fit a run with gradient checkpointing:

    1. Measure the activations, as in [Where the memory goes](/course/open/sys-memory-accounting?lang=en), and see how much must go.
    2. Split the layers into segments; about $\sqrt{n}$ layers each keeps the least. A transformer block keeps many tensors inside, so one block per segment, which keeps only its input, already saves most, and is often simplest.
    3. Run every segment through `checkpoint(..., use_reentrant=False)`.
    4. Measure again: the kept memory, and the time per step. The FLOPs are about a third more; the time is often more than that.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 and step 4 on one linear layer and a ReLU, with `saved_tensors_hooks` as in [Where the memory goes](/course/open/sys-memory-accounting?lang=en): what the forward pass keeps, without and with a checkpoint around the layer and the ReLU.
    """)
    return


@app.cell
def example_one_segment():
    import torch
    from torch.autograd.graph import saved_tensors_hooks
    from torch.utils.checkpoint import checkpoint

    _layer = torch.nn.Linear(100, 100)
    _x = torch.randn(10, 100)
    _places = {p.data_ptr() for p in _layer.parameters()}

    def _kept(run):
        _sizes = {}

        def _pack(t):
            if t.data_ptr() not in _places:
                _sizes[t.data_ptr()] = t.numel() * t.element_size()
            return t

        with saved_tensors_hooks(_pack, lambda t: t):
            _out = run()
        _out.sum().backward()
        return sum(_sizes.values())

    def _segment(x):
        return torch.relu(_layer(x))

    print("without:", _kept(lambda: _segment(_x)), "bytes")
    print("with a checkpoint:", _kept(lambda: checkpoint(_segment, _x, use_reentrant=False)), "bytes")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Without a checkpoint, the forward pass keeps the input of the layer, for the gradient of its weights, and the output of the ReLU, for the slope of the ReLU: 2 tensors of 10 × 100 float32 numbers, 8,000 bytes. With a checkpoint it keeps only the input, 4,000 bytes; the ReLU's output is worked out again in the backward pass.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_checkpointed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Segments of k blocks

        The box starts with `kept_bytes(run, blocks)`, the measurement of the worked example for a list of blocks, and `run_blocks(blocks, x)`, which runs `x` through the blocks in order, each a linear layer with a ReLU after it. Complete `checkpointed(blocks, x, k)`: run the blocks in segments of `k`, each segment through `checkpoint`, and give back the output. The check runs 8 linear layers of 32 numbers on 16 examples: the output must be the same as without checkpoints, and in segments of 2 the forward pass must keep 4 segment inputs, 8,192 bytes. It also tries segments of 3, and checks that the gradient reaches the first block.
        """),
        L.task("Make `checkpointed` run every segment of `k` blocks through `checkpoint`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_checkpointed():
    box_checkpointed = L.code_box("checkpointed", lines=30)
    box_checkpointed
    return (box_checkpointed,)


@app.cell(hide_code=True)
def check_checkpointed(box_checkpointed):
    result_checkpointed = L.check("checkpointed", box_checkpointed.value)
    result_checkpointed
    return (result_checkpointed,)


@app.cell(hide_code=True)
def hints_checkpointed():
    L.hints(
        "checkpointed",
        [
            "A loop with `range(0, len(blocks), k)`; a segment is `blocks[start:start + k]`, and `checkpoint` passes everything after the function on to it.",
        ],
        why="`for start in range(0, len(blocks), k): x = checkpoint(run_blocks, blocks[start:start + k], x, use_reentrant=False)`, and `return x`. The last segment is shorter when `k` does not divide the number of blocks.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The best segment length

        Complete `best_k(n)`: for $n$ layers, the segment length $k$ from 1 to $n$ with the least memory, $\lceil n / k \rceil + k$ units, and the smallest $k$ when two are equal. The check calls `best_k(16)`, which should give `4`, and tries 1 layer and numbers of layers that segments do not divide.
        """),
        L.task("Make `best_k` give the segment length with the least activation memory."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-k", lines=9)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-k", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-k",
        [
            "Keep the best `k` so far and its sum; replace them only when a new `k` gives a smaller sum, so that the first, smallest `k` stays on a tie.",
        ],
        why="Loop `k` over `range(1, n + 1)`, work out `math.ceil(n / k) + k`, and keep the `k` with the smallest; or `min(range(1, n + 1), key=lambda k: math.ceil(n / k) + k)`, which keeps the first on a tie. For 16 layers, `k = 4` gives 4 + 4 = 8.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your checkpoints on 16 layers

    The cell below runs 16 linear layers of 512 numbers, each with a ReLU, on 256 examples, with your `checkpointed` once it is right, else the course's: for every segment length, what the forward pass keeps, and the time of a forward and backward pass on your CPU. It takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_segments(box_checkpointed, result_checkpointed):
    _d = C.demo(box_checkpointed.value if result_checkpointed.state == "correct" else None)
    _base = _d["rows"][0][2]
    _who = "your `checkpointed`" if _d["yours"] else "the course's `checkpointed`"
    _lines = [f"With {_who}:", "",
              "| segments of | kept bytes | time |", "|---|---|---|"]
    for _k, _bytes, _t in _d["rows"]:
        _lines.append(f"| {'no checkpoints' if _k == '-' else _k} | {C.grouped(_bytes, 'en')} | {L.number(_t / _base, 2)}× |")
    _slower = all(_t > _base for _, _, _t in _d["rows"][1:])
    _end = ("Every segment length took more time than no checkpoints: the extra forward passes, and some overhead for every checkpoint." if _slower else
            "The extra forward passes and the overhead of every checkpoint cost time, but a busy computer makes timings vary: run the cell again if a segment length came out as fast as no checkpoints.")
    _lines += ["", "Time as a multiple of the run without checkpoints. The kept bytes halve each time the segments double. " + _end]
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def text_exercise_same():
    mo.md(r"""
    ### Exercise 3. Which costs more compute
    """)
    return


@app.cell(hide_code=True)
def choice_same_cell():
    choice_same = L.choice(
        "same-compute",
        question="You checkpoint a network of 16 layers once in segments of 1 layer, and once in segments of 4. Which adds more FLOPs to a training step?",
        options={
            "one": "Segments of 1: more checkpoints, more recomputing",
            "four": "Segments of 4: longer segments to recompute",
            "right": "About the same: every layer runs forward once more",
        },
        correct="right",
        why={
            "one": "Each layer is in exactly one segment, and each segment runs its forward pass once more: the number of segments does not change the FLOPs.",
            "four": "A longer segment recomputes more layers, but there are fewer segments: together, every layer once.",
            "right": "The segment length changes the memory, not the FLOPs. Only very short segments cost a little more time, through the overhead of each checkpoint.",
        },
    )
    choice_same
    return (choice_same,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The activation memory of [Where the memory goes](/course/open/sys-memory-accounting?lang=en), measured with the same `saved_tensors_hooks`, now traded for compute: the inputs of [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en) are kept only at the start of every segment. With mixed precision and gradient accumulation, it is one of the usual ways to make a run fit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same, result_best, result_checkpointed):
    L.wrap_up(result_checkpointed, result_best, choice_same)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
