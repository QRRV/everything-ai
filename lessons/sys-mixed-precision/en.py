# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-mixed-precision. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-mixed-precision", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en) and [A reusable training loop](/course/open/nn-trainer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Which part of FP32 does BF16 keep?",
        options={
            "digits": "Its 23 mantissa bits, so its fine steps",
            "right": "Its 8 exponent bits, so its range",
            "both": "Both, packed into half the bits",
        },
        correct="right",
        why={
            "digits": "BF16 has only 7 mantissa bits: its steps near 1 are 1/128. See [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en).",
            "both": "16 bits cannot hold 8 exponent bits and 23 mantissa bits. See [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en).",
            "right": "So BF16 rarely overflows or underflows, but its steps are coarse. This lesson keeps the numbers that need fine steps in FP32.",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="In your trainer, what does `opt.step()` change?",
        options={
            "grads": "The gradients of the parameters",
            "right": "The parameters themselves",
            "loss": "The loss of the next mini-batch",
        },
        correct="right",
        why={
            "grads": "The gradients come from `backward()`; `opt.zero_grad()` clears them. See [A reusable training loop](/course/open/nn-trainer?lang=en).",
            "loss": "The loss changes because the parameters changed; the step itself only touches the parameters. See [A reusable training loop](/course/open/nn-trainer?lang=en).",
            "right": "The optimizer adds a small update to every parameter. Whether that update survives depends on the format the parameters are stored in.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en), a weight of 1.0 in BF16 never moved: each small update rounded away. Yet large models train in 16 bits. The trick is not to put everything in 16 bits. **Mixed precision**, published by Paulius Micikevicius and colleagues in 2018, keeps a master copy of the weights in FP32 and does the matrix multiplications in 16 bits. PyTorch's `torch.autocast` does the bookkeeping, so a training step needs only one extra line.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights-format",
        question="You run the forward pass of a training step inside `torch.autocast(..., dtype=torch.bfloat16)`. In which format are the weights of your model after the step?",
        options={
            "bf16": "BF16, like the calculations",
            "right": "FP32, as before",
            "mixed": "Some layers in BF16, some in FP32",
        },
        reveal="**FP32, as before.** Autocast changes only the format of the calculations inside its block: a linear layer gets a BF16 copy of its weights and its input for the multiplication, and the weights themselves stay FP32. Their gradients come out in FP32 too, and the optimizer updates the FP32 weights, so a small update is not lost.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: three kinds of numbers.** A training step has weights, activations and gradients. The activations are the outputs of the layers. The optimizer also keeps numbers of its own for each weight, such as the two averages of AdamW. Mixed precision keeps the weights in FP32, as the **master weights**, and the optimizer's averages too. It does the matrix multiplications of the forward and the backward pass in BF16.

        **Step 2: autocast.** Put the forward pass and the loss inside `with torch.autocast(device_type=..., dtype=torch.bfloat16):`. Inside it, operations on autocast's list for 16 bits, such as `nn.Linear`, matrix multiplications and convolutions, turn their inputs into BF16 copies and calculate in BF16. Operations on its list for FP32, which need range or fine steps, such as the cross-entropy loss, calculate in FP32; on an NVIDIA GPU that list also has softmax, logarithms and sums. Other operations, such as ReLU, keep the format of their input. Call `backward()` after the block: each operation's backward runs in the format of its forward, and the gradients of FP32 weights come out in FP32. The optimizer step is unchanged.

        **Step 3: FP16 needs loss scaling.** With FP16 instead of BF16, small gradients fall below its range: below about $3 \times 10^{-8}$ they become 0. **Loss scaling** multiplies the loss by a big factor $S$, such as 65,536, before `backward()`, so every gradient is $S$ times bigger and stays in range; before the step it divides them by $S$ again. If a gradient became `inf` or `nan`, it skips that step and halves $S$; after 2,000 good steps in a row, it doubles $S$ again. `torch.amp.GradScaler` does all of this. BF16 has the range of FP32 and needs none of it.

        **Step 4: what you save.** Many of the activations that the backward pass needs are kept in BF16, which saves memory, though less than half: some of them stay FP32. On a GPU with tensor cores the matrix multiplications run several times faster. The weights, their gradients and the optimizer's averages stay FP32, as large as before, and the BF16 copies of the weights come on top; the next lesson counts it all.

        The picture: one training step in mixed precision.
        """),
        mo.Html(C.flow_picture({"title": "One step in mixed precision", "weights": "master weights, FP32", "copy": "a BF16 copy for each matrix multiplication",
                                "forward": "forward pass: matrix multiplications in BF16", "loss": "loss, FP32",
                                "backward": "backward pass, each operation in the format of its forward", "grads": "gradients of the weights, FP32",
                                "update": "optimizer step on the master weights, FP32",
                                "alt": "Seven boxes from top to bottom: master weights in FP32; a BF16 copy for each matrix multiplication; the forward pass with the matrix multiplications in BF16; the loss in FP32; the backward pass, each operation in the format of its forward; the gradients of the weights in FP32; the optimizer step on the master weights in FP32."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **mixed precision** | training with the calculations in 16 bits and the master weights in FP32 | |
        | **master weights** | the FP32 weights that the optimizer updates; the 16-bit numbers are copies | |
        | **autocast** | PyTorch's switch that runs the operations in a block in a lower format where that is safe | `with torch.autocast(device_type, dtype=torch.bfloat16):` |
        | **loss scaling** | multiplying the loss by a big factor before `backward()`, and dividing the gradients by it before the step, so small FP16 gradients do not become 0 | `torch.amp.GradScaler()` |
        """),
        L.watch_out("Turning the model itself into BF16, with `model.to(torch.bfloat16)`, is not mixed precision: then the weights that the optimizer updates are BF16 too, and small updates are lost."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train in mixed precision:

    1. Keep the model and the optimizer as they are, in FP32.
    2. Put the forward pass and the loss inside `torch.autocast` with `dtype=torch.bfloat16`.
    3. Call `backward()` and the optimizer step after the block, as before.
    4. With FP16 instead, add a `GradScaler` for the loss scaling.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Autocast is new: one linear layer as the model, a forward hook as in [Looking inside a training run](/course/open/nn-activation-stats?lang=en) that reports the format the layer calculates in, and the formats of the loss, the weights and their gradients.
    """)
    return


@app.cell
def example_autocast():
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    _model = torch.nn.Linear(4, 3)
    _model.register_forward_hook(lambda m, i, o: print("the layer calculates in", o.dtype))
    _x, _y = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _logits = _model(_x)
        _loss = F.cross_entropy(_logits, _y)
    _loss.backward()
    print("logits", _logits.dtype, " loss", _loss.dtype)
    print("weights", _model.weight.dtype, " gradients", _model.weight.grad.dtype)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The layer calculates in BF16, so the logits are BF16; the cross-entropy loss is FP32, because autocast runs it in FP32. The weights and their gradients are FP32. On an NVIDIA GPU you write `device_type="cuda"`, on a Mac `"mps"`; the rest stays the same.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A step in mixed precision

        Complete `train_step(model, opt, x, y, device_type)`: one training step with the forward pass and the cross-entropy loss under autocast in BF16, and the model left in FP32. It gives back the loss as a number. The check runs it on one linear layer, looks at the format the layer calculates in, the format of the weights and the loss it gives back, and tries a small update and two steps in a row.
        """),
        L.task("Complete `train_step(model, opt, x, y, device_type)`: the forward pass and the loss under autocast in BF16, the model in FP32."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("train-step", lines=11)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("train-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "train-step",
        [
            "The two lines of the forward pass and the loss go, indented, under one `with` line, as in the worked example, with `device_type` instead of `\"cpu\"`.",
        ],
        why="The `with` block covers the forward pass and the loss, where autocast chooses the format of each operation; `backward()` and `opt.step()` stay outside it, and the model is never converted.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### FP32 against mixed precision on your machine

    The button trains an MLP of 784, 1,024, 1,024 and 10 on the course's 6,000 digits, three epochs twice from the same start: once in plain FP32, once with your `train_step` (until Exercise 1 is right, the course's). Whether mixed precision is faster depends on your hardware. A GPU with fast BF16, such as an NVIDIA GPU from 2020 or later, does the multiplications on its tensor cores: the bigger the matrices, the bigger the gain, and these are small. A CPU without fast BF16, as in many laptops, has to imitate BF16 with other instructions, and can take many times as long. It takes from a few seconds to a few minutes.
    """)
    return


@app.cell(hide_code=True)
def compare_button_cell():
    compare_button = mo.ui.run_button(label="Train both")
    compare_button
    return (compare_button,)


@app.cell(hide_code=True)
def demo_compare(box_step, compare_button, result_step):
    if not compare_button.value:
        _out = mo.md("*Press the button to train both.*")
    else:
        _step, _yours = C.train_step_of(box_step.value if result_step.state == "correct" else None)
        _runs = {"fp32": "FP32", "mixed": "mixed precision"}
        _r = C.compare(_step, show=lambda _run, _i, _n: mo.output.replace(mo.md(f"*{_runs[_run]}: epoch {_i + 1} of {_n}*")))
        _where = {"cuda": "your GPU", "mps": "your Mac's GPU", "cpu": "your CPU"}[_r["device"]]
        _n = lambda _x, _d: C.number(_x, _d, L.lang)
        _who = "your `train_step`" if _yours else "the course's `train_step`"
        _f, _m = _r["fp32"], _r["mixed"]
        _ratio = _f["seconds"] / _m["seconds"]
        _speed = (f"Here mixed precision is {_n(_ratio, 1)} times as fast as FP32." if _ratio >= 1.1 else
                  f"Here mixed precision takes {_n(1 / _ratio, 1)} times as long as FP32." if _ratio <= 1 / 1.1 else
                  "Here the two are about as fast.")
        _moved = ("Your GPU does not calculate in BF16 under autocast (on a Mac that needs macOS 14 or later), so both runs are on your CPU. "
                  if _r["moved"] else "")
        _out = mo.md(f"""
    | | FP32 | mixed precision |
    |---|---|---|
    | seconds per epoch | {_n(_f['seconds'], 2)} | {_n(_m['seconds'], 2)} |
    | test digits right | {_n(100 * _f['accuracy'], 1)}% | {_n(100 * _m['accuracy'], 1)}% |
    | first layer calculates in | {_f['calculates']} | {_m['calculates']} |
    | weights after training | {_f['weights']} | {_m['weights']} |

    {_moved}On {_where}, with {_who}. {_speed} Both runs start from the same weights and get the mini-batches in the same order; the accuracies differ a little, because BF16 rounds the calculations. The time per epoch leaves out the first epoch, which also warms up.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_stays():
    mo.md(r"""
    ### Exercise 2. What stays in FP32?
    """)
    return


@app.cell(hide_code=True)
def choice_stays_cell():
    choice_stays = L.choice(
        "what-stays",
        question="You train with autocast in BF16 and AdamW. Which numbers stay in FP32 the whole time?",
        options={
            "acts": "The activations that the backward pass needs",
            "right": "The weights and AdamW's two averages",
            "all": "None: autocast turns everything into BF16",
        },
        correct="right",
        why={
            "acts": "Those come out of BF16 multiplications and are kept in BF16: that is where the memory saving comes from.",
            "all": "Autocast changes only the calculations inside its block; the weights and AdamW's averages are never converted.",
            "right": "The master weights and AdamW's averages stay FP32, so every update lands in fine steps. They do not shrink: the next lesson counts their memory.",
        },
    )
    choice_stays
    return (choice_stays,)


@app.cell(hide_code=True)
def text_exercise_scaling():
    mo.md(r"""
    ### Exercise 3. Why scale the loss?
    """)
    return


@app.cell(hide_code=True)
def explain_scaling():
    explain_scaling_cell = L.explain(
        "scaling",
        question="In a backward pass in FP16, a gradient should come out as $10^{-8}$. What happens to it without loss scaling, and what does a scale of $S = 65{,}536$ change? Why does BF16 not need this?",
        model_answer="**Without scaling:** FP16 turns numbers below about $3 \\times 10^{-8}$ into 0, and keeps few digits below about $6 \\times 10^{-5}$, so $10^{-8}$ becomes 0: that part of the gradient is lost. **With $S = 65{,}536$:** the loss is multiplied by $S$ before `backward()`, so every gradient is $S$ times bigger: $10^{-8} \\times 65{,}536 \\approx 6.6 \\times 10^{-4}$, well within the range of FP16. Before the step, the gradients are divided by $S$ again, in FP32. If a gradient overflows to `inf` or becomes `nan`, the step is skipped and $S$ halved. **BF16** has the 8 exponent bits of FP32, so $10^{-8}$ fits as it is.",
        points=[
            "you say that without scaling the gradient becomes 0, below the range of FP16",
            "you say that scaling the loss by $S$ makes every gradient $S$ times bigger, about $6.6 \\times 10^{-4}$ here, and that it is divided back before the step",
            "you say that BF16 has the range of FP32, so it needs no scaling",
        ],
    )
    explain_scaling_cell
    return (explain_scaling_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The formats come from [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en), the training step from [A reusable training loop](/course/open/nn-trainer?lang=en). The next lesson, *Where the memory goes*, adds up what a training run keeps in memory: the weights, their gradients, what the optimizer keeps for them and the activations, and how much of it mixed precision saves.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stays, explain_scaling_cell, result_step):
    L.wrap_up(result_step, choice_stays, explain_scaling_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
