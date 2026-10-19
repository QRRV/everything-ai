# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-memory-accounting. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-memory-accounting", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Adam](/course/open/opt-adam?lang=en) and [Mixed-precision training](/course/open/sys-mixed-precision?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_adam():
    quick_adam = L.choice(
        "quick-adam",
        question="Besides each parameter and its gradient, how many numbers per parameter does Adam carry from step to step?",
        options={
            "one": "One: its own learning rate",
            "right": "Two: an average of its gradients and one of their squares",
            "none": "None: it only uses the newest gradient and the learning rate",
        },
        correct="right",
        why={
            "one": "Adam's learning rate is one number for all parameters. What it keeps per parameter are two averages. See [Adam](/course/open/opt-adam?lang=en).",
            "none": "Adam keeps an average of the gradients and an average of their squares for every parameter. See [Adam](/course/open/opt-adam?lang=en).",
            "right": "Two numbers for every parameter, so twice as many numbers as the model itself.",
        },
    )
    quick_adam
    return (quick_adam,)


@app.cell(hide_code=True)
def quick_check_master():
    quick_master = L.choice(
        "quick-master",
        question="You switch a training run from FP32 to mixed precision with autocast. What happens to the memory of the weights?",
        options={
            "half": "It halves, because the calculations are in BF16",
            "right": "It stays the same, because the weights stay FP32",
            "double": "It doubles, because every weight now has a copy next to it",
        },
        correct="right",
        why={
            "half": "The calculations are in BF16, but the master weights stay FP32. See [Mixed-precision training](/course/open/sys-mixed-precision?lang=en).",
            "double": "Autocast's BF16 copies are half the size of the FP32 weights, and are gone after the backward pass; the FP32 weights stay as they were. See [Mixed-precision training](/course/open/sys-mixed-precision?lang=en).",
            "right": "The master weights are as large as before. Mixed precision saves on the activations, not on the weights.",
        },
    )
    quick_master
    return (quick_master,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A run that stops with "out of memory" after twenty minutes has wasted twenty minutes. Before you start, you can add up what a training run keeps in memory, and know whether it fits. The sums are simple, and their results often surprise: a model of 7 billion parameters fits on a 24 GB GPU to use in BF16, but fine-tuning all of it, training it further on your own data, needs well over 100 GB.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt():
    guess_gpt = L.guess(
        "gpt-memory",
        question="The GPT of [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en) has about 10.8 million parameters. How much memory do its weights, their gradients and AdamW's state take in FP32?",
        options={
            "weights": "About 43 MB",
            "right": "About 173 MB",
            "gb": "About 1 GB",
        },
        reveal="**About 173 MB.** Each parameter takes 4 bytes for the weight, 4 for its gradient and 8 for AdamW's two averages: 16 bytes. $10.8 \\times 10^6 \\times 16 \\approx 1.73 \\times 10^8$ bytes. 43 MB is the weights alone. The activations come on top, and for a small model they are often the largest part.",
    )
    guess_gpt
    return (guess_gpt,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: four parts.** During training, memory holds:

        - the **weights**: 4 bytes per parameter in FP32;
        - their **gradients**: as many, 4 bytes each;
        - the **optimizer state**: what the optimizer carries from step to step, per parameter. Plain SGD keeps nothing, momentum one number (4 bytes), Adam and AdamW two (8 bytes);
        - the **activations**: what the forward pass keeps for the backward pass, such as the input of every linear layer.

        **Step 2: 16 bytes per parameter.** With AdamW, the first three parts add up to $4 + 4 + 8 = 16$ bytes per parameter. That holds in FP32, and also in mixed precision, where the master weights, their gradients and AdamW's state stay FP32.

        **Step 3: the activations.** Their memory depends less on the number of parameters than on what flows through the model: it grows with the number of windows in a batch, the number of tokens per window, the number of layers and the width $d$; the attention scores grow with the square of the window length, and the logits of the output head with the size of the vocabulary. You can measure them: `torch.autograd.graph.saved_tensors_hooks` calls a function of yours for every tensor the forward pass keeps for the backward pass.

        **Step 4: what helps where.** A smaller batch, with gradient accumulation to keep the effective batch size, shrinks the activations; so does mixed precision, though by less than half, because softmax, sums and the loss stay FP32. An optimizer with less state, or one that stores it in 8 bits, shrinks the optimizer state. A smaller model shrinks the weights and gradients; training only a few of the parameters leaves out the gradients and optimizer state of the rest. Real runs also need some memory for PyTorch itself and lose some to fragmentation, so keep 10 to 20% free.

        The picture: using a model of 7 billion parameters in BF16, left, against fine-tuning all of it with AdamW, right, with dashed lines at the memory of GPUs of 24 and 80 GB. The numbers in the bars are GB.
        """),
        mo.Html(C.bars_picture({"title": "A model of 7 billion parameters", "y": "GB", "weights": "weights", "grads": "gradients", "state": "AdamW state",
                                "bars": ["using", "training"], "gpu24": "24 GB", "gpu80": "80 GB", "lang": L.lang,
                                "alt": "Two bars. Using it in BF16: 14 GB of weights, below the 24 GB line. Training with AdamW: 28 GB of weights, 28 GB of gradients and 56 GB of AdamW state stacked, 112 GB before any activations, above the lines for GPUs of 24 and 80 GB."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **optimizer state** | the numbers an optimizer carries per parameter from step to step, such as Adam's two averages | `opt.state` |
        | **activation memory** | the memory of the tensors the forward pass keeps for the backward pass | |
        """),
        L.watch_out("A GB in this course is $10^9$ bytes. Some tools show GiB, $2^{30}$ bytes, about 7% more per unit, and the memory of a GPU sold as 80 GB is close to 80 GiB: a little more room than the sums here say."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To know before a run whether it fits:

    1. Count the parameters.
    2. Multiply by the bytes per parameter: 4 for the weight, 4 for its gradient, and the optimizer's state, 8 for AdamW.
    3. Measure the activations for one batch, or take them from a run with a smaller batch: they grow in step with the batch.
    4. Add it up, and keep 10 to 20% of the GPU free.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Measuring activations is new: a small MLP, and two functions for `saved_tensors_hooks`. It calls the first for every tensor the forward pass keeps, and keeps what that function gives back, here the tensor itself; the second gives it back to the backward pass, here unchanged. `t.numel()` is the number of numbers in a tensor `t`, `t.element_size()` the bytes of each, and `t.data_ptr()` the place in memory where its numbers start. The function leaves out the parameters, which are counted already, and counts every place once, because two operations can keep the same tensor.
    """)
    return


@app.cell
def example_saved():
    import torch
    from torch.autograd.graph import saved_tensors_hooks

    _hidden, _out = torch.nn.Linear(64, 256), torch.nn.Linear(256, 10)
    _parameters = list(_hidden.parameters()) + list(_out.parameters())
    _places = {p.data_ptr() for p in _parameters}
    _kept = {}

    def _pack(t):
        if t.data_ptr() not in _places:
            _kept[t.data_ptr()] = t.numel() * t.element_size()
        return t

    with saved_tensors_hooks(_pack, lambda t: t):
        _loss = _out(torch.relu(_hidden(torch.randn(32, 64)))).sum()
    print("activations kept for backward:", sum(_kept.values()), "bytes")
    print("parameters:", sum(p.numel() * 4 for p in _parameters), "bytes")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The MLP keeps 40,960 bytes for a batch of 32: the input, 32 × 64 numbers, and the output of ReLU, 32 × 256 numbers, which both the ReLU and the second layer need. Its parameters take 76,840 bytes. With a batch of 320, the activations take ten times as much, and the weights stay the same.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gb():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Add it up

        Write `training_gb(n_params, optimizer, activation_gb)`: the memory of a training run in GB, $10^9$ bytes, in FP32 or mixed precision: the weights, their gradients and the state of `optimizer`, which is `"sgd"`, `"momentum"` or `"adamw"`, plus the activations, which come in GB already. The check calls `training_gb(7e9, "adamw", 0)`, which should give 112, and tries other models, optimizers and activations.
        """),
        L.task("Write `training_gb(n_params, optimizer, activation_gb)`: the weights, gradients and optimizer state of `n_params` parameters in GB, plus `activation_gb`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gb():
    box_gb = L.code_box("training-gb", lines=7)
    box_gb
    return (box_gb,)


@app.cell(hide_code=True)
def check_gb(box_gb):
    result_gb = L.check("training-gb", box_gb.value)
    result_gb
    return (result_gb,)


@app.cell(hide_code=True)
def hints_gb():
    L.hints(
        "training-gb",
        [
            "Add the bytes of one parameter first, steps 1 and 2 of the recipe; the rest is multiplying, converting to GB, and adding the activations.",
        ],
        why="With AdamW that is $4 + 4 + 8 = 16$ bytes per parameter: 112 GB for 7 billion parameters, before a single activation.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your GPT's activations

    The button measures your `GPT` from [Assemble your GPT](/course/open/gpt-assemble?lang=en), at a small size: 2 blocks, $d = 64$, windows of 64 tokens. Where a module of yours is missing, the course's version stands in. It counts what the forward pass keeps for batches of 1 to 64 windows, next to the 16 bytes per parameter of training with AdamW. It takes a second.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Measure my GPT")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(measure_button):
    if not measure_button.value:
        _out = mo.md("*Press the button to measure.*")
    else:
        _m = C.measure()
        if _m == "error":
            _out = mo.md("*Your GPT stopped with an error at these sizes: does it still work in [Assemble your GPT](/course/open/gpt-assemble?lang=en)?*")
        else:
            _fixed = 16 * _m["params"]
            _n = lambda _x, _d=2: C.number(_x, _d, L.lang)
            _rows = "\n".join(f"| {_b} | {_n(_v / 1e6)} MB | {_n(_v / _fixed, 1)}× |" for _b, _v in _m["rows"])
            _stand = (" The course's version stood in for: " + ", ".join(f"`{_s}`" for _s in _m["stand_ins"]) + ".") if _m["stand_ins"] else ""
            _out = mo.md(f"""
    Your GPT has {_n(_m['params'], 0)} parameters: {_n(_fixed / 1e6)} MB of weights, gradients and AdamW state.{_stand}

    | windows per batch | activations | times the 16 bytes per parameter |
    |---|---|---|
    {_rows}

    The activations grow in step with the batch. For a model this small, they are the largest part from a handful of windows on.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_fine():
    mo.md(r"""
    ### Exercise 2. Fine-tuning on one GPU
    """)
    return


@app.cell(hide_code=True)
def choice_fine_cell():
    choice_fine = L.choice(
        "fine-tune",
        question="You want to fine-tune all parameters of a model of 7 billion parameters with AdamW, in mixed precision, on one GPU of 24 GB. Does it fit?",
        options={
            "bf16": "Yes: in BF16 the model takes only 14 GB",
            "batch": "Yes, with a batch of one window",
            "right": "No, not even without activations",
        },
        correct="right",
        why={
            "bf16": "14 GB is the model in BF16 for using it. Training keeps FP32 master weights, their gradients and AdamW's state: 16 bytes per parameter.",
            "batch": "A small batch shrinks only the activations; the 112 GB of weights, gradients and AdamW state stay.",
            "right": "Weights, gradients and AdamW state alone take $7 \\times 10^9 \\times 16 = 112$ GB. People fit this on one GPU by training only a few of the parameters, with the rest kept in BF16 or fewer bits: a later lesson.",
        },
    )
    choice_fine
    return (choice_fine,)


@app.cell(hide_code=True)
def text_exercise_oom():
    mo.md(r"""
    ### Exercise 3. Out of memory
    """)
    return


@app.cell(hide_code=True)
def explain_oom():
    explain_oom_cell = L.explain(
        "out-of-memory",
        question="Training your GPT stops with \"out of memory\" on a GPU. Name three things you can change, and for each one which part of the memory it shrinks, and what it costs.",
        model_answer="**A smaller batch, with gradient accumulation:** shrinks the activations, which grow in step with the batch; accumulation keeps the effective batch size the same, at the cost of more steps per update and a little more time. **Shorter windows:** shrinks the activations, and the attention scores even with the square of the length; the model then sees less context at once. **Mixed precision:** keeps much of the activations in BF16, so they shrink, though by less than half, and it often runs faster on a GPU; the cost is coarser steps in the calculations, and on a GPU without BF16 you need FP16 with loss scaling. The weights, gradients and optimizer state stay as large. **An optimizer with less state,** such as SGD with momentum instead of AdamW, or one that stores its state in 8 bits: shrinks the optimizer state; it may train worse or need retuning. **A smaller model:** shrinks everything, at the cost of what the model can learn.",
        points=[
            "you name at least three changes",
            "for each, you name the part it shrinks: activations, optimizer state, or weights and gradients",
            "for each, you name a cost, such as more time, less context, worse training or a weaker model",
        ],
    )
    explain_oom_cell
    return (explain_oom_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Adam's two averages come from [Adam](/course/open/opt-adam?lang=en), the FP32 master weights from [Mixed-precision training](/course/open/sys-mixed-precision?lang=en), and gradient accumulation from [A reusable training loop](/course/open/nn-trainer?lang=en). The next lesson, *Estimating training compute: the 6ND rule*, adds up the other budget of a run: its FLOPs, and so its time.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fine, explain_oom_cell, result_gb):
    L.wrap_up(result_gb, choice_fine, explain_oom_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
