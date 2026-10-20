# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-lora. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-lora", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Rank: how many directions a matrix has](/course/open/math-matrix-rank?lang=en) and [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rank():
    quick_rank = L.choice(
        "quick-rank",
        question="You add to a weight matrix of shape (96, 96) the product `B @ A`, with `B` of shape (96, 8) and `A` of shape (8, 96). What is the highest rank that this change can have?",
        options={
            "full": "96",
            "sum": "104",
            "right": "8",
        },
        correct="right",
        why={
            "full": "Every row of `B @ A` is a mix of the 8 rows of `A`, so it has at most 8 directions. See [Rank: how many directions a matrix has](/course/open/math-matrix-rank?lang=en).",
            "sum": "The rank of a product is at most its smallest inner size: the 8 columns of `B` and rows of `A`, not 96 and 8 added up. See [Rank: how many directions a matrix has](/course/open/math-matrix-rank?lang=en).",
            "right": "A (96, 8) matrix times an (8, 96) matrix has rank at most 8. Today such a product is all a model learns.",
        },
    )
    quick_rank
    return (quick_rank,)


@app.cell(hide_code=True)
def quick_check_padding():
    quick_padding = L.choice(
        "quick-padding",
        question="In SFT, why do the targets of the padding in a batch get the label -100?",
        options={
            "stop": "So that the model learns to stop after its answer",
            "right": "So that they count for nothing in the loss",
            "id": "Because -100 is the id of the padding token",
        },
        correct="right",
        why={
            "stop": "The model learns to stop from the end-of-text at the end of every answer, with its label counted. The padding after it should teach nothing. See [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).",
            "id": "-100 is no token at all: it is the label that `F.cross_entropy` skips. The inputs are padded with the end-of-text. See [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).",
            "right": "`F.cross_entropy` skips every target of -100. Today you fine-tune on the same batches, with far fewer numbers that learn.",
        },
    )
    quick_padding
    return (quick_padding,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Full fine-tuning trains every number of the model. AdamW keeps two numbers of its own for each one that learns, and there is a gradient for each too: for a model of billions of parameters that is several copies of the model in memory. And every fine-tuned version is a whole new model to store. **LoRA**, low-rank adaptation (Hu and others, 2021), freezes the model and learns for each weight matrix only a small correction of low rank. On GPT-3, with 175 billion parameters, it can train up to 10,000 times fewer numbers than full fine-tuning with Adam, with a third of the GPU memory.

    **An everyday example.** You write your notes in the margin of a big book instead of rewriting the book. The book stays as it was; the notes are small, you can copy them into another copy of the same book, and rub them out.
    """)
    return


@app.cell(hide_code=True)
def predict_lora():
    _m, _f = C.MEASURED["yours"], C.T.MEASURED["yours"]
    guess_lora = L.guess(
        "lora",
        question=f"In the SFT project, the course's model of the capstone's demo run did {_f['after']['used']} of {_f['after']['total']} test instructions after {C.number(C.STEPS, 0, L.lang)} steps of SFT on all its {C.number(_m['params'], 0, L.lang)} numbers. LoRA of rank {_m['rank']} freezes them all and trains {C.number(_m['trainable'], 0, L.lang)} new ones, on the same examples for the same number of steps. How many instructions does it do?",
        options={
            "fewer": f"Far fewer: about {round(_f['after']['used'] / 3)}",
            "right": f"About as many: {round(_f['after']['used'] * 2 / 3)} to {round(_f['after']['used'] * 4 / 3)}",
            "more": f"Far more: about {2 * _f['after']['used']}",
        },
        reveal=f"**{_m['after']['used']} of {_m['after']['total']}**, against {_f['after']['used']} for full SFT. With {C.number(100 * _m['trainable'] / _m['params'], 1, L.lang)}% as many numbers that learn, and a learning rate five times as high, a few more; Hu and others also found LoRA as good as full fine-tuning or better. They guessed that the change fine-tuning makes to a weight matrix has a low rank; for this model and these instructions, an update of rank {_m['rank']} in every linear layer of the blocks is enough.",
    )
    guess_lora
    return (guess_lora,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **A frozen weight and a low-rank update.** A linear layer computes $W x$. LoRA keeps $W$ as it is, and adds a matrix $B A$: $A$ has $r$ rows, $B$ has $r$ columns, so $B A$ has the shape of $W$ but a rank of at most $r$, the **rank** of the update. Only $A$ and $B$ learn. The update is scaled by $\alpha / r$. Hu and others keep $\alpha$ fixed when they change the rank, which helps them keep the learning rate; many take $\alpha$ twice the rank instead, a scale of 2, as the demo does:
    """)
    return


@app.cell(hide_code=True)
def formula_lora():
    L.formula(r"h = W x + \frac{\alpha}{r} \, B A x")
    return


@app.cell(hide_code=True)
def picture_lora():
    _d, _r = C.MEASURED["yours"]["d"], C.RANK
    mo.vstack([
        mo.md(f"The picture, to scale: a weight matrix of the course's model of the capstone's demo run, {_d} by {_d}, and an update of rank {_r}. The frozen weight is grey; what learns has the colour of what a model learns."),
        C.lora_picture(_d, _r, {"title": f"A frozen weight and an update of rank {_r}",
                                "w": f"W: {_d} × {_d}, frozen: {C.number(_d * _d, 0, L.lang)} numbers",
                                "ba": f"B: {_d} × {_r}, A: {_r} × {_d}: {C.number(2 * _d * _r, 0, L.lang)} numbers that learn",
                                "alt": f"A big grey square W of {_d} by {_d} numbers, plus a tall thin matrix B and a wide thin matrix A in orange: {C.number(2 * _d * _r, 0, L.lang)} numbers that learn."}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    **Start at zero.** $A$ starts random, $B$ at zero (Hu and others, 2021). Then $B A$ is zero, and the model starts exactly as it was. $B$ gets a gradient from the first step, because $A$ is not zero; once $B$ moves, $A$ learns too.

    **Merge, or switch off.** After training, $W + \frac{\alpha}{r} B A$ is one matrix of the shape of $W$: merged in, the update costs nothing when the model writes. Kept apart, the update is an **adapter**: a small file, and switched off, the model is the one you started with.

    **Where.** In most of their experiments, Hu and others put LoRA only on the query and the value of attention. Dettmers and others (2023) found that LoRA needs to be on every linear layer of the blocks to match full fine-tuning. Your `add_lora` does that: seven layers per block of your ModernGPT. The embedding and the head stay frozen.

    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **LoRA** | low-rank adaptation: fine-tuning only a low-rank update $B A$ of each frozen weight matrix | `LoRALinear(base, r, alpha)` |
    | $A$, $B$ | the two thin matrices of the update: $A$ of shape $(r, \text{in})$, $B$ of shape $(\text{out}, r)$ | `self.A`, `self.B` |
    | $r$ | the **rank** of the update: the number of rows of $A$ and of columns of $B$ | `r` |
    | $\alpha$ | alpha: with $r$, the scale $\alpha / r$ of the update | `alpha / r` |
    | **adapter** | the update of a model kept apart from it, which can be switched on and off | `layer.on = False` |
    | **merge** | adding the scaled update into the weight, so that one matrix is left | `W + scale * B @ A` |

    ### Recipe

    To fine-tune a model with LoRA:

    1. Freeze every parameter of the model.
    2. Put a LoRA layer around every linear layer of its blocks: the frozen layer, $A$ random, $B$ zero.
    3. Train only $A$ and $B$, with the step and the examples of full fine-tuning, at a higher learning rate.
    4. Keep the adapters, or merge them into the weights.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A weight of shape (4, 4) and an update of rank 1. The numbers are random.
    """)
    return


@app.cell
def example_lora():
    torch.manual_seed(0)
    _W = torch.randn(4, 4)
    _A = torch.randn(1, 4)
    _B = torch.zeros(4, 1)
    _x = torch.randn(4)
    print(torch.allclose(_W @ _x + _B @ _A @ _x, _W @ _x))   # B is zero: nothing changes yet
    _B = torch.tensor([[1.0], [0.5], [0.0], [-1.0]])         # as after some training
    print(_W.numel(), _A.numel() + _B.numel())
    print(np.linalg.matrix_rank((_B @ _A).numpy()))
    _merged = _W + _B @ _A
    print(torch.allclose(_merged @ _x, _W @ _x + _B @ (_A @ _x)))   # merged: one matrix, the same output
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At the start $B$ is zero, so the layer gives what $W$ gives. After some training the update has 8 numbers against the 16 of $W$, and a rank of 1: every row of $B A$ is a multiple of $A$. Merged into one matrix, it gives the same output. For a matrix of 96 by 96 and a rank of 8, it is 1,536 numbers against 9,216.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A LoRA layer

        Complete `LoRALinear(base, r, alpha)`: the layer of step 2. It keeps the linear layer `base`, frozen, and makes `A` of shape `(r, base.in_features)`, random, and `B` of shape `(base.out_features, r)`, zero, both as `nn.Parameter`. Its `forward` gives what `base` gives, plus, when `self.on` is True, the update `x @ self.A.T @ self.B.T` times `self.scale`. The check makes `LoRALinear(nn.Linear(6, 4), 2, 4)` and tries it on three inputs of 6 numbers, and more.
        """),
        L.task("Make `LoRALinear` start as its base, learn only `A` and `B`, and switch off."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("lora-linear", lines=20)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("lora-linear", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "lora-linear",
        [
            "Freezing a parameter `p` is `p.requires_grad = False`, for every parameter of the base. A tensor becomes a parameter of your own with `nn.Parameter(...)`; `torch.zeros` and `torch.randn` make tensors of the shapes the task gives, and dividing the random numbers by `base.in_features ** 0.5` keeps them small.",
            "`x` has `in_features` numbers in its last dimension. `x @ self.A.T` takes them to `r` numbers, `@ self.B.T` to `out_features`: the shape of what `base(x)` gives, also for a batch of windows.",
        ],
        why="With `B` at zero the layer starts as its base. Only `A` and `B` need a gradient, and `self.on` lets a later lesson switch the update off.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_add():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Adapters in your model

        Complete `add_lora(model, r, alpha)`: steps 1 and 2. Freeze every parameter of the model, then put a `LoRALinear` around each of the seven linear layers of every block: `query`, `key`, `value` and `out` of its attention `a`, and `gate`, `up` and `down` of its MLP `m`. Give back the list of the new layers. The check builds a small ModernGPT of 2 blocks, `ModernGPT(40, 16, 2, 1, 2)`, and calls `add_lora(model, 2, 4)`.
        """),
        L.task("Make `add_lora` freeze the model and give every linear layer of its blocks an adapter."),
    ])
    return


@app.cell(hide_code=True)
def exercise_add():
    box_add = L.code_box("add-lora", lines=16)
    box_add
    return (box_add,)


@app.cell(hide_code=True)
def check_add(box_add):
    result_add = L.check("add-lora", box_add.value)
    result_add
    return (result_add,)


@app.cell(hide_code=True)
def hints_add():
    L.hints(
        "add-lora",
        [
            "Freeze first, as in your `LoRALinear`, but for `model.parameters()`. The `A` and `B` you make after that are new parameters, and they learn.",
            "`a.query = LoRALinear(a.query, r, alpha)` puts an adapter around one layer: the new layer takes the place of the old one in the model, and keeps it as its `base`.",
        ],
        why="The order matters: what you freeze after adding the adapters, freezes the adapters too. Seven adapters per block is what Dettmers and others found that LoRA needs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Merge the update

        Complete `merge_lora(layer)`: step 4. Give back a new `nn.Linear` whose weight is the weight of `layer.base` plus `layer.scale` times `layer.B @ layer.A`, with the bias of the base when it has one. The base itself stays as it was. The check merges a `LoRALinear(nn.Linear(6, 4), 2, 4)` whose `B` is no longer zero, and more.
        """),
        L.task("Make `merge_lora` give one linear layer that does what the layer and its update do."),
    ])
    return


@app.cell(hide_code=True)
def exercise_merge():
    box_merge = L.code_box("merge-lora", lines=12)
    box_merge
    return (box_merge,)


@app.cell(hide_code=True)
def check_merge(box_merge):
    result_merge = L.check("merge-lora", box_merge.value)
    result_merge
    return (result_merge,)


@app.cell(hide_code=True)
def hints_merge():
    L.hints(
        "merge-lora",
        [
            "`layer.B @ layer.A` has the shape of the weight: (out_features, in_features). The merged weight is the weight of the base plus that product, scaled as in `forward`.",
            "`merged.weight.copy_(...)` puts numbers into the new layer, under `torch.no_grad()`; `merged.bias.copy_(base.bias)` does the same for the bias.",
        ],
        why="The update is linear, so it can be added into the weight: one matrix gives the same output, and the model writes as fast as before.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _m = C.MEASURED["yours"]
    _r = _m["rank"]
    mo.vstack([
        mo.md(f"""
        ### Fine-tune your model with LoRA

        The button does what the demo of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en) did, with your `sft_example`, `sft_batch` and `pretrain_step`, but on your model of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en) with your `add_lora` of the rank you choose and alpha twice the rank: {C.number(C.STEPS, 0, L.lang)} steps at a learning rate of {C.number(C.LR, 3, L.lang)}, five times that of full SFT, because the update starts at zero and has few numbers. Then the model answers the same {C.T.TESTS} test instructions, counted with your `words_used`, and your `merge_lora` merges every adapter, to compare the logits. On the course's CPU, with one thread, rank {_r} took about {round((_m['seconds'] + _m['other_seconds']) / 60)} minutes for its model of the capstone's demo run; the course's own model, about five times as big, takes up to about ten minutes on two threads, at rank 16. The adapters are kept in `work/projects/llm/lora.pt`.

        On a CPU, and with a model this small, a step of LoRA is not faster than one of full SFT: the backward pass still goes through every layer. LoRA saves memory, and file size: the optimizer keeps its two numbers only for what learns.

        While it trains, the chart shows the loss of every step, with your run of full SFT on the same model dashed beside it, if you have done one.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def rank_cell():
    rank_pick = mo.ui.dropdown(options={str(_r): _r for _r in C.RANKS}, value=str(C.RANK), label="Rank")
    lora_button = mo.ui.run_button(label="Fine-tune with LoRA")
    mo.hstack([rank_pick, lora_button], align="end", justify="start")
    return lora_button, rank_pick


@app.cell(hide_code=True)
def demo_lora(box_add, box_layer, box_merge, lora_button, rank_pick, result_add, result_layer, result_merge):
    adapted = None
    _note = None
    _live = None
    if not all(_r.state == "correct" for _r in (result_layer, result_add, result_merge)):
        _note = "*Finish exercises 1 to 3 first: the demo fine-tunes with your `LoRALinear` and `add_lora`, and merges with your `merge_lora`.*"
    elif lora_button.value:
        _live = L.live({"title": "Your LoRA run: the loss of every step", "x": "step", "y": "loss",
                        "alt": f"The loss of every step of your LoRA run so far, in grey, and its mean over the last {max(1, C.STEPS // 50)} steps as the thicker line; dashed, the loss of your run of full SFT on the same model, if you have done one."}, C.STEPS, low=0.0)

        def _progress(_p):
            if _p["phase"] == "pairs":
                _live.reference(_p["full"], "full SFT")
                _live.line(f"**{C.number(_p['pairs'], 0, L.lang)} examples** · adapters of rank {rank_pick.value} …")
            elif _p["phase"] == "after":
                _live.line(f"**After LoRA** · instruction {_p['done']} of {_p['total']} · every word used: {_p['used']}")
            else:
                _live.extend(_p["fresh"], f"**Step {C.number(_p['step'], 0, L.lang)} of {C.number(_p['steps'], 0, L.lang)}** · loss {C.number(_p['loss'], 3, L.lang)} · {C.number(_p['seconds'] / 60, 1, L.lang)} minutes")
        _r = C.run(rank_pick.value, _progress)
        if _r == "run":
            _note = "*The demo stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
        elif _r == "lora":
            _note = "*Your adapters stop with an error on the model or when they are merged, change what it gives before any training, or leave nothing to learn. Do your `LoRALinear`, `add_lora` and `merge_lora` still pass their checks?*"
        elif _r == "memory":
            _note = "*The memory of your device is full. Close other programs, and press the button again.*"
        elif _r == "diverged":
            _note = "*The loss is no longer a number: the training ran off. Try a lower rank, or press the button again.*"
        else:
            adapted = _r
    if _note and _live is not None and _live.xs:
        _live.line(_note)   # a run that stops after its first step keeps its chart, with the note under it
    else:
        mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (adapted,)


@app.cell(hide_code=True)
def demo_result(adapted):
    _m = C.MEASURED["yours"]
    _run = adapted if adapted is not None else C.saved() or _m
    _mine = adapted is not None or C.saved() is not None
    _f = C.full_sft(_run.get("base", _run.get("source")) if _mine else None)
    _same = (_f["source"] == "yours") == _mine   # both rows on one model: the learner's two runs, or the course's two
    _rows = [f"| full SFT | {C.number(_f['params'], 0, L.lang)} | {C.number(2 * _f['params'], 0, L.lang)} | **{_f['used']}** of {_f['total']} |",
             f"| LoRA, rank {_run['rank']} | {C.number(_run['trainable'], 0, L.lang)} | {C.number(2 * _run['trainable'], 0, L.lang)} | **{_run['after']['used']}** of {_run['after']['total']} |"]
    _whose = "your model" if _mine and _run.get("base", _run.get("source")) == "yours" else "the course's model" if _mine else "the course's model of the capstone's demo run"
    _stand = "" if _mine else "\n\nThe course's runs stand in until you fine-tune your own."
    _sft = "" if _f["source"] == "yours" else " The row of full SFT is the course's run of the demo of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en)."
    _gap = f" Merged, the model gives the same logits up to {C.small(_run['gap'], L.lang)}." if _mine else ""
    _size = f" The adapters take {C.number(_run['bytes'] / 1000, 0, L.lang)} kB in `lora.pt`; the whole model takes {C.number(4 * _run['params'] / 1e6, 1, L.lang)} MB in float32."
    _before = f" (before SFT: {_f['before']} of them)" if _same else ""
    mo.md(f"The answers of {_whose} to {_run['after']['total']} held-out instructions{_before}:\n\n"
          "| | numbers that learn | numbers AdamW keeps | every word used |\n|---|---|---|---|\n" + "\n".join(_rows) + _stand
          + f"\n\n{_sft}{_gap}{_size}")
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(f"""
    ### Your LoRA run, step by step

    As in the project on SFT, the button lets your model answer the first test instruction six times while it trains: before the first step and after every {C.number(C.T.STEPS // C.T.SNAPSHOTS, 0, L.lang)} steps. Here you go through those moments: the loss along the run, with a dashed line at the moment you choose, and what your model wrote then. If you have done full SFT on the same model, that run is beside it as a thin dashed curve.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(adapted):
    _run = adapted if adapted is not None else C.saved()
    if _run is None:
        _out = mo.md("*Once you have fine-tuned your model with LoRA, you can go through its run here.*")
    elif not _run.get("snapshots"):
        _out = mo.md("*Your saved run has no moments to go through: it was made before the lesson kept them. Fine-tune with LoRA again to go through its run here.*")
    else:
        _labels = {"title": "The loss along your LoRA run", "x": "step", "y": "loss", "lang": L.lang, "reference": "full SFT",
                   "alt": "The loss of your LoRA run, the mean of every 50 steps, with a dashed line at the chosen moment, and thin and dashed the loss of your run of full SFT on the same model, if you have done one.",
                   "start": "Before the first step", "after": "After {step} steps", "used": "every word used", "not_used": "not every word used",
                   "stopped": "stopped by itself", "not_stopped": "did not stop by itself"}
        _full = C.full_sft_losses(_run.get("base", _run.get("source")))
        _out = L.explore("replay", "step", lambda _v: C.T.replay_frame(_run, int(_v), _labels, _full), start=_run["snapshots"][-1]["step"],
                         values=[_s["step"] for _s in _run["snapshots"]], role="",
                         task=f"At which moments does your model's answer to *{C.T.instruction(_run['snapshots'][0].get('words') or C.T.tests()[0]).splitlines()[0]}* use every word asked for, and stop by itself?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(f"""
    As in the project on SFT, whether this one answer uses every word can come and go from one moment to the next: one answer is one draw. The thin dashed curve, if it is there, is your full SFT: with every number of the model free to move, it usually fits its examples more closely, and its loss ends lower than that of LoRA. That does not by itself mean that the model's own stories use the words more often: that is what the table above counts, on {C.T.TESTS} held-out instructions.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Exercise 4. Why zero?
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "why-zero",
        question="LoRA starts `B` at zero and `A` random. Why not both at zero?",
        options={
            "memory": "Zeros take less memory, so one zero matrix is enough",
            "right": "With both at zero, neither ever gets a gradient other than zero",
            "random": "Both at zero would work as well; random `A` is only a habit",
        },
        correct="right",
        why={
            "memory": "A matrix of zeros takes as much memory as any other. The reason is the gradient: the gradient of `B` goes through `A`, and that of `A` through `B`.",
            "random": "With both at zero, the gradient of `B` is zero because `A` is, and that of `A` because `B` is: nothing would ever learn.",
            "right": "One of them at zero is enough for the model to start as it was; the other must be random, so that the first gets a gradient.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The low rank of [Rank: how many directions a matrix has](/course/open/math-matrix-rank?lang=en), now as what a model learns; the gradients of matrix products of [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en), which reach `B` through `A`; and the examples, batches and check of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), now for adapters. Your project `llm` has `lora.py`, `add_lora.py` and `merge_lora.py` now, and after the demo your adapters in `lora.pt`. In a later lesson, DPO, an adapter is what learns, and the same model with its adapters switched off is the model it is held against.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_layer, result_add, result_merge, choice_zero):
    L.wrap_up(result_layer, result_add, result_merge, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
