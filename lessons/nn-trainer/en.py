# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-trainer. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-trainer", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en) and [Gradient clipping](/course/open/opt-gradient-clipping?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sched():
    quick_sched = L.choice(
        "quick-sched",
        question="A schedule that is counted in steps sets the learning rate of every step. When do you call `scheduler.step()`?",
        options={
            "epoch": "Once at the end of every epoch, however many steps it had",
            "right": "After every `opt.step()`, so that the next step gets its learning rate",
            "forward": "Before every forward pass, so that the loss uses the new learning rate",
        },
        correct="right",
        why={
            "epoch": "A schedule counted in steps would then move one step per epoch, and its warmup and decay would take far too long. See [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en).",
            "forward": "The loss does not use the learning rate; only the step of the optimizer does. See [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en).",
            "right": "One step of the optimizer, one step of the schedule. Today that pair goes into a loop you write once, with a twist: not every mini-batch gets a step.",
        },
    )
    quick_sched
    return (quick_sched,)


@app.cell(hide_code=True)
def quick_check_clip():
    quick_clip = L.choice(
        "quick-clip",
        question="You clip at a norm of 1. One step has a gradient with norm 0.5, the next one norm 3. What happens to them?",
        options={
            "both": "Both are scaled to norm 1, so that every step is the same size",
            "half": "Both are halved, because the limit is 1",
            "right": "The first stays as it is, the second goes down to norm 1",
        },
        correct="right",
        why={
            "both": "Clipping only makes a gradient smaller; one below the limit stays as it is. See [Gradient clipping](/course/open/opt-gradient-clipping?lang=en).",
            "half": "Clipping scales by `max_norm / norm`, and only when the norm is above the limit. See [Gradient clipping](/course/open/opt-gradient-clipping?lang=en).",
            "right": "Below the limit, nothing changes; above it, the gradient keeps its direction and gets norm 1. Today, clipping becomes one line of a loop that every project uses.",
        },
    )
    quick_clip
    return (quick_clip,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every project so far had its own training loop, and every new idea added lines to it: evaluation, a training log, checkpoints, clipping, a schedule. Copied into every project, such a loop copies its mistakes too, and a fix in one copy is missing in the others. You write it once, test it once, and import it everywhere. And one new idea goes in: on a small GPU, a large model leaves room for only a small mini-batch, while training goes better with a large one. **Gradient accumulation** adds up the gradients of several small mini-batches before one step.
    """)
    return


@app.cell(hide_code=True)
def predict_accum():
    guess_accum = L.guess(
        "four-small",
        question="A model gets one mini-batch of 64 examples, with `backward()` of the mean loss. Another copy of the same model gets the same 64 examples as four mini-batches of 16, with `backward()` of each mean loss divided by 4, and no step or `zero_grad()` in between. How do the two gradients compare?",
        options={
            "four": "The second is about four times as large",
            "right": "They are the same, up to rounding",
            "quarter": "The second is a quarter as large",
        },
        reveal="**The same, up to rounding.** `backward()` adds every new gradient to the one already there. Each of the four mean losses is the mean over 16 examples; a quarter of each, added up, is the mean over all 64. In the worked example below, the largest difference is less than 0.000001.",
    )
    guess_accum
    return (guess_accum,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Gradient accumulation** with $k$ mini-batches of $B$ examples: for each mini-batch, `backward()` of its mean loss divided by $k$; after every $k$ mini-batches, one step. The gradients add up, so the step is that of one mini-batch of $kB$ examples, the **effective batch size**:
        """),
        L.formula(r"[[grad:\nabla \mathcal{L}_{kB}]] = \frac{1}{[[hyper:k]]} \sum_{j=1}^{[[hyper:k]]} [[grad:\nabla \mathcal{L}_{B,\,j}]]"),
        mo.md(r"""
        A **trainer** is the loop around it, written once: for every epoch, train with accumulation, clipping and a schedule; then measure the loss on the validation set; write a line in the log; and save the parameters when the validation loss is the lowest so far. Everything it needs, it gets as arguments: the model, the two loaders, the number of epochs, the peak learning rate, the warmup in steps of the optimizer, `max_norm`, $k$ and where to save. Here it uses AdamW, and `torch.optim.lr_scheduler.LambdaLR`, which multiplies the peak learning rate at every step by what a function of the step gives: the warmup and cosine factor of [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en).

        The loss line is written so that it also fits a language model, whose logits have a time axis: shape (batch, time, vocabulary). `logits.reshape(-1, logits.shape[-1])` puts all predictions one under the other: the `-1` means as many rows as it takes, and `yb.reshape(-1)` their targets, so that `F.cross_entropy` gets the shapes it wants for both kinds of model.

        The exercises use three new bits of Python; they are in the table too.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **gradient accumulation** | | adding up the gradients of several mini-batches before one step | `(loss / accum).backward()` |
        | **effective batch size** | | the number of examples per step: the size of a mini-batch times $k$ | `batch_size * accum` |
        | $k$ | k | the number of mini-batches per step | `accum` |
        | **trainer** | | a training loop written once, with evaluation, a log, checkpoints, clipping and a schedule | `Trainer(...).fit()` |
        | `%` | modulo | the remainder after dividing: `7 % 3` is 1, so `k % 4 == 0` is `True` when `k` is a multiple of 4 | `k % accum == 0` |
        | `or` | | `True` when the condition on its left or the one on its right is `True`, or both | `k % accum == 0 or k == len(loader)` |
        | `is None` | | `True` when a variable is `None`; `is not None` is the opposite: there is something | `if sched is not None:` |
        """),
        L.watch_out("With gradient accumulation, the schedule counts steps, not mini-batches: an epoch of 94 mini-batches with $k = 4$ has 24 steps, the last one after only 2 mini-batches. And a batch normalization layer still sees only the small mini-batch: accumulation does not make its statistics come from more examples."),
        mo.md(r"""
        The picture: the trainer on the MLP of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), 8 epochs of AdamW with a peak learning rate of 0.003, warmup during the first epoch and cosine decay to 0, and clipping at 1. Left, the log: the training loss is the mean over the epoch, so in epoch 1 it lies above the validation loss, measured at its end; then both fall, the validation loss from 0.50 to 0.27. Right, the learning rate at the start and at the end of every epoch. After epoch 8, the network reads 915 of the 1,000 test digits right.
        """),
        mo.Html(C.history_picture({"loss_title": "Loss after every epoch", "lr_title": "Learning rate", "x": "epoch", "loss": "loss", "lr": "learning rate",
                                   "train": "training", "val": "validation",
                                   "alt": "Left: the training loss after every epoch falls from 1.25 to 0.11, the validation loss from 0.50 to 0.27. Right: the learning rate, close to 0 at the start, 0.003 after the warmup of epoch 1, then falling along a cosine to 0 after epoch 8."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A trainer, epoch after epoch:

    1. For every mini-batch: its loss, divided by $k$, and `backward()`; the gradients add up.
    2. After every $k$ mini-batches, and after the last one: clip, step, set the gradients to zero, and step the schedule.
    3. After every epoch: the mean loss on the validation set, in eval mode and without a graph, and back to train mode.
    4. A line in the log with the epoch, both losses and the learning rate; save the parameters when the validation loss is the lowest so far.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1, checked: the gradient of one mini-batch of 64 against that of four mini-batches of 16, for an `nn.Linear(4, 3)` with random data. The comments with `#` say which step of the recipe each part is. Step 1 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_accumulation():
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(0)
    _model = nn.Linear(4, 3)
    _X = torch.randn(64, 4)
    _y = (_X[:, 0] > 0).long() + (_X[:, 1] > 0).long()
    # to compare: the gradient of one mini-batch of 64
    F.cross_entropy(_model(_X), _y).backward()
    _full = _model.weight.grad
    _model.zero_grad()
    # 1. four mini-batches of 16: backward for each, without a step or zero_grad in between
    for _k in range(4):
        _loss = F.cross_entropy(_model(_X[16 * _k:16 * _k + 16]), _y[16 * _k:16 * _k + 16])
        # complete: divide the loss by 4 first
        _loss.backward()
    print([round(v, 4) for v in _full[0].tolist()])
    print([round(v, 4) for v in _model.weight.grad[0].tolist()])
    print(round((_model.weight.grad - _full).abs().max().item(), 6))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first line is row 0 of the gradient of the mini-batch of 64: 0.2481, 0.118, −0.0296 and −0.0886. `_model.zero_grad()` empties the gradient for the second way, and leaves `_full` as it was. As it is, every `backward()` of the four adds its whole gradient: row 0 is 0.9922, 0.4718, −0.1185 and −0.3546, four times as large, and the largest difference is 0.744157. Completed, with `(_loss / 4).backward()`, the rows are the same and the largest difference prints `0.0`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_divide():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why divide by k?
    """)
    return


@app.cell(hide_code=True)
def choice_divide_cell():
    choice_divide = L.choice(
        "why-divide",
        question="With gradient accumulation, why is each loss divided by $k$ before `backward()`?",
        options={
            "right": "So that the added-up gradient is a mean, as for one large mini-batch",
            "memory": "To save memory: a smaller loss takes less room in the graph",
            "clip": "So that the gradient always stays below the limit of the clipping, whatever happens",
        },
        correct="right",
        why={
            "memory": "A loss is one number either way. What saves memory is the small mini-batch itself.",
            "clip": "Dividing makes the gradient k times smaller than without, not always small; clipping still has its own job.",
            "right": "`backward()` adds up. Without dividing, k mini-batches give k times the gradient of a mean, and with SGD the step is k times larger; with the 1/k, the step equals that of one mini-batch of k times as many examples.",
        },
    )
    choice_divide
    return (choice_divide,)


@app.cell(hide_code=True)
def text_exercise_epoch():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One epoch, with everything

        Complete `train_epoch(model, loader, opt, sched, max_norm, accum)`: one epoch by steps 1 and 2 of the recipe. The loss line is there, and `k` counts the mini-batches. Replace the `...`: `backward()` of the loss divided by `accum`; and when `k` is a multiple of `accum`, or the last mini-batch (`k == len(loader)`): clip when `max_norm` is not `None`, take the step, set the gradients to zero, and step the schedule when `sched` is not `None`. The check runs an epoch of 5 mini-batches with `accum = 2`, then the same with clipping and a schedule, and then a model with logits of shape (4, 3, 5), as a language model gives them. When it is right, the check saves it in your project, as `work/projects/nn/train_epoch.py`.
        """),
        L.task("Make `train_epoch` train one epoch with gradient accumulation, clipping and a schedule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_epoch():
    box_epoch = L.code_box("train-epoch", lines=26)
    box_epoch
    return (box_epoch,)


@app.cell(hide_code=True)
def check_epoch(box_epoch):
    result_epoch = L.check("train-epoch", box_epoch.value)
    result_epoch
    return (result_epoch,)


@app.cell(hide_code=True)
def hints_epoch():
    L.hints(
        "train-epoch",
        [
            "First the line of the completed worked example, with `accum` instead of 4. Then one `if` with `%` and `or`, and under it, in this order: clipping, `opt.step()`, `opt.zero_grad()`, and the schedule.",
            "Clipping and the schedule are optional: each gets its own `if ... is not None:`, with the line of [Gradient clipping](/course/open/opt-gradient-clipping?lang=en) and `sched.step()`.",
        ],
        why="Dividing by `accum` makes the gradients of the mini-batches add up to a mean. The step comes after every `accum` mini-batches, and after the last one, so that no gradient is lost. Clipping goes before the step, because the step uses the gradients, and the schedule after it, once per step of the optimizer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_trainer(result_epoch):
    result_epoch   # made again after the check of exercise 2, which saves your train_epoch
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The trainer

        Complete the class `Trainer`. `__init__` is there: it keeps everything, makes AdamW and the schedule, and an empty `self.history`; `train_epoch` comes from your project. Replace the two `...`:

        - `evaluate()` gives the mean loss on `self.val_loader`, by step 3: `self.model.eval()`, a loop under `torch.no_grad()` with the same loss line as in `train_epoch`, and `self.model.train()` at the end.
        - `fit()` does step 4 for every epoch: `train_epoch` with the optimizer, the schedule, `max_norm` and `accum` of the trainer; then `evaluate()`; then a dict in `self.history` with `"epoch"` (from 1), `"train_loss"`, `"val_loss"` and `"lr"`, the learning rate of `self.opt.param_groups[0]`; and `torch.save(self.model.state_dict(), self.path)` when the validation loss is the lowest so far and `self.path` is not `None`.

        The check trains small networks on random data: it compares the log, loads the saved parameters, and measures twice with dropout. When it is right, the check saves it as `work/projects/nn/trainer.py`.
        """),
        L.course_versions(),
        L.task("Make `Trainer` evaluate, keep a log and save the best parameters, epoch after epoch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_trainer():
    box_trainer = L.code_box("trainer", lines=44)
    box_trainer
    return (box_trainer,)


@app.cell(hide_code=True)
def check_trainer(box_trainer):
    result_trainer = L.check("trainer", box_trainer.value)
    result_trainer
    return (result_trainer,)


@app.cell(hide_code=True)
def hints_trainer():
    L.hints(
        "trainer",
        [
            "`evaluate` is the loop of [Reading loss curves](/course/open/torch-loss-curves?lang=en) for the validation set, with the loss line of `train_epoch`; add `.item()` of every loss to a total and divide by `len(self.val_loader)`.",
            "In `fit`, `best` starts at `math.inf`, infinity, above any loss. A dict is written as `{\"epoch\": epoch + 1, ...}`; the save goes under `if val_loss < best:`, under which you also set `best = val_loss`.",
        ],
        why="`evaluate` switches to eval mode, so that dropout is off while it measures, and back to train mode at the end, or the next epoch would train without dropout; `torch.no_grad()` builds no graph, which saves time and memory. `fit` reads the learning rate after `train_epoch`, so the log has the learning rate at the end of the epoch, as in the picture, and it saves only when the validation loss is lower than every one before.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loop of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) grows into a trainer, with the validation loss of [Reading loss curves](/course/open/torch-loss-curves?lang=en), the best checkpoint of [Saving and loading a model](/course/open/torch-checkpoints?lang=en), the clipping of [Gradient clipping](/course/open/opt-gradient-clipping?lang=en), and AdamW of [AdamW: weight decay done right](/course/open/opt-adamw?lang=en) with the warmup and cosine decay of [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en). From now on, a project imports it: `from nn.trainer import Trainer`. Larger libraries, such as PyTorch Lightning, have the same kind of trainer, with callbacks: functions you hand it, which it calls at set moments, such as after every epoch. In the next lesson, *Project: make a deep network train*, a deep MLP fails at first, and you make it train step by step, with this trainer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_divide, result_epoch, result_trainer):
    L.wrap_up(choice_divide, result_epoch, result_trainer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
