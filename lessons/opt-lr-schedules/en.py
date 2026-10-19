# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson opt-lr-schedules. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow cosine annealing

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-lr-schedules", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [AdamW](/course/open/opt-adamw?lang=en) and [Reading loss curves](/course/open/torch-loss-curves?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_half():
    quick_half = L.choice(
        "quick-half",
        question="AdamW first shrinks every weight by the fraction $\\eta\\lambda$, then takes Adam's step, which is about $\\eta$ in size. You halve the learning rate $\\eta$. What happens?",
        options={
            "step-only": "Only Adam's step halves: the shrinking depends on $\\lambda$ alone",
            "right": "Both the shrinking and Adam's step halve: each goes with $\\eta$",
            "same": "Nothing: Adam rescales every step to the size of the gradient anyway",
        },
        correct="right",
        why={
            "step-only": "The fraction is $\\eta\\lambda$, so half the $\\eta$ is half the shrinking too. See [AdamW](/course/open/opt-adamw?lang=en).",
            "same": "Adam divides by $\\sqrt{\\hat{s}}$ to take out the size of the gradient, not the size of $\\eta$: every step is $\\eta$ times $\\hat{m} / \\sqrt{\\hat{s}}$. See [Adam](/course/open/opt-adam?lang=en).",
            "right": "Both go with $\\eta$: the shrinking is $\\eta\\lambda w$, and the step is $\\eta$ times $\\hat{m} / (\\sqrt{\\hat{s}} + \\varepsilon)$. So changing $\\eta$ during training changes every step directly.",
        },
    )
    quick_half
    return (quick_half,)


@app.cell(hide_code=True)
def quick_check_jumping():
    quick_jumping = L.choice(
        "quick-jumping",
        question="A training loss jumps up and down, and grows. What is the most likely cause?",
        options={
            "right": "A learning rate that is too large: steps overshoot",
            "overfitting": "Overfitting: the model learns its examples by heart",
            "small": "A learning rate that is too small: the model is stuck",
        },
        correct="right",
        why={
            "overfitting": "Overfitting shows in the validation loss, which rises while the training loss falls; it does not make the training loss jump. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "small": "A learning rate that is too small gives a loss that hardly moves, not one that jumps. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "Steps that are too large overshoot the bottom, and the loss jumps. A learning rate that is far too large can even leave the loss stuck at the loss of guessing. Today you see how to change the learning rate during training.",
        },
    )
    quick_jumping
    return (quick_jumping,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    One learning rate for the whole training is a compromise. At the start, the weights are random and the gradients can be large and point in odd directions: big steps there can wreck the training. In the middle, big steps make the most progress. At the end, every mini-batch gives a slightly different gradient, and big steps keep the weights bouncing around the bottom of the valley instead of settling in it. A **learning-rate schedule** changes the learning rate during training, and almost every large model is trained with one.
    """)
    return


@app.cell(hide_code=True)
def predict_warmup():
    guess_warmup = L.guess(
        "with-warmup",
        question="The MLP of 10 residual blocks from [Residual connections](/course/open/nn-residual?lang=en) trains for one epoch with AdamW at a learning rate of 0.03, 30 times Adam's usual 0.001. Its loss shoots up in the first 20 steps and ends at 2.35, about the loss of guessing: it reads 131 of the 1,000 test digits right. Now the learning rate starts near 0 and climbs evenly to 0.03 over the first 30 steps, and then stays at 0.03. What do you think happens?",
        options={
            "later": "It still gets stuck, only some steps later: the peak is the same",
            "right": "It trains: after one epoch it reads more than 800 of the test digits right",
            "stuck": "It hardly learns: the steps at the start are too small",
        },
        reveal="**It trains: 827 of the 1,000 test digits right.** Without warmup, the first steps, as large as all the others, push almost every ReLU of the first layer below 0 for every image: those neurons are dead, and from then on the network gives nearly the same answer for every digit. With warmup, the first steps are small while the network leaves its random start, and by the time they are large, the weights can take them. With a cosine decay after the warmup, it reads 869.",
    )
    guess_warmup
    return (guess_warmup,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A schedule gives the learning rate as a function of the step $t$, from 0 to the last step $T - 1$. The most used one today starts with a **warmup**: over the first $W$ steps, the learning rate climbs evenly from almost 0 to its peak $\eta_{\max}$:
        """),
        L.formula(r"[[hyper:\eta_t]] = [[hyper:\eta_{\max}]] \, \frac{t + 1}{W} \qquad (t < W)"),
        mo.md(r"""
        After that comes **cosine decay**: the learning rate falls from the peak to a minimum $\eta_{\min}$ along half a cosine, with $r = (t - W)/(T - W)$:
        """),
        L.formula(r"[[hyper:\eta_t]] = [[hyper:\eta_{\min}]] + \tfrac{1}{2} ([[hyper:\eta_{\max}]] - [[hyper:\eta_{\min}]]) \bigl(1 + \cos(\pi r)\bigr)"),
        mo.md(r"""
        $r$ runs from 0 right after the warmup to 1 at step $T$, just after the last step. You only need the shape of $\cos$ between 0 and $\pi$: it falls from 1 to −1, slowly at both ends and fastest in the middle. So $\tfrac{1}{2}(1 + \cos(\pi r))$ falls from 1 to 0 in the same way, and the learning rate falls from the peak to the minimum: slowly at first, then faster, and slowly again at the end, where the weights settle. In Python, `math.cos` and `math.pi` from the `math` module give $\cos$ and $\pi$.

        Older schedules are simpler: **step decay** multiplies the learning rate by, say, 0.1 every few epochs, and **exponential decay** multiplies it by a factor like 0.99 every step. The transformer of 2017 used a warmup, followed by a learning rate that falls as one over the square root of the step. A third kind looks at the validation loss and lowers the learning rate when it stops improving: `torch.optim.lr_scheduler.ReduceLROnPlateau`. In PyTorch, an optimizer keeps its settings per parameter group, as in [AdamW](/course/open/opt-adamw?lang=en): each group is a dict in the list `opt.param_groups`, and its learning rate is `group["lr"]`. A scheduler sets it: `torch.optim.lr_scheduler` has `LinearLR` for a warmup and `CosineAnnealingLR` for the decay (PyTorch calls it cosine annealing; after its last step it climbs again), `SequentialLR` puts them one after the other, and you call `scheduler.step()` after every `opt.step()`.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **learning-rate schedule** | | the learning rate as a function of the step | `lr_at(step)` |
        | **warmup** | | the first steps, in which the learning rate climbs from almost 0 to its peak | `peak * (step + 1) / warmup` |
        | **cosine decay** | | the learning rate falling from the peak to a minimum along half a cosine | `0.5 * (1 + math.cos(math.pi * r))` |
        | $\cos$ | cos | the cosine: between 0 and $\pi$ it falls from 1 to −1, slowly at both ends | `math.cos(x)` |
        | $\eta_t$ | eta t | the learning rate at step $t$ | `group["lr"]` |
        | **step decay**, **exponential decay** | | multiplying the learning rate by a factor every few epochs, or every step | `lr * 0.1`, `lr * 0.99` |
        """),
        L.watch_out("The schedule needs the total number of steps $T$ from the start: if you train longer than planned, the cosine has already reached its minimum. And the warmup and the peak go together: a warmup does not save a peak that is far too large. When large models are pretrained, the warmup is usually 1% of all steps or less, in fine-tuning a few percent; the 30 of 94 here is long, because one epoch is short."),
        mo.md(r"""
        The picture: left, the three schedules for one epoch of 94 mini-batches; after its warmup, the schedule with warmup only stays at the peak, on the line of the constant one. The two without decay are dashed, the one with cosine decay is solid, here and on the right. Right, the training loss of the network from the guess with each schedule, averaged over the last 10 mini-batches, on a log scale. With a constant 0.03, the loss shoots up and ends at 2.35, and the network reads 131 test digits right. With 30 steps of warmup, it ends at 0.36 and reads 827. With warmup and cosine decay, the last steps are small, the loss ends a little lower, at 0.34, and it reads 869.
        """),
        mo.Html(C.schedule_picture({"lr_title": "Learning rate", "loss_title": "Training loss", "x": "mini-batch", "lr": "learning rate", "loss": "loss",
                                    "constant": "constant", "warmup": "warmup", "cosine": "+ cosine", "both": "warmup, and warmup + cosine",
                                    "alt": "Left: three learning-rate schedules over 94 mini-batches: constant at 0.03; warmup from 0 to 0.03 over 30 steps, then constant; the same warmup, then a cosine down to 0. Right, on a log scale, the training loss with each. With the constant learning rate it shoots up to about 30 and ends at about 2.3; the two schedules with warmup fall to about 0.35, close together."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A warmup and cosine schedule:

    1. Choose the peak $\eta_{\max}$, the number of warmup steps $W$, the total number of steps $T$, and a minimum $\eta_{\min}$, often 0 or a tenth of the peak.
    2. Steps $0$ to $W - 1$: $\eta_t = \eta_{\max}\,(t + 1)/W$.
    3. Steps $W$ to $T - 1$: $r = (t - W)/(T - W)$ and $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi r))$. From $T$ on: $\eta_{\min}$.
    4. Before every step of the optimizer, set $\eta_t$ in every parameter group: `group["lr"] = ...`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The schedule of the picture (peak 0.03, 30 warmup steps, 94 steps in all, minimum 0), worked out at a few steps and at one step after the end. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again. As it is, the learning rate stays at the peak after the warmup.
    """)
    return


@app.cell
def example_schedule():
    import math

    # 1. the numbers of the schedule
    _peak = 0.03
    _warmup = 30
    _total = 94
    _minimum = 0.0
    for _t in [0, 14, 29, 40, 62, 90, 100]:
        # 2. warmup
        if _t < _warmup:
            _lr = _peak * (_t + 1) / _warmup
        # 3. from _total on: the minimum
        elif _t >= _total:
            _lr = _minimum
        # 3. complete: with _r = (_t - _warmup) / (_total - _warmup), the cosine decay from _peak to _minimum
        else:
            _lr = _peak
        print(_t, round(_lr, 6))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The warmup gives 0.001 at step 0, 0.015 at step 14 and the full 0.03 at step 29; step 100, after the end, gets the minimum, 0. As it is, the cell stays at 0.03 in between. With the cosine, step 40 has $r = 10/64 \approx 0.16$ and a learning rate of 0.028229, still close to the peak; step 62 is halfway, $r = 0.5$, where $\cos(\pi/2) = 0$ gives exactly half the peak, 0.015; and step 90 has 0.000288, almost nothing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_decay():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why decay at the end
    """)
    return


@app.cell(hide_code=True)
def choice_decay_cell():
    choice_decay = L.choice(
        "why-decay",
        question="Why do almost all schedules make the learning rate small toward the end of training?",
        options={
            "larger": "The gradients get larger near the end of training, and smaller steps make up for that",
            "right": "Small steps let the weights settle, instead of bouncing around the bottom on noisy gradients",
            "overfit": "Small steps at the end undo the overfitting caused by the large steps before them",
        },
        correct="right",
        why={
            "larger": "Near the bottom of the loss, the gradients usually get smaller, not larger. What stays is their noise from one mini-batch to the next.",
            "overfit": "Small steps do not undo anything; overfitting is about what the model learns, and is fought with a validation set and regularization.",
            "right": "Every mini-batch gives a slightly different gradient. With a large learning rate, each noisy step throws the weights a little to one side or the other of the bottom; smaller steps throw them less far, and the loss goes down further. In the picture, warmup with cosine decay ends lower than warmup only, and reads 869 digits instead of 827.",
        },
    )
    choice_decay
    return (choice_decay,)


@app.cell(hide_code=True)
def text_exercise_lr():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The learning rate at a step

        Write `lr_at(step, peak, warmup, total, minimum)`: the learning rate of the warmup and cosine schedule at step `step`, by steps 2 and 3 of the recipe, also for steps from `total` on. The check calls `lr_at(30, 0.002, 10, 50, 0.0)`, which should give `0.001`, halfway down, and tries warmup steps, the last steps, steps after the end, and a minimum above 0.
        """),
        L.task("Make `lr_at` give the learning rate of the warmup and cosine schedule at a step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lr():
    box_lr = L.code_box("lr-at", lines=10)
    box_lr
    return (box_lr,)


@app.cell(hide_code=True)
def check_lr(box_lr):
    result_lr = L.check("lr-at", box_lr.value)
    result_lr
    return (result_lr,)


@app.cell(hide_code=True)
def hints_lr():
    L.hints(
        "lr-at",
        [
            "Three parts, each with its own `return`: `if step < warmup:`, then `if step >= total:`, and the rest.",
            "For the decay, first `r = (step - warmup) / (total - warmup)`; then step 3 of the recipe, with `math.cos` and `math.pi`: only the part above the minimum decays.",
        ],
        why="`if step < warmup: return peak * (step + 1) / warmup`; `if step >= total: return minimum`; `r = (step - warmup) / (total - warmup)`; `return minimum + 0.5 * (1 + math.cos(math.pi * r)) * (peak - minimum)`. In the example, $r = 20/40 = 0.5$, $\\cos(\\pi/2) = 0$, so $0.5 \\cdot 1 \\cdot 0.002 = 0.001$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scheduler():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A scheduler for an optimizer

        Write a class `WarmupCosine`, a scheduler as PyTorch has them, with a minimum of 0. `WarmupCosine(opt, peak, warmup, total)` keeps the optimizer and the numbers, starts a step counter at 0, and already sets the learning rate of step 0 in every group of `opt.param_groups`. `step()`, called after every `opt.step()`, raises the counter by one and sets the learning rate of that next step in every group. You may copy your `lr_at` into the box and use it. The check makes an SGD optimizer with two parameter groups, attaches your scheduler, and writes down the learning rate before every step, with a short schedule and with another one.
        """),
        L.task("Make `WarmupCosine` set the learning rate of the schedule before every step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scheduler():
    box_scheduler = L.code_box("warmup-cosine", lines=22)
    box_scheduler
    return (box_scheduler,)


@app.cell(hide_code=True)
def check_scheduler(box_scheduler):
    result_scheduler = L.check("warmup-cosine", box_scheduler.value)
    result_scheduler
    return (result_scheduler,)


@app.cell(hide_code=True)
def hints_scheduler():
    L.hints(
        "warmup-cosine",
        [
            "Setting the learning rate is the same in `__init__` and in `step`: a loop `for group in self.opt.param_groups:` with `group[\"lr\"] = ...`. A method of its own for it saves writing it twice.",
            "`step()` comes after a step of the optimizer, so first the counter one up, then the learning rate for that new count.",
        ],
        why="`__init__`: `self.opt = opt`, the three numbers, `self.t = 0`, and set the learning rate. `step`: `self.t = self.t + 1`, and set it again. Setting it: for each group, `group[\"lr\"] = lr_at(self.t, self.peak, self.warmup, self.total, 0.0)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The jumping loss of [Reading loss curves](/course/open/torch-loss-curves?lang=en) comes from steps that are too large, and the first steps from a random start are the most dangerous: in the guess, they turned the ReLUs of the first layer into dead neurons, as in [Looking inside a training run](/course/open/nn-activation-stats?lang=en). Warmup keeps those steps small. Because the steps of [AdamW](/course/open/opt-adamw?lang=en) go with its learning rate, a schedule sets the size of every step directly; AdamW with warmup and cosine decay is how most transformers are trained. In the next lesson, *Gradient clipping*, you guard against the other danger: a single gradient that is suddenly huge, at any point in training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_decay, result_lr, result_scheduler):
    L.wrap_up(choice_decay, result_lr, result_scheduler)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
