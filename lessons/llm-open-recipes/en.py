# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-open-recipes. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-open-recipes", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en) and [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_read():
    quick_read = L.choice(
        "quick-read",
        question="Your `TokenWindows` opens 7 shards with `np.memmap`. When are the ids of window 500 read from the disk?",
        options={
            "right": "When a DataLoader asks for window 500: `__getitem__` reads its ids then",
            "start": "When the Dataset is made: `__init__` reads every window once, at the start",
            "never": "Never: a memory-mapped file keeps all its ids in memory from the moment it is opened",
        },
        correct="right",
        why={
            "start": "`__init__` only opens the shards and lists where every window starts. See [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en).",
            "never": "A memory-mapped file stays on the disk; indexing it reads those ids. See [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en).",
            "right": "The demo of this lesson trains from your shards that way.",
        },
    )
    quick_read
    return (quick_read,)


@app.cell(hide_code=True)
def quick_check_swiglu():
    quick_swiglu = L.choice(
        "quick-swiglu",
        question="What does the gate of SwiGLU in the MLP of your ModernGPT do?",
        options={
            "next": "It chooses which block the tokens go through next",
            "norm": "It normalizes the input of the MLP, as RMSNorm does",
            "right": "It scales every number of the other branch, so the MLP can let some parts through and hold others back",
        },
        correct="right",
        why={
            "next": "Every token goes through every block, in order. The gate works inside one MLP: see [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).",
            "norm": "The normalization comes before the MLP. The gate multiplies: `F.silu(self.gate(x)) * self.up(x)`. See [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).",
            "right": "nanochat, the recipe of this lesson, uses another MLP: ReLU squared, without a gate.",
        },
    )
    quick_swiglu
    return (quick_swiglu,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A paper tells you what a team found; a fully open training recipe lets you do it yourself. nanochat, by Andrej Karpathy, is such a recipe: its code, its data and every setting are public, and one script runs it all, from the tokenizer to a model you can chat with. In 2019, training GPT-2 cost about 43,000 dollars. nanochat keeps a leaderboard of how fast its recipe beats GPT-2's score on CORE, a set of benchmarks, on one computer with 8 H100 GPUs, and every entry names the commit of the code that did it. Reading such a recipe end to end shows which choices a real run makes, and lets you change one of them and see what it does.

    **An everyday example.** A photo of a loaf of bread tells you it can be done. A recipe with the flour, the amounts, the times and the oven temperature lets you bake it, and bake it again with less yeast to see what the yeast does.
    """)
    return


@app.cell(hide_code=True)
def predict_speedrun():
    guess_speedrun = L.guess(
        "speedrun",
        question="In 2019, training GPT-2 cost about 43,000 dollars. How long do you think nanochat's best recipe of March 2026 needs, on one computer with 8 H100 GPUs, to beat GPT-2's score on CORE?",
        options={
            "month": "About a month",
            "day": "About a day",
            "right": "Under two hours",
        },
        reveal="**Under two hours: 1.65 hours, the record of 14 March 2026 on nanochat's leaderboard.** Its first entry, of January 2026, took 3.04 hours. Lower-precision numbers (FP8), a larger batch, better data and two rounds of automated research, with changes found by AI agents, nearly halved the time in two months, and each change is in the code of its entry.",
    )
    guess_speedrun
    return (guess_speedrun,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A training recipe** is everything a run needs, written down: the data, the tokenizer, the model, the optimizer and its schedule, how long to train, how to evaluate, and the order of the phases. nanochat's is one script, `runs/speedrun.sh`. Without its setup, its logging options and the lines that wait for the download, and with the course's comments instead of its own, these are its steps (nanochat, MIT license, © 2025 Andrej Karpathy; at commit `92d63d4`):

        ```bash
        python -m nanochat.dataset -n 8            # download the first 8 shards of the pretraining data
        python -m nanochat.dataset -n 170 &        # and the rest of the 170 shards, while the tokenizer trains
        python -m scripts.tok_train                # train the tokenizer on about 2 billion characters
        python -m scripts.tok_eval                 # evaluate it
        torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
        torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
        ```

        **The phases.** The data: nanochat's own repacked copy of NVIDIA's ClimbMix, web text in parquet shards, downloaded as needed. The tokenizer: byte-level BPE with a vocabulary of $2^{15}$ = 32,768 ids, a BOS token at the start of every document, and digits split into groups of at most 2. Pretraining: `base_train`. Then evaluation, supervised fine-tuning for chat (`chat_sft`), and its evaluation. This lesson reads the pretraining; later lessons do the rest.

        **One knob for the size.** `--depth` sets the number of blocks, and from it all the rest: the width is the depth times 64, rounded up to a multiple of 128, the head size; the heads are the width divided by 128. The scaling parameters, the matrices of the blocks and the output head, are about $12 \, L \, d^2 + V d$ for $L$ blocks of width $d$ and a vocabulary of $V$. The tokens to train on are a fixed number times those parameters, 8 in the speedrun. The batch grows with the tokens, as a power law, and the steps are the tokens divided by the batch.

        **The schedule.** The learning rate rises in a straight line for 40 steps, stays at its peak, and in the last 65% of the steps falls in a straight line to 5% of the peak: a warmdown, instead of the cosine of [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en).

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **training recipe** | | everything a training run needs, written down so that others can run it: data, tokenizer, model, optimizer, schedule, length, evaluation and the order of the phases | `runs/speedrun.sh` |
        | **warmdown** | | the end of a schedule in which the learning rate falls in a straight line to a small fraction of its peak | `lr_multiplier(step, ...)` |
        | **depth** | | the number of blocks of a model; nanochat sets the width, the heads and the length of the run from it | `--depth=24` |
        """),
        mo.md(r"""
        The picture: nanochat's schedule over the 300 steps of this lesson's demo, next to a cosine schedule with the same warmup.
        """),
        mo.Html(C.schedule_picture({"title": "Learning rate, as a share of its peak", "x": "step", "y": "share of the peak", "lang": L.lang,
                                    "nanochat": "nanochat", "cosine": "cosine",
                                    "alt": "Two lines over 300 steps. nanochat's rises to 1 in 40 steps, stays at 1 up to step 105, and falls in a straight line to 0.05 at step 300. The cosine rises the same way and then falls in a half cosine to 0."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a training recipe:

    1. Find the script that runs it all, and list its steps in order.
    2. For every step: which data it reads, and which settings it gets.
    3. Find the rule that sizes the run: the model, the tokens, the batch and the steps.
    4. Find the schedule of the learning rate.
    5. Put every part next to what you built yourself: the same, or new?
    6. Change one setting at a time, and measure what it does.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 for the speedrun, `--depth=24`, by nanochat's rules, with a vocabulary of $2^{15}$.
    """)
    return


@app.cell
def example_speedrun():
    import math

    _depth, _vocab, _ratio = 24, 2 ** 15, 8
    # the width: depth times 64, rounded up to a multiple of 128; one head per 128
    _dim = math.ceil(_depth * 64 / 128) * 128
    _heads = int(_dim / 128)
    # the scaling parameters, and the tokens to train on
    _params = 12 * _depth * _dim * _dim + _vocab * _dim
    _tokens = _ratio * _params
    # the batch, from nanochat's reference d12: 2 ** 19 tokens, growing as the tokens to the power 0.383
    _ref = _ratio * (12 * 12 * 768 * 768 + _vocab * 768)
    _batch = 2 ** round(math.log2(2 ** 19 * (_tokens / _ref) ** 0.383))
    print(_dim, _heads, _params, _tokens, _batch, int(_tokens / _batch))
    return


@app.cell(hide_code=True)
def text_after_example():
    _s = C.speedrun()
    mo.md(f"""
    Depth 24 gives a width of {C.number(_s['dim'], 0, L.lang)}, already a multiple of 128, so {_s['heads']} heads, and {C.number(_s['params'], 0, L.lang)} scaling parameters by the formula; nanochat counts {C.number(729810624, 0, L.lang)}, as a few small gates are not in it. 8 tokens per parameter make {C.number(_s['tokens'] / 1e9, 2, L.lang)} billion tokens. The reference model, depth 12, trains on `2 ** 19` tokens per step; the speedrun has {C.number(_s['tokens'] / (8 * C.size_of(12, 2 ** 15)[2]), 1, L.lang)} times its tokens, so its batch is `2 ** 19` times that to the power 0.383, rounded to a power of 2: `2 ** 20`, {C.number(_s['batch'], 0, L.lang)} tokens. That gives {C.number(_s['steps'], 0, L.lang)} steps. The whole run follows from one number, which is why a leaderboard entry can say "d24" or "d26" and mean a whole run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What you built, and what is new

        `nanochat/gpt.py` lists what makes nanochat's model different from the GPT-2 of 2019. Here are six of those parts, and three more of its recipe. Put every part in its place: does your ModernGPT of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en) have it too, or is it new in nanochat?
        """),
        L.task("Put every part of nanochat's recipe in its place."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    sort_features = L.sort("features")
    sort_features
    return (sort_features,)


@app.cell(hide_code=True)
def check_features(sort_features):
    result_features = L.check("features", sort_features)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def text_exercise_size():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The size from the depth

        Complete `size(depth, vocab)`: step 3 of the recipe for the model. It gives back the width, the depth times 64 rounded up to a multiple of 128; the number of heads, one per 128 numbers of the width; and the scaling parameters, `12 * depth * dim * dim + vocab * dim`. The check calls `size(20, 32768)`, and others.
        """),
        L.task("Make `size` give nanochat's width, heads and parameters for a depth."),
    ])
    return


@app.cell(hide_code=True)
def exercise_size():
    box_size = L.code_box("size", lines=12)
    box_size
    return (box_size,)


@app.cell(hide_code=True)
def check_size(box_size):
    result_size = L.check("size", box_size.value)
    result_size
    return (result_size,)


@app.cell(hide_code=True)
def hints_size():
    L.hints(
        "size",
        [
            "Rounding up to a multiple of 128 is: divide by 128, round up with `math.ceil`, and multiply by 128 again.",
            "Depth 5 gives 320, which is not a multiple of 128: rounded up, 384, so 3 heads.",
        ],
        why="With whole heads of 128, every model of the family has heads of the same size, whatever its depth: a deeper model gets more heads, not bigger ones.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lr():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. nanochat's schedule

        Complete `lr_multiplier(step, steps, warmup, warmdown_ratio, final)`: step 4, as `get_lr_multiplier` in nanochat's `scripts/base_train.py` does it. It gives the share of the peak learning rate at step `step`, counted from 0, of a run of `steps` steps. During the first `warmup` steps, it rises in a straight line, from `1 / warmup` at step 0 to 1. Then it is 1, up to and including step `steps - warmdown`, where `warmdown` is `round(warmdown_ratio * steps)`. After that, it falls in a straight line, from 1 at step `steps - warmdown` to `final` at step `steps`, one past the last. The check calls `lr_multiplier(10, 100, 20, 0.5, 0.1)`, and others.
        """),
        L.task("Make `lr_multiplier` give nanochat's schedule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lr():
    box_lr = L.code_box("lr-multiplier", lines=12)
    box_lr
    return (box_lr,)


@app.cell(hide_code=True)
def check_lr(box_lr):
    result_lr = L.check("lr-multiplier", box_lr.value)
    result_lr
    return (result_lr,)


@app.cell(hide_code=True)
def hints_lr():
    L.hints(
        "lr-multiplier",
        [
            "Three parts: `step < warmup`, then `step <= steps - warmdown`, then the rest.",
            "In the warmdown: how many steps are left at step `step`, up to step `steps`? Divided by `warmdown`, that is the share still to come: 1 at its start.",
        ],
        why="The plateau at the peak learns fast; the straight warmdown at the end lowers the loss a good deal in its last steps. And the length of the warmdown follows the run, so a longer run needs no new schedule.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _dim, _heads, _ = C.size_of(C.DEPTH, C.D.B.VOCAB)
    mo.vstack([
        mo.md(f"""
        ### A few hundred steps of the recipe

        The button below trains your ModernGPT with what this lesson read from nanochat: the size of depth {C.DEPTH} by nanochat's rule, a width of {_dim} with {_heads} heads, and your `lr_multiplier` with nanochat's defaults, a warmup of {C.SCHEDULE['warmup']} steps and a warmdown over the last {round(100 * C.SCHEDULE['warmdown'])}% to {round(100 * C.SCHEDULE['final'])}% of the peak, for {C.STEPS} steps of {C.BATCH} windows of {C.BLOCK} tokens. It reads the windows with your `TokenWindows` from the shards of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), and holds the last shard back to validate on. The optimizer is AdamW, at a peak of {C.number(C.PEAK, 3, L.lang)}, not Muon. On the course's computer, on its CPU, the run took about {round(C.MEASURED['seconds'] / 60)} minutes, on 2 threads; on a GPU, it is much faster. While it trains, a chart shows the loss of every step, with the course's run dashed beside it.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train for 300 steps")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, box_lr, result_lr):
    _m = C.MEASURED
    _labels = {"title": "Training loss, every 10 steps", "x": "step", "y": "loss per token", "lang": L.lang}
    _course = mo.Html(C.curve_picture([(_m["curve"], True)], {**_labels, "key": "- - the course's run",
                                                                "alt": f"The training loss of the course's run, from {C.number(_m['curve'][0], 1, L.lang)} after 10 steps down to {C.number(_m['curve'][-1], 2, L.lang)} after 300."}))
    _verdict = f"The course's run ended at a validation loss of {C.number(_m['val'], 2, L.lang)} per token, {C.number(_m['per_char'], 2, L.lang)} per character, in {round(_m['seconds'] / 60)} minutes on 2 threads. The course's GPT on tokens of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) reached {C.number(C.BPE_RUN['per_char'], 2, L.lang)} per character after {C.BPE_RUN['epochs']} epochs, in {round(C.BPE_RUN['minutes'])} minutes on one thread. That GPT was smaller, 1.3 million parameters with 128 tokens of context, and had another tokenizer; per character the two can be compared. These {C.STEPS} steps see less than the training shards once, and already get close."
    if result_lr.state != "correct":
        _out = mo.vstack([mo.md("*Finish exercise 3 first, then press the button: the run uses your `lr_multiplier`. Until then, the course's run:*"), _course, mo.md(_verdict)])
    elif not train_button.value:
        _out = mo.vstack([mo.md("*Press the button to start the run. Until then, the course's run:*"), _course, mo.md(_verdict)])
    else:
        _live = L.live({"title": "Your run: the training loss of every step", "x": "step", "y": "loss per token",
                        "alt": "The training loss of every step of your run so far, in grey, and its mean over the last 10 steps as the thicker line; dashed, the course's run."}, C.STEPS, window=10)
        _live.reference(C.course_curve(), "the course's run", role="loss")
        _live.line(f"**Training** · step 1 of {C.STEPS} …")   # the course's run shows from the press of the button

        def _progress(_p):
            _live.extend(_p["fresh"], f"**Step {_p['step']} of {_p['steps']}** · training loss {C.number(_p['loss'], 2, L.lang)} per token · {round(_p['seconds'])} s")
        _r = C.run(run_code(box_lr.value).names["lr_multiplier"], _progress)
        if _r == "run":
            _out = _live.stopped("*The run stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*")
        else:
            _on = "the CPU" if _r["device"] == "cpu" else "the GPU"
            _out = mo.vstack([
                mo.Html(C.curve_picture([(_m["curve"], True), (_r["curve"], False)], {**_labels, "key": "— your run   - - the course's",
                                                                                      "alt": f"The training loss of your run, from {C.number(_r['curve'][0], 1, L.lang)} after 10 steps down to {C.number(_r['curve'][-1], 2, L.lang)} after 300, next to the course's run."})),
                mo.md(f"""Your ModernGPT of {C.number(_r['params'], 0, L.lang)} parameters trained for {C.STEPS} steps on {_on}, in {round(_r['seconds'])} seconds. Its validation loss on the held-back shard is **{C.number(_r['val'], 2, L.lang)} per token**, {C.number(_r['per_char'], 2, L.lang)} per character. {_verdict}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_recipe():
    mo.md(r"""
    ### Exercise 4. Why a recipe is open
    """)
    return


@app.cell(hide_code=True)
def choice_recipe_cell():
    choice_recipe = L.choice(
        "recipe",
        question="Every entry of nanochat's leaderboard names the commit of the code that made it. What does that let you do?",
        options={
            "best": "Know that the newest commit always trains the best model",
            "right": "Run any record again, and see which changes of code or settings made it faster",
            "private": "Keep the data private, as the commit holds only the code and not where the data comes from",
        },
        correct="right",
        why={
            "best": "A newer commit can be faster and worse in another way; the leaderboard lists the score of each entry, and the code lets you check it.",
            "private": "The commit holds the code that downloads the data, so the data is as public as the code.",
            "right": "That is what makes a recipe open: you can rerun it, and change one thing at a time, as in step 6 of the recipe.",
        },
    )
    choice_recipe
    return (choice_recipe,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The ModernGPT of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en), now sized by nanochat's rule; the shards of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), now trained from; and the schedules of [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en), now with a warmdown. Later lessons fit a scaling law for your model and pretrain it for longer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_features, result_size, result_lr, choice_recipe):
    L.wrap_up(result_features, result_size, result_lr, choice_recipe)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
