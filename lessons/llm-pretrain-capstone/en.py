# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-pretrain-capstone. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-pretrain-capstone", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en) and [Project: fit your own scaling law](/course/open/llm-fit-scaling-law?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mfu():
    quick_mfu = L.choice(
        "quick-mfu",
        question="A run trains 20,000 tokens per second, and every token costs 120 million FLOPs. The fastest multiplication this GPU does reaches 12,000 billion FLOP/s. What is the MFU of the run?",
        options={
            "tenth": "2%",
            "right": "20%",
            "half": "50%",
        },
        correct="right",
        why={
            "tenth": "20,000 times 120 million is 2,400 billion FLOP/s, a fifth of the 12,000 billion: count the zeros once more. See [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en).",
            "half": "The run needs 20,000 times 120 million, 2,400 billion FLOP/s; that is a fifth of the peak. See [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en).",
            "right": "In this project you measure the MFU of your own run, on your own computer.",
        },
    )
    quick_mfu
    return (quick_mfu,)


@app.cell(hide_code=True)
def quick_check_optimal():
    quick_optimal = L.choice(
        "quick-optimal",
        question="For a budget, your `compute_optimal` gives 1 million parameters at 20 tokens per parameter. What does it give for four times that compute?",
        options={
            "params": "4 million parameters on the same tokens",
            "tokens": "1 million parameters on four times the tokens",
            "right": "2 million parameters on twice the tokens",
        },
        correct="right",
        why={
            "params": "At a fixed number of tokens per parameter, both grow: the compute is 6 times 20 times the parameters squared. See [Project: fit your own scaling law](/course/open/llm-fit-scaling-law?lang=en).",
            "tokens": "At a fixed number of tokens per parameter, the model grows as well: the compute is 6 times 20 times the parameters squared. See [Project: fit your own scaling law](/course/open/llm-fit-scaling-law?lang=en).",
            "right": "The parameters and the tokens each grow with the square root of the compute. In this project the compute is what your computer does in 15 minutes.",
        },
    )
    quick_optimal
    return (quick_optimal,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Everything you built for your LLM comes together here: your tokenizer and its shards, your ModernGPT, the recipe of an open project, your scaling law, mixed precision and the MFU. Labs pretrain this way at a large scale: they measure what their computers do, size a model for the budget, train it, and measure the result. Eldan and Li (2023) showed with TinyStories that models of fewer than 10 million parameters can already write fluent, consistent stories. In this project you pretrain such a model yourself, in a budget of 15 minutes, and let it write.

        **An everyday example.** You have a quarter of an hour to cook. You first check how hot your stove gets, then choose a dish that is done in that time, and you taste it at the end.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           5, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_lower():
    _m = C.MEASURED
    _run = _m["run"]
    guess_lower = L.guess(
        "lower",
        question=f"The course planned this project's run on its own CPU: a model of depth {_run['depth']}, {C.number(_run['params'], 0, L.lang)} parameters, on {C.number(_run['tokens'] / 1e6, 1, L.lang)} million tokens. In [Project: fit your own scaling law](/course/open/llm-fit-scaling-law?lang=en), the model of depth {C.S.BIG}, {C.number(C.S.MEASURED['big']['params'], 0, L.lang)} parameters on {C.number(C.S.MEASURED['big']['tokens'] / 1e6, 2, L.lang)} million tokens, reached a validation loss of {C.number(C.S.MEASURED['big']['val'], 2, L.lang)} on the same validation stories. What does the smaller model of this run reach?",
        options={
            "higher": "A higher loss: it has fewer parameters",
            "same": "About the same loss",
            "right": "A lower loss",
        },
        reveal=f"**A lower loss: {C.number(_run['val'], 2, L.lang)} against {C.number(C.S.MEASURED['big']['val'], 2, L.lang)}.** The model is smaller, but it sees {C.number(_run['tokens'] / C.S.MEASURED['big']['tokens'], 0, L.lang)} times as many tokens: {C.number(_run['tokens'] / _run['params'], 0, L.lang)} per parameter instead of {C.number(C.S.MEASURED['big']['tokens'] / C.S.MEASURED['big']['params'], 1, L.lang)}. For a budget, a smaller model on more data can beat a bigger one on too little.",
    )
    guess_lower
    return (guess_lower,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A time budget is a compute budget.** Before the run, a **probe** measures this computer: a few steps of the model of depth 6, which give its tokens per second. Times the 6 FLOPs per parameter and token that Chinchilla counts, $6N$, that is the FLOP/s a plan can count on; and times the 15 minutes of the run, that is the compute $C$.

        **The size: compute-optimal, then the nearest model.** Your `compute_optimal` splits $C$ into parameters and tokens at 20 tokens per parameter. The family of [Project: fit your own scaling law](/course/open/llm-fit-scaling-law?lang=en) has a model for every depth, so the run takes the depth whose parameters $N$ come closest, and spends the whole budget on it: $C / 6N$ tokens.

        **One step of the run.** The learning rate follows the schedule of [Reading an open training recipe](/course/open/llm-open-recipes?lang=en); the forward pass and the loss run under autocast in BF16, as in [Mixed-precision training](/course/open/sys-mixed-precision?lang=en), but only where that is faster: the probe measures both; and the gradients are clipped to a norm of 1, as in [Gradient clipping](/course/open/opt-gradient-clipping?lang=en).

        **The run itself.** After every tenth of its steps, the run measures its loss on 64 windows of the validation stories of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), and keeps its state, so that a run that stops can go on. At the end it measures the loss on all of them, and its MFU, as in [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en).

        ### New words

        | Word | What it means | In code |
        |---|---|---|
        | **time budget** | the time a run may take; times the FLOP/s of the computer, a compute budget | `flops_per_second * seconds` |
        | **probe** | a few steps that measure what a computer does, before the run is planned | 8 timed calls of `pretrain_step` |
        """),
        mo.md(r"""
        The picture: the validation loss of the course's run on its own CPU, after every tenth of its steps, against the tokens it had seen.
        """),
        mo.Html(C.curve_picture([(_m["run"]["history"], False)], C.S.MEASURED["big"]["val"],
                                {"title": "Validation loss during the run", "x": "tokens seen", "y": "validation loss per token",
                                 "reference": f"depth {C.S.BIG}, scaling law", "lang": L.lang,
                                 "alt": f"The validation loss falls from {C.number(_m['run']['history'][0]['val'], 2, L.lang)} after {C.number(_m['run']['history'][0]['tokens'] / 1e6, 1, L.lang)} million tokens to {C.number(_m['run']['history'][-1]['val'], 2, L.lang)} after {C.number(_m['run']['tokens'] / 1e6, 1, L.lang)} million, below the {C.number(C.S.MEASURED['big']['val'], 2, L.lang)} of depth {C.S.BIG} of the scaling law."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To pretrain a model in a time budget:

    1. Turn the stories into shards; a fast computer sees them more than once.
    2. Measure this computer with a probe, in FP32 and in BF16, and keep the faster.
    3. Plan: the compute of the budget, split compute-optimally, and the nearest model.
    4. Train: every step at the learning rate of the schedule, with clipped gradients; validate now and then, and keep the state.
    5. Measure the result, the loss, the tokens per second and the MFU, and let the model write.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 in numbers: the plan the course made on its own CPU, from the FLOP/s its probe measured.
    """)
    return


@app.cell
def example_plan():
    _flops = 7.38e10    # FLOP/s of the course's probe on its CPU, 2 threads: tokens per second times 6N
    _compute = _flops * 15 * 60
    _params = (_compute / (6 * 20)) ** 0.5    # compute-optimal at 20 tokens per parameter
    _sizes = {2: 260649, 3: 575337, 4: 1110633, 5: 1940457, 6: 3141801}
    for _depth in _sizes:
        print(_depth, _sizes[_depth], round(abs(_sizes[_depth] - _params)))
    _best = 3    # the depth with the smallest difference
    print(round(_compute / 1e12, 1), round(_params), round(_compute / (6 * _sizes[_best])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The budget is about $66 \cdot 10^{12}$ FLOPs, and compute-optimal that is about 744,000 parameters. Depth 3, with 575,337 parameters, comes closest: 169,000 below, against 367,000 above for depth 4. Spent on depth 3, the whole budget is about 19.2 million tokens, some 33 per parameter: the model is a little smaller than compute-optimal, so it gets a little more data.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The plan

        Complete `plan_run(flops_per_second, seconds, sizes, tokens_per_param)`: step 3. `sizes` gives the parameters of every depth of the family, such as `{2: 260649, 3: 575337}`. Work out the compute of the budget and its compute-optimal parameters with your own `compute_optimal`, find the depth whose parameters come closest, and give back that depth and the tokens that spend the whole compute on it, as a whole number. The check calls `plan_run(8e12, 900, {6: 3141801, 7: 4782537, 8: 6938601, 9: 9691305, 10: 13100457}, 20)`, for a computer a hundred times faster than the course's CPU, and others.
        """),
        L.task("Make `plan_run` give the depth and the tokens of a compute-optimal run in the budget."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plan():
    box_plan = L.code_box("plan-run", lines=12)
    box_plan
    return (box_plan,)


@app.cell(hide_code=True)
def check_plan(box_plan):
    result_plan = L.check("plan-run", box_plan.value)
    result_plan
    return (result_plan,)


@app.cell(hide_code=True)
def hints_plan():
    L.hints(
        "plan-run",
        [
            "The compute is `flops_per_second * seconds`. `params, _ = compute_optimal(compute, tokens_per_param)` gives the parameters; the `_` leaves out its tokens, which belong to exactly that size.",
            "Walk through `sizes` and keep the best depth so far, as your `best_floor` kept its best floor: a depth is better when `abs(sizes[depth] - params)` is smaller.",
        ],
        why="No model of the family is exactly compute-optimal. The nearest one, with the whole budget, comes close to the lowest loss for that compute. Labs plan the same way: from the compute they have, not from the model they would like.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One step of the run

        Complete `pretrain_step(model, opt, x, y, lr, device_type, mixed)`: one step of step 4. Set the learning rate `lr` in every group of the optimizer, run the forward pass and the loss under autocast in BF16 only when `mixed` is True, and clip the gradients to a norm of at most 1.0 between the backward pass and the step. The check runs your step on a small layer with sequences of 5 positions, as a GPT gets them, and tries, among other things, whether a huge gradient is clipped.
        """),
        L.task("Make `pretrain_step` take one step at the learning rate it gets, in BF16 only when asked, with clipped gradients."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("pretrain-step", lines=16)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("pretrain-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "pretrain-step",
        [
            "`opt.param_groups` is a list of dicts, each with its own `\"lr\"`. `torch.autocast(..., enabled=mixed)` switches autocast on or off: with `enabled=False` it does nothing.",
            "A GPT gives logits of shape (batch, positions, vocabulary), and `F.cross_entropy` wants one row per position: `logits.reshape(-1, logits.shape[-1])` and `y.reshape(-1)`, as in the loss line of [A reusable training loop](/course/open/nn-trainer?lang=en). The clipping comes after `loss.backward()` and before `opt.step()`, as in [Gradient clipping](/course/open/opt-gradient-clipping?lang=en).",
        ],
        why="The schedule only works when every step uses its learning rate; clipping keeps one odd mini-batch from throwing the model off; and autocast only pays off where the hardware multiplies BF16 faster than FP32.",
    )
    return


@app.cell(hide_code=True)
def text_demo_data():
    _d = C.MEASURED["data"]
    mo.vstack([
        mo.md(f"""
        ### Your run, step 1: the stories

        The button downloads the first 200 MB of the training stories of TinyStories once, {C.number(_d['stories'], 0, L.lang)} stories, and turns them into shards with your `encode_fast` and `write_shards`, {C.number(C.CHUNK / 1e6, 0, L.lang)} million characters at a time, into `work/projects/llm/{C.FOLDER}/`. On the course's computer, with the course's merge rules, that gave {C.number(_d['ids'] / 1e6, 1, L.lang)} million tokens in about {round(_d['seconds'] / 60)} minutes. The validation stories stay those of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en): no story of the run is among them.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def data_button_cell():
    data_button = mo.ui.run_button(label="Get the stories")
    data_button
    return (data_button,)


@app.cell(hide_code=True)
def demo_data(data_button):
    _note = None
    if data_button.value:
        def _progress(_p):
            if _p["phase"] == "download":
                mo.output.replace(mo.md(f"**Downloading** · {C.number(_p['done'] / 1e6, 0, L.lang)} of {C.number(_p['total'] / 1e6, 0, L.lang)} MB"))
            else:
                mo.output.replace(mo.md(f"**Encoding** · {C.number(_p['stories'], 0, L.lang)} of {C.number(_p['total'], 0, L.lang)} stories · {C.number(_p['ids'] / 1e6, 1, L.lang)} million tokens · {round(_p['seconds'])} s"))
        _r = C.prepare(_progress)
        if _r == "download":
            _note = "*The download did not work. Is there a connection to the internet? Press the button again to try once more.*"
        elif _r == "run":
            _note = "*Encoding stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
    _info = C.shards_info()
    if _info is None:
        _out = mo.md(_note or "*No shards of the run yet. Press the button.*")
    else:
        _whose = "your merge rules" if _info["merges"] == "yours" else "the course's merge rules"
        _out = mo.md(f"Your shards are there: **{C.number(_info['ids'] / 1e6, 1, L.lang)} million tokens** of {C.number(_info['stories'], 0, L.lang)} stories, encoded with {_whose}.")
    _out
    return


@app.cell(hide_code=True)
def text_demo_measure():
    mo.md(f"""
    ### Your run, step 2: measure and plan

    The button times {C.PROBE_STEPS} steps of your `pretrain_step` on the model of depth {C.PROBE}, in FP32 and in BF16, keeps the faster, and plans a run of {C.MINUTES} minutes with your `plan_run`. The plan is kept in your project: to go on with a run that stopped, press *Pretrain* again without measuring again.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Measure this computer")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(box_plan, box_step, measure_button, result_plan, result_step):
    _note = None
    _ready = result_plan.state == "correct" and result_step.state == "correct"
    if not _ready:   # no mo.stop: the course's run below shows meanwhile
        _note = "*Finish exercises 1 and 2 first: the run plans with your `plan_run` and trains with your `pretrain_step`.*"
    elif measure_button.value:
        mo.output.replace(mo.md("**Measuring** · in FP32 and in BF16 …"))
        _m = C.measure(run_code(box_step.value).names["pretrain_step"], lambda _p: None)
        if _m == "run" or C.save_plan(_m, run_code(box_plan.value).names["plan_run"]) == "run":
            _note = "*The measurement stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
    plan = C.saved_plan() if _ready else None
    if _note is not None:
        _out = mo.md(_note)
    elif plan is None:
        _out = mo.md("*Not measured yet. Press the button.*")
    else:
        _fast = "BF16" if plan["mixed"] else "FP32"
        _where = "the CPU" if plan["device"] == "cpu" else plan["name"]
        _other = plan["tps_bf16"] if not plan["mixed"] else plan["tps_fp32"]
        _out = mo.md(f"""On {_where}, your step trained **{C.number(plan['tps'], 0, L.lang)} tokens per second** in {_fast}{'' if _other is None else f', against {C.number(_other, 0, L.lang)} in the other format'}: {C.number(plan['flops'] / 1e9, 1, L.lang)} billion FLOP/s at 6N. In {C.MINUTES} minutes that is $\\text{{{C.number(plan['compute'] / 1e12, 0, L.lang)}}} \\cdot 10^{{12}}$ FLOPs. Your `plan_run` chooses depth **{plan['depth']}**, {C.number(plan['params'], 0, L.lang)} parameters, on **{C.number(plan['tokens'] / 1e6, 1, L.lang)} million tokens**: {C.number(plan['steps'], 0, L.lang)} steps of {plan['batch']} windows.""")
    _out
    return (plan,)


@app.cell(hide_code=True)
def train_button_cell(plan):
    train_button = mo.ui.run_button(label=f"Pretrain depth {plan['depth']} on {C.number(plan['tokens'] / 1e6, 1, L.lang)} million tokens" if plan else "Pretrain", disabled=plan is None)
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_step, plan, train_button):
    _labels = {"title": "Validation loss during the run", "x": "tokens seen", "y": "validation loss per token",
               "reference": f"depth {C.S.BIG}, scaling law", "lang": L.lang, "alt": "The validation loss of your run so far."}
    trained = None
    _note = None
    if train_button.value and plan is None:
        _note = "*Measure this computer first: the run follows its plan.*"
    elif train_button.value:
        _seen = []

        def _progress(_p):
            if _p["phase"] == "eval":
                _seen.append(_p)
            _parts = [mo.md(f"**Step {C.number(_p['step'], 0, L.lang)} of {C.number(_p['steps'], 0, L.lang)}** · {C.number(_p['seconds'] / 60, 1, L.lang)} minutes")]
            if _seen:
                _parts.append(mo.Html(C.curve_picture([(_seen, False)], C.S.MEASURED["big"]["val"], _labels)))
            mo.output.replace(mo.vstack(_parts))
        _r = C.run_plan(plan, run_code(box_step.value).names["pretrain_step"], _progress)
        if _r == "data":
            _note = "*There are no shards of the run yet: press* Get the stories *first.*"
        elif _r == "merges":
            _note = "*Your merge rules changed after you got the stories: press* Get the stories *again.*"
        elif _r == "memory":
            _note = "*The GPU ran out of memory: this model, with mini-batches of 64 windows, does not fit next to what else uses it. Close other programs that use the GPU, and press* Pretrain *again; until then, the course's run stands in below.*"
        elif isinstance(_r, str):
            _note = "*The run stops with an error, or its loss is no longer a number. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
        else:
            trained = _r
    mo.md(_note) if _note else mo.md("*Done: your model is in `work/projects/llm/capstone.pt`.*") if trained else None
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _m = C.MEASURED
    _info = C.saved()
    _mine = _info is not None
    _run = _info if _mine else _m["run"]
    _whose = "Your model" if _mine else "The course's model"
    _stand = "" if _mine else " Until you pretrain your own, the course's run stands in."
    _on = "the CPU" if _run["device"] == "cpu" else _run["name"]
    _runs = [(_run["history"], False)] + ([(_m["run"]["history"], True)] if _mine else [])
    _plan = C.saved_plan() if _mine else None
    _probe = _plan["flops"] if _plan else _m["measure"]["flops"]
    _did = _run["tps"] * 6 * _run["params"]
    mo.vstack([
        mo.Html(C.curve_picture(_runs, C.S.MEASURED["big"]["val"],
                                {"title": "Validation loss during the run", "x": "tokens seen", "y": "validation loss per token",
                                 "reference": f"depth {C.S.BIG}, scaling law", "lang": L.lang,
                                 "alt": f"The validation loss of {'your run, with the course run dashed' if _mine else 'the course run'}: at the end {C.number(_run['val'], 2, L.lang)} per token."})),
        mo.md(f"""{_whose} of depth {_run['depth']}, {C.number(_run['params'], 0, L.lang)} parameters, trained for {C.number(_run['seconds'] / 60, 0, L.lang)} minutes on {_on}, on {C.number(_run['tokens'] / 1e6, 1, L.lang)} million tokens: {C.number(_run['tps'], 0, L.lang)} tokens per second, an MFU of {C.number(100 * _run['mfu'], 0, L.lang)}%. Its validation loss is **{C.number(_run['val'], 3, L.lang)}** per token, against {C.number(C.S.MEASURED['big']['val'], 3, L.lang)} for depth {C.S.BIG} of the scaling law. The plan counted on {C.MINUTES} minutes: at depth {_run['depth']}, the steps did {C.number(_did / 1e9, 1, L.lang)} billion FLOP/s at 6N, against {C.number(_probe / 1e9, 1, L.lang)} billion for the probe of depth {C.PROBE}, so the time differs from the plan.{_stand}"""),
        mo.md(f"It writes, from *{C.START}*:\n\n" + C.quote(trained["story"] if trained else _info["story"] if _mine else _m["story"])),
    ])
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(f"""
    ### The run, measurement by measurement

    Every time the run measures its validation loss, it also lets the model write: {C.SNAP} tokens from *{C.START}*, drawn with the same seed each time. Here you go through those moments: the validation loss along the run, with a dashed line at the moment you choose, and what the model wrote then.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _info = trained if trained is not None else C.saved()
    _mine = _info is not None and bool(C.moments(_info["history"]))
    _run = _info if _mine else C.MEASURED["run"]
    _whose = "your run" if _mine else "the course's run"
    if _mine:
        _note = ""
    elif _info is not None:
        _note = "*Your saved run was made before the lesson kept what the model wrote, so the course's run stands in here. Pretrain again to go through your own.*"
    else:
        _note = "*Until you pretrain your own model, you go through the course's run here.*"
    _labels = {"title": f"Validation loss during {_whose}", "x": "tokens seen", "y": "validation loss per token", "lang": L.lang,
               "alt": f"The validation loss of {_whose} against the tokens seen, on a log scale, with a dashed line at the chosen moment.",
               "after": "After {tokens} million tokens", "val": "validation loss"}
    _ks = C.moments(_run["history"])   # numbered from 1 on the slider
    _block = L.explore("replay", "measurement", lambda _v: C.replay_frame(_run["history"], _ks[int(_v) - 1], _labels), start=len(_ks), values=list(range(1, len(_ks) + 1)), role="",
                       task="Does the model already write sentences at the first measurement? What still goes wrong there that goes right at the last?")
    mo.vstack([mo.md(_note), _block]) if _note else _block
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(f"""
    In the course's run (your own model may write other stories), the model writes sentences from the first measurement on, after {C.number(C.MEASURED['run']['history'][0]['tokens'] / 1e6, 1, L.lang)} million tokens: names, *One day*, the start of what someone says. After that the validation loss falls ever more slowly, from {C.number(C.MEASURED['run']['history'][0]['val'], 2, L.lang)} to {C.number(C.MEASURED['run']['history'][-1]['val'], 2, L.lang)}, and what it buys is harder to see: whether the story holds together. At first Tim becomes Tom, who is then a little girl; at the end Tim has a toy bear named Lily, and they play together. In between, the model writes *Tim had a toy boy named Tim* more than once: every moment is one draw, and a slightly different model takes another turn somewhere. The loss counts {C.number(C.QUICK * C.BLOCK, 0, L.lang)} tokens of the validation stories at once, in {C.QUICK} windows; one story of {C.SNAP} tokens shows one path.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measure():
    mo.md(r"""
    ### Exercise 3. Why measure?
    """)
    return


@app.cell(hide_code=True)
def choice_measure_cell():
    _p = C.MEASURED["measure"]
    choice_measure = L.choice(
        "why-measure",
        question=f"The probe of the course's CPU trained {C.number(_p['tps_fp32'], 0, L.lang)} tokens per second in FP32 and {C.number(_p['tps_bf16'], 0, L.lang)} in BF16. Why does the run measure both, instead of always taking BF16?",
        options={
            "right": "Whether BF16 is faster depends on the hardware, and on this CPU it was slower",
            "precision": "BF16 makes the loss less precise, so the run takes it only when the loss stays the same",
            "memory": "BF16 takes more memory on a CPU, so the model may not fit",
        },
        correct="right",
        why={
            "right": f"Mixed precision is a way to go faster, and only a measurement tells whether it does on your computer. On the course's CPU, a step in BF16 took {C.number(_p['tps_fp32'] / _p['tps_bf16'], 1, L.lang)} times as long.",
            "precision": "Under autocast the weights and the loss stay FP32, as in [Mixed-precision training](/course/open/sys-mixed-precision?lang=en), and the probe never compares losses: it only times the steps, to keep the faster format.",
            "memory": f"A BF16 number takes half the bytes of an FP32 number, on every device. The question is speed: on the course's CPU, BF16 was {C.number(_p['tps_fp32'] / _p['tps_bf16'], 1, L.lang)} times slower.",
        },
    )
    choice_measure
    return (choice_measure,)


@app.cell(hide_code=True)
def text_longer():
    _c = C.course_info()
    mo.vstack([
        mo.md(f"""
        ## Wrap-up

        **A longer run of the same code.** The course trained its own model with this code for {C.number(_c['hours'], 1, L.lang)} hours on its CPU: depth {_c['depth']}, {C.number(_c['params'], 0, L.lang)} parameters, compute-optimal on {C.number(_c['tokens'] / 1e6, 0, L.lang)} million tokens of stories for young children from Cosmopedia, because the course shares no model trained on TinyStories. Its losses cannot be compared with yours: other stories, other merge rules. It writes, from *{C.START}*:
        """),
        mo.md(C.quote(_c["story"])),
        mo.md(r"""
        The lessons after this one go on with your model, or with the course's when you have none: they measure it, and teach it to follow instructions.

        **How this connects.** Your `compute_optimal` of [Project: fit your own scaling law](/course/open/llm-fit-scaling-law?lang=en), now sizing a real run; the step of [Mixed-precision training](/course/open/sys-mixed-precision?lang=en), now with the schedule of [Reading an open training recipe](/course/open/llm-open-recipes?lang=en) and the clipping of [Gradient clipping](/course/open/opt-gradient-clipping?lang=en); the MFU of [Profiling: tokens per second and MFU](/course/open/sys-profiling-mfu?lang=en), now of your own run; and the shards of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), now of 200 MB of stories. Your project `llm` has `plan_run.py` and `pretrain_step.py` now, and after the run your model in `capstone.pt`, with its sizes and merge rules in `capstone.json`.
        """),
    ])
    return


@app.cell(hide_code=True)
def wrap_up(result_plan, result_step, choice_measure):
    L.wrap_up(result_plan, result_step, choice_measure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
