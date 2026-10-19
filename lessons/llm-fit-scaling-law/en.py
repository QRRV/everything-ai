# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-fit-scaling-law. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-fit-scaling-law", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Scaling laws](/course/open/llm-scaling-laws?lang=en) and [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Three runs give the losses 3.2, 2.7 and 2.45, and you take a floor of 2.0. What do you fit a straight line through, against the log of the sizes?",
        options={
            "loss": "The log of the loss: the floor only matters for big models",
            "rest": "The loss minus the floor: 1.2, 0.7 and 0.45, without a log",
            "right": "The log of the loss minus the floor: the logs of 1.2, 0.7 and 0.45",
        },
        correct="right",
        why={
            "loss": "Without the floor, the curve bends on a log-log plot and the line predicts too low a loss for bigger runs. See [Scaling laws](/course/open/llm-scaling-laws?lang=en).",
            "rest": "A power law is a straight line only on a log-log plot: both axes need a log. See [Scaling laws](/course/open/llm-scaling-laws?lang=en).",
            "right": "This lesson goes one step further: which floor makes your own runs the straightest line?",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="Your ModernGPT has a width of d = 160 and 5 query heads. How many numbers does each query head work with?",
        options={
            "right": "32: the width is shared out over the heads",
            "all": "160: every head sees the whole width",
            "each": "800: five heads of 160 each",
        },
        correct="right",
        why={
            "all": "The queries of all heads together have d numbers: each head gets d / h of them. See [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).",
            "each": "The heads split the width; they do not each get a copy of it. See [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).",
            "right": "In this lesson every model has heads of 32 numbers, whatever its size.",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A big training run can cost months and millions; you want to know what it will give before you start. The makers of GPT-4 fitted a scaling law with a floor, $L(C) = a C^b + c$ in the compute $C$, through models trained the same way with at most 10,000 times less compute, and predicted GPT-4's final loss with it shortly after the run started, without using any of its results. The prediction was highly accurate (OpenAI, 2023). In this project you do the same at a small scale: train four small versions of your ModernGPT, fit a law through them, predict the loss of a bigger one, and only then train it.

        **An everyday example.** Before a long walk you time yourself over 1, 2 and 4 kilometres, and from that you predict the 10. You know that tiredness will make the last kilometres slower: the prediction from short walks is a little too fast.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           5, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_close():
    _m = C.MEASURED
    _small, _big = _m["small"], _m["big"]
    guess_close = L.guess(
        "close",
        question=f"The course trained four small ModernGPTs, from {C.number(_small[0]['params'] / 1e6, 2, L.lang)} to {C.number(_small[-1]['params'] / 1e6, 2, L.lang)} million parameters, on the same stories, and fitted a law with a floor through their losses. It predicts the loss of a model of {C.number(_big['params'] / 1e6, 2, L.lang)} million before it is trained. How far off do you think the prediction is?",
        options={
            "tenths": "It misses by a few tenths",
            "right": "It misses by a few hundredths",
            "fails": "Not close: the bigger model learns something new, which no law of small ones can see",
        },
        reveal=f"**By a few hundredths: the law predicted {C.number(_m['predicted'], 2, L.lang)}, and the model reached {C.number(_big['val'], 2, L.lang)}.** The prediction is {C.number(abs(_big['val'] - _m['predicted']), 3, L.lang)} too {'low' if _m['predicted'] < _big['val'] else 'high'}: most likely because a model that sees the same data as the small ones gets fewer tokens per parameter, and learns less than the law of the small ones expects.",
    )
    guess_close
    return (guess_close,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A family of models.** To compare sizes, only the size may change. This project makes every model from one knob, its depth $L$: a width of $d = 32 L$, and one head per 32 numbers. Every run trains on the same windows of the same stories, one epoch over the training shards of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), with the same schedule and the same seed.

        **The law, with a floor you do not know.** [Scaling laws](/course/open/llm-scaling-laws?lang=en) fitted $L(N) = E + a N^k$ with the floor $E$ given. For your own runs nobody gives you $E$. So try floors, and keep the one that makes the runs the straightest line on a log-log plot: the smallest mean squared distance between the logs of $L - E$ and the line through them. Four runs pin a floor down only roughly, so the further you predict, the less you can trust it.

        **Compute-optimal.** A run of $N$ parameters on $D$ tokens costs about $C = 6 N D$ FLOPs. For a given compute, a model can be too big for its data or too small: Hoffmann and others (2022) found the lowest loss at about 20 tokens per parameter. In this project every model sees the same tokens, so the bigger the model, the fewer tokens per parameter it gets: the big ones are far from compute-optimal, and their loss can fall less than the law of the small ones expects.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **compute-optimal** | the split of a compute budget over parameters and tokens that gives the lowest loss; for Chinchilla, about 20 tokens per parameter | `compute_optimal(c, 20)` |
        | **family of models** | models made from one recipe, of which only the size differs, so that their losses can be compared | `width = 32 * depth` |
        """),
        mo.md(r"""
        The picture: the course's four small runs, the law with the floor that fits them best, its prediction for the bigger model, and that model's real loss.
        """),
        mo.Html(C.law_picture(_m["small"], _m["floor"], _m["big"], _m["predicted"],
                              {"title": "Validation loss against size", "x": "parameters", "y": "validation loss per token", "lang": L.lang,
                               "measured": "measured", "predicted": "predicted", "law": "the law",
                               "alt": f"Four measured losses, from {C.number(_m['small'][0]['val'], 2, L.lang)} at {C.number(_m['small'][0]['params'] / 1e6, 2, L.lang)} million parameters to {C.number(_m['small'][-1]['val'], 2, L.lang)} at {C.number(_m['small'][-1]['params'] / 1e6, 2, L.lang)} million, with a dashed law through them. At {C.number(_m['big']['params'] / 1e6, 2, L.lang)} million, the law predicts {C.number(_m['predicted'], 2, L.lang)}, and the model reached {C.number(_m['big']['val'], 2, L.lang)}."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To predict the loss of a big run from small ones:

    1. Make a family of models from one knob.
    2. Train every size on the same data, with the same schedule and seed.
    3. Train four sizes, each about twice the one before.
    4. Find the floor that makes the losses the straightest line on a log-log plot.
    5. Fit the law with that floor, and predict the loss of the big model before you train it.
    6. Train the big one, and compare.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 in numbers: the parameters, tokens per parameter and compute of the course's four small runs and the big one.
    """)
    return


@app.cell
def example_budget():
    _runs = [(2, 260649), (3, 575337), (4, 1110633), (5, 1940457), (6, 3141801)]
    _tokens = 6138 * 256   # one epoch: the 6,138 windows of 256 tokens of the training shards
    for _depth, _params in _runs:
        # tokens per parameter, and the compute by the rule 6ND, in units of 10 ** 12 FLOPs
        print(_depth, _params, round(_tokens / _params, 1), round(6 * _params * _tokens / 1e12, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Every size is 1.6 to 2.2 times the one before, and every run sees the same 1,571,328 tokens. So the tokens per parameter fall from 6.0 for depth 2 to 0.5 for depth 6, far below Chinchilla's 20: these runs have more parameters than their data can feed. The compute grows with the parameters, from about $2.5 \cdot 10^{12}$ FLOPs for depth 2 to $30 \cdot 10^{12}$ for depth 6: the big run costs about four fifths of the four small ones together.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_floor():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The straightest line

        Complete `best_floor(x, loss, floors)`: step 4. For every floor of `floors`, fit the straight line through `np.log10(x)` and `np.log10(np.array(loss) - floor)`, and measure how far the logs lie from it: the mean of their squared differences. Give back the floor with the smallest error. The check calls `best_floor([1e5, 3e5, 1e6, 3e6], [3.4, 2.9, 2.55, 2.35], [0.0, 0.5, 1.0, 1.5, 2.0, 2.2])`, and others.
        """),
        L.task("Make `best_floor` give the floor that makes the straightest line."),
    ])
    return


@app.cell(hide_code=True)
def exercise_floor():
    box_floor = L.code_box("best-floor", lines=14)
    box_floor
    return (box_floor,)


@app.cell(hide_code=True)
def check_floor(box_floor):
    result_floor = L.check("best-floor", box_floor.value)
    result_floor
    return (result_floor,)


@app.cell(hide_code=True)
def hints_floor():
    L.hints(
        "best-floor",
        [
            "The fit is the one of [Scaling laws](/course/open/llm-scaling-laws?lang=en): `k, b = np.polyfit(log_x, log_rest, 1)`. The line's own values are `k * log_x + b`.",
            "Keep two names up to date in the loop: the best floor so far and its error. Replace them when an error is smaller, or when there is none yet.",
        ],
        why="Too low a floor bends the logs one way, too high a floor the other way; in between lies the floor whose logs are a straight line. That is the floor the runs themselves point to.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The prediction

        Complete `predict(x_new, x, loss, floor)`: step 5. Fit the line through the logs of `x` and of `loss - floor`, with slope $k$ and intercept $b$, and give back the floor plus the power law at `x_new`: $E + 10^b \cdot x_{\text{new}}^k$. The check calls `predict(1e7, [1e5, 3e5, 1e6, 3e6], [3.4, 2.9, 2.55, 2.35], 1.5)`, and others.
        """),
        L.task("Make `predict` give the loss the law predicts at `x_new`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict", lines=10)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict",
        [
            "`np.polyfit` gives the slope first, then the intercept, which is the log of $a$.",
            "The law gives the part of the loss above the floor; the prediction is the floor plus that part.",
        ],
        why="The prediction rests on two guesses: that the law goes on as it did, and that the floor is right. The further `x_new` lies from the runs, the more both matter.",
    )
    return


@app.cell(hide_code=True)
def text_demo_small():
    _m = C.MEASURED
    mo.vstack([
        mo.md(f"""
        ### Your four small models

        The button below trains your ModernGPT at depths {', '.join(str(_d) for _d in C.SMALL[:-1])} and {C.SMALL[-1]}: one epoch each over the training shards, mini-batches of {C.BATCH} windows of {C.BLOCK} tokens, AdamW with the warmup, plateau and warmdown of [Reading an open training recipe](/course/open/llm-open-recipes?lang=en), seed 0, and the last shard to validate on. The runs are kept in `work/projects/llm/{C.RUNS}`. On the course's computer, on 2 CPU threads, the four took about {round(sum(_r['seconds'] for _r in _m['small']) / 60)} minutes together; a GPU is much faster. Until you train your own, the course's runs stand in.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def small_button_cell():
    small_button = mo.ui.run_button(label="Train the four small models")
    small_button
    return (small_button,)


@app.cell(hide_code=True)
def demo_small(small_button):
    _failed = False
    if small_button.value:
        def _progress(_p):
            mo.output.replace(mo.md(f"**Depth {_p['depth']}** · step {_p['step']} of {_p['steps']} · loss {C.number(_p['loss'], 2, L.lang)} per token · {round(_p['seconds'])} s"))
        for _depth in C.SMALL:
            _r = C.train(_depth, _progress)
            if _r == "run":
                _failed = True
                break
            C.save_run(_r)
    _saved = C.saved_runs()
    _mine = all(_d in _saved for _d in C.SMALL)
    small_runs = [_saved[_d] for _d in C.SMALL] if _mine else C.MEASURED["small"]
    _whose = "Your runs" if _mine else "The course's runs, until you train your own"
    _rows = "\n".join(f"| {_r['depth']} | {C.number(_r['params'], 0, L.lang)} | {C.number(_r['tokens'] / _r['params'], 1, L.lang)} | {C.number(_r['val'], 3, L.lang)} |" for _r in small_runs)
    _note = [mo.md("*The run stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*")] if _failed else []
    mo.vstack(_note + [mo.md(f"{_whose}:\n\n| depth | parameters | tokens per parameter | validation loss per token |\n|---|---|---|---|\n{_rows}")])
    return (small_runs,)


@app.cell(hide_code=True)
def demo_law(box_floor, box_predict, result_floor, result_predict, small_runs):
    mo.stop(not (result_floor.state == "correct" and result_predict.state == "correct"),
            mo.md("*Finish exercises 1 and 2 first: this part fits the law with your `best_floor` and your `predict`, and the big model trains only once you have predicted its loss.*"))
    _x = [_r["params"] for _r in small_runs]
    _loss = [_r["val"] for _r in small_runs]
    law_floor = run_code(box_floor.value).names["best_floor"](_x, _loss, C.floors_for(_loss))
    big_params = C.MEASURED["big"]["params"]
    law_prediction = float(run_code(box_predict.value).names["predict"](big_params, _x, _loss, law_floor))
    _plain = float(run_code(box_predict.value).names["predict"](big_params, _x, _loss, 0.0))
    mo.md(f"""Your `best_floor` picks a floor of **{C.number(law_floor, 2, L.lang)}** for these runs. With it, your `predict` predicts a validation loss of **{C.number(law_prediction, 3, L.lang)}** per token for the model of depth {C.BIG}, with {C.number(big_params, 0, L.lang)} parameters. Without a floor, a straight line through the losses themselves, it would say {C.number(_plain, 3, L.lang)}.""")
    return (law_floor, law_prediction)


@app.cell(hide_code=True)
def big_button_cell(law_prediction):
    big_button = mo.ui.run_button(label=f"Train the model of depth {C.BIG}: your law says {C.number(law_prediction, 3, L.lang)}")
    big_button
    return (big_button,)


@app.cell(hide_code=True)
def demo_big(big_button, law_floor, law_prediction, small_runs):
    _m = C.MEASURED
    _failed = False
    if big_button.value:
        def _progress(_p):
            mo.output.replace(mo.md(f"**Depth {_p['depth']}** · step {_p['step']} of {_p['steps']} · loss {C.number(_p['loss'], 2, L.lang)} per token · {round(_p['seconds'])} s"))
        _r = C.train(C.BIG, _progress)
        if _r == "run":
            _failed = True
        else:
            C.save_run(_r)
    _saved = C.saved_runs()
    _big = _saved.get(C.BIG) or _m["big"]
    _whose = "Your model" if C.BIG in _saved else "The course's model"
    _stand = "" if C.BIG in _saved else " Until you train your own, the course's model stands in."
    _diff = _big["val"] - law_prediction
    _note = [mo.md("*The run stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*")] if _failed else []
    mo.vstack(_note + [
        mo.Html(C.law_picture(small_runs, law_floor, _big, law_prediction,
                              {"title": "Validation loss against size", "x": "parameters", "y": "validation loss per token", "lang": L.lang,
                               "measured": "measured", "predicted": "predicted", "law": "the law",
                               "alt": f"The four small runs, the law through them, the prediction of {C.number(law_prediction, 2, L.lang)} for depth {C.BIG}, and its real loss of {C.number(_big['val'], 2, L.lang)}."})),
        mo.md(f"""{_whose} of depth {C.BIG} reached a validation loss of **{C.number(_big['val'], 3, L.lang)}** per token, against the {C.number(law_prediction, 3, L.lang)} your law predicted: {C.number(abs(_diff), 3, L.lang)} {'higher' if _diff > 0 else 'lower'} than predicted. It got {C.number(_big['tokens'] / _big['params'], 1, L.lang)} tokens per parameter, the smallest of all five.{_stand}"""),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_optimal():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A compute-optimal split

        Complete `compute_optimal(compute, tokens_per_param)`. A run costs `compute = 6 * params * tokens` FLOPs; at the compute-optimal split, `tokens = tokens_per_param * params`. Solve for the parameters, and give back the parameters and the tokens. The check calls `compute_optimal(6e18, 20)`, and others. Once it is right, it is saved in your project: the capstone uses it to size its model.
        """),
        L.task("Make `compute_optimal` split a compute budget into parameters and tokens."),
    ])
    return


@app.cell(hide_code=True)
def exercise_optimal():
    box_optimal = L.code_box("compute-optimal", lines=8)
    box_optimal
    return (box_optimal,)


@app.cell(hide_code=True)
def check_optimal(box_optimal):
    result_optimal = L.check("compute-optimal", box_optimal.value)
    result_optimal
    return (result_optimal,)


@app.cell(hide_code=True)
def hints_optimal():
    L.hints(
        "compute-optimal",
        [
            "Put `tokens = tokens_per_param * params` into `compute = 6 * params * tokens`: the compute is `6 * tokens_per_param * params ** 2`.",
            "Solve that for `params`: first divide, then undo the square.",
        ],
        why="Twice the compute does not double the model: at a fixed number of tokens per parameter, the parameters and the tokens each grow with the square root of the compute.",
    )
    return


@app.cell(hide_code=True)
def demo_optimal(box_optimal, result_optimal):
    mo.stop(result_optimal.state != "correct", mo.md("*Finish exercise 3 first: this part uses your `compute_optimal`.*"))
    _big = C.MEASURED["big"]
    _compute = 6 * _big["params"] * _big["tokens"]
    _p, _t = run_code(box_optimal.value).names["compute_optimal"](_compute, C.CHINCHILLA)
    mo.md(f"""The run of depth {C.BIG} costs about ${C.number(_compute / 1e12, 0, L.lang)} \\cdot 10^{{12}}$ FLOPs. Your `compute_optimal` splits that budget, at Chinchilla's {C.CHINCHILLA} tokens per parameter, into a model of **{C.number(_p, 0, L.lang)} parameters** on **{C.number(_t / 1e6, 1, L.lang)} million tokens**: a model a sixth of the size, on more stories than the course has. With little data, a bigger model is not the best use of compute; the capstone gets more data before it grows.""")
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.md(r"""
    ### Exercise 4. Why too low?
    """)
    return


@app.cell(hide_code=True)
def choice_low_cell():
    choice_low = L.choice(
        "why-low",
        question="The law through the four small runs predicted the big model's loss a little too low. Which reason fits these runs best?",
        options={
            "seed": "Chance: another seed would have put the big model right on the law",
            "right": "Too little data: the big model got the fewest tokens per parameter",
            "floor": "The floor was too low: with a higher one, every prediction is right",
        },
        correct="right",
        why={
            "seed": "Chance moves a loss a little, and one run per size cannot rule it out. But the big run differs from the small ones in a way the law does not know about: its tokens per parameter, the fewest of all.",
            "floor": "A higher floor bends the law and raises the prediction, but the floor is chosen to fit the small runs; it cannot know what so few tokens per parameter do to the big one.",
            "right": "Most likely. A law of small runs is a guide for runs made the same way, and the big run differs most: the same data for far more parameters, 0.5 tokens per parameter, far outside the range Chinchilla fitted its formula on. Labs scale the data with the model, as `compute_optimal` does.",
        },
    )
    choice_low
    return (choice_low,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The laws of [Scaling laws](/course/open/llm-scaling-laws?lang=en), now fitted through your own runs, with a floor they pick themselves; your ModernGPT of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en), now in five sizes; and the shards of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), now the data every size sees, with the schedule of [Reading an open training recipe](/course/open/llm-open-recipes?lang=en). The capstone uses your `compute_optimal` to size the small LLM you pretrain.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_floor, result_predict, result_optimal, choice_low):
    L.wrap_up(result_floor, result_predict, result_optimal, choice_low)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
