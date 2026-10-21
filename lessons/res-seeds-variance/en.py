# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-seeds-variance. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-seeds-variance", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en) and [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="In your ablation table, the GPT without positions and the full GPT ended close together. Each row is one run. What is still missing before you can say whether taking the positions out made a difference?",
        options={
            "bigger": "A run with a bigger GPT",
            "right": "More runs of each row, with other seeds",
            "longer": "A longer run of the GPT without positions",
        },
        correct="right",
        why={
            "bigger": "A bigger GPT is another setup; it says nothing about chance in this one. See [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).",
            "longer": "A longer run is also another setup. See [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).",
            "right": "One run per row is one roll of the dice: with other seeds, each number moves a little. This lesson measures how much.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_interval():
    quick_interval = L.choice(
        "quick-interval",
        question="A 95% bootstrap interval of a mean runs from 2.39 to 2.41. What does it say?",
        options={
            "runs": "95% of all single values lie between 2.39 and 2.41",
            "true": "The true mean is 2.40",
            "right": "How far the mean would move with other samples like this one",
        },
        correct="right",
        why={
            "runs": "The single values spread wider than their mean: the interval is about the mean. See [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en).",
            "true": "The interval does not pin the mean down to one number; it gives a range. See [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en).",
            "right": "The bootstrap draws new samples from the one you have, and the middle 95% of their means is the interval.",
        },
    )
    quick_interval
    return (quick_interval,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Change one thing at a time](/course/open/res-hypothesis-control?lang=en), the small GPT without a warmup ended a little lower than with one. Is that the warmup, or chance? The seed decides the starting weights of a model and the order of its mini-batches, and with another seed the same run ends at another number. Before you say that a change helps, you need to know how much a number moves when nothing changes but the seed.

    **An everyday example.** You weigh yourself, step off, and step on again: the scale shows 70.2, then 70.4. You did not gain 200 grams. Before you trust a change of 200 grams, you want to know how much the scale moves on its own.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "five-seeds",
        question="You train the small GPT of Change one thing at a time five times, with seeds 0 to 4 and nothing else different. How far apart do you think the lowest and the highest validation loss lie?",
        options={
            "same": "Not at all: the settings and the data are the same",
            "right": "A few hundredths",
            "tenths": "A few tenths, as far as a change of the learning rate",
        },
        reveal="**A few hundredths.** In the course's runs, seeds 0 to 4 ended at 2.390, 2.392, 2.417, 2.394 and 2.401: from lowest to highest 0.027 apart, with nothing changed but the seed. That is more than the warmup changed in Change one thing at a time, 0.004 to 0.008.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Run-to-run variance** is how much the result of the same run changes when only the seed changes. The seed decides the starting weights and the order of the mini-batches; both change where the training ends up. With one run, you cannot tell a real effect from this noise.

        So you train each setting with several seeds, the same seeds for every setting, and report three things:

        - the **mean** over the seeds: the number you report as the result;
        - the **spread**: the lowest and the highest run;
        - a **95% bootstrap interval** of the mean, as in [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en): how far the mean itself could move with other seeds.

        Then you compare a difference between two settings with this noise. If their intervals overlap a lot, the intervals do not show a difference: with other seeds, the two means could come out the other way around. That is not proof that there is none. Five seeds give a rough interval, often too narrow; it is still far better than one number. And the same seeds for both settings pair the runs, seed by seed: a small difference that two intervals miss can still show up pair by pair, as a later lesson shows.

        The picture: the course's runs of the small GPT with seeds 0 to 4, with a warmup and without. Each point is one run; the bar is the 95% bootstrap interval of the mean, the tick the mean.
        """),
        mo.Html(C.strip_picture([("with warmup", C.REFERENCE["control"]), ("without warmup", C.REFERENCE["warmup=0"])],
                                {"title": "Five seeds each", "x": "validation loss after 6 epochs", "lang": L.lang, "alt": "Two rows of five points. With warmup, the runs lie from 2.390 to 2.417, mean 2.399, interval of the mean 2.392 to 2.409. Without warmup, from 2.384 to 2.412, mean 2.394, interval 2.387 to 2.403. The two intervals overlap for the most part."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **run-to-run variance** | how much the result of the same run changes with only another seed |
        """),
        L.watch_out("A 95% interval of the mean is not where 95% of the runs land: the runs themselves spread wider. It says how sure you can be of the mean."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To report a result that holds up:

    1. Choose the seeds: at least 3, here 5, the same for every setting.
    2. Train each setting with each seed; nothing else changes.
    3. Report for each setting the mean, the lowest and the highest run, and the 95% bootstrap interval of the mean.
    4. Compare: if the intervals of two settings overlap a lot, the intervals do not show a difference.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 with made-up validation losses: five seeds of one setting; step 4 follows below the cell. The comments say which part of step 3 each line is. It is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_summary():
    import numpy as np

    _losses = np.array([1.652, 1.678, 1.664, 1.662, 1.671])
    # 3. the mean, and the lowest and the highest run
    print("mean:", _losses.mean(), " lowest:", _losses.min(), " highest:", _losses.max())
    # 3. the 95% bootstrap interval of the mean: the means of 10,000 resamples, and their middle part
    _rng = np.random.default_rng(0)
    _means = []
    for _ in range(10000):
        _means.append(np.mean(_rng.choice(_losses, size=len(_losses))))
    _low, _high = np.quantile(_means, [0.25, 0.75])  # complete: the middle 95%, [0.025, 0.975]
    print("interval:", _low, "to", _high)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The five runs lie 0.026 apart, from 1.652 to 1.678. With the quantiles 0.25 and 0.75 the interval holds the middle half of the means: much too narrow. With 0.025 and 0.975 it holds the middle 95%, about 1.658 to 1.672: narrower than the spread of the runs, because a mean of five moves less than one run. **Step 4:** a second setting gets its own interval in the same way. If the two overlap a lot, the intervals cannot tell the settings apart.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_summary():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A summary of several seeds

        Complete `seed_summary(losses, n_boot, seed)`: step 3 of the recipe as a function. It gives back three numbers: the mean of `losses`, and the low and the high end of the 95% bootstrap interval of that mean, from `n_boot` resamples with a random number generator made with `seed`. Replace the two `...`. The check calls `seed_summary([2.41, 2.38, 2.44, 2.40, 2.39], 10000, 0)`, and tries other runs. A bootstrap is random, so the check allows the small differences that another way of drawing gives.
        """),
        L.task("Make `seed_summary(losses, n_boot, seed)` give back the mean of `losses` and the two ends of the 95% bootstrap interval of that mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_summary():
    box_summary = L.code_box("seed-summary", lines=10)
    box_summary
    return (box_summary,)


@app.cell(hide_code=True)
def check_summary(box_summary):
    result_summary = L.check("seed-summary", box_summary.value)
    result_summary
    return (result_summary,)


@app.cell(hide_code=True)
def hints_summary():
    L.hints(
        "seed-summary",
        [
            "The same steps as in the worked example, with the names `losses`, `n_boot` and `rng`: `n_boot` times the mean of a resample, drawn with replacement; then the two quantiles of those means.",
            "In the loop: `means.append(np.mean(rng.choice(___, size=len(losses))))`. After it: `low, high = np.quantile(___, [___, ___])`.",
        ],
        why="```python\nimport numpy as np\n\n\ndef seed_summary(losses, n_boot, seed):\n    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n        means.append(np.mean(rng.choice(losses, size=len(losses))))\n    low, high = np.quantile(means, [0.025, 0.975])\n    return [np.mean(losses), low, high]\n```\n\nEach resample draws as many runs as there are, with replacement, so a run can come twice or not at all. The middle 95% of the means of the resamples is the interval of the mean.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your GPT with five seeds

    The button trains the small GPT of [Change one thing at a time](/course/open/res-hypothesis-control?lang=en) five times, with seeds 0 to 4, with your `GPT`, `gpt_loaders` and `Trainer` on the CPU: about half a minute to a minute per run, so a few minutes in all. The rest of the lesson waits while it trains, and the cell's stop button ends it. Meanwhile a chart shows the validation loss after every epoch, of each seed in turn, with the runs of the earlier seeds dashed beside it. The five end close together, but not at the same number; the picture after the runs zooms in on where they end.
    """)
    return


@app.cell(hide_code=True)
def seeds_button_cell():
    seeds_button = mo.ui.run_button(label="Train your GPT with five seeds")
    seeds_button
    return (seeds_button,)


@app.cell(hide_code=True)
def demo_seeds(box_summary, result_summary, seeds_button):
    if not seeds_button.value:
        _out = mo.md("*Press the button to start the five runs.*")
    else:
        _done, _now = [], {}   # the curves of the seeds before; the live chart of the seed that trains now

        def _hear(_i, _e):
            if _now.get("seed") != _i:
                if "live" in _now:
                    _done.append(list(zip(_now["live"].xs, _now["live"].ys)))
                _now["seed"], _now["live"] = _i, L.live({"title": f"Seed {_i}: the validation loss", "x": "epoch", "y": "validation loss",
                                                          "alt": f"The validation loss after every epoch so far of the run with seed {_i}{'; dashed, the runs of the earlier seeds' if _done else ''}."}, _e["epochs"])
                for _c in _done:
                    _now["live"].reference(_c, "", role="loss")
            if "val_loss" in _e:
                _now["live"].add(_e["epoch"], _e["val_loss"], f"**Seed {_i}** · epoch {_e['epoch']} of {_e['epochs']} · validation loss {C.number(_e['val_loss'], 3, L.lang)}")
            else:
                _now["live"].line(f"**Seed {_i}** · epoch {_e['epoch']} of {_e['epochs']} · batch {_e['batch']} of {_e['batches']}")

        mo.output.replace(mo.md("*Seed 0 · starting …*"))   # until the first batches of seed 0
        _runs = C.run_seeds(None, _hear)
        if isinstance(_runs, str):
            _note = "*You stopped the runs. Press the button to start again.*" if _runs == "stopped" else "*A run stops with an error. Do your modules in `work/projects/llm/` and `work/projects/nn/` still pass their checks? Delete a file you changed, and check its exercise again.*"
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)   # runs that stop keep their chart so far
        else:
            _summary, _yours = C.summary_of(box_summary.value if result_summary.state == "correct" else None)
            _mean, _low, _high = _summary(_runs, 10000, 0)
            _who = "" if _yours else " Exercise 1 is not right yet, so the course's `seed_summary` made the summary."
            _out = mo.vstack([
                mo.Html(C.strip_picture([("yours, with warmup", _runs), ("course, with warmup", C.REFERENCE["control"]), ("course, without warmup", C.REFERENCE["warmup=0"])],
                                        {"title": "Five seeds each", "x": "validation loss after 6 epochs", "lang": L.lang,
                                         "alt": f"Three rows of five runs each. Your runs lie from {C.number(min(_runs), 3, L.lang)} to {C.number(max(_runs), 3, L.lang)}, with a mean of {C.number(_mean, 3, L.lang)}."})),
                mo.md(f"Your five runs ended at {', '.join(C.number(_v, 3, L.lang) for _v in _runs)}: a spread of {C.number(max(_runs) - min(_runs), 3, L.lang)}, a mean of {C.number(_mean, 3, L.lang)}, and a 95% interval of the mean from {C.number(_low, 3, L.lang)} to {C.number(_high, 3, L.lang)}.{_who}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.md(r"""
    ### Exercise 2. Does warmup matter here?
    """)
    return


@app.cell(hide_code=True)
def choice_overlap_cell():
    choice_overlap = L.choice(
        "overlap",
        question="The course's five seeds: with warmup, mean 2.399 and 95% interval 2.392 to 2.409; without warmup, mean 2.394 and interval 2.387 to 2.403. What can you say from these two intervals?",
        options={
            "lower": "Without warmup is better here: its mean is lower",
            "none": "Warmup makes no difference here: the intervals overlap",
            "right": "They cannot tell the two apart: they overlap a lot",
        },
        correct="right",
        why={
            "lower": "Its mean is lower by 0.005, less than a mean of five runs moves by chance: the two intervals alone do not show it. See *The idea*.",
            "none": "Overlapping intervals do not show that there is no difference. With every one of the five seeds, the run without warmup ended lower: seed by seed, the runs can show what two intervals miss. See *The idea*.",
            "right": "As two groups, the difference of 0.005 is smaller than the noise between seeds. But that the run without warmup ended lower with every one of the five seeds is a hint that the intervals miss something; a later lesson tests it, by comparing the runs seed by seed.",
        },
    )
    choice_overlap
    return (choice_overlap,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Exercise 3. Report your result
    """)
    return


@app.cell(hide_code=True)
def explain_report():
    explain_report_cell = L.explain(
        "report",
        question="Write down the result of your five runs as you would in a paper: the setup, the number of seeds, the mean, the spread and the interval, and one sentence on what it means. No demo run yet? Use the course's numbers in the picture of The idea.",
        model_answer="**Setup:** the small GPT of Change one thing at a time, with warmup, 6 epochs on the first quarter of Tiny Shakespeare, seeds 0 to 4. **Result:** a validation loss of 2.399 on average, from 2.390 to 2.417, with a 95% bootstrap interval of the mean from 2.392 to 2.409. **Meaning:** two runs of this setup can end 0.03 apart by chance alone, so a difference between two settings of a few thousandths, measured with one run each, says nothing.",
        points=[
            "you name the setup and the five seeds",
            "you give the mean, the spread from the lowest to the highest run, and the 95% interval of the mean",
            "you say what it means: how much a single run can move by chance, so how big a difference has to be before it says something",
        ],
    )
    explain_report_cell
    return (explain_report_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Change one thing at a time](/course/open/res-hypothesis-control?lang=en) and [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en) each compared one run with one run; now you know how much one run moves by chance. The interval comes from [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en). With several seeds per setting, you have many runs to keep track of: the next lesson, *Keeping an experiment log*, shows how to write every run down.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_overlap, explain_report_cell, result_summary):
    L.wrap_up(result_summary, choice_overlap, explain_report_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
