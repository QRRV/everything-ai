# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-hypothesis-control. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-hypothesis-control", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en) and [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_wrong():
    quick_wrong = L.choice(
        "quick-wrong",
        question="A hypothesis: \"With a warmup, my GPT ends its run with a lower validation loss than without.\" Which result would show that it is wrong?",
        options={
            "longer": "The run with a warmup takes a few seconds longer",
            "right": "The run with a warmup ends at the same or a higher validation loss",
            "train": "Without a warmup, the training loss falls faster in the first epoch",
        },
        correct="right",
        why={
            "longer": "The hypothesis is about the validation loss at the end, not about time. See [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en).",
            "train": "The hypothesis says what happens at the end of the run, and to the validation loss. A faster start does not prove it wrong. See [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en).",
            "right": "The hypothesis expects a lower validation loss at the end; the same or a higher one proves it wrong.",
        },
    )
    quick_wrong
    return (quick_wrong,)


@app.cell(hide_code=True)
def quick_check_val():
    quick_val = L.choice(
        "quick-val",
        question="You compare two runs of your GPT. Why do you look at their validation loss, and not at their training loss?",
        options={
            "lower": "The validation loss is always lower than the training loss",
            "faster": "The validation loss takes less time to compute",
            "right": "The validation loss is measured on text never trained on",
        },
        correct="right",
        why={
            "lower": "It is usually higher, not lower: the model has not seen that text. See [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en).",
            "faster": "Speed is not the reason. See [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en).",
            "right": "A model can learn its training text by heart; the validation loss shows how well it does on new text.",
        },
    )
    quick_val
    return (quick_val,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Say your GPT ended its run at a validation loss of 1.66. You want it lower, so you raise the learning rate, make the model deeper and make the windows longer, all at once, and the next run ends at 1.60. Which change did it? Perhaps one helped and another hurt. Perhaps it was none of them, and the run just went differently. You cannot tell. To know what a change does, you change one thing, and compare with a run without it.

    **An everyday example.** You bake bread with new flour, a hotter oven and less salt, and it tastes better. Next week you want to bake it again: which of the three do you keep?
    """)
    return


@app.cell(hide_code=True)
def predict_lr():
    guess_lr = L.guess(
        "three-times",
        question="A small GPT trains for 6 epochs with a learning rate of 0.001. A second run is the same in every way, but with a learning rate of 0.003. What do you think its validation loss does after the 6 epochs?",
        options={
            "right": "It is clearly lower",
            "same": "It is about the same",
            "higher": "It is clearly higher: too big a step",
        },
        reveal="**Clearly lower.** In the course's runs with 3 seeds, the control ended at 2.39 to 2.42, and the runs with 0.003 at 2.08 to 2.10: lower with every seed, by about 0.3. For this small GPT, which trains for only 6 short epochs, 0.001 is too careful: with bigger steps it learns more in the same time. Whether 0.003 is also better for your bigger GPT, this experiment does not tell you: it tested this setup.",
    )
    guess_lr
    return (guess_lr,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An experiment compares runs. The **control run** is your setup as it is, without the change you want to test. A **variant** is a copy of it in which you change exactly one setting: the **experimental variable**, which school science calls the independent variable. Everything else stays the same: the data, the sizes of the model, the number of epochs, the code. What you measure, here the validation loss at the end, you compare between the control and the variant. <!-- termen:ok -->

        Why one setting? If the variant differs in two, a difference in the result has two possible causes, and you cannot tell them apart. A run with a higher learning rate *and* an extra epoch that ends lower tells you nothing about the learning rate.

        Give both runs the same **seed**. The seed decides the starting weights of the model and the order of the mini-batches. With the same seed, a variant with another learning rate or warmup starts from the same weights as the control and gets the mini-batches in the same order: the setting is the only difference. A change to the model itself, such as more layers, uses the random numbers differently, so there the order of the mini-batches changes too. And one run per setting is still one roll of the dice: a later lesson repeats every run with several seeds.

        In code, settings fit in a dictionary. `dict(control)` makes a new dictionary with the same pairs; then you change one pair in it. `variant = control` would not: it gives the same dictionary a second name, so a change to the variant changes the control too.
        """),
        mo.Html(C.settings_picture(C.CONTROL, {**C.CONTROL, "lr": 0.003},
                                   {"title": "A control run and a variant", "setting": "setting", "control": "control", "variant": "variant", "changed": "changed",
                                    "alt": "A table of nine settings. The control run and the variant are the same in every setting except the learning rate, lr: 0.001 in the control, 0.003 in the variant, marked as changed."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **control run** | the run of an experiment that is your setup as it is, without the change you test; every other run is compared with it |
        | **experimental variable** | the one setting you change on purpose between the runs of an experiment |
        | **variant** | a run that differs from the control run in the experimental variable only |
        """),
        L.watch_out("A control run is not a baseline. A baseline is another method you compare with, as in [Always beat a simple baseline](/course/open/ml-baselines?lang=en); a control run is your own setup without the change. An experiment often needs both."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To test what one change does:

    1. Write down the hypothesis: which setting you change, to what, what you expect, and what you measure.
    2. Write down the control run: every setting of the run you have now.
    3. Make the variant: a copy of the control with that one setting changed.
    4. Train both, with the same seed, and compare what you measure.
    5. Before you conclude anything, check that the two runs differ in that one setting only.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    **1. The hypothesis:** with a learning rate of 0.003 instead of 0.001, the small GPT of this lesson ends its 6 epochs with a lower validation loss. Steps 2, 3 and 5 in code; the comments say which step each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_variant():
    # 2. the control run: every setting of the run
    _control = {"lr": 0.001, "warmup": 100, "layers": 2, "d": 64, "heads": 4, "block": 64, "batch": 32, "epochs": 6, "seed": 0}
    # 3. the variant: a new dictionary with the same pairs, and one setting changed
    _variant = _control  # complete: dict(_control)
    _variant["lr"] = 0.003
    # 5. check: in which settings do the two runs differ?
    _differ = []
    for _name in _control:
        if _control[_name] != _variant[_name]:
            _differ.append(_name)
    print("control lr:", _control["lr"], " variant lr:", _variant["lr"])
    print("differ in:", _differ)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, `_variant = _control` gives the same dictionary a second name, so the next line changes the control too: both have a learning rate of 0.003, step 5 finds no difference at all, and your control run is gone. Completed, `dict(_control)` makes a new dictionary: the control keeps its 0.001, and the two runs differ in `lr` and in nothing else. Then step 4, training both, can show what the learning rate does. The demo below does step 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_change():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A variant with one change

        Make `one_change(control, name, value)`: a new dictionary with every pair of `control`, in which the setting `name` has the value `value`. `control` itself must stay as it was. Replace the two `...`. The check calls `one_change(control, "warmup", 0)` with the control of the worked example, and tries other settings and another dictionary.
        """),
        L.task("Make `one_change(control, name, value)` give back a copy of `control` with only the setting `name` changed to `value`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_change():
    box_change = L.code_box("one-change", lines=7)
    box_change
    return (box_change,)


@app.cell(hide_code=True)
def check_change(box_change):
    result_change = L.check("one-change", box_change.value)
    result_change
    return (result_change,)


@app.cell(hide_code=True)
def hints_change():
    L.hints(
        "one-change",
        [
            "Two steps, as in step 3 of the worked example: first a new dictionary with the same pairs as `control`, then the one pair changed, with the names `name` and `value`.",
            "`variant = dict(___)`, then `variant[___] = ___`.",
        ],
        why="```python\ndef one_change(control, name, value):\n    variant = dict(control)\n    variant[name] = value\n    return variant\n```\n\n`dict(control)` makes a new dictionary with the same pairs, so changing the variant leaves `control` as it was. `variant[name] = value` changes the one setting.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Step 4: run the experiment

    Pick the setting to change; the control run has a learning rate of 0.001, a warmup of 100 steps and 2 layers. The button trains the control run of the worked example, and then the variant that your `one_change` makes of it: two runs of a small GPT of about 110,000 parameters (210,000 with 4 layers), with your `GPT`, `gpt_loaders` and `Trainer`, on the first quarter of Tiny Shakespeare, on the CPU. Each run takes about half a minute to a minute, a variant with 4 layers about twice as long. While it trains, the rest of the lesson waits; the stop button of the cell ends it.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    change_choice = mo.ui.dropdown(
        options={"learning rate 0.003": 0, "no warmup": 1, "4 layers": 2},
        value="learning rate 0.003", label="Change:")
    run_button = mo.ui.run_button(label="Run the experiment")
    mo.vstack([change_choice, run_button], align="start")
    return change_choice, run_button


@app.cell(hide_code=True)
def demo_experiment(box_change, change_choice, result_change, run_button):
    _labels = {"title": "Validation loss", "x": "epoch", "y": "loss", "lang": L.lang, "control": "control", "variant": "variant"}
    _name, _value = C.CHANGES[change_choice.value]
    if not run_button.value:
        _out = mo.md("*Pick a change and press the button.*")
    else:
        _fn, _yours = C.one_change_of(box_change.value if result_change.state == "correct" else None)
        _variant = _fn(dict(C.CONTROL), _name, _value)
        _differ = C.changed(C.CONTROL, _variant)
        if _differ != [_name]:
            _out = mo.md(f"*The variant differs from the control in {', '.join(f'`{_d}`' for _d in _differ) or 'nothing'}, not in `{_name}` alone. Check exercise 1 again.*")
        else:
            _runs = []
            for _which, _settings in (("the control run", C.CONTROL), ("the variant", _variant)):
                _h = C.run(_settings, lambda _share, _w=_which: mo.output.replace(mo.md(f"*Training {_w}: {round(100 * _share)}%*")))
                if isinstance(_h, str):
                    break
                _runs.append(_h)
            if len(_runs) < 2:
                _out = mo.md("*You stopped the experiment before both runs were done. Press the button to start it again.*" if _h == "stopped" else "*A run stops with an error. Do your modules in `work/projects/llm/` and `work/projects/nn/` still pass their checks? Delete a file you changed, and check its exercise again.*")
            else:
                _c, _v = _runs[0][-1]["val_loss"], _runs[1][-1]["val_loss"]
                _ref = C.REFERENCE.get(f"{_name}={_value}")
                _course = "" if not _ref else f" In the course's runs with seeds 0, 1 and 2, the control ended at {C.number(_ref['control'][0], 3, L.lang)} to {C.number(_ref['control'][1], 3, L.lang)}, and this variant lower than the control with every seed, by {C.number(_ref['lower'][0], _ref['lower'][2], L.lang)} to {C.number(_ref['lower'][1], _ref['lower'][2], L.lang)}."
                _who = "" if _yours else " Exercise 1 is not right yet, so the course's `one_change` made the variant."
                _out = mo.vstack([
                    mo.Html(C.curves_picture(_runs[0], _runs[1], {**_labels, "alt": f"The validation loss after every epoch: the control ends at {C.number(_c, 3, L.lang)}, the variant at {C.number(_v, 3, L.lang)}."})),
                    mo.md(f"After {len(_runs[0])} epochs, the control run ended at a validation loss of {C.number(_c, 3, L.lang)}, and the variant, with `{_name} = {_value}`, at {C.number(_v, 3, L.lang)}.{_course}{_who}"),
                ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_confound():
    mo.md(r"""
    ### Exercise 2. What changed?
    """)
    return


@app.cell(hide_code=True)
def choice_confound_cell():
    choice_confound = L.choice(
        "confound",
        question="Your control run trains 6 epochs with a learning rate of 0.001 and seed 0. A variant trains 8 epochs with a learning rate of 0.003 and seed 0, and ends with a lower validation loss. What can you conclude?",
        options={
            "lr": "The higher learning rate helps: that is what you wanted to test",
            "right": "Not which change helped: the variant changed two things",
            "epochs": "Training longer helps: the variant had two epochs more",
        },
        correct="right",
        why={
            "lr": "The variant also trained two epochs longer, so the extra epochs could explain the difference just as well. See *The idea*.",
            "epochs": "The variant also had a higher learning rate, so that could explain the difference just as well. See *The idea*.",
            "right": "Two settings changed, so the difference has two possible causes. Make two variants, each with one change, and compare each with the control.",
        },
    )
    choice_confound
    return (choice_confound,)


@app.cell(hide_code=True)
def text_exercise_design():
    mo.md(r"""
    ### Exercise 3. Design an experiment

    Your hypothesis: a warmup of 100 steps gives the small GPT of this lesson a lower validation loss after 6 epochs than no warmup at all.
    """)
    return


@app.cell(hide_code=True)
def explain_design():
    explain_design_cell = L.explain(
        "design",
        question="Design the experiment with the recipe: write down the control run, the variant, what stays the same, what you measure, and which result would show the hypothesis wrong.",
        model_answer="**Control run:** the settings of the worked example, with `warmup` 100. **Variant:** the same settings with `warmup` 0, made with `one_change`. **The same:** the data, the learning rate, the sizes of the GPT, the batch size, the 6 epochs and the seed. **Measured:** the validation loss after epoch 6. **Wrong if:** the variant without warmup ends at the same or a lower validation loss than the control.\n\nThe course did this experiment with seeds 0, 1 and 2: without warmup, the run ended a little lower each time, by 0.004 to 0.008. For this small GPT, at this learning rate, the hypothesis is wrong; that is a result too. Whether so small a difference is more than chance, a later lesson shows.",
        points=[
            "the control has a warmup of 100 steps and the variant none, and nothing else differs",
            "both runs use the same seed and the same 6 epochs",
            "you say what you measure, and which result proves the hypothesis wrong",
        ],
    )
    explain_design_cell
    return (explain_design_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en) said that an experiment changes something on purpose and keeps the rest the same; now you do that with the GPT of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en). In the next lesson, *Ablation: what happens if you take a part out?*, the one change is a whole part of your GPT, left out. And because one run per setting is one roll of the dice, a lesson after that repeats every run with several seeds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_confound, explain_design_cell, result_change):
    L.wrap_up(result_change, choice_confound, explain_design_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
