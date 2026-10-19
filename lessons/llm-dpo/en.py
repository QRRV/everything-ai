# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-dpo. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("llm-dpo", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [DPO: preferences without a reward model](/course/open/llm-dpo-loss?lang=en) and [LoRA: fine-tuning with a low-rank update](/course/open/llm-lora?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_margin():
    quick_margin = L.choice(
        "quick-margin",
        question="After some steps of DPO, the policy makes the chosen answer of a pair 2 nats less likely than the reference model does, and the rejected answer 6 nats less likely. Is the loss of that pair now below ln 2, or above?",
        options={
            "above": "Above: the chosen answer got less likely, and DPO wanted it up",
            "right": "Below: the margin is -2 - (-6) = 4, above 0",
            "same": "Exactly ln 2: both answers moved, so nothing changed",
        },
        correct="right",
        why={
            "above": "The loss only looks at the margin: what the chosen answer gained on the reference model, minus what the rejected one gained. -2 minus -6 is 4. See [DPO: preferences without a reward model](/course/open/llm-dpo-loss?lang=en).",
            "same": "Both moved, but not by the same amount: the margin is 4, and the loss is ln 2 only at a margin of 0. See [DPO: preferences without a reward model](/course/open/llm-dpo-loss?lang=en).",
            "right": "Keep it in mind: the loss sees only the margin, not which of the two answers moved.",
        },
    )
    quick_margin
    return (quick_margin,)


@app.cell(hide_code=True)
def quick_check_off():
    quick_off = L.choice(
        "quick-off",
        question="A `LoRALinear` with `on = False`: what does it give for an input `x`?",
        options={
            "zero": "Zeros: without its adapter the layer is switched off",
            "update": "Only the update of its adapter, `x @ A.T @ B.T` times the scale",
            "right": "What the layer it wraps gives, as if there were no adapter",
        },
        correct="right",
        why={
            "zero": "`on` switches only the adapter: `forward` always starts with the output of the layer it wraps. See [LoRA: fine-tuning with a low-rank update](/course/open/llm-lora?lang=en).",
            "update": "The update is what `on` leaves out; the output of the layer it wraps stays. See [LoRA: fine-tuning with a low-rank update](/course/open/llm-lora?lang=en).",
            "right": "So with every adapter off, the model is the model it was before LoRA.",
        },
    )
    quick_off
    return (quick_off,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The lesson about DPO trained on one pair, for twenty steps. A real run trains on many pairs, and needs the reference model's log-probabilities of all of them: from a second copy of the model in memory, or from a pass over every pair before the training starts. With LoRA, one copy gives both, on the way. The policy is the SFT model with adapters; switch the adapters off, and the SFT model is back, as the reference model. And the pairs need no people: for instructions with requested words, `words_used` says which answers are better.

    **An everyday example.** A painter paints on a sheet of glass laid over a finished painting. To see what has changed, the painter lifts the glass: underneath, the painting is as it was.
    """)
    return


@app.cell(hide_code=True)
def predict_moved():
    _y, _c = C.MEASURED["yours"]["moved"], C.MEASURED["course"]["moved"]
    guess_moved = L.guess(
        "moved",
        question="DPO on preference pairs from a checker, one pass over them. Afterwards, which moved further from the reference model: the chosen answers of the pairs, or the rejected ones?",
        options={
            "chosen": "The chosen ones: DPO pulls them up, and that is what it is for",
            "right": "The rejected ones: they fall much further than the chosen ones move",
            "same": "Both about as far: one up, the other down by the same amount",
        },
        reveal=f"**The rejected ones.** In the course's runs, on the first {C.MOVED} pairs: on its model of the demo run, the chosen answers moved {C.signed(_y['chosen'], 1, L.lang)} nats on average and the rejected ones {C.signed(_y['rejected'], 1, L.lang)}; on its own SFT model, {C.signed(_c['chosen'], 1, L.lang)} and {C.signed(_c['rejected'], 1, L.lang)}. The loss only asks for the margin, and pushing the rejected answers down gives it as well as pulling the chosen ones up: on the second model even the chosen ones became less likely. Still, the model now writes more answers that use every word: answers to other instructions than those of the pairs.",
    )
    guess_moved
    return (guess_moved,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Pairs from a checker.** The course's SFT model wrote four answers to each of a thousand instructions, such as *Write a story with the words: mirror and sun.* Your `words_used` says which answers use every requested word, and your `make_pairs` of [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en) makes a pair of every answer that passes and every one that fails. Two passes or two fails say nothing.

        **The reference model for free.** With LoRA, the policy is the SFT model with an adapter on every linear layer of its blocks. Each `LoRALinear` has its switch `on`: with every adapter off, the policy is the SFT model again, the reference model, without a second copy in memory. `B` starts at zero, so at the first step the two are the same; after it, only the adapters have changed.

        **One step of DPO** takes a batch of pairs: four log-probabilities each, with `answer_logprob` of [DPO: preferences without a reward model](/course/open/llm-dpo-loss?lang=en), the policy's with the adapters on and the reference model's with them off; the loss of `dpo_loss`; and a step of AdamW on the adapters only.
        """),
        draw.flow(
            [("Answers", ["4 to each of", "1,000 instructions"]),
             ("Pairs", ["chosen, rejected", "by words_used"]),
             ("DPO", ["adapters on:", "the policy", "adapters off:", "the reference model"])],
            forward=["make_pairs", "dpo_step"],
            back=[],
            title="From answers to a better model"),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **on-policy** | trained on answers that the model being trained wrote itself, before training | |
    | **off-policy** | trained on answers another model wrote; DPO works with both | `data/course-dpo-answers.json` |
    | `layer.on` | the switch of one adapter: `True` for the policy, `False` for the reference model | `layer.on = False` |
    | `torch.stack(xs)` | one tensor of a list of tensors, one more dimension; the gradient goes back to each | `torch.stack(pc)` |

    The answers come from the course's SFT model. If yours is the course's, the pairs are on-policy: the model you train wrote such answers itself, before training. If you fine-tuned your own in [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), they are off-policy. DPO trains on them all the same, but answers far from what your model would write teach it less: that is why the DPO paper takes as its reference model the SFT model that wrote the answers, where it can, to keep that gap small (Rafailov and others, 2023, section 4). The course's run on its model of the demo run is such an off-policy run.

    ### Recipe

    To train your model with DPO:

    1. Let a model write a few answers to every instruction, and let a checker say which pass: every answer that passes and every one that fails make a preference pair.
    2. Put LoRA adapters on your SFT model: the policy.
    3. Every step, for a batch of pairs: both answers under the policy, both under the reference model with every adapter off, the DPO loss, and a step of the optimizer on the adapters.
    4. One pass over the pairs, and ask the test instructions again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One of the thousand instructions, *Write a story with the words: mirror and sun.*, and the verdict on each of its four answers.
    """)
    return


@app.cell
def example_pairs():
    _item = C.items()[1]
    print(_item["words"])
    for _answer in _item["answers"]:
        _found = {_w.strip(".,!?;:\"'") for _w in _answer.lower().split()}
        print(all(_w in _found for _w in _item["words"]), _answer[:60] + " ...")
    return


@app.cell(hide_code=True)
def text_after_example():
    _c = C.MEASURED["course"]
    mo.md(f"""
    The first and the third answer use both words, the second and the fourth do not: *Sunnyville* is not *sun*. So this instruction gives 2 × 2 = 4 pairs: the first answer against the second and against the fourth, and the third against both of them. An instruction whose four answers all pass, or all fail, gives none: of the thousand, {C.number(_c['mixed'], 0, L.lang)} give pairs, {C.number(_c['pairs'], 0, L.lang)} in all.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Pairs from a checker

        Complete `dpo_pairs(items)`. Every item is a dict with the requested `"words"` of an instruction and the `"answers"` to it. Give back a list of `(words, chosen, rejected)` for every pair your `make_pairs` makes from the answers and the verdicts of your `words_used`. The check calls `dpo_pairs` with two instructions of three answers each, and more.
        """),
        L.task("Make `dpo_pairs` turn the checker's verdicts into preference pairs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("dpo-pairs", lines=10)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("dpo-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "dpo-pairs",
        [
            "Which function of your project says whether an answer uses every requested word, and what does `make_pairs` need from it, for each answer?",
            "Each pair of `make_pairs` is two answers. What does the pair still need, so that the instruction can be written in front of both?",
        ],
        why="The checker turns four answers into verdicts, and `make_pairs` turns the verdicts into pairs; the words come along, as DPO needs the instruction to work out the log-probability of each answer after it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_reference():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The reference model under the adapters

        Complete `reference_logprob(model, layers, ids, start)`. `model` is your SFT model with adapters, `layers` the list of `LoRALinear` layers your `add_lora` gave back, and `ids` and `start` as in `answer_logprob`. Give back the log-probability of the answer under the reference model: the model with every adapter off. Afterwards, the adapters are on again. The check uses a small model whose adapters have learned something already.
        """),
        L.task("Make `reference_logprob` switch the adapters off, work out the log-probability, and switch them on again."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reference():
    box_reference = L.code_box("reference-logprob", lines=12)
    box_reference
    return (box_reference,)


@app.cell(hide_code=True)
def check_reference(box_reference):
    result_reference = L.check("reference-logprob", box_reference.value)
    result_reference
    return (result_reference,)


@app.cell(hide_code=True)
def hints_reference():
    L.hints(
        "reference-logprob",
        [
            "Each `LoRALinear` has a switch that its `forward` looks at. What must it be for the reference model, and for which layers?",
            "After this function the adapters must go on learning. What should the switches be when it is done?",
        ],
        why="Under the adapters the SFT model is still there, untouched: the adapters are the only numbers that learn. Switching them off gives the reference model, and no second copy of the model is needed.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. One step of DPO

        Complete `dpo_step(model, layers, opt, batch, beta)`. `batch` is a list of `(chosen, rejected, start)`: the ids of the instruction with each answer, and where the answers start. Work out the four log-probabilities of every pair, the loss with your `dpo_loss`, and take one step with the optimizer `opt`. Give back the loss as a number. The check takes three steps on a small model with two pairs, and looks at the loss of each and at how far `B` moved from zero, and more.
        """),
        L.task("Make `dpo_step` take one step of DPO on a batch of pairs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("dpo-step", lines=16)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("dpo-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "dpo-step",
        [
            "The rejected answer goes through the policy as the chosen one does. Which of your functions gives both answers under the reference model, and why may those be plain numbers?",
            "Every training loop of the course ends a step with the same three lines. Which, and in which order?",
        ],
        why="Only the policy's log-probabilities carry a gradient, back to the adapters; the reference model's are fixed numbers to compare with.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _y, _c, _r = C.MEASURED["yours"], C.MEASURED["course"], C.RANK
    mo.vstack([
        mo.md(f"""
        ### Train your model with DPO

        The button puts adapters of rank {_r} on your SFT model of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), or on the course's when you have none, makes the pairs with your `dpo_pairs`, and runs your `dpo_step` once over all of them, in batches of {C.BATCH}, at $\\beta = {C.number(C.BETA, 1, L.lang).replace(',', '{,}')}$ and a learning rate of {C.number(C.LR, 3, L.lang)}. Then the model answers the {C.T.TESTS} test instructions of that lesson again, and the demo measures how far the answers of the first {C.MOVED} pairs moved. On the course's CPU, with one thread, that took about {round((_y['seconds'] + _y['other_seconds']) / 60)} minutes for its model of the demo run, and about {round((_c['seconds'] + _c['other_seconds']) / 60)} minutes on two threads for the course's own SFT model, which is about five times as big. The adapters are kept in `work/projects/llm/dpo.pt`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def button_cell():
    dpo_button = mo.ui.run_button(label="Train with DPO")
    dpo_button
    return (dpo_button,)


@app.cell(hide_code=True)
def demo_dpo(dpo_button, result_pairs, result_reference, result_step):
    trained = None
    _note = None
    _k = C.RANK   # the rank of the adapters
    if not all(_r.state == "correct" for _r in (result_pairs, result_reference, result_step)):
        _note = "*Finish exercises 1 to 3 first: the demo trains with your `dpo_pairs`, `reference_logprob` and `dpo_step`.*"
    elif dpo_button.value:
        def _progress(_p):
            if _p["phase"] == "pairs":
                mo.output.replace(mo.md(f"**{C.number(_p['pairs'], 0, L.lang)} pairs** from {C.number(_p['instructions'], 0, L.lang)} instructions · adapters of rank {_k} …"))
            elif _p["phase"] == "after":
                mo.output.replace(mo.md(f"**After DPO** · instruction {_p['done']} of {_p['total']} · every word used: {_p['used']}"))
            else:
                mo.output.replace(mo.md(f"**Step {C.number(_p['step'], 0, L.lang)} of {C.number(_p['steps'], 0, L.lang)}** · loss {C.number(_p['loss'], 3, L.lang)} · {C.number(_p['seconds'] / 60, 1, L.lang)} minutes"))
        _r = C.run(_progress)
        _note = {
            "run": "*The demo stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*",
            "pairs": "*Your `dpo_pairs` stops with an error on the course's answers, or gives no list of `(words, chosen, rejected)`. Does it still pass its check?*",
            "lora": "*Your `add_lora` stops with an error on the model. Does it still pass its check in the lesson about LoRA?*",
            "step": "*Your `dpo_step` or `reference_logprob` stops with an error on the model, or gives no number. Do they still pass their checks?*",
            "memory": "*The memory of your device is full. Close other programs, and press the button again.*",
            "diverged": "*The loss is no longer a number: the training ran off. Press the button again.*",
        }.get(_r) if isinstance(_r, str) else None
        trained = None if isinstance(_r, str) else _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _run = C.keep(trained) if trained is not None else C.saved() or C.MEASURED["yours"]
    _mine = trained is not None or C.saved() is not None
    _whose = ("your SFT model" if _run["base"] == "yours" else "the course's SFT model") if _mine else "the course's model of the demo run"
    _stand = "" if _mine else "\n\n*The course's run stands in until you train your own.*"
    _ex = _run["example"]
    _after = _ex["after"] if len(_ex["after"]) < 600 else _ex["after"][:600] + " ..."
    mo.vstack([
        mo.md(f"How many answers of {_whose} to the {_run['after']['total']} test instructions use every requested word:\n\n"
              f"| | every word used |\n|---|---|\n| SFT | **{_run['before']['used']}** of {_run['before']['total']} |\n"
              f"| SFT and DPO | **{_run['after']['used']}** of {_run['after']['total']} |\n\n"
              f"{C.number(_run['pairs'], 0, L.lang)} pairs, {C.number(_run['steps'], 0, L.lang)} steps. Against the reference model, the chosen answers of the first {C.MOVED} pairs moved {C.signed(_run['moved']['chosen'], 1, L.lang)} nats on average, the rejected ones {C.signed(_run['moved']['rejected'], 1, L.lang)}.{_stand}"),
        mo.Html(C.losses_picture(_run["losses"], _run["every"], {"title": "The DPO loss along the run", "x": "step", "y": "loss", "lang": L.lang,
                                                  "alt": f"The DPO loss along the run: from 0.693 to about {C.number(sum(_run['losses'][-5:]) / len(_run['losses'][-5:]), 2, L.lang)}."})),
        mo.md(("**" + ("A test instruction it did after DPO and not before" if _ex["better"] else "The first test instruction") + f"**: *{C.T.instruction(_ex['words']).splitlines()[0]}*\n\n> {_after}")),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _y, _c = C.MEASURED["yours"], C.MEASURED["course"]
    mo.md(f"""
    For the course's model of the demo run, DPO took the test instructions whose answer used every requested word from {_y['before']['used']} to {_y['after']['used']} of {C.T.TESTS}; for the course's own SFT model, from {_c['before']['used']} to {_c['after']['used']}. The pairs never ask for these test instructions, and the checker is the same one that counts them: DPO taught the model what `words_used` rewards. That is what it should learn here, but it is also why a check that can be gamed is a risk, as reward hacking showed in [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_off():
    mo.md(r"""
    ### Exercise 4. Why off?
    """)
    return


@app.cell(hide_code=True)
def choice_off_cell():
    choice_off = L.choice(
        "why-off",
        question="`B` starts at zero, so at the first step the policy gives the same log-probabilities with its adapters on as with them off. Why switch them off for the reference model at all?",
        options={
            "right": "After the first step the adapters have learned, and the reference model has to stay the SFT model",
            "speed": "With the adapters off the model runs faster, and the reference model is needed on every step",
            "grad": "With the adapters on, the reference model would get a gradient and learn along with the policy",
        },
        correct="right",
        why={
            "speed": "An adapter is small: switching it off saves little work. The reason is what the reference model stands for: the model the policy started as.",
            "grad": "`dpo_step` turns the reference model's log-probabilities into plain numbers, and `reference_logprob` works inside `torch.no_grad()`: no gradient goes through them, adapters on or off. The reason is what the reference model stands for: the model the policy started as.",
            "right": "With the adapters on, the reference model would move along with the policy: the margin would always be 0 and the loss ln 2, however far the policy went, and nothing would hold it near the SFT model.",
        },
    )
    choice_off
    return (choice_off,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The pairs of [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en), now made from a thousand instructions; the loss of [DPO: preferences without a reward model](/course/open/llm-dpo-loss?lang=en), now on many pairs; and the adapters of [LoRA: fine-tuning with a low-rank update](/course/open/llm-lora?lang=en), now also giving the reference model. Next: when a program can check the answer, the check can be the reward itself, as in the training of reasoning models.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pairs, result_reference, result_step, choice_off):
    L.wrap_up(result_pairs, result_reference, result_step, choice_off)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
