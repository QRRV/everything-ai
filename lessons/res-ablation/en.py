# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-ablation. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("res-ablation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Change one thing at a time](/course/open/res-hypothesis-control?lang=en) and [Telling attention where things are](/course/open/attn-positions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_control():
    quick_control = L.choice(
        "quick-control",
        question="In an experiment with your GPT, which run is the control run?",
        options={
            "change": "The run with the change you want to test",
            "right": "Your run without the change, the same in every other way",
            "other": "A run of another kind of model, such as the bigram model",
        },
        correct="right",
        why={
            "change": "That is the variant. See [Change one thing at a time](/course/open/res-hypothesis-control?lang=en).",
            "other": "A different kind of model is a baseline, not a control. See [Change one thing at a time](/course/open/res-hypothesis-control?lang=en).",
            "right": "Every variant is compared with it, so it differs from each variant in one setting only.",
        },
    )
    quick_control
    return (quick_control,)


@app.cell(hide_code=True)
def quick_check_positions():
    quick_positions = L.choice(
        "quick-positions",
        question="Your GPT adds a row of its table `self.pos` to the embedding of every token. What does that row tell the blocks?",
        options={
            "meaning": "What the token means",
            "right": "At which place in the window the token is",
            "next": "How likely each token is to come right after it",
        },
        correct="right",
        why={
            "meaning": "That is the row of the table of tokens, `self.tok`. See [Telling attention where things are](/course/open/attn-positions?lang=en).",
            "next": "That comes out at the end, from the head. See [Telling attention where things are](/course/open/attn-positions?lang=en).",
            "right": "The table has a row for every place; without it, one block could not tell where a token stands.",
        },
    )
    quick_positions
    return (quick_positions,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A paper presents a model with many parts, and shows that it works. But which parts does it need? Papers answer that with an ablation: take one part out, train again in exactly the same way, and measure. A table of such runs, the full model first, shows what each part adds. In [Change one thing at a time](/course/open/res-hypothesis-control?lang=en) the one change was a setting; here it is a part of your GPT, taken out. In this project step you do it with the position embedding of your own GPT.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_positions():
    guess_positions = L.guess(
        "no-positions",
        question="Take the position embedding out of your GPT: it then sees which characters came before, but not where they stand. What do you think happens to its validation loss after the same run?",
        options={
            "right": "About the same: attention still sees every character before it",
            "worse": "Clearly worse: it no longer knows in which order the characters came",
            "guess": "As bad as guessing evenly among the 65 characters",
        },
        reveal="**About the same, after the whole run.** In the course's runs of the CPU plan, a GPT of 5 blocks trained for 10 epochs with seeds 0 to 2, the full GPT ended at 1.660 to 1.674 and the GPT without positions at 1.663 to 1.668: no difference bigger than the spread between seeds. Without positions it did learn more slowly: after 2 epochs it was at 2.23 to 2.26, the full GPT at 2.06 to 2.09. With several blocks and the causal mask, a GPT can work out the order by itself. In 2022, Adi Haviv and his colleagues found the same for much bigger language models: without positions, they came close to models with them, though a little behind ([Haviv et al., 2022](https://arxiv.org/abs/2203.16634)). On the GPU plan, a bigger GPT trained for 20 epochs, the course ran each GPT once, with seed 0. The GPT without positions was behind after every epoch but the first: by 0.25 after 6 epochs, and by 0.014 at the end: 1.538 against 1.524. One run per row cannot tell whether 0.014 is more than chance: the next lesson shows how much one run moves.",
    )
    guess_positions
    return (guess_positions,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **ablation** takes one part of a model out, trains the model again in exactly the same way, and measures how much worse it gets. It is an experiment as in the last lesson: the full model is the control run, the model without the part is the variant, and everything else stays the same: the data, the sizes, the epochs, the seed. Several ablations together, each taking out another part, make an **ablation study**, which papers show as a table: the full model in the first row, then one row per part taken out.

        To take the position embedding out, you could rewrite your GPT. There is a smaller way: set every number of its table to 0, so that it adds nothing to the tokens, and **freeze** it, so that training cannot learn the numbers back. In PyTorch a parameter freezes when its `requires_grad` is `False`: it keeps no slopes, so the optimizer leaves it alone. The rest of your GPT, its table of tokens, its blocks and its head, trains as before.

        Without positions, attention still sees every character before it, but not in which order. One block cannot tell "abc" from "bac" at the last place: it sees the same three characters. With more blocks, a GPT can work the order out again, through the causal mask: the first place sees one character, the second two, and so on, and later blocks can tell the places apart by that.
        """),
        mo.Html(C.flow_picture({"title": "Your GPT, with the positions taken out",
                                "tok": "table of tokens", "pos": "table of places", "pos_note": "every number 0, frozen", "plus": "+",
                                "blocks": "the blocks", "head": "norm and head", "out": "the next character",
                                "alt": "The parts of your GPT from top to bottom: the table of tokens and the table of places, added; the table of places crossed out, every number 0 and frozen; then the blocks, then the norm and the head, then the next character."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **ablation** | taking one part of a model out, training again in the same way, and measuring how much worse it gets | |
        | **ablation study** | several ablations, each taking out another part, often shown as a table | |
        | **freeze** | keep parameters fixed while the rest trains | `p.requires_grad = False` |
        """),
        L.watch_out("An ablation shows what a part adds in this setup: this model, this data, this many epochs. With a bigger model or a longer run, the same part can matter more, or less. And one run per row is one roll of the dice, as the next lessons show."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out what one part of a model adds:

    1. Choose the part, and write down what you expect without it.
    2. The control run: the full model, trained as usual.
    3. The variant: the same model with the part taken out, its numbers set to 0 and frozen.
    4. Train the variant exactly as the control: the same data, sizes, epochs and seed. Compare the validation losses.
    5. Write the result as a row of an ablation table, and say what it does not show.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 and 3 on a small GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en), with 1 block and 4 places, that has not been trained. **1. The expectation:** without positions, it gives the same logits at the last place for "abc" and for "bac". Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_order():
    import torch

    _model = C.tiny_gpt()
    _abc = torch.tensor([[0, 1, 2]])
    _bac = torch.tensor([[1, 0, 2]])
    with torch.no_grad():
        # the logits at the last place, for "abc" and for "bac"
        _before = torch.allclose(_model(_abc)[0, -1], _model(_bac)[0, -1])
        # 3. take the positions out: every number of the table of places to 0
        _model.pos.weight[0] = 0  # complete: every row, [:], not only the first
        _after = torch.allclose(_model(_abc)[0, -1], _model(_bac)[0, -1])
    print("the same with positions:", _before)
    print("the same without positions:", _after)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With positions, the last place gets different logits for "abc" and "bac": the first two characters stand in other places. With only row 0 set to 0, row 1 still marks the second place, so they still differ. With every row 0, the two give the same logits: one block sees the same three characters, and nothing says where. In the worked example the model does not train, so freezing does not matter yet; in your run it does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_function():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Take the positions out

        Complete `no_positions(model)`: step 3 of the recipe for any GPT of yours. Set every number of `model.pos.weight` to 0, without a graph, and freeze it. The function changes the model itself and gives nothing back. The check calls it on the GPT of the worked example, also with a step of training after it, and on a GPT with 2 blocks. When it is right, the check saves it in your project, as `work/projects/llm/ablate.py`.
        """),
        L.course_versions(),
        L.task("Make `no_positions(model)` set the table of places of `model` to 0 and freeze it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_function():
    box_nopos = L.code_box("no-positions", lines=7)
    box_nopos
    return (box_nopos,)


@app.cell(hide_code=True)
def check_function(box_nopos):
    result_nopos = L.check("no-positions", box_nopos.value)
    result_nopos
    return (result_nopos,)


@app.cell(hide_code=True)
def hints_function():
    L.hints(
        "no-positions",
        [
            "Two lines, as in step 3 of the worked example and in *The idea*: one sets every number of the table to 0, under `with torch.no_grad():`; the other makes it stop training.",
            "`model.pos.weight[___] = 0` under `with torch.no_grad():`, then `model.pos.weight.requires_grad = ___`.",
        ],
        why="```python\nimport torch\n\n\ndef no_positions(model):\n    with torch.no_grad():\n        model.pos.weight[:] = 0\n    model.pos.weight.requires_grad = False\n```\n\n`[:]` sets every row to 0, and `torch.no_grad()` lets you change a parameter in place. With `requires_grad` `False`, the table gets no slopes, so the optimizer leaves it at 0.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _p = C.T.plan()
    _gpu = "the GPU of your Mac" if _p["device"] == "mps" else f"your GPU, {_p['name']}"
    _where = "on the CPU, the smaller GPT of the CPU plan" if _p["device"] == "cpu" else f"on {_gpu}, the GPT of {C.T.grouped(_p['params'], L.lang)} parameters"
    _time = f" On the course's computer, a CPU with 4 cores, such a run took about {round(C.T.REFERENCE['cpu'][0]['minutes'])} minutes." if _p["device"] == "cpu" else ""
    mo.md(f"""
    ### Step 4: the run without positions

    The button trains your GPT once more, {_where}, exactly as in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en): the same plan, {_p['epochs']} epochs and seed 0, with your `no_positions` applied before it trains.{_time} While it trains, the cell shows the loss after every epoch, and the rest of the lesson waits; the stop button of the cell ends the run early. Below, your run stands next to the run with positions: yours, if you trained one on this computer, or else the course's.
    """)
    return


@app.cell(hide_code=True)
def ablation_button_cell():
    ablation_button = mo.ui.run_button(label="Train your GPT without positions")
    ablation_button
    return (ablation_button,)


@app.cell(hide_code=True)
def demo_ablation(ablation_button, result_nopos):
    result_nopos  # run again when exercise 1 saves your no_positions
    _labels = {"title": "Your run without positions", "x": "epoch", "y": "loss", "lang": L.lang, "guess": "guessing evenly", "mlp": "your MLP", "train": "training", "val": "validation"}

    def _progress(_p):
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · about {_m} {'minutes' if _m > 1 else 'minute'} left"
        _parts = [mo.md(f"**Epoch {_p['epoch']} of {_p['epochs']}** · step {_p['step']} of {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.T.history_picture(_p["history"], {**_labels, "alt": "The training and validation loss of your run without positions so far, after every epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    _result = C.train(_progress) if ablation_button.value else C.last_run()
    if _result is None and result_nopos.state == "correct" and not ablation_button.value:
        _out = mo.md("*Press the button to start the run.*")
    elif _result is None:
        _out = mo.md("*Finish exercise 1 first, then press the button.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*The run stops with an error. Do your modules in `work/projects/llm/` and `work/projects/nn/` still pass their checks? Delete a file you changed, and check its exercise again.*",
            "memory": "*Your GPU ran out of memory. Close other programs that use it, such as a game or another notebook, and press the button again.*",
            "stopped": "*You stopped the run during its first epoch, so nothing of it was saved. Press the button to start again.*",
        }[_result])
    else:
        _control, _who = C.control()
        _h = _result["history"]
        _stopped = f" You stopped the run after epoch {len(_h)} of {_result['epochs']}, so compare with care." if _result["stopped"] else ""
        _stand = "" if not _result.get("stand_ins") else " The course's version stood in for " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + ", as the note at exercise 1 says."
        if _control is None:
            _out = mo.md(f"Without positions, your GPT reached a validation loss of {C.number(_result['best'], 3, L.lang)} after epoch {_result['best_epoch']}.{_stopped}{_stand} Train your GPT with positions in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en) to compare.")
        else:
            _whose = "your run of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en)" if _who == "yours" else "the course's run of the same plan, with seed 0"
            _d = _result["best"] - _control["best"]
            _diffs = C.course_differences("cpu" if _result["device"] == "cpu" else "gpu")
            _course = "" if not _diffs else f" In the course's {'runs' if len(_diffs) > 1 else 'run'} of this plan{', seed by seed' if len(_diffs) > 1 else ', with seed 0'}, the difference without positions minus with them was {', '.join(C.signed(_x, 3, L.lang) for _x in _diffs)}."
            _rows = [("full GPT", C.number(_control["best"], 3, L.lang)), ("without positions", C.number(_result["best"], 3, L.lang))]
            _out = mo.vstack([
                mo.Html(C.pair_picture(_control["history"], _h, {"title": "Validation loss", "x": "epoch", "y": "loss", "lang": L.lang, "guess": "guessing evenly",
                                                                  "with": "with positions", "without": "without",
                                                                  "alt": f"The validation loss after every epoch, with positions down to {C.number(_control['best'], 2, L.lang)} and without positions down to {C.number(_result['best'], 2, L.lang)}."},
                                       _result["epochs"])),
                mo.Html(C.table_picture(_rows, {"model": "model", "loss": "lowest validation loss"})),
                mo.md(f"The full GPT is {_whose}. Without positions, your GPT ended {C.number(abs(_d), 3, L.lang)} {'higher' if _d > 0 else 'lower'}.{_stopped}{_course}{_stand}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_table():
    mo.md(r"""
    ### Exercise 2. Reading an ablation table
    """)
    return


@app.cell(hide_code=True)
def choice_table_cell():
    choice_table = L.choice(
        "table",
        question="An ablation table as a paper would show it, made up for this question: the full model 1.52; without the MLPs 1.95; without layer normalization 1.71; without the position embedding 1.53. Each row was trained once. What can you say?",
        options={
            "right": "The MLPs add the most here; the position embedding may add nothing, or a little",
            "pos": "The position embedding can be left out of every model, since it changed the loss by only 0.01",
            "order": "Layer normalization matters more than the MLPs",
        },
        correct="right",
        why={
            "pos": "In this table it made a difference of 0.01, in one run of one setup. That is no proof that it can go from every model. See the watch-out in *The idea*.",
            "order": "Without the MLPs the loss rose more, 0.43, than without layer normalization, 0.19. See *The idea*.",
            "right": "Each row is compared with the full model: the bigger the rise, the more the part adds. A rise of 0.01 in one run is too small to say much, as the runs without positions in this lesson show.",
        },
    )
    choice_table
    return (choice_table,)


@app.cell(hide_code=True)
def text_exercise_shows():
    mo.md(r"""
    ### Exercise 3. What does your ablation show?
    """)
    return


@app.cell(hide_code=True)
def explain_shows():
    explain_shows_cell = L.explain(
        "what-it-shows",
        question="Write down your ablation as a row of a table, with the full GPT above it. No run of your own? Take the course's, on the CPU plan with seed 0: 1.662 for the full GPT; 1.668 without positions. Then: what does it show about the position embedding, and what does it not show?",
        model_answer="**Table:** full GPT 1.662; without positions 1.668 (the course's runs of the CPU plan with seed 0). **It shows** that in this setup, a GPT of 5 blocks trained for 10 epochs, the position embedding adds nothing measurable at the end: the difference, 0.006, is no bigger than how far runs with other seeds land apart. With positions it did learn faster in the first epochs. **It does not show** that positions never matter: with one block, longer texts or other data it can be different, and each row is one run.",
        points=[
            "you give both rows, with their validation losses",
            "you say what the difference means for the position embedding, in this setup",
            "you name a limit: one run per row, or only this model, data and length of training",
        ],
    )
    explain_shows_cell
    return (explain_shows_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Telling attention where things are](/course/open/attn-positions?lang=en) gave your GPT its positions; now you measured what they add. The experiment is the one of [Change one thing at a time](/course/open/res-hypothesis-control?lang=en), with a part taken out instead of a setting changed. Each row of your table is one run. In the next lesson, *One run is not a result: seeds and variance*, you repeat a run with other seeds, and see how much a number moves by chance alone.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_table, explain_shows_cell, result_nopos):
    L.wrap_up(result_nopos, choice_table, explain_shows_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
