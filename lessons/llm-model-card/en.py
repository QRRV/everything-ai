# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-model-card. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-model-card", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where does this data come from?](/course/open/data-provenance?lang=en) and [Red-team your own chat model](/course/open/llm-safety-eval?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_datasheet():
    quick_datasheet = L.choice(
        "quick-datasheet",
        question="What does the datasheet of a dataset write down that the table of data itself does not show?",
        options={
            "right": "Its source, its license, consent, and its limits",
            "stats": "The mean and the spread of every column",
            "models": "Which models were trained on it, and how well they did",
        },
        correct="right",
        why={
            "stats": "Those you compute from the table itself; a datasheet holds what the table cannot tell. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "models": "How well models did belongs on their own cards: a datasheet is about the data. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "A model card does the same for a model: it says what its files and its scores cannot say by themselves.",
        },
    )
    quick_datasheet
    return (quick_datasheet,)


@app.cell(hide_code=True)
def quick_check_filter():
    quick_filter = L.choice(
        "quick-filter",
        question="With a filter of 11 words in front, the course's model refused 14 of the 20 harmful requests. What else did the filter do?",
        options={
            "only": "Nothing: it stopped only harmful requests",
            "jail": "It stopped all 10 jailbreaks too",
            "right": "It stopped 4 harmless requests, and no jailbreak",
        },
        correct="right",
        why={
            "only": "It also stopped harmless requests, such as *How do I kill a Python process?* See [Red-team your own chat model](/course/open/llm-safety-eval?lang=en).",
            "jail": "Not one: a jailbreak is written so that the filter does not see its words. See [Red-team your own chat model](/course/open/llm-safety-eval?lang=en).",
            "right": "A card that said only *it refuses 14 of 20 harmful requests* would hide both.",
        },
    )
    quick_filter
    return (quick_filter,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You trained a chat model, and someone else may want to use it. Without a word from you, they do not know that it writes only stories for young children, that it never refuses, or that *60 of 100 test instructions* means less than it seems. A **model card** (Mitchell and others, 2019) comes with the model and says what it is for and what not, what it was trained on, how it was tested and with what results, and where it fails. On Hugging Face, the card is the file README.md of the model's repository, with a header the hub reads, as on the card of the digit reader of [Using a pretrained model](/course/open/torch-pretrained?lang=en).

    **An everyday example.** The leaflet in a box of medicine says what it is for, how much to take, its side effects, and when not to take it. You would not take a pill from a box without one.
    """)
    return


@app.cell(hide_code=True)
def predict_section():
    guess_section = L.guess(
        "section",
        question="Liang and others (2024) analysed 32,111 model cards on Hugging Face. Which section did they find filled in most often?",
        options={
            "limits": "Limitations: what the model cannot do",
            "right": "Training: how the model was trained",
            "eval": "Evaluation: how well the model does",
        },
        reveal=f"**Training: in {C.number(74.3, 1, L.lang)}% of the cards.** Evaluation was filled in on only {C.number(15.4, 1, L.lang)}% of them, and Limitations on {C.number(17.4, 1, L.lang)}%: two of the four sections filled in least. What a reader needs most, to decide whether a model fits their task, is just what cards often leave out.",
    )
    guess_section
    return (guess_section,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Facts, not hopes.** Everything on a card comes from a file a run wrote, or from a test you ran. A number names its test and its count: *60 of 100 test instructions*, not *60%*, and with the clean subset next to it.

        **What it is for, and what not.** The **intended use** says who the model is for and for what. Next to it stands what it is not for: what someone might try, and should not.

        **Limits come from the tests.** What the last training step made worse, and what is still too weak, are both findings; and so is what you did not test.

        **A header the hub reads.** At the top of README.md stands the card's **metadata**: between two lines `---`, a line `key: value` for each field, and for a list a line `key:` with a line `- item` for each item. That format is **YAML**. The hub shows the license from it, and finds the model by its datasets, language and tags. The digit reader's card begins like this:
        """),
        mo.Html(C.yaml_block(C.digit_reader_header())),
        mo.md(r"""
        **Licenses.** The card names the license of the model, and that of every dataset it learned from. The licenses of the data can set conditions on what you make with it: read each before you share.
        """),
        draw.flow(
            [("Your runs", ["capstone, SFT, DPO", "a .json file each"]),
             ("Your tests", ["verifiable instructions", "and a red-team set"]),
             ("Model card", ["header and sections", "README.md"])],
            forward=["three models", "scores, limits"],
            back=[],
            title="From your runs to a model card"),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **intended use** | who a model is for, and for what; the card says what it is not for next to it | `C.USES` |
    | **metadata** | data about data: on a model card, the header with its license, datasets, language and tags | `card_header(fields, lists)` |
    | **YAML** | the text format of that header: `key: value`, and a line `- item` for each item of a list | `"license: cc-by-4.0"` |

    ### Recipe

    To write a model card:

    1. Gather the facts of every run from its files: the model, its size, its data with their licenses, its training.
    2. Write what the model is for, and what it is not for.
    3. Give every test with its count, before and after each training step, and the clean subset next to it.
    4. Write the limits your tests found, and what you did not test.
    5. Put the license, the datasets, the language and the tags in the header.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 for the course's chat model: the facts of its three runs, from the files the runs wrote. `C.infos("course")` gives a dict per run.
    """)
    return


@app.cell
def example_facts():
    _info = C.infos("course")
    _base, _sft, _dpo = _info["base"], _info["sft"], _info["dpo"]
    print("model:", _base["params"], "parameters, and", _dpo["trainable"], "in the adapters of rank", _dpo["rank"])
    print("pretraining:", _base["tokens"], "tokens of", _base["stories"], "stories, in", _base["steps"], "steps")
    print("SFT:", _sft["pairs"], "pairs of an instruction and a story, in", _sft["steps"], "steps")
    print("DPO:", _dpo["pairs"], "preference pairs from", _dpo["instructions"], "instructions, in", _dpo["steps"], "steps")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Every number on these lines comes from a file a run wrote: nothing is typed in by hand, so nothing is guessed. What the files cannot say is how well the model does, and where it fails: for that you need your tests.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_header():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The header

        Complete `card_header(fields, lists)`. `fields` holds the fields with one value, and `lists` those with a list of values. Give back the header as one string: a line `---`, a line `key: value` for each field, for each list a line `key:` and a line `- item` for each item, and a closing line `---`. The check reads your header as the hub does, and calls `card_header({"license": "mit"}, {"language": ["en", "nl"]})`, and others.
        """),
        L.task("Make `card_header` write the header of a model card."),
    ])
    return


@app.cell(hide_code=True)
def exercise_header():
    box_header = L.code_box("card-header", lines=11)
    box_header
    return (box_header,)


@app.cell(hide_code=True)
def check_header(box_header):
    result_header = L.check("card-header", box_header.value)
    result_header
    return (result_header,)


@app.cell(hide_code=True)
def hints_header():
    L.hints(
        "card-header",
        [
            "A list takes two kinds of line: one for its key, and one for each of its items. Which loop goes inside which?",
            "The header ends the way it begins. What is its last line?",
        ],
        why="The hub reads the header to show the license and to find the model by its datasets and language; a header it cannot read is no header.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_limits():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The limits

        Complete `limits(before, after, totals, enough)`. `before` and `after` give, for every kind of a test, how many passed before and after the last training step, and `totals` how many there were of that kind. Give back a list of the kinds that are a limit: those that got worse, and those whose share after is below `enough`. The check calls `limits({"rhyme": 10, "names": 35, "short": 6}, {"rhyme": 25, "names": 30, "short": 14}, {"rhyme": 50, "names": 40, "short": 20}, 0.6)`, and others.
        """),
        L.task("Make `limits` find the kinds that got worse or are too weak."),
    ])
    return


@app.cell(hide_code=True)
def exercise_limits():
    box_limits = L.code_box("limits", lines=8)
    box_limits
    return (box_limits,)


@app.cell(hide_code=True)
def check_limits(box_limits):
    result_limits = L.check("limits", box_limits.value)
    result_limits
    return (result_limits,)


@app.cell(hide_code=True)
def hints_limits():
    L.hints(
        "limits",
        [
            "Two reasons make a kind a limit. What do you compare for each?",
            "Either reason is enough. Which word joins two conditions so that one of them is enough?",
        ],
        why="A card that names only what got better hides what a reader needs most: where the model fails.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### The card of your chat model

        The lesson writes the card from the files of the three runs and the scores of the two tests: of your own chat model when its capstone, SFT and DPO runs are all yours, and else of the course's. Your `card_header` writes its header, and your `limits` finds its limits, below the share you set. Write what the model is for and what not in the boxes, in English, as the card is. Your own models take both tests when you press the button: on the course's CPU, with one thread, that took about {round((C.E.MEASURED['yours']['seconds'] + C.F.MEASURED['yours']['seconds']) / 60)} minutes for its models of the demo runs. The button below the card saves it as `work/projects/llm/{C.CARD}`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def card_inputs():
    _yours = C.source() == "yours"
    uses_box = mo.ui.text_area(value="" if _yours else C.USES, label="What it is for", full_width=True)
    not_for_box = mo.ui.text_area(value="" if _yours else C.NOT_FOR, label="What it is not for", full_width=True)
    license_box = mo.ui.text(value="" if _yours else "cc-by-4.0", label="License of the model", full_width=True)
    enough_slider = mo.ui.slider(10, 90, step=10, value=50, label="Enough, in %: a kind below this share is a limit", show_value=True)
    measure_button = mo.ui.run_button(label="Let my models take both tests", disabled=not _yours)
    mo.vstack([uses_box, not_for_box, license_box, enough_slider, measure_button])
    return enough_slider, license_box, measure_button, not_for_box, uses_box


@app.cell(hide_code=True)
def demo_measure(measure_button):
    scores = C.measured(C.source())
    _note = None
    if measure_button.value:
        _names = {"base": "before post-training", "sft": "after SFT", "dpo": "after SFT and DPO"}
        _tests = {"chat": "verifiable instructions", "safety": "red-team set"}

        def _progress(_p):
            mo.output.replace(mo.md(f"**{_tests[_p['test']]}** · {_names[_p['model']]} · prompt {_p['done']} of {_p['total']} · {C.number(_p['seconds'] / 60, 1, L.lang)} minutes"))
        _r = C.measure(_progress)
        if _r == "run":
            _note = "*Your models cannot answer: a module of your project stops with an error. Do your modules in `work/projects/llm/` still pass their checks?*"
        else:
            scores = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (scores,)


@app.cell(hide_code=True)
def demo_card(scores, uses_box, not_for_box, license_box, enough_slider, box_header, box_limits, result_header, result_limits):
    _course = {"card_header": run_code(C.EXERCISES["card-header"].solution).names["card_header"],
               "limits": run_code(C.EXERCISES["limits"].solution).names["limits"]}
    _boxes = {"card_header": (box_header, result_header), "limits": (box_limits, result_limits)}
    _fns, _notes = {}, []
    for _k, (_box, _result) in _boxes.items():
        _fns[_k] = run_code(_box.value).names.get(_k) if _result.state == "correct" else None
        if _fns[_k] is None:
            _fns[_k] = _course[_k]
            _notes.append(f"*The course's `{_k}` stands in until yours passes its check.*")
    _src = C.source()
    _enough = enough_slider.value / 100   # the slider is in %
    _fields, _lists = C.fields(_src, license_box.value)
    try:
        _head = _fns["card_header"](_fields, _lists)
        _ok = C._header(_head) is not None
    except Exception:
        _ok = False
    if not _ok:   # yours stops on these fields, or writes a header the hub cannot read: the course's stands in
        _head = _course["card_header"](_fields, _lists)
        _notes.append("*Your `card_header` writes no header the hub can read for these fields: the course's stands in.*")
    _found = []
    if scores is not None:
        _c = scores["chat"]
        _totals = {_k: _c["total"][_k] for _k in C.CHAT if _c["total"][_k]}
        try:
            _found = [_k for _k in _fns["limits"](_c["sft"], _c["dpo"], _totals, _enough) if _k in C.CHAT]
        except Exception:
            _found = _course["limits"](_c["sft"], _c["dpo"], _totals, _enough)
    card_text = C.card(_src, scores, _head, uses_box.value, not_for_box.value, _found, _enough, license_box.value)
    _body = card_text[len(_head.rstrip("\n")):].replace("\n## ", "\n#### ").replace("\n# ", "\n### ")   # its headings below the lesson's
    _whose = "your chat model" if _src == "yours" else "the course's chat model"
    mo.vstack([
        mo.md(f"**The card of {_whose}.** Its header:"),
        mo.Html(C.yaml_block(_head)),
        mo.Html(f"<div style='border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.2rem 1rem; overflow-x: auto'>{mo.md(_body).text}</div>"),
        *[mo.md(_n) for _n in _notes],
    ])
    return (card_text,)


@app.cell(hide_code=True)
def save_cell():
    save_button = mo.ui.run_button(label="Save the card")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def demo_save(save_button, card_text):
    _msg = None
    if save_button.value:
        try:
            _msg = f"*Saved as `{C.save(card_text)}`: put it in the repository of your model as README.md when you share it.*"
        except OSError:
            _msg = "*The card could not be saved: is the folder `work/projects/llm/` there, and may the course write in it?*"
    mo.output.replace(mo.md(_msg)) if _msg else mo.output.clear()
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The files of the runs fill the sections on the model, its data and its training. The scores add what those files cannot say: how the model does on words never asked for together in training, what the last step made worse or left too weak, and whether it refuses. Those are the parts most cards leave out. What the model is for and what not, no program can write: those are yours, and so is the license: read the licenses of its data first.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_how_well():
    mo.md(r"""
    ### Exercise 3. How well does it do?
    """)
    return


@app.cell(hide_code=True)
def choice_how_well_cell():
    choice_how_well = L.choice(
        "how-well",
        question="Which line under *Evaluation* tells a reader what the 60 of 100 of the course's chat model is worth?",
        options={
            "vague": "It follows instructions well: 60% on our test",
            "right": "60 of 100 test instructions, most also trained on; 3 of the 20 that were not",
            "up": "60 of 100 test instructions after DPO, up from 24 after SFT and 1 before",
        },
        correct="right",
        why={
            "vague": "A share without its test and its count gives a reader nothing to check, and *well* is a hope, not a measurement.",
            "up": "True, but most of those instructions were also training instructions: without the clean subset, a reader takes 60 of 100 for what the model does with new ones.",
            "right": "A number with its test, its count, and the clean subset next to it: a reader sees what the 60 is worth.",
        },
    )
    choice_how_well
    return (choice_how_well,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The datasheet of [Where does this data come from?](/course/open/data-provenance?lang=en), now for a model; the model card of [Using a pretrained model](/course/open/torch-pretrained?lang=en), now written by you; and the tests of [Did post-training help?](/course/open/llm-eval-chat?lang=en) and [Red-team your own chat model](/course/open/llm-safety-eval?lang=en), now as its evaluation and its limits. Next: turning the formula of a paper into code.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_header, result_limits, choice_how_well):
    L.wrap_up(result_header, result_limits, choice_how_well)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
