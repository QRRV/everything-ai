# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-experiment-log. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-experiment-log", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en) and [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_seeds():
    quick_seeds = L.choice(
        "quick-seeds",
        question="You compare two settings with five seeds each. Which seeds does the second setting get?",
        options={
            "right": "The same five as the first",
            "new": "Five new ones, so that you see more chance",
            "any": "It does not matter which",
        },
        correct="right",
        why={
            "new": "New seeds bring other starting weights and another order of the mini-batches into the comparison. See [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
            "any": "The seed alone moves the result by a few hundredths. See [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
            "right": "With the same seeds, the two settings start from the same random numbers, so you can compare their runs seed by seed.",
        },
    )
    quick_seeds
    return (quick_seeds,)


@app.cell(hide_code=True)
def quick_check_ablation():
    quick_ablation = L.choice(
        "quick-ablation",
        question="In an ablation, you train your GPT without one part. What stays the same as in the run of the full GPT?",
        options={
            "data": "Only the data and the number of epochs",
            "right": "All the rest",
            "nothing": "Nothing has to: it is another model anyway",
        },
        correct="right",
        why={
            "data": "If the settings or the seed differ too, a difference has more than one cause. See [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).",
            "nothing": "Then you cannot tell what the missing part did. See [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en).",
            "right": "Data, settings, epochs and seed stay the same; only the part you take out differs, so a difference comes from that part or from chance.",
        },
    )
    quick_ablation
    return (quick_ablation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the last three lessons you trained your GPT many times: a control run and its variants in [Change one thing at a time](/course/open/res-hypothesis-control?lang=en), a GPT without positions in [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en), five seeds in [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en). In a month, someone asks which settings gave 2.39, and with which version of your code. A checkpoint holds weights, not settings, and a notebook forgets its numbers when it closes.

    The code itself is part of an experiment too. In 2018, Peter Henderson and his colleagues showed that public implementations of the same learning algorithms, run with the same settings, gave clearly different results ([Henderson et al., 2018](https://arxiv.org/abs/1709.06560)). So you also write down which code ran.

    **An everyday example.** A baker who tries twelve breads and writes nothing down knows afterwards that the seventh was the best, but not what was in it. A baker who writes down the flour, the water, the oven and the time of each bread can bake the seventh again.
    """)
    return


@app.cell(hide_code=True)
def predict_left():
    guess_left = L.guess(
        "runs-left",
        question="If you pressed every button, those three lessons trained your GPT eight times or more. How many of those runs can you still look up now, with their settings and their result?",
        options={
            "all": "All of them: the course saves every run",
            "right": "One at most",
            "none": "None at all",
        },
        reveal="**One at most.** Of those runs, your project keeps only one: the best run of the GPT without positions, in `work/projects/llm/gpt_nopos.json`. Even that file has no learning rate or warmup; those are only in your code. (Next to it, `gpt_chars.json` keeps the best run of your full GPT from Project: train your GPT on a book.) The runs of Change one thing at a time and of the five seeds lived only in the notebook: once it closed, their numbers were gone.",
    )
    guess_left
    return (guess_left,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **experiment log** is a file with one line for every run you make: the code version, all settings, the seed and the result. The code that trains writes the line itself, right after the run, and a line is never changed afterwards. Then the log holds what ran, not what you remember.

        Each line is written in **JSON**, a text format for data such as dictionaries and lists: `json.dumps(run)` turns a dictionary into one line of text, and `json.loads(line)` turns that line back into a dictionary. A file with one JSON line per run is called **JSON Lines**, often with the extension `.jsonl`. `open(path, "a")` opens a file to add to its end; `"w"` empties it first.

        The **code version** says which code ran. Here it is a **hash** of the code: a short string of letters and digits computed from the text, the same for the same text, and a completely different one when a single character changes. `hashlib.sha256` computes it, and the first 8 characters are enough to tell your versions apart. If you keep your code in Git, the commit hash does the same job. Tools such as MLflow and Weights & Biases keep a log like this for you, with a website to browse it; underneath, it is the same idea.

        The picture: one line of the course's log, with its four parts.
        """),
        mo.Html(C.line_picture({"title": "One run, one line", "code": "code version", "settings": "all settings", "seed": "seed", "val_loss": "result",
                                "under": "The dots stand for the five other settings; the real line holds all eight.",
                                "alt": "One line of JSON with four parts: the code version, a dictionary with all settings, the seed 0 and the validation loss 2.3902."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **experiment log** | a file with one line per run: code version, settings, seed and result, written by the code that trains |
        | **JSON** | a text format for data such as dictionaries and lists |
        | **JSON Lines** | a file with one JSON line per record, often `.jsonl` |
        | **hash** | a short string of letters and digits computed from a text: the same for the same text, a different one when anything changes |
        | **code version** | which code a run ran with, such as the hash of that code |
        """),
        L.watch_out("A line you type in later by hand holds what you remember. Let the code that trains write the line, right after the run, with the settings it really used."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To keep an experiment log:

    1. Before the run, collect the code version, all settings, also those you did not change, and the seed.
    2. After the run, note the result, such as the validation loss.
    3. Add the run as one JSON line to the end of the log, with `"a"`; never change a line that is there.
    4. To answer a question, read the lines back with `json.loads` and pick the runs you need.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two runs of one setting, with seeds 0 and 1, into a new log, and back out. The comments say which step each part is. It is not finished: run the cell as it is, then complete it where the comment says, and run it again. `.encode()` turns the text into bytes, which is what `sha256` takes, and `.hexdigest()` gives the hash back as text.
    """)
    return


@app.cell
def example_log():
    import hashlib
    import json

    _path = C.scratch_path()  # a new file in a temporary folder, not in your project
    _code = "lr = 0.001\nwarmup = 100\n"  # stands in for the code of the runs
    # 1. the code version: the first 8 characters of the hash of the code
    _version = hashlib.sha256(_code.encode()).hexdigest()[:8]
    for _seed, _loss in [(0, 2.3902), (1, 2.3924)]:
        # 1-2. the record of one run, with its result
        _run = {"code": _version, "settings": {"lr": 0.001, "warmup": 100}, "seed": _seed, "val_loss": _loss}
        # 3. one JSON line at the end of the log
        with open(_path, "w") as _f:  # complete: "a" adds to the end, "w" empties the file first
            _f.write(json.dumps(_run) + "\n")
    # 4. read every line back
    with open(_path) as _f:
        _log = [json.loads(_line) for _line in _f]
    print("runs in the log:", len(_log))
    for _r in _log:
        print(_r)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With `"w"`, the log holds one run, seed 1: each `open` emptied the file, so the run with seed 0 is gone. With `"a"`, it holds both, each on a line of its own. Change one character of `_code`, say `0.001` into `0.002`, and run the cell again: the code version is a completely different one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Write a run to the log

        Complete `log_run(path, settings, seed, val_loss, version)`: steps 1 to 3 of the recipe as a function. It makes the record of one run, a dictionary with the keys `"code"` (the version), `"settings"`, `"seed"` and `"val_loss"`, and adds it as one JSON line to the end of the file at `path`. Replace the two `...`. The check calls it twice on a new file and reads the lines back, and tries a file that already holds a run.
        """),
        L.task("Make `log_run(path, settings, seed, val_loss, version)` add one JSON line with the keys `\"code\"`, `\"settings\"`, `\"seed\"` and `\"val_loss\"` to the end of the file at `path`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_log():
    box_log = L.code_box("log-run", lines=8)
    box_log
    return (box_log,)


@app.cell(hide_code=True)
def check_log(box_log):
    result_log = L.check("log-run", box_log.value)
    result_log
    return (result_log,)


@app.cell(hide_code=True)
def hints_log():
    L.hints(
        "log-run",
        [
            "The record is a dictionary, as in the worked example, with `version` under the key `\"code\"`. Then open the file with `\"a\"` and write the record as one line.",
            "`run = {\"code\": version, \"settings\": ___, \"seed\": ___, \"val_loss\": ___}`, then `with open(path, ___) as f:` and under it `f.write(json.dumps(___) + \"\\n\")`.",
        ],
        why="```python\nimport json\n\n\ndef log_run(path, settings, seed, val_loss, version):\n    run = {\"code\": version, \"settings\": settings, \"seed\": seed, \"val_loss\": val_loss}\n    with open(path, \"a\") as f:\n        f.write(json.dumps(run) + \"\\n\")\n```\n\n`\"a\"` adds to the end, so every run that was there stays. `json.dumps` makes one line of the dictionary, and `\"\\n\"` ends that line, so the next run starts on a line of its own.",
    )
    return


@app.cell(hide_code=True)
def text_version():
    mo.md(r"""
    ### Your own code version

    The cell below computes the code version of your GPT project: the hash of all `.py` files in `work/projects/llm/` and `work/projects/nn/`, the code your GPT trains with. Each time a check saves new code there, or you change a file yourself, the version changes.
    """)
    return


@app.cell(hide_code=True)
def demo_version():
    _version, _files = C.project_version()
    if _version is None:
        _out = mo.md("*Your project has no files in `work/projects/llm/` yet. They come with the lessons of the GPT, such as [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en).*")
    else:
        _out = mo.md(f"The {_files} files of your GPT project have code version **`{_version}`** now. A run you log today gets this version; a run after a change of your code gets another one.")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_find():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Find runs back

        The course's log holds its own 16 runs of the small GPT, from [Change one thing at a time](/course/open/res-hypothesis-control?lang=en) and [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en), with all their settings. Complete `find_runs(log, wanted)`: step 4 of the recipe. It gives back, in the order of the log, every run whose settings have the value of `wanted` for every key of `wanted`. `wanted` holds only the settings you ask about, such as `{"warmup": 0}`. Replace the `...`. The check calls `find_runs(log, {"warmup": 0})`, which should give the 5 runs without warmup, and asks other questions too.
        """),
        L.task("Make `find_runs(log, wanted)` give back every run of `log` whose settings have the value of `wanted` for every key of `wanted`, in the order of the log."),
    ])
    return


@app.cell(hide_code=True)
def exercise_find():
    box_find = L.code_box("find-runs", lines=8)
    box_find
    return (box_find,)


@app.cell(hide_code=True)
def check_find(box_find):
    result_find = L.check("find-runs", box_find.value)
    result_find
    return (result_find,)


@app.cell(hide_code=True)
def hints_find():
    L.hints(
        "find-runs",
        [
            "For each run, start with `same = True`. Go through the keys of `wanted`; when the run's setting differs from `wanted[key]`, set `same = False`. Keep the run if `same` is still `True`.",
            "`for key in wanted:` with `if run[\"settings\"][key] != wanted[key]:` and `same = False` under it; after that loop, `if same:` and `found.append(run)`.",
        ],
        why="```python\ndef find_runs(log, wanted):\n    found = []\n    for run in log:\n        same = True\n        for key in wanted:\n            if run[\"settings\"][key] != wanted[key]:\n                same = False\n        if same:\n            found.append(run)\n    return found\n```\n\nA run fits when no key of `wanted` has another value in its settings. An empty `wanted` asks for nothing, so then every run fits.",
    )
    return


@app.cell(hide_code=True)
def text_demo_find():
    mo.md(r"""
    ### Ask the course's log

    Choose a question: your `find_runs` picks the runs out of the course's log. Until Exercise 2 is right, the course's `find_runs` does it.
    """)
    return


@app.cell(hide_code=True)
def question_cell():
    question = mo.ui.dropdown(
        options={"the control run": "control", "learning rate 0.003": "lr=0.003", "no warmup": "warmup=0", "4 layers": "layers=4"},
        value="no warmup",
        label="Which runs?",
    )
    question
    return (question,)


@app.cell(hide_code=True)
def demo_find(box_find, question, result_find):
    _find, _yours = C.find_runs_of(box_find.value if result_find.state == "correct" else None)
    _wanted = C.QUESTIONS[question.value]
    _runs = _find(C.LOG, _wanted)
    _who = "Your `find_runs`" if _yours else "The course's `find_runs`"
    if _runs:
        _mean = sum(_r["val_loss"] for _r in _runs) / len(_runs)
        _out = mo.md(f"{_who} finds {len(_runs)} runs with `{C.as_code(_wanted)}`, with a mean validation loss of {C.number(_mean, 3, L.lang)}:\n\n" + C.runs_table(_runs, L.lang))
    else:
        _out = mo.md(f"{_who} finds no runs with `{C.as_code(_wanted)}`.")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_settings():
    mo.md(r"""
    ### Exercise 3. Which settings go in?
    """)
    return


@app.cell(hide_code=True)
def choice_settings_cell():
    choice_settings = L.choice(
        "all-settings",
        question="You try a learning rate of 0.003 instead of 0.001, and change nothing else. Which settings go into the line of this run?",
        options={
            "changed": "Only `lr`, the one you changed",
            "right": "All of them, also those you did not change",
            "defaults": "Those that differ from the defaults in your code",
        },
        correct="right",
        why={
            "changed": "Then the line does not say what the other settings were. If you change one of them later, you can no longer tell which runs had which.",
            "defaults": "Defaults live in your code, and code changes: after a change, a line that leans on them means something else. See step 1 of the recipe.",
            "right": "A line with every setting stands on its own: it says what each setting was, also after the defaults in your code have changed, and you can find it back by any setting.",
        },
    )
    choice_settings
    return (choice_settings,)


@app.cell(hide_code=True)
def text_exercise_version():
    mo.md(r"""
    ### Exercise 4. Two runs, two numbers
    """)
    return


@app.cell(hide_code=True)
def explain_version():
    explain_version_cell = L.explain(
        "code-version",
        question="Two lines in your log have the same settings and the same seed. One ended at 2.3902, the other at 2.4317, and their code versions are `4f9a1c2e` and `b81d07a3`. What do you conclude, and how would you have read these two lines without a code version?",
        model_answer="**What it shows:** the code changed between the two runs. With the same settings and the same seed, the difference of 0.04 can come from that change, not only from chance; on the same computer, the same code with the same settings and seed usually ends at the same number again. The hash does not say what changed: for that you need the old code, which is why people keep their code in Git and log the commit. **Without a code version**, it would look as if the same run ended at two numbers. You might call 0.04 noise, or trust the wrong run, without knowing which code gave which number.",
        points=[
            "you say that the code changed between the two runs",
            "you say that the difference can come from that change, and not only from chance",
            "you say what you would have thought without the version: the same run with two results, noise or a mistake you cannot trace",
        ],
    )
    explain_version_cell
    return (explain_version_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The runs in the course's log come from [Change one thing at a time](/course/open/res-hypothesis-control?lang=en) and [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en); the file is read as in [Reading a text file](/course/open/py-files?lang=en). The next lesson, *Fair comparisons*, asks whether the baseline in your log got as much care as your own method.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_settings, explain_version_cell, result_find, result_log):
    L.wrap_up(result_log, result_find, choice_settings, explain_version_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
