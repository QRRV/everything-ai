# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-train-chars. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-train-chars", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Assemble your GPT](/course/open/gpt-assemble?lang=en) and [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_window():
    quick_window = L.choice(
        "quick-window",
        question="Your GPT gets a window of 256 ids, and as targets the same window one id further. From how many predictions does it learn?",
        options={
            "one": "1: the id after the window",
            "right": "256: one at every place of the window",
            "fewer": "255: the last place has no target in the window",
        },
        correct="right",
        why={
            "one": "That was your MLP of [A neural language model that looks further back](/course/open/lm-mlp-context?lang=en): one context, one target. Your GPT guesses at every place. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "fewer": "The targets are the window one id further, so the last place has a target too: the id right after the window. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "right": "256 predictions, each with a target of its own: today a mini-batch of 64 windows gives 16,384 of them.",
        },
    )
    quick_window
    return (quick_window,)


@app.cell(hide_code=True)
def quick_check_copy():
    quick_copy = L.choice(
        "quick-copy",
        question="A mini-batch `X` is on the CPU, and your model on the GPU. You write `X.to(\"cuda\")` on a line of its own, and then `model(X)`. What happens?",
        options={
            "moved": "It works: `X` is on the GPU now",
            "right": "An error: `X` itself is still on the CPU",
            "slow": "It works, but slowly: PyTorch copies `X` to the GPU at every call",
        },
        correct="right",
        why={
            "moved": "`.to(device)` gives back a moved copy, and leaves `X` where it was. Keep the copy: `X = X.to(device)`. See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "slow": "PyTorch moves nothing by itself, also not slowly: the model and `X` are on two devices, and that is an error. See [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en).",
            "right": "Today your `gpt_loaders` keeps the moved copies, in a `TensorDataset` on the device.",
        },
    )
    quick_copy
    return (quick_copy,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your GPT has all its parts, and knows nothing yet: its loss is about 4.17, as if it guessed evenly among 65 characters. In this project step it learns from Tiny Shakespeare, about a million characters of plays. On a GPU with enough memory, that is your GPT of 10.8 million parameters from [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en); on a computer without one, a smaller GPT of about a million parameters, on the CPU. Afterwards your project holds a trained GPT, and the next lesson lets it write.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_loss():
    guess_loss = L.guess(
        "final-loss",
        question="Your MLP of [A neural language model that looks further back](/course/open/lm-mlp-context?lang=en) got to a loss of 2.20 after its 1,000 steps, on text it had not trained on. Where do you think the validation loss of your GPT ends after this run?",
        options={
            "mlp": "About 2.1: a little below the MLP, as it is the same book",
            "zero": "Close to 0: by then it knows the book by heart",
            "right": "Clearly below 2: it looks back much further than the MLP",
        },
        reveal="**Clearly below 2.** In the course's runs of the smaller GPT on a CPU, with 3 seeds, the validation loss ended at 1.66 to 1.67; the GPT of 10.8 million parameters went to 1.52 in a run of the course. Your MLP looked back 3 characters; your GPT looks back 128 or 256, and so learns words, names and the shape of a play. Close to 0 it never gets: the text it validates on is new to it, and what comes next in a play is never certain.",
    )
    guess_loss
    return (guess_loss,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **run** is one training from start to end, with fixed settings and a fixed seed. A run of your GPT has five steps, and you wrote almost every piece of it already:

        - **the tokens:** the characters of the book, sorted, so character $i$ has id $i$: 65 of them;
        - **the split:** the first 90% of the ids to train on, the last 10% to validate on. Not shuffled before the split, unlike in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en): windows next to each other are one text, so the windows to validate on are a part of the book the GPT never reads, not windows in between ones it trains on;
        - **the mini-batches:** windows of `block` ids from each part, with your `gpt_examples`, as tensors on the device. All the windows of the book and their targets are less than 20 MB of ids, so they go to the device at once, and no mini-batch has to move during training;
        - **the model:** your GPT on the device, and your `Trainer` of [A reusable training loop](/course/open/nn-trainer?lang=en): AdamW, a warmup and cosine decay, clipping, and the parameters with the lowest validation loss saved;
        - **the run:** every epoch, windows of the whole book, in a new order; after every epoch the loss on the windows to validate on.

        In every epoch after the first, the demo starts the windows at another place of the book, a random number of characters later. With the same windows in every epoch, the bigger GPT learned them by heart: in a run of the course its validation loss went up again after 8 epochs. The same text, cut at other places, gives it new examples in every epoch.

        How much work that is, you know from [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en): training your GPT of 10.8 million parameters on one token of a window of 256 costs about 68 million FLOPs, attention included. A mini-batch of 64 windows is 16,384 tokens, so about $10^{12}$ FLOPs per step, and 20 epochs of 62 steps are about $1.4 \cdot 10^{15}$.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **run** | one training from start to end, with fixed settings and a fixed seed |
        """),
        L.watch_out("`.to(device)` moves the parameters of your GPT, not the tensors its `forward` makes along the way: the causal mask in your attention, made with `torch.ones(n, n)`, and the places in your GPT, `torch.arange(n)`. Those are made on the CPU, and on a GPU PyTorch stops with an error. The demo below runs your GPT inside `with torch.device(device):`, which makes new tensors on the device. In code of your own, you can write `torch.arange(n, device=ids.device)`."),
        mo.md(r"""
        The picture: the training loss (solid) and the validation loss (dashed) after every epoch, in one of the course's runs of the smaller GPT on the CPU, with seed 0. Both drop fast from 4.17, the loss of guessing evenly, and pass the 2.20 of your MLP in the second epoch; then they go down slower, as the learning rate decays. In the first two epochs the training loss is the higher one: it is the mean over the steps of an epoch, while the GPT was still learning fast. From the third epoch on, the validation loss stays above it: text the GPT never saw is harder than text it learns from. The runs with seeds 1 and 2 ended within 0.02 of this one.
        """),
        mo.Html(C.history_picture(C.reference("cpu"), {"title": "A run of the smaller GPT on the CPU", "x": "epoch", "y": "loss", "lang": "en",
                                                                    "guess": "guessing evenly", "mlp": "your MLP", "train": "training", "val": "validation", "alt": "The training and validation loss after each of the 10 epochs of the course's run: the validation loss falls from 2.38 after epoch 1 to 1.66 after epoch 10, far below the dashed lines of guessing evenly at 4.17 and of the MLP at 2.20; the training loss ends at 1.47."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A run of a GPT on the characters of a text:

    1. The characters of the text, sorted, as tokens; the text as ids.
    2. The first 90% of the ids to train on, the last 10% to validate on.
    3. Windows of `block` ids from each part, as tensors on the device, in mini-batches: shuffled to train on, in order to validate on.
    4. A GPT for these characters on the device, and a `Trainer` with the settings of the run and a path for the best parameters.
    5. Train with `fit()`, and watch both losses after every epoch: the checkpoint keeps the parameters with the lowest validation loss.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 and 2 for the first lines of Tiny Shakespeare, with windows of 4: how many ids and windows each part gets. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_split():
    _text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n"
    # 1. the characters, sorted, and the text as ids
    _chars = sorted(set(_text))
    _index = {}
    for _i in range(len(_chars)):
        _index[_chars[_i]] = _i
    _ids = []
    for _ch in _text:
        _ids.append(_index[_ch])
    print(len(_chars), "characters,", len(_ids), "ids")
    # 2. the first 90% to train on, the last 10% to validate on
    _cut = len(_ids)  # complete: int(0.9 * len(_ids))
    _train, _val = _ids[:_cut], _ids[_cut:]
    # windows of 4 side by side, as long as one more id follows: as your gpt_examples makes them
    print("to train on:", len(_train), "ids,", len(range(0, len(_train) - 4, 4)), "windows")
    print("to validate on:", len(_val), "ids,", len(range(0, len(_val) - 4, 4)), "windows")
    print(_text[_cut:])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The lines have 30 different characters, and 81 ids. As it is, all 81 ids go to training, 20 windows, and nothing is left to validate on. Completed, 72 ids give 17 windows to train on, and the last 9, `, speak.` and a new line, give 2 windows to validate on: text the GPT never trains on. For the whole book, that is about a million ids to train on, and a hundred thousand to validate on.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which parameters are saved?
    """)
    return


@app.cell(hide_code=True)
def choice_best_cell():
    choice_best = L.choice(
        "best-epoch",
        question="A run lasts 30 epochs. The training loss goes down every epoch; the validation loss is lowest after epoch 18, and goes up a little after that. Which parameters are in the checkpoint at the end?",
        options={
            "last": "Those after epoch 30: the last ones",
            "right": "Those after epoch 18: the lowest validation loss",
            "train": "Those after epoch 30: their training loss is the lowest",
        },
        correct="right",
        why={
            "last": "Your Trainer saves only when the validation loss is the lowest so far: the last time was after epoch 18.",
            "train": "The training loss is measured on the windows the GPT learns from: low there can mean it learned those by heart. The checkpoint goes by the validation loss.",
            "right": "After epoch 18 your GPT learned the windows it trains on by heart, little by little, and got worse on text it never saw: overfitting. The checkpoint keeps the best.",
        },
    )
    choice_best
    return (choice_best,)


@app.cell(hide_code=True)
def text_exercise_loaders():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The mini-batches

        Complete `gpt_loaders(ids, block, batch_size, device)`: steps 2 and 3 of the recipe, for any list of ids, so that a later project step can use it for the tokens of your own tokenizer too. Its first lines import your `gpt_examples` from your project. The check calls `gpt_loaders(list(range(50)), 4, 2, "cpu")`, whose mini-batches to validate on should be one window, `[45, 46, 47, 48]` with targets `[46, 47, 48, 49]`, and then tries the order of the mini-batches and PyTorch's `"meta"` device, which stands in for a GPU, as in [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en). When it is right, the check saves it in your project, as `work/projects/llm/gpt_loaders.py`.
        """),
        L.task("Make `gpt_loaders` split the ids, and give mini-batches of windows on the device: shuffled to train on, in order to validate on."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loaders():
    box_loaders = L.code_box("gpt-loaders", lines=16)
    box_loaders
    return (box_loaders,)


@app.cell(hide_code=True)
def check_loaders(box_loaders):
    result_loaders = L.check("gpt-loaders", box_loaders.value)
    result_loaders
    return (result_loaders,)


@app.cell(hide_code=True)
def hints_loaders():
    L.hints(
        "gpt-loaders",
        [
            "`cut` as in step 2 of the completed worked example. Then your `gpt_examples` once for each part: the ids before `cut`, and the ids from `cut` on.",
            "The windows and the targets of one part: two tensors on `device`, together in a `TensorDataset`, as in [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en). A `DataLoader` of `batch_size` for each part; only one of the two is shuffled.",
        ],
        why="The windows to validate on must be text the GPT never trains on: the end of the book. Shuffling matters only for training, where the order of the mini-batches changes the steps; the validation loss is the mean over all windows, in any order.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run(result_loaders):
    result_loaders  # the note changes when exercise 2 saves your gpt_loaders
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The run

        Complete `gpt_run(text, sizes, batch_size, epochs, device, path)`: step 4 of the recipe, a `Trainer` that is ready to train your GPT on the characters of `text`. Step 1, the characters and the ids, is given; its first lines import your `GPT`, your `gpt_loaders` and your `Trainer` from your project. The check calls `gpt_run(text, (16, 2, 1, 8), 4, 3, "cpu", path)` for a short text, with `path` a file in a temporary folder, and looks at the logits of 3 ids and at what the Trainer got; then it trains for 2 epochs, and tries the `"meta"` device. When it is right, the check saves it in your project, as `work/projects/llm/gpt_run.py`.
        """),
        L.course_versions(),
        L.task("Make `gpt_run` give a Trainer with your mini-batches and your GPT on the device, ready to train."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("gpt-run", lines=21)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("gpt-run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "gpt-run",
        [
            "Your `gpt_loaders` with `ids`, `block`, `batch_size` and `device`; then your `GPT` with the number of different characters and the four sizes, moved to `device`.",
            "`Trainer(model, train_loader, val_loader, epochs, ...)`, with the settings of the comment as keyword arguments, as in [A reusable training loop](/course/open/nn-trainer?lang=en): `lr`, `warmup`, `max_norm` and `path`.",
        ],
        why="The vocabulary is the number of different characters, `len(chars)`, not the length of the text. And all the tensors of one calculation must be on one device: the mini-batches and the parameters.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _p = C.plan()
    _d, _h, _layers, _block = _p["sizes"]
    if _p["device"] == "cpu":
        _where = "This computer has no GPU that fits the bigger GPT" + (f" ({_p['small']} has too little memory)" if _p["small"] else "") + ", so a smaller version runs on the CPU"
        _time = f" On the course's computer, a CPU with 4 cores, the run took about {round(C.REFERENCE['cpu'][0]['minutes'])} minutes."
    else:
        _where = f"Your computer has a GPU, {_p['name']}, so your GPT trains on it"
        _time = ""
    mo.md(f"""
    ### What your project can do now

    The button below starts a run on Tiny Shakespeare with your `gpt_run`, and so with your `gpt_loaders`, your `GPT` and your `Trainer`. {_where}: a GPT of {C.grouped(_p['params'], L.lang)} parameters, with $d = {_d}$, {_h} heads, {_layers} blocks and windows of {_block} characters, {_p['epochs']} epochs of mini-batches of {_p['batch']}.{_time} While it trains, the cell shows the loss after every epoch and the time left, and the rest of the lesson waits. The stop button of the cell ends the run early; the best parameters so far stay saved, in `work/projects/llm/gpt_chars.pt`.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your GPT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_loaders, result_run):
    result_loaders, result_run  # run again when exercise 2 or 3 saves your module
    _labels = {"title": "Your run", "x": "epoch", "y": "loss", "lang": L.lang, "guess": "guessing evenly", "mlp": "your MLP", "train": "training", "val": "validation"}

    def _progress(_p):
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · about {_m} {'minutes' if _m > 1 else 'minute'} left"
        _parts = [mo.md(f"**Epoch {_p['epoch']} of {_p['epochs']}** · step {_p['step']} of {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.history_picture(_p["history"], {**_labels, "alt": "The training and validation loss of your run so far, after every epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    if train_button.value:
        _result = C.train(_progress)
    else:
        _result = C.last_run()
    _ready = result_loaders.state == "correct" and result_run.state == "correct"
    if _result is None and _ready and not train_button.value:
        _out = mo.md("*Press the button to start the run.*")
    elif _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first, then press the button: the run uses your `gpt_loaders` and your `gpt_run`.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*The run stops with an error. Do your modules in `work/projects/llm/` and `work/projects/nn/` still pass their checks? Delete a file you changed, and check its exercise again.*",
            "memory": "*Your GPU ran out of memory. Close other programs that use it, such as a game or another notebook, and press the button again.*",
            "stopped": "*You stopped the run during its first epoch, so nothing of it was saved. Press the button to start again.*",
        }[_result])
    else:
        _h = _result["history"]
        _plan = "gpu" if _result["device"] != "cpu" else "cpu"
        _limit = C.number(C.THRESHOLD[_plan], 2, L.lang)
        if _result["stopped"]:
            _verdict = f"You stopped the run after epoch {len(_h)} of {_result['epochs']}; the course's runs of this GPT, with all {_result['epochs']} epochs, ended below {_limit}. Press the button for a whole run."
        elif _result["best"] <= C.THRESHOLD[_plan]:
            _verdict = f"The course's runs of this GPT ended below {_limit}, and so did yours: your GPT learned what it should."
        else:
            _verdict = f"The course's runs of this GPT ended below {_limit}, and yours did not. Check exercises 2 and 3 again, and those of the lessons in the note at exercise 3."
        if _result.get("kept"):
            _saved = f" Your earlier run got a lower validation loss, {C.number(_result['kept']['best'], 3, L.lang)} against {C.number(_result['best'], 3, L.lang)}, so `work/projects/llm/gpt_chars.pt` keeps the parameters of that run."
        else:
            _saved = " Those parameters are in `work/projects/llm/gpt_chars.pt` now."
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + ", as the note at exercise 3 says."
        _on = {"cpu": "the CPU", "mps": "the Apple GPU"}.get(_result["device"], _result["name"])
        _rows = []
        for _g in _result.get("guesses") or []:
            _top = ", ".join(f"`{C.shown_char(_c)}` {C.number(100 * _pr, 0, L.lang)}%" for _c, _pr in _g["top"])
            _rows.append(f"| …{C.shown_text(_g['read'])} | `{C.shown_char(_g['next'])}` | {_top} |")
        _table = "" if not _rows else ("\n\nWhat your GPT expects next, at three places near the end of the book, in the text it validates on and never trained on (␣ is a space, ↵ a new line):\n\n"
                                       "| it read | next | your GPT's three guesses |\n|---|---|---|\n" + "\n".join(_rows))
        _out = mo.vstack([
            mo.Html(C.history_picture(_h, {**_labels, "alt": f"The training and validation loss of your run after every epoch: the validation loss is lowest after epoch {_result['best_epoch']}, at {C.number(_result['best'], 2, L.lang)}."}, _result["epochs"])),
            mo.md(f"""Your GPT of {C.grouped(_result['params'], L.lang)} parameters trained for {len(_h)} {'epochs' if len(_h) > 1 else 'epoch'} on {_on}, in {C.number(_result['minutes'], 1, L.lang)} minutes: {C.grouped(_result['tokens'], L.lang)} tokens, about ${C.sci(_result['flops'], L.lang)}$ FLOPs. Its validation loss went from {C.number(_h[0]['val_loss'], 2, L.lang)} after the first epoch to **{C.number(_result['best'], 2, L.lang)}** after epoch {_result['best_epoch']}.{_saved} Guessing evenly gives 4.17, and your MLP got 2.20. {_verdict}{_who}{_table}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en), your Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en), the device of [Why AI uses GPUs, and the same code on yours](/course/open/torch-gpu?lang=en) and the FLOPs of [Counting parameters and FLOPs per token](/course/open/gpt-count?lang=en), together in one run; the checkpoint keeps the parameters of the epoch with the lowest validation loss, against the overfitting of [Overfitting and underfitting](/course/open/ml-overfitting?lang=en). Your project `llm` now has `gpt_loaders.py` and `gpt_run.py`, and after the run a trained GPT in `gpt_chars.pt`. Next: *Generate text with your GPT*, where your sampling loop of [Sampling: let the model write](/course/open/lm-sampling?lang=en) writes with it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_best, result_loaders, result_run):
    L.wrap_up(choice_best, result_loaders, result_run)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
