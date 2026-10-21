# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-train-bpe. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-train-bpe", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en) and [Generate text with your GPT](/course/open/gpt-generate?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_end():
    quick_end = L.choice(
        "quick-end",
        question="Your tokenizer has 1,000 merge rules. Which id does end-of-text get?",
        options={
            "thousand": "1000: the first id after the 1,000 rules",
            "first": "256: the first id after the bytes",
            "right": "1256: after the bytes and the rules",
        },
        correct="right",
        why={
            "thousand": "The 256 bytes come first, with ids 0 to 255, and the rules take 256 to 1255. See [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en).",
            "first": "256 is the first merge rule. End-of-text needs an id no rule has: `256 + len(merges)`. See [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en).",
            "right": "`256 + len(merges)`: so your GPT of this lesson has a vocabulary of 1,257 ids.",
        },
    )
    quick_end
    return (quick_end,)


@app.cell(hide_code=True)
def quick_check_tau():
    quick_tau = L.choice(
        "quick-tau",
        question="Your GPT writes with temperature 0.8 instead of 1. What changes?",
        options={
            "right": "The likeliest ids get likelier still: safer text",
            "random": "Every id gets more equal chances: wilder text",
            "nothing": "Nothing: the temperature only matters during training",
        },
        correct="right",
        why={
            "random": "That is a temperature above 1. Below 1, the logits are divided by less than 1, so the largest grow apart from the rest. See [Generate text with your GPT](/course/open/gpt-generate?lang=en).",
            "nothing": "The temperature divides the logits before the softmax, every time the GPT writes a step. See [Generate text with your GPT](/course/open/gpt-generate?lang=en).",
            "right": "This lesson's demo writes its story with 0.8.",
        },
    )
    quick_tau
    return (quick_tau,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(f"""
        ## Why this matters

        Your GPT of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en) reads and writes one character at a time: its window of 128 characters is about 25 words. Your tokenizer of [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en) cuts text into pieces of several characters: on {C.grouped(5_000_000, L.lang)} characters of short stories, a token holds {C.number(C.CHARS_PER_TOKEN, 1, L.lang)} characters on average. Then the same window holds {C.number(C.CHARS_PER_TOKEN, 1, L.lang)} times as much story, and every step of training learns from {C.number(C.CHARS_PER_TOKEN, 1, L.lang)} times as much text.

        But your `train_pre_bpe` goes through every piece of the text in every round. The first 90% of the stories have {C.grouped(C.PIECES, L.lang)} pieces, so 1,000 rounds take it {C.NAIVE_MINUTES} minutes on the course's computer. This lesson makes your tokenizer fast enough, and then trains your GPT on its tokens.

        **An everyday example.** At the self-checkout, twelve cans of the same soup: you scan one and type × 12, instead of scanning all twelve.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_speed():
    guess_speed = L.guess(
        "speed",
        question=f"Of the {C.grouped(C.PIECES, L.lang)} pieces, many are the same: \" the\" alone occurs {C.grouped(C.TOP_PIECES[2][1], L.lang)} times. A BPE that goes through every different piece once, and counts each of its pairs as often as the piece occurs, learns the same 1,000 rules. How much faster do you think it is?",
        options={
            "two": "About 2 times: it still does 1,000 rounds",
            "ten": "About 10 times: most pieces are different",
            "right": "About 100 times: there are few different pieces",
        },
        reveal=f"**About {C.SPEEDUP} times: {C.FAST_SECONDS} seconds instead of {C.NAIVE_MINUTES} minutes**, on the course's computer. The stories have only {C.grouped(C.DISTINCT, L.lang)} different pieces, {C.SPEEDUP_PIECES} times fewer than all {C.grouped(C.PIECES, L.lang)}: children's stories use few words, again and again. Each round still counts pairs and merges, but in {C.grouped(C.DISTINCT, L.lang)} pieces.",
    )
    guess_speed
    return (guess_speed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Count every different piece once.** Cut the text into pieces, as in [Pre-tokenization](/course/open/tok-pre-tokenization?lang=en), and count how often each piece occurs, in a dict `counts_of`. A pair inside a piece that occurs 3 times then counts 3 times: add the piece's count, not 1. The pair counts are exactly those of your `train_pre_bpe`, and so are the merge rules, in the same order, from far fewer pieces.

        **Encode every different piece once.** The same for encoding: a piece goes through `encode_part` the first time you meet it, and its ids go into a dict `known`. Every next time, you take them from there. Remembering a result to use again is **caching**.

        **Your GPT on tokens.** The same GPT and Trainer as on characters, with 1,257 ids instead of the characters: only the embedding and the output layer get larger. Its loss is per token, and a token holds several characters, so to compare it with a GPT on characters, spread it over the characters: loss per character = loss per token × tokens ÷ characters.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **caching** | | remembering a result, so that the same question later needs no work | `known[piece]` |
        | **characters per token** | | how many characters of a text a token holds on average: the length of the text divided by its number of tokens | `len(text) / len(ids)` |
        """),
        L.watch_out("Add to a long list with `append`. `ids = ids + more` builds a whole new list every time: with 1.5 million ids, that takes about an hour instead of seconds."),
        mo.md(r"""
        The picture: the pieces of "the the the cat cow cup", the example of exercise 1. Left every piece, right every different piece once with its count. The pair `t h` is in "the" once and in "␣the" twice: 1 + 2 = 3, as on the left.
        """),
        mo.Html(C.pieces_picture({"title": "Every piece, or every different piece once", "all": "every piece", "once": "every different piece once",
                                  "count": "count", "pair": "pair t h: 1 + 2 = 3",
                                  "alt": "Left, the six pieces the, ␣the, ␣the, ␣cat, ␣cow and ␣cup. Right, the five different pieces with their counts: the 1, ␣the 2, ␣cat 1, ␣cow 1, ␣cup 1. The pair t h counts 1 + 2 = 3 times."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train your GPT on your own tokens:

    1. Count the pieces: every piece of every part between two end-of-texts, with how often it occurs.
    2. Train BPE on those pieces, each pair counted as often as its piece occurs: the same rules as your `train_pre_bpe`.
    3. Encode the whole text with every different piece encoded once, and end-of-text as id `256 + len(merges)`.
    4. Train your GPT on the ids, with a vocabulary of `257 + len(merges)`, and compare its loss per character.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 on the first 20,000 characters of the stories: how many pieces, how many different ones, and the most common. Then the check: the pair counts from every different piece once, each with its count, are those from every piece.
    """)
    return


@app.cell
def example_counts():
    import re

    _text = C.text()[:20000]
    _pieces = []
    for _part in _text.split("<|endoftext|>"):
        for _piece in re.findall(C.PATTERN, _part):
            _pieces.append(_piece)
    # 1. every different piece once, with how often it occurs
    _counts_of = {}
    for _piece in _pieces:
        if _piece in _counts_of:
            _counts_of[_piece] = _counts_of[_piece] + 1
        else:
            _counts_of[_piece] = 1
    print(len(_pieces), "pieces,", len(_counts_of), "different")
    # repr shows a piece in quotes, so that you see a space at its start
    for _piece in sorted(_counts_of, key=_counts_of.get, reverse=True)[:5]:
        print(repr(_piece), _counts_of[_piece])
    # the pairs of every piece, and the pairs of every different piece, counted as often as it occurs
    _every = {}
    for _piece in _pieces:
        _ids = list(_piece.encode("utf-8"))
        for _i in range(len(_ids) - 1):
            _pair = (_ids[_i], _ids[_i + 1])
            if _pair in _every:
                _every[_pair] = _every[_pair] + 1
            else:
                _every[_pair] = 1
    _once = {}
    for _piece in _counts_of:
        _ids = list(_piece.encode("utf-8"))
        for _i in range(len(_ids) - 1):
            _pair = (_ids[_i], _ids[_i + 1])
            if _pair in _once:
                _once[_pair] = _once[_pair] + _counts_of[_piece]
            else:
                _once[_pair] = _counts_of[_piece]
    print(_every == _once)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(f"""
    On 20,000 characters, {C.grouped(C.FIRST_PIECES['pieces'], L.lang)} pieces are only {C.grouped(C.FIRST_PIECES['different'], L.lang)} different ones, and the full stops, the commas and \" and\" are the most common. The two dicts of pair counts are the same: `True`. On all the stories, the gap is far larger: {C.grouped(C.PIECES, L.lang)} pieces, {C.grouped(C.DISTINCT, L.lang)} different. `sorted(..., key=_counts_of.get, reverse=True)` puts the pieces in order of their count, the largest first; you do not need it in the exercises.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bpe():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. BPE on every different piece once

        Complete `train_bpe_fast(text, n)`: steps 1 and 2 of the recipe. The rounds are those of your `train_pre_bpe`, with `words` and `weights` in place of the pieces: `words[k]` holds the ids of a different piece, and `weights[k]` how often it occurs. The check calls `train_bpe_fast("the the the cat cow cup", 2)`, and other texts, also one of 300,000 characters, which must be done within 2 seconds.
        """),
        L.task("Make `train_bpe_fast` count every different piece once, and every pair as often as its piece occurs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bpe():
    box_bpe = L.code_box("train-bpe-fast", lines=36)
    box_bpe
    return (box_bpe,)


@app.cell(hide_code=True)
def check_bpe(box_bpe):
    result_bpe = L.check("train-bpe-fast", box_bpe.value)
    result_bpe
    return (result_bpe,)


@app.cell(hide_code=True)
def hints_bpe():
    L.hints(
        "train-bpe-fast",
        [
            "Step 1 is the first loop of the worked example, inside a loop over the parts of `text.split(END)`.",
            "For every piece of `re.findall(PATTERN, part)`: one more in `counts_of`, or 1 for a piece you meet for the first time. In the rounds, a pair adds `weights[k]` to its count.",
        ],
        why="A pair inside a piece occurs as often as the piece does, so adding the piece's count gives the same count as going through every piece of the text. The dicts keep the order in which pieces and pairs first appear, so even a tie between two pairs ends as in your `train_pre_bpe`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_encode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Encode every different piece once

        Complete `encode_fast(text, merges)`: step 3 of the recipe. It gives the same ids as your `encode` of [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en), and it uses your `encode_part` from there. The check calls `encode_fast("hi hi<|endoftext|>hi", {(104, 105): 256})`, and other texts, also long ones with 300 rules, which must be done within 2 seconds.
        """),
        L.task("Make `encode_fast` encode every different piece once, and take its ids from `known` every time it occurs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_encode():
    box_encode = L.code_box("encode-fast", lines=18)
    box_encode
    return (box_encode,)


@app.cell(hide_code=True)
def check_encode(box_encode):
    result_encode = L.check("encode-fast", box_encode.value)
    result_encode
    return (result_encode,)


@app.cell(hide_code=True)
def hints_encode():
    L.hints(
        "encode-fast",
        [
            "Two parts: a piece you meet for the first time goes into `known`; then, every time, its ids go into `ids`.",
            "`if piece not in known:` with `encode_part(piece, merges)`; after the `if`, a loop over `known[piece]`.",
        ],
        why="Every different piece goes through all the merge rules once, and its ids are in `known` from then on. `append` adds one id to the end of the same list, so 1.5 million ids take a moment, not an hour.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_bpe, result_encode):
    result_bpe, result_encode  # the note changes when exercise 1 or 2 saves your module
    _p = C.plan()
    _d, _h, _layers, _block = _p["sizes"]
    if _p["device"] == "cpu":
        _where = "This computer has no GPU that fits the bigger GPT" + (f" ({_p['small']} has too little memory)" if _p["small"] else "") + ", so a smaller version runs on the CPU"
        _time = f" On the course's computer, a CPU with 4 cores, the run took about {round(C.REFERENCE['cpu'][0]['minutes'])} minutes, on one thread."
    else:
        _where = f"Your computer has a GPU, {_p['name']}, so your GPT trains on it"
        _time = ""
    mo.vstack([
        mo.md(f"""
        ### What your project can do now

        The button below trains your tokenizer, your `train_bpe_fast`, with {C.grouped(C.MERGES, L.lang)} rules on the first 90% of the stories, encodes all of them with your `encode_fast`, and trains your GPT on their tokens, with your `gpt_loaders` and your `Trainer`, as in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en). {_where}: a GPT of {C.grouped(_p['params'], L.lang)} parameters, with $d = {_d}$, {_h} heads, {_layers} blocks and windows of {_block} tokens, {_p['epochs']} epochs of mini-batches of {_p['batch']}. Then it writes a story that starts with "{C.START}".{_time} While it trains, the cell shows the loss after every epoch and the time left, and the rest of the lesson waits. The stop button of the cell ends the run early; the best parameters so far stay saved, in `work/projects/llm/gpt_bpe.pt`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your tokenizer and your GPT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_bpe, result_encode):
    result_bpe, result_encode  # run again when exercise 1 or 2 saves your module
    _labels = {"title": "Your run", "x": "epoch", "y": "loss per token", "lang": L.lang, "guess": "guessing evenly", "train": "training", "val": "validation"}

    def _progress(_p):
        if _p["phase"] == "tokenizer":
            mo.output.replace(mo.md("**Training your tokenizer** on the first 90% of the stories…"))
            return
        if _p["phase"] == "encode":
            mo.output.replace(mo.md(f"**Encoding the stories** with your `encode_fast` · your tokenizer took {C.number(_p['seconds_bpe'], 0, L.lang)} seconds"))
            return
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
    _ready = result_bpe.state == "correct" and result_encode.state == "correct"
    if _result is None and _ready and not train_button.value:
        _out = mo.md("*Press the button to start the run.*")
    elif _result is None:
        _out = mo.md("*Finish exercises 1 and 2 first, then press the button: the run uses your `train_bpe_fast` and your `encode_fast`.*")
    if _result is None and C.trained()["source"] == "course":
        _c = C.trained()
        _out = mo.vstack([_out, mo.md(f"""Until then, the course's own GPT on tokens: the same GPT and tokenizer as on the CPU plan, trained by the course on {C.grouped(_c['chars'], L.lang)} characters of stories from Cosmopedia, whose license lets the course share a model trained on them, and not on TinyStories. On these stories, harder than TinyStories, its validation loss is {C.number(_c['best'], 2, L.lang)} per token, {C.number(_c['best_per_char'], 3, L.lang)} per character. It wrote, with temperature 0.8:

> {_c['story'].replace(chr(10), chr(10) + '> ')}""")])
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*The run stops with an error. Do your modules in `work/projects/llm/` and `work/projects/nn/` still pass their checks? Delete a file you changed, and check its exercise again.*",
            "memory": "*Your GPU ran out of memory. Close other programs that use it, such as a game or another notebook, and press the button again.*",
            "stopped": "*You stopped the run during its first epoch, so nothing of it was saved. Press the button to start again.*",
        }[_result])
    else:
        _h = _result["history"]
        _plan = "gpu" if _result["device"] != "cpu" else "cpu"
        _chars = C.CHARS[_plan] if _plan in C.CHARS else C.CHARS["cpu"]
        _on = {"cpu": "the CPU", "mps": "the Apple GPU"}.get(_result["device"], _result["name"])
        _ratio = _result["val_chars"] / _result["val_tokens"]
        _row = "about as many steps" if _plan in C.CHARS else "the CPU plan"
        _steps = "about as many steps as your run on the CPU plan" if _plan in C.CHARS else "about as many steps as a run of this lesson on the CPU plan, not as your run on the GPU plan, with its bigger GPT"
        _limit = C.number(C.THRESHOLD[_plan], 2, L.lang)
        if _result["stopped"]:
            _verdict = f"You stopped the run after epoch {len(_h)} of {_result['epochs']}; the course's runs of this GPT, with all {_result['epochs']} epochs, ended below {_limit} per token."
        elif _result["best"] <= C.THRESHOLD[_plan]:
            _verdict = f"The course's runs of this GPT ended below {_limit} per token, and so did yours: your tokenizer and your GPT learned what they should."
        else:
            _verdict = f"The course's runs of this GPT ended below {_limit} per token, and yours did not. Check exercises 1 and 2 again, and those of the lessons your GPT and Trainer come from."
        _saved = "" if not _result.get("kept") else f" Your earlier run got a lower validation loss, so `work/projects/llm/gpt_bpe.pt` keeps the parameters of that run."
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + "."
        _story = "" if not _result.get("story") else "\n\nThe story your GPT wrote, with temperature 0.8:\n\n> " + _result["story"].replace("\n", "\n> ")
        _out = mo.vstack([
            mo.Html(C.history_picture(_h, {**_labels, "alt": f"The training and validation loss of your run after every epoch: the validation loss is lowest after epoch {_result['best_epoch']}, at {C.number(_result['best'], 2, L.lang)} per token."}, _result["epochs"])),
            mo.md(f"""Your GPT of {C.grouped(_result['params'], L.lang)} parameters trained for {len(_h)} {'epochs' if len(_h) > 1 else 'epoch'} on {_on}, in {C.number(_result['minutes'], 1, L.lang)} minutes, tokenizer and encoding included. The stories became {C.grouped(_result['tokens'], L.lang)} tokens, {C.number(_ratio, 2, L.lang)} characters per token on the part it validates on. Its validation loss went down to **{C.number(_result['best'], 2, L.lang)} per token** after epoch {_result['best_epoch']}: **{C.number(_result['best_per_char'], 3, L.lang)} per character**.{_saved} {_verdict}

| | loss per character |
|---|---|
| your GPT on tokens | {C.number(_result['best_per_char'], 3, L.lang)} |
| the course's GPT on characters, the same stories, {_row} | {C.number(_chars['best'], 3, L.lang)} |

The GPT on characters is the course's run of the CPU plan of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en) on these stories, for {_chars['epochs']} epochs: {_steps}.{_who}{_story}

The course's GPT on characters wrote, from the same start:

> {_chars['story'].replace(chr(10), chr(10) + '> ')}"""),
        ])
    trained_run = _result
    _out
    return (trained_run,)


@app.cell(hide_code=True)
def text_replay():
    mo.md(f"""
    ### The run, epoch by epoch

    After every epoch, the run lets your GPT write {C.SNAP} tokens from *{C.START}*, drawn with the same seed each time. Here you go through those moments: the losses after every epoch, with a dashed line at the epoch you choose, and what the GPT wrote then.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained_run):
    _run = trained_run if isinstance(trained_run, dict) else C.last_run()
    _mine = _run is not None and len(C.moments(_run)) >= 2   # the slider needs two moments
    if _mine:
        _note = ""
    elif _run is not None and "snapshots" in _run:
        _note = "*Your run kept what your GPT wrote after one epoch at most: too little to go through, so the course's run of the CPU plan stands in here. Train again without stopping to go through your own.*"
    elif _run is not None:
        _note = "*Your saved run was made before the lesson kept what the GPT wrote, so the course's run of the CPU plan stands in here. Train again to go through your own.*"
    else:
        _note = "*Until you train your own GPT, you go through the course's run of the CPU plan here.*"
    if not _mine:
        _run = C.course_run(C.REFERENCE["cpu"][0])
    _labels = {"title": "Your run" if _mine else "The course's run", "x": "epoch", "y": "loss per token", "lang": L.lang, "guess": "guessing evenly",
               "train": "training", "val": "validation", "after": "After epoch {epoch}", "said_val": "validation loss",
               "alt": "The training and validation loss after every epoch, with a dashed line at the chosen epoch."}
    _k = C.moments(_run)
    _block = L.explore("replay", "epoch", lambda _v: C.replay_frame(_run, int(_v), _labels), start=_k[-1], values=_k, role="",
                       task="After which epoch does the GPT stop making up words that do not exist?")
    mo.vstack([mo.md(_note), _block]) if _note else _block
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(f"""
    In the course's run (your own GPT may write other stories), the GPT writes sentences with names and with what people say from the first epoch on, but it also makes up words: *adventally* after epoch 1, *fationhone* after epoch 3, *hiddench* after epoch 7. A token is often a piece of a word, so the GPT can glue pieces together that do not belong together. Every moment is one draw of {C.SNAP} tokens: after epoch 4, it happens to make up none. From epoch 8 on, every word exists: the *legle* at the end of epoch 8 is where the {C.SNAP} tokens run out, and *legless* starts with the same three tokens. Meanwhile, the validation loss falls only a little more: from {C.number(C.REFERENCE['cpu'][0]['history'][7][2], 2, L.lang)} after epoch 8 to {C.number(C.REFERENCE['cpu'][0]['history'][9][2], 2, L.lang)} after epoch 10. Whether the story makes sense is another matter: Tim goes to the park with his mom and his mom.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_char():
    mo.md(r"""
    ### Exercise 3. Per token or per character
    """)
    return


@app.cell(hide_code=True)
def choice_char_cell():
    _r = C.REFERENCE["cpu"][0]
    choice_char = L.choice(
        "per-char",
        question=f"In the course's run of the demo above, the GPT on tokens ends at a validation loss of {C.number(_r['best'], 2, L.lang)} per token; its tokens hold {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} characters on average. The course's GPT on characters, on the same stories, ends at {C.number(C.CHARS['cpu']['best'], 2, L.lang)} per character. Which predicts the stories better?",
        options={
            "chars": f"The GPT on characters: {C.number(C.CHARS['cpu']['best'], 2, L.lang)} is less than {C.number(_r['best'], 2, L.lang)}",
            "right": f"The GPT on tokens: {C.number(_r['best'], 2, L.lang)} per token is {C.number(_r['best_per_char'], 2, L.lang)} per character",
            "same": "You cannot compare them: they have different vocabularies",
        },
        correct="right",
        why={
            "chars": f"A token holds {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} characters, so one loss per token is spread over that many characters: {C.number(_r['best'], 2, L.lang)} ÷ {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} = {C.number(_r['best_per_char'], 2, L.lang)} per character.",
            "same": "Per character, you can: both give a probability to the same stories, so the loss of the whole text compares, spread over its characters. Spread the loss per token over the characters a token holds.",
            "right": f"{C.number(_r['best'], 2, L.lang)} ÷ {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} = {C.number(_r['best_per_char'], 2, L.lang)} per character, against {C.number(C.CHARS['cpu']['best'], 2, L.lang)}. With about as many steps, the GPT on tokens read about {C.number(_r['val_chars'] / _r['val_tokens'], 1, L.lang)} times as much story.",
        },
    )
    choice_char
    return (choice_char,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Your tokenizer of [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en), [Pre-tokenization](/course/open/tok-pre-tokenization?lang=en) and [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en), now fast enough for megabytes; your GPT and run of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en), now on its tokens; and the temperature of [Generate text with your GPT](/course/open/gpt-generate?lang=en) for its story. Your project `llm` now has `fast_bpe.py` and `fast_encode.py`, and after the run a GPT on tokens in `gpt_bpe.pt`, with its merge rules in `gpt_bpe.json`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_bpe, result_encode, choice_char):
    L.wrap_up(result_bpe, result_encode, choice_char)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
