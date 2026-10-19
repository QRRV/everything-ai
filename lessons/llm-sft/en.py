# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-sft. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-sft", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Chat templates](/course/open/llm-chat-template?lang=en) and [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_turn():
    quick_turn = L.choice(
        "quick-turn",
        question="A chat model writes its answer, one token at a time. How does the loop that lets it write know that the answer is done?",
        options={
            "right": "The model writes its end-of-turn token",
            "length": "After a fixed number of tokens",
            "dot": "At the first full stop",
        },
        correct="right",
        why={
            "length": "A fixed number is only a limit, so that a loop always ends. A chat model ends its own turn: the loop stops at its end-of-turn token. See [Chat templates](/course/open/llm-chat-template?lang=en).",
            "dot": "An answer can have many sentences. The model ends its own turn with its end-of-turn token. See [Chat templates](/course/open/llm-chat-template?lang=en).",
            "right": "The model learned to write that token at the end of every answer. Today you see where it learns that.",
        },
    )
    quick_turn
    return (quick_turn,)


@app.cell(hide_code=True)
def quick_check_lr():
    quick_lr = L.choice(
        "quick-lr",
        question="You train on with every parameter of a network that was already trained. Which learning rate do you take?",
        options={
            "bigger": "A bigger one than for the first training",
            "same": "The same one as for the first training",
            "right": "A smaller one than for the first training",
        },
        correct="right",
        why={
            "bigger": "With a big step, the trained weights move too far and lose what they learned. See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
            "same": "The trained weights are already good; the learning rate of the first training moves them too far. See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
            "right": "Small steps keep what the network learned. Today you fine-tune your language model the same way.",
        },
    )
    quick_lr
    return (quick_lr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your model of the capstone continues text. Give it *Write a story with the word: kite.* and it writes on: a story perhaps, with a kite or without one, or another instruction. A chat model is a base model that was trained further, first on examples of an instruction with a good answer. That training is **supervised fine-tuning**, SFT. InstructGPT, a GPT-3 that OpenAI trained to follow instructions (Ouyang and others, 2022), starts with it, and the open recipe of nanochat in [Reading an open training recipe](/course/open/llm-open-recipes?lang=en) has it too.

    **An everyday example.** A new cook has read a thousand recipes: read out the start of one, and they can finish it. On their first days in a restaurant, they watch the orders come in, each with the dish that was made for it. After a few hundred, they cook what is ordered, and stop when the plate is full.
    """)
    return


@app.cell(hide_code=True)
def predict_after():
    _m = C.MEASURED["yours"]
    guess_after = L.guess(
        "after",
        question=f"The course gave its model of the capstone's demo run {_m['before']['total']} instructions, such as *Write a story with the words: kite and park.* Its stories used every requested word for {_m['before']['used']} of them. Then {C.number(C.STEPS, 0, L.lang)} steps of SFT on examples made from other stories, about {round(_m['seconds'] / 60)} minutes on one thread of the course's CPU. For how many of the same instructions now?",
        options={
            "same": "About as many: a few minutes of training change little",
            "right": "Several times as many, but far from all",
            "all": "Nearly all of them",
        },
        reveal=f"**{_m['after']['used']} of {_m['after']['total']}.** The same model, the same instructions: a few minutes of examples taught it to answer an instruction with a story that uses the words. Its stories are often still odd: a model of {C.number(_m['params'], 0, L.lang)} parameters learned only so much in {round(C.P.MEASURED['run']['seconds'] / 60)} minutes of pretraining.",
    )
    guess_after
    return (guess_after,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Supervised fine-tuning.** Write every example as one text: an instruction, and a good answer to it, in a fixed template. Then train the pretrained model on those texts as in pretraining: predict every next token, with the same loss and the same step. Only the data differ, and as in [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en), the steps are fewer and the learning rate lower. Training on instructions is also called **instruction tuning**.

    **The template.** Your tokenizer has one special token: the end-of-text. So the course's template is plain text: the instruction, and on a line of its own `Story:`, the generation prompt. The answer starts on the next line and ends with the end-of-text, which here is the end-of-turn token. The chat template of a real model, as in [Chat templates](/course/open/llm-chat-template?lang=en), does the same with special tokens around every message.

    **Loss masking.** The instruction is given: the model does not have to learn to write it. So every token of the instruction gets the label **-100**, a number that is no token. `F.cross_entropy` skips every target of -100, and takes the mean over the others: the loss, and so every gradient, comes from the answer and its end-of-text only.
    """)
    return


@app.cell(hide_code=True)
def picture_example():
    mo.vstack([
        mo.md(f"The picture: the example *{C.PICTURE[1]}* for the word *{C.PICTURE[0][0]}*, in the tokens of the model that this lesson fine-tunes, with the label of every token under it. ↵ is a new line, ␣ a space."),
        C.example_picture({"title": "An example, and the label of every token", "end": "end",
                           "alt": "An instruction, an answer and an end-of-text cut into tokens. Under the tokens of the instruction the label is -100; under the answer and the end-of-text, their own ids."}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_batch():
    mo.md(r"""
    **Shift and pad.** As in pretraining, the inputs are the ids without the last one, and the targets the labels without the first: at every place, the model predicts the next token. The examples differ in length, and a tensor is a rectangle, so every row of a batch is **padded** to the longest: the inputs with the end-of-text, though any id would do, and the targets with -100, so that the padding counts for nothing. Padding goes at the end: with the causal mask, a token only looks at the tokens before it, so no real token ever sees the padding.

    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **supervised fine-tuning**, SFT | training a pretrained model on examples of an instruction with a good answer | |
    | **instruction tuning** | another name for SFT on instructions | |
    | **loss masking** | giving the targets that should not count the label -100: the instruction, and the padding | `[-100] * len(prompt_ids)` |
    | -100 | the label that `F.cross_entropy` skips: its `ignore_index` | `F.cross_entropy(logits, y)` |
    | **padding** | filling the shorter rows of a batch up to the length of the longest | `[end_id] * pad` |

    ### Recipe

    To fine-tune a base model on instructions:

    1. Write every example in the template: the ids of the instruction, of the answer, and the end-of-text.
    2. Give every id a label: -100 for the instruction, the ids themselves for the answer and the end.
    3. Make batches: the inputs without the last id, the targets without the first label, padded at the end with the end-of-text and -100.
    4. Train with the step of pretraining, at a lower learning rate, for a few hundred to a few thousand steps.
    5. Test on instructions the model never saw, with a check that a program can do.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An instruction of three ids, `[5, 9, 2]`, and an answer of two, `[7, 3]`; the end-of-text is 0. The ids are made up, and so are the logits of the model at the five places.
    """)
    return


@app.cell
def example_mask():
    _prompt, _answer, _end = [5, 9, 2], [7, 3], 0
    _ids = _prompt + _answer + [_end]
    _labels = [-100] * len(_prompt) + _answer + [_end]
    _x, _y = _ids[:-1], _labels[1:]
    print(_x)
    print(_y)
    torch.manual_seed(0)
    _logits = torch.randn(len(_x), 10)   # 10 tokens in the vocabulary
    print(round(F.cross_entropy(_logits, torch.tensor(_y)).item(), 4))
    print(round(F.cross_entropy(_logits[2:], torch.tensor(_y[2:])).item(), 4))   # only the last three places
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At place 2 the model has seen 5, 9 and 2, and must predict 7, the first token of the answer; at place 4 it must predict the end. The first two places have the label -100, so the loss is the mean of the last three only: the same number as when you leave the first two out.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_example():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. An example and its labels

        Complete `sft_example(prompt_ids, answer_ids, end_id)`: steps 1 and 2. Give back the ids of the whole example and a label for every id. The check calls `sft_example([31, 8, 14, 2], [50, 51, 52], 99)`, and others.
        """),
        L.task("Make `sft_example` give the ids and the labels of an example."),
    ])
    return


@app.cell(hide_code=True)
def exercise_example():
    box_example = L.code_box("sft-example", lines=7)
    box_example
    return (box_example,)


@app.cell(hide_code=True)
def check_example(box_example):
    result_example = L.check("sft-example", box_example.value)
    result_example
    return (result_example,)


@app.cell(hide_code=True)
def hints_example():
    L.hints(
        "sft-example",
        [
            "`+` puts lists together: `[1, 2] + [3]` is `[1, 2, 3]`. `end_id` is one number, so `[end_id]` is a list of one.",
            "`[-100] * 3` is `[-100, -100, -100]`: one -100 for every id of the instruction, then the answer and the end, just as in the ids.",
        ],
        why="The labels say what counts in the loss: -100 for what was given, the ids themselves for what the model should write, up to the end-of-text, so that it learns to stop.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_batch():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A batch

        Complete `sft_batch(examples, end_id)`: step 3. `examples` is a list of `(ids, labels)`, as your `sft_example` gives them. Give back two tensors: the inputs `x` and the targets `y`, a row per example, shifted by one and padded at the end to the longest: the inputs with `end_id`, the targets with -100. The check runs your function on two examples of different lengths, and more.
        """),
        L.task("Make `sft_batch` turn examples of different lengths into one batch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batch():
    box_batch = L.code_box("sft-batch", lines=12)
    box_batch
    return (box_batch,)


@app.cell(hide_code=True)
def check_batch(box_batch):
    result_batch = L.check("sft-batch", box_batch.value)
    result_batch
    return (result_batch,)


@app.cell(hide_code=True)
def hints_batch():
    L.hints(
        "sft-batch",
        [
            "The longest example has `max(len(ids) for ids, labels in examples)` ids, and one fewer inputs: its last id is only a target. A row of `len(ids) - 1` inputs needs `pad = n - (len(ids) - 1)` more.",
            "`[7, 3, 0][:-1]` is `[7, 3]`, and `[7, 3] + [0] * 2` is `[7, 3, 0, 0]`: a row without its last id, then the padding. The targets go the same way, from `labels[1:]`, with `-100` as the padding.",
        ],
        why="A tensor needs rows of one length. The targets are shifted by one, as in pretraining, and the padding gets -100, so that it counts for nothing in the loss.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_words():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Did it do what was asked?

        Complete `words_used(story, words)`: step 5. Give back `True` when every word of `words` is in the story as a whole word, and else `False`. The requested words are in small letters; in the story a word can start with a capital, and have punctuation before or after it, such as `"Kite!`. *Kites* is another word than *kite*. The check calls `words_used("Tom flew his kite in the park.", ["kite", "park"])`, and others.
        """),
        L.task("Make `words_used` say whether a story uses every requested word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("words-used", lines=8)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("words-used", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "words-used",
        [
            "`story.lower().split()` gives the words of the story in small letters. `word.strip(\".,!?;:\\\"'\")` takes punctuation off both ends of a word, as in [Keeping test data out of training data](/course/open/llm-decontamination?lang=en).",
            "Put the stripped words in a set with `found.add(...)`. Then a loop over `words` that gives back `False` as soon as one is not in `found`; after the loop, `True`.",
        ],
        why="A program can only check what it can see: whole words, whatever their capitals and punctuation. The demo counts with your function how many answers do what was asked.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _m = C.MEASURED["yours"]
    mo.vstack([
        mo.md(f"""
        ### Fine-tune your model

        The button writes an instruction for every story of `data/tinystories-small.txt` but the last {C.HELD} that names one of the {len(C.WORDS)} nouns of the course's list: the story cut after its last sentence that fits in {C.MAX_LEN} tokens with its instruction, and one or two of its nouns as the requested words. It makes the examples with your `sft_example`: about {C.number(round(_m['pairs'], -2), 0, L.lang)}. Then your model of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en), or the course's when you have none, answers {C.TESTS} instructions made from the last {C.HELD} stories, which none of the examples come from, and your `words_used` counts the answers that use every word. With only {len(C.WORDS)} words to ask for, most of these instructions are also among the examples, each with another story: what counts is whether the model's own story uses the words. Then {C.number(C.STEPS, 0, L.lang)} steps of SFT, with your `sft_batch` and your `pretrain_step` of the capstone at a learning rate of {C.number(C.LR, 3, L.lang)}, a third of that of the pretraining, and the same {C.TESTS} instructions again.

        Your `pretrain_step` needs no change: `F.cross_entropy` skips every target of -100 by itself. On the course's CPU, with one thread, the button took about {round((_m['seconds'] + _m['other_seconds']) / 60)} minutes in all for its model of the capstone's demo run, of {C.number(_m['params'], 0, L.lang)} parameters; the course's own model is about five times as big, and takes about four times as long. The fine-tuned model is kept in `work/projects/llm/sft.pt`, for the lessons after this one.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def tune_button_cell():
    tune_button = mo.ui.run_button(label="Fine-tune your model")
    tune_button
    return (tune_button,)


@app.cell(hide_code=True)
def demo_tune(box_batch, box_example, box_words, result_batch, result_example, result_words, tune_button):
    tuned = None
    _note = None
    if not all(_r.state == "correct" for _r in (result_example, result_batch, result_words)):
        _note = "*Finish exercises 1 to 3 first: the demo fine-tunes with your `sft_example` and `sft_batch`, and counts with your `words_used`.*"
    elif tune_button.value:
        def _progress(_p):
            if _p["phase"] == "pairs":
                mo.output.replace(mo.md(f"**{C.number(_p['pairs'], 0, L.lang)} examples** · the model answers the instructions …"))
            elif _p["phase"] in ("before", "after"):
                _when = "Before SFT" if _p["phase"] == "before" else "After SFT"
                mo.output.replace(mo.md(f"**{_when}** · instruction {_p['done']} of {_p['total']} · every word used: {_p['used']}"))
            else:
                mo.output.replace(mo.md(f"**Step {C.number(_p['step'], 0, L.lang)} of {C.number(_p['steps'], 0, L.lang)}** · loss {C.number(_p['loss'], 3, L.lang)} · {C.number(_p['seconds'] / 60, 1, L.lang)} minutes"))
        _r = C.run(run_code(box_example.value).names["sft_example"], run_code(box_batch.value).names["sft_batch"],
                   run_code(box_words.value).names["words_used"], _progress)
        if _r == "run":
            _note = "*The demo stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
        elif _r == "example":
            _note = "*Your `sft_example` stops with an error on the stories, or gives ids and labels of other lengths. Does it work for an answer of a hundred ids?*"
        elif _r == "batch":
            _note = "*Your `sft_batch` stops with an error on a batch of the stories. Does it work for 16 examples of different lengths?*"
        elif _r == "words":
            _note = "*Your `words_used` stops with an error on an answer of the model, or gives something else than `True` or `False`. Does it work for a story without any of the words?*"
        elif _r == "memory":
            _note = "*The memory of your device is full. Close other programs, and press the button again.*"
        elif _r == "diverged":
            _note = "*The loss is no longer a number: the training ran off. Is the learning rate of your `pretrain_step` the one it is given?*"
        else:
            tuned = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (tuned,)


@app.cell(hide_code=True)
def demo_result(tuned):
    _m = C.MEASURED["yours"]
    _info = C.saved()
    _run = tuned if tuned is not None else _info if _info is not None else _m
    _base = tuned["source"] if tuned is not None else _info["base"] if _info is not None else None
    _whose = {"yours": "your model", "course": "the course's model", None: "the course's model of the capstone's demo run"}[_base]
    _rows = "\n".join(f"| {_label} | **{_run[_when]['used']}** of {_run[_when]['total']} | {_run[_when]['stopped']} of {_run[_when]['total']} |"
                      for _label, _when in (("before SFT", "before"), ("after SFT", "after")))
    _stand = "" if _base is not None else "\n\nThe course's run stands in until you fine-tune your own."
    _example = _run["example"] if tuned is None else C.example(tuned["before"], tuned["after"])
    mo.vstack([
        mo.md(f"The answers of {_whose} to the {_run['after']['total']} test instructions:\n\n| | every word used | stopped by itself |\n|---|---|---|\n{_rows}{_stand}"),
        mo.md(("The first of these instructions that the model did after SFT and not before: " if _example.get("better") else "The first of these instructions: ") + f"*{C.instruction(_example['words']).splitlines()[0]}*\n\nBefore SFT:\n\n" + C.P.quote(_example["before"] or "…")),
        mo.md("After SFT:\n\n" + C.P.quote(_example["after"] or "…")),
    ])
    return


@app.cell(hide_code=True)
def text_ask():
    mo.md(r"""
    ### Ask your model

    Type one or two words, and let your fine-tuned model write a story with them. Words that the stories of TinyStories often use work best.
    """)
    return


@app.cell(hide_code=True)
def ask_cell():
    ask_box = mo.ui.text(value="kite park", label="Your words", full_width=True)
    ask_button = mo.ui.run_button(label="Write")
    mo.hstack([ask_box, ask_button], align="end")
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, tuned):
    _note = None
    if ask_button.value:
        _words = C.asked_words(ask_box.value)
        _t = C.tools()
        if not _words:
            _note = "*Type a word first, in the letters a to z.*"
        elif _t == "run":
            _note = "*Writing stops with an error. Do your modules in `work/projects/llm/` still pass their checks?*"
        else:
            _model, _info, _ = (tuned["model"], C.saved(), "yours") if tuned is not None else C.load()
            _merges = {tuple(_pair): 256 + _i for _i, _pair in enumerate(_info["merges"])}
            _text, _ended = C.write_for(_model, _merges, _words, _t["encode_fast"], _t["decode"])
            _whose = "" if C.saved() is not None else " The course's SFT model wrote this: fine-tune your own first."
            _note = f"*{C.instruction(_words).splitlines()[0]}*{_whose}\n\n" + C.P.quote(_text or "…") + ("" if _ended else "\n\n*It did not stop by itself within " + str(C.LENGTH) + " tokens.*")
    mo.md(_note) if _note else None
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ### Exercise 4. The end of an answer
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "no-end",
        question="You forget the end-of-text at the end of every answer in the examples, and fine-tune. What happens when the model answers an instruction?",
        options={
            "error": "Generating stops with an error: the model has no token to end with",
            "nothing": "Nothing changes: it learned the end-of-text in pretraining",
            "right": "It does not stop by itself: it writes on after its story",
        },
        correct="right",
        why={
            "error": "Nothing breaks: the model still has its end-of-text token. It only never learned to write it after an answer, so the loop goes on to its limit.",
            "nothing": f"Before SFT, the course's model did stop by itself in {C.MEASURED['yours']['before']['stopped']} of {C.MEASURED['yours']['before']['total']} answers. After SFT without the end-of-text, it stopped in {C.MEASURED['no-end']['stopped']} of {C.MEASURED['no-end']['total']}: no example ever showed it an end after an answer.",
            "right": "The model learns to stop from the examples: every answer that ends with the end-of-text, with its label counted, teaches it where a turn is over.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The fine-tuning of [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en), now of a language model, with the same step as your pretraining in [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en); the generation prompt and the end-of-turn token of [Chat templates](/course/open/llm-chat-template?lang=en), now in a template of your own; and a check that a program can do, of what was asked. Your project `llm` has `sft_example.py`, `sft_batch.py` and `words_used.py` now, and after the demo your fine-tuned model in `sft.pt`. Next, LoRA fine-tunes the same way with far fewer numbers to learn, and DPO uses your `words_used` to tell a better answer from a worse one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_example, result_batch, result_words, choice_end):
    L.wrap_up(result_example, result_batch, result_words, choice_end)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
