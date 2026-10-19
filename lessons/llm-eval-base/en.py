# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-eval-base. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-eval-base", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Perplexity](/course/open/lm-perplexity?lang=en) and [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ppl():
    quick_ppl = L.choice(
        "quick-ppl",
        question="On a text it never saw, a model has an average loss of 2.0 nats per token. About what is its perplexity?",
        options={
            "hundred": "100: 10 to the power 2",
            "two": "2: the loss itself",
            "right": "7.4: e to the power 2",
        },
        correct="right",
        why={
            "two": "The loss is the log of the perplexity, in nats: the perplexity is $e^{2.0}$. See [Perplexity](/course/open/lm-perplexity?lang=en).",
            "hundred": "A loss in nats uses the natural log, so the perplexity is $e^{2.0}$, not $10^{2.0}$. See [Perplexity](/course/open/lm-perplexity?lang=en).",
            "right": "$e^{2.0} \\approx 7.4$: as if the model chooses between about 7 tokens each time. Today you see why that number depends on the tokens.",
        },
    )
    quick_ppl
    return (quick_ppl,)


@app.cell(hide_code=True)
def quick_check_base():
    quick_base = L.choice(
        "quick-base",
        question="You ask your model of the capstone, a base model, \"What is two plus two?\", and it writes on with a story. What does that tell you about how well it learned its stories?",
        options={
            "right": "Little: a base model only learned to continue text",
            "badly": "That it learned them badly: a good model answers 4",
            "more": "That it needs more training on the same stories",
        },
        correct="right",
        why={
            "badly": "Answering is what a chat model is trained for; a base model continues the text, however well it learned. See [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
            "more": "More of the same stories makes it better at stories, not at answering: a chat model learns that from training on conversations. See [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
            "right": "So you do not test a base model with questions, but with how well it predicts text: its loss on text it never saw.",
        },
    )
    quick_base
    return (quick_base,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You pretrained a model. Is it any good? A number on its own says little: you want to hold it against other models, such as GPT-2. But every model cuts text into its own tokens, and a loss or a perplexity per token depends on how big those tokens are. So the loss of a base model can be given per byte of text instead: The Pile, a large open dataset of text, measures how well models predict it in bits per byte, because that does not depend on how a tokenizer cuts the text (Gao and others, 2020).

    **An everyday example.** Two runners each say how many steps they needed for a race. One takes small steps, the other big ones: only the time on the same course tells who is faster.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt2():
    _m = C.MEASURED
    guess_gpt2 = L.guess(
        "gpt2",
        question=f"In the course's run of the capstone, a model of {C.number(_m['yours']['params'], 0, L.lang)} parameters trained on TinyStories has a perplexity of {C.number(_m['yours']['ppl'], 1, L.lang)} per token on {C.STORIES} stories it never saw. GPT-2 small has {C.number(_m['gpt2']['params'] / 1e6, 0, L.lang)} million parameters and was trained on web pages. What is its perplexity per token on the same stories, with its own tokens?",
        options={
            "lower": "Lower: a model that size predicts better",
            "right": "Higher",
            "same": "About the same",
        },
        reveal=f"**Higher: {C.number(_m['gpt2']['ppl'], 1, L.lang)}.** Per token, the small model looks much better. But GPT-2's tokens are bigger: it needs {C.number(_m['gpt2']['tokens'], 0, L.lang)} tokens for these stories, the small model {C.number(_m['yours']['tokens'], 0, L.lang)}. Each of GPT-2's guesses covers more text, so each one is harder. Which model predicts the stories better, this lesson measures.",
    )
    guess_gpt2
    return (guess_gpt2,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Score a base model on text it never saw.** Cut the text into the model's own tokens, and in windows as long as the ones it trained on, let it predict every token from the ones before it in its window. The loss of each token, in nats, says how surprised the model was.

    **A number per token depends on the tokens.** A tokenizer with big tokens needs few of them, and each prediction covers more text: its loss per token is higher, even for a model that predicts the text just as well. A perplexity per token is only comparable between models with the same tokenizer.
    """)
    return


@app.cell(hide_code=True)
def picture_cut():
    mo.vstack([
        mo.md("The picture: a sentence of the stories this lesson scores, cut into the tokens of the model the first button scores, and into GPT-2's. The bytes are the same; GPT-2 needs fewer tokens for them, so each of its guesses covers more text."),
        C.cut_picture({"title": "The same {n} bytes, other tokens", "ours": "the model of the first button: {n} tokens", "gpt2": "GPT-2: {n} tokens",
                       "alt": "A sentence of {n} bytes cut into the tokens of two tokenizers: GPT-2 needs fewer."}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_bpb():
    mo.md(r"""
    **Bits per byte.** Add the losses of all tokens, the surprise of the whole text. Turn nats into bits, dividing by $\ln 2$, and spread them over the bytes of the text in UTF-8: that is the number of **bits per byte**. The text is the same for every model, and so are its bytes, whatever the tokens:
    """)
    return


@app.cell(hide_code=True)
def formula_bpb():
    L.formula(r"\text{bits per byte} = \frac{\sum_i \text{loss}_i}{\ln 2 \cdot \text{bytes}}")
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    A model that knew nothing, guessing every one of the 256 values of a byte as likely, would need 8 bits per byte; a lower number means a better model of the text. Per character works too for text of plain letters, as in [Perplexity](/course/open/lm-perplexity?lang=en), but a character can take up to 4 bytes in UTF-8: bytes are the same unit for every text.

    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **bits per byte** | the total loss of a text in bits, spread over its bytes in UTF-8; comparable between tokenizers | `sum(losses) / math.log(2) / n_bytes` |
    | $\ln 2$ | the nats in one bit, about 0.693 | `math.log(2)` |

    ### Recipe

    To compare base models on a text:

    1. Take a text that none of the models trained on.
    2. Cut it into each model's own tokens, and score every token in windows of the model's own length.
    3. Add the losses, turn them into bits, and divide by the bytes of the text.
    4. Compare the bits per byte; a perplexity per token only between the same tokens.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One text, "The cat sat.", 12 bytes, scored by two models with other tokenizers: one cuts it into 12 characters, the other into 4 words. The numbers are made up.
    """)
    return


@app.cell
def example_bpb():
    _n_bytes = len("The cat sat.".encode("utf-8"))
    _chars = [1.1] * 12    # a loss per character
    _words = [3.0] * 4     # a loss per word: each word is harder to guess than one character
    for _losses in (_chars, _words):
        _mean = sum(_losses) / len(_losses)
        print(len(_losses), round(math.exp(_mean), 1), round(sum(_losses) / math.log(2) / _n_bytes, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Per token, the model of characters looks far better: a perplexity of 3.0 against 20.1. But per byte, the model of words is better: 1.443 bits against 1.587. It is surprised by the whole text a little less, 12.0 nats against 13.2, and only the bytes are the same for both.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_losses():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The loss of every token

        Complete `token_losses(model, ids, block)`: step 2. Walk through `ids` in windows of `block + 1` ids that overlap by one: the inputs are a window without its last id, the targets the window without its first. Give back the loss of every target, one number each, in a list. The check tries your function on a small model whose logits depend on how far into its window a token is, with `ids = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5]` and `block = 4`, and more.
        """),
        L.task("Make `token_losses` give the loss of every token of `ids` but the first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_losses():
    box_losses = L.code_box("token-losses", lines=14)
    box_losses
    return (box_losses,)


@app.cell(hide_code=True)
def check_losses(box_losses):
    result_losses = L.check("token-losses", box_losses.value)
    result_losses
    return (result_losses,)


@app.cell(hide_code=True)
def hints_losses():
    L.hints(
        "token-losses",
        [
            "`window = ids[start:start + block + 1]`. A model takes a batch, so `x = torch.tensor([window[:-1]])`, and `model(x)[0]` gives the logits of that one window: one row per input.",
            "`F.cross_entropy(logits, y, reduction=\"none\")` gives a loss per row instead of their mean; `.tolist()` turns them into numbers to add to `losses`.",
        ],
        why="Every token is scored once, from at most `block` tokens before it, as the model saw text during training. The first token has nothing before it, so it gets no loss.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bpb():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Bits per byte

        Complete `bits_per_byte(losses, pieces)`: step 3. `losses` are the losses in nats of the scored tokens, `pieces` the text each of them stands for. Count the bytes of all pieces in UTF-8, and give back the total loss in bits per byte. The check calls `bits_per_byte([1.2, 0.7, 2.3], ["The", " café", " is"])`, and others.
        """),
        L.task("Make `bits_per_byte` turn losses per token into bits per byte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bpb():
    box_bpb = L.code_box("bits-per-byte", lines=8)
    box_bpb
    return (box_bpb,)


@app.cell(hide_code=True)
def check_bpb(box_bpb):
    result_bpb = L.check("bits-per-byte", box_bpb.value)
    result_bpb
    return (result_bpb,)


@app.cell(hide_code=True)
def hints_bpb():
    L.hints(
        "bits-per-byte",
        [
            "`piece.encode(\"utf-8\")` gives the bytes of a piece, as in [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en); `len` counts them. `é` is one character, but two bytes.",
            "The total loss is `sum(losses)`, in nats; divided by `math.log(2)` it is in bits.",
        ],
        why="The bytes of a text are the same for every tokenizer, so the bits per byte of two models on the same text can be compared, however they cut it.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### Your model against GPT-2

        The text is the last {C.STORIES} stories of `data/tinystories-small.txt`, {C.number(len(C.text().replace(C.B.END, '').encode('utf-8')), 0, L.lang)} bytes, with an end-of-text token between every two stories: it stands for no text, so it counts 0 bytes, but its loss counts, for both models. The stories lie in the validation shard of [Project: a pretraining data pipeline](/course/open/llm-data-pipeline?lang=en), so your model never trained on them, and GPT-2 (2019) is older than TinyStories (2023). The first button scores your model of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en) with your `token_losses`, or the course's model when you have none; that takes about a minute. The second downloads GPT-2 small once, 548 MB, as in [Look inside a real LLM's next-token probabilities](/course/open/lm-peek-real-llm?lang=en), and scores it: on the course's CPU, with one thread, that took about {round(C.MEASURED['gpt2']['seconds'] / 60)} minutes.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def yours_button_cell():
    yours_button = mo.ui.run_button(label="Score your model")
    yours_button
    return (yours_button,)


@app.cell(hide_code=True)
def demo_yours(box_losses, result_losses, yours_button):
    yours = None
    _note = None
    if result_losses.state != "correct":   # no mo.stop: the table after this cell shows the course's rows meanwhile
        _note = "*Finish exercise 1 first: the demo scores with your `token_losses`.*"
    elif yours_button.value:
        mo.output.replace(mo.md("**Scoring your model** …"))
        _r = C.score_yours(run_code(box_losses.value).names["token_losses"])
        if _r == "run":
            _note = "*Scoring stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
        elif _r == "losses":
            _note = "*Your `token_losses` stops with an error on this model, or gives another number of losses than there are tokens to score. Does it work on a text of thousands of tokens?*"
        else:
            yours = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()   # also when scoring is done: no stale progress line
    return (yours,)


@app.cell(hide_code=True)
def gpt2_button_cell():
    gpt2_button = mo.ui.run_button(label="Score GPT-2" if C.gpt2_ready() else "Download and score GPT-2 (548 MB)")
    gpt2_button
    return (gpt2_button,)


@app.cell(hide_code=True)
def demo_gpt2(box_losses, gpt2_button, result_losses):
    gpt2 = None
    _note = None
    if result_losses.state != "correct":
        _note = "*Finish exercise 1 first: the demo scores with your `token_losses`.*"
    elif gpt2_button.value:
        _why = C.download_gpt2(lambda _done, _total: mo.output.replace(mo.md(f"**Downloading GPT-2** · {round(_done / 1e6)} of {round(_total / 1e6)} MB")))
        if _why is not None:
            _note = {"net": "*The download did not work. Is there a connection to the internet? Press the button again to try once more.*",
                     "pinned": "*The hub gave another file than the course pinned. Try again later.*",
                     "room": "*The download stopped: is there room on the disk for 548 MB?*"}[_why]
        else:
            mo.output.replace(mo.md("**Scoring GPT-2** · this takes a few minutes …"))
            _r = C.score_gpt2(run_code(box_losses.value).names["token_losses"])
            if _r == "run":
                _note = "*Scoring GPT-2 stops with an error in your `token_losses`. Does it work on a model whose windows are 1,024 tokens long?*"
            else:
                gpt2 = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (gpt2,)


@app.cell(hide_code=True)
def demo_table(box_bpb, gpt2, result_bpb, yours):
    mo.stop(result_bpb.state != "correct", mo.md("*Finish exercise 2 first: the table gives the bits per byte with your `bits_per_byte`.*"))
    _m = C.MEASURED
    _bpb = run_code(box_bpb.value).names["bits_per_byte"]
    _rows = []
    for _name, _mine, _stand in (("yours", yours, _m["yours"]), ("gpt2", gpt2, _m["gpt2"])):
        if _mine is not None:
            _label = "GPT-2 small" if _name == "gpt2" else "Your model" if _mine["source"] == "yours" else "The course's model, on Cosmopedia"
            _value = _bpb(_mine["losses"], _mine["pieces"])
        else:
            _label = "GPT-2 small, the course's run" if _name == "gpt2" else "The course's run of the capstone, on TinyStories"
            _mine, _value = _stand, _stand["bpb"]
        _rows.append(f"| {_label} | **{C.number(_value, 3, L.lang)}** | {C.number(_mine['params'], 0, L.lang)} | {C.number(_mine['tokens'], 0, L.lang)} | {C.number(_mine['loss'], 3, L.lang)} | {C.number(_mine['ppl'], 2, L.lang)} |")
    mo.md("| model | bits per byte | parameters | tokens | loss per token | perplexity |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
          + "\n\nA row of the course stands in until you score that model yourself.")
    return


@app.cell(hide_code=True)
def text_exercise_better():
    mo.md(r"""
    ### Exercise 3. Which model is better?
    """)
    return


@app.cell(hide_code=True)
def choice_better_cell():
    _m = C.MEASURED
    choice_better = L.choice(
        "which-better",
        question=f"On these stories, the course's run of the capstone has a perplexity of {C.number(_m['yours']['ppl'], 1, L.lang)} per token and {C.number(_m['yours']['bpb'], 3, L.lang)} bits per byte; GPT-2 small {C.number(_m['gpt2']['ppl'], 1, L.lang)} per token and {C.number(_m['gpt2']['bpb'], 3, L.lang)} bits per byte. Which model predicts these stories better?",
        options={
            "small": "The small model: its perplexity is lower",
            "cannot": "Neither: their tokenizers differ, so they cannot be compared",
            "right": "GPT-2: it needs fewer bits per byte of the same text",
        },
        correct="right",
        why={
            "small": "Its tokens are smaller, so each of its guesses is easier: a perplexity per token can only be compared between the same tokens.",
            "cannot": "Per token they cannot, but the text and its bytes are the same for both: bits per byte compare them.",
            "right": f"GPT-2 is surprised by the whole text a little less. With some 200 times the parameters, trained on web pages, it is only {C.number(100 * (1 - _m['gpt2']['bpb'] / _m['yours']['bpb']), 0, L.lang)}% better per byte on stories like the ones the small model trained on.",
        },
    )
    choice_better
    return (choice_better,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The perplexity of [Perplexity](/course/open/lm-perplexity?lang=en), now compared with bits per byte; the base model of [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en), tested on how well it predicts text, not on questions; and your model of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en), now next to GPT-2. Later lessons teach your model to follow instructions, and measure that with checks a program can verify.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_losses, result_bpb, choice_better):
    L.wrap_up(result_losses, result_bpb, choice_better)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
