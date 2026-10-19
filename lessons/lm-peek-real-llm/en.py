# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-peek-real-llm. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-peek-real-llm", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en) and [Many classes: softmax](/course/open/ml-softmax?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="In your bigram table of probabilities for Tiny Shakespeare, what does the row of \"q\" hold?",
        options={
            "column": "The probability of \"q\" after each of the 65 characters",
            "right": "The probability of each of the 65 characters to come after \"q\"",
            "counts": "How often each character comes after \"q\", as whole numbers",
        },
        correct="right",
        why={
            "column": "That is the column of \"q\": the row is about what comes after it. See [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).",
            "counts": "Those are the counts, the row of the count matrix before you divide by its sum. See [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).",
            "right": "A probability distribution over the next character, which adds up to 1. A real language model gives such a row too, over its tokens.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Softmax turns a model's 50,257 logits into probabilities. What do those add up to?",
        options={
            "many": "50,257: one for each token",
            "logits": "The sum of the logits: softmax keeps the total",
            "right": "1: whatever the logits are",
        },
        correct="right",
        why={
            "many": "Each probability is between 0 and 1, and softmax divides by the sum of all exponentials, so together they are 1. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "logits": "Softmax divides every exponential by their sum, so the total is always 1, not the sum of the logits. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "Softmax divides every exponential by their sum. With 50,257 tokens, most probabilities are tiny.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your bigram table predicts the next character from one character: after "e", always the same row, whether the text was "the" or "she". A real language model reads the whole text before it. GPT-2, an open model OpenAI published in 2019, learned from 8 million web pages, with 124 million parameters. For every text, it computes a new row: a probability for each of the 50,257 tokens of its vocabulary. In this lesson you load it and look at that row.

    **An everyday example.** The keyboard of your phone suggests three words while you type. It has such a row too, and shows you the three words with the largest numbers in it.
    """)
    return


@app.cell(hide_code=True)
def predict_capital():
    guess_capital = L.guess(
        "capital",
        question="GPT-2 reads \"The capital of France is\". Which token do you think it gives the highest probability to come next?",
        options={
            "paris": "\" Paris\"",
            "right": "\" the\"",
            "france": "\" France\"",
        },
        reveal=f"**\" the\", with {C.share(C.REFERENCE['The capital of France is'][0][1], 'en')}.** \" Paris\" gets {C.share(C.REFERENCE['The capital of France is'][4][1], 'en')}, in fifth place, after \" now\", \" a\" and \" France\". GPT-2 does not answer a question: it continues the text the way texts on the web go on, such as \"The capital of France is the city of…\" or \"…is now…\". Which tokens are likely depends on everything before them.",
    )
    guess_capital
    return (guess_capital,)


@app.cell(hide_code=True)
def text_idea():
    _cat = C.REFERENCE["The cat sat on the"]
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A row for every text.** GPT-2 turns a text of $t$ tokens into logits of shape $(t, 50257)$: at every position, a logit for every token of its vocabulary, for the token that comes after that position. The last row is about the token after the whole text. Softmax turns that row into probabilities that add up to 1: the **next-token distribution**. Your bigram table had one fixed row per character; GPT-2 computes its row anew from all the tokens before, with 12 transformer blocks, which you build in later lessons. It reads up to 1,024 tokens back.

        **The most likely tokens.** Of the 50,257 probabilities, you look at the few largest: `torch.topk(probs, k)` gives the $k$ largest values, `.values`, and their places, `.indices`, which are token IDs. `enc.decode([i])` turns token ID `i` back into its piece of text, as in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).

        **Download once.** The course does not ship GPT-2: its file is 548 MB. The button in the worked example downloads it once from Hugging Face, from `openai-community`, where Hugging Face keeps the GPT-2 that OpenAI published, at a fixed version, and checks that every byte is the course's pinned file. After that, the lesson works without a connection. Its license is MIT, as its model card says.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **next-token distribution** | | the probabilities a language model gives each token of its vocabulary to come after a text | `torch.softmax(logits[-1], dim=0)` |
        | `torch.softmax(z, dim=0)` | torch softmax | the softmax of [Many classes: softmax](/course/open/ml-softmax?lang=en) in PyTorch, along dimension 0 of `z` | |
        | `torch.topk(p, k)` | torch top k | the $k$ largest values of `p`, in `.values`, and their places, in `.indices` | `torch.topk(probs, 10)` |
        """),
        L.watch_out("The next token comes from the last row, `logits[-1]`. The first row, `logits[0]`, is what GPT-2 expects after only the first token."),
        mo.md(r"""
        The picture: the 10 most likely tokens GPT-2 gives after "The cat sat on the", each with its probability. A space shows as ␣: in GPT-2, it belongs to the token after it. Together these 10 get less than half: the other 50,247 tokens share the rest.
        """),
        mo.Html(C.token_bars("The cat sat on the", _cat, {"title": "GPT-2, after \"The cat sat on the\"",
                             "alt": "Ten bars, the most likely first: " + ", ".join(f"{C.piece(_t)} {C.share(_p, 'en')}" for _t, _p in _cat) + "."}, "en")),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To see what a language model expects after a text:

    1. The tokens: `ids = enc.encode(text)`.
    2. The logits: `logits = gpt2(torch.tensor(ids))`, inside `with torch.no_grad():`, of shape $(t, 50257)$.
    3. The next-token distribution: `probs = torch.softmax(logits[-1], dim=0)`.
    4. The most likely tokens: `top = torch.topk(probs, k)`, and `enc.decode([i])` for each token ID `i` in `top.indices`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The four steps for the text "Once upon a time". `gpt2` is GPT-2 small, which the cell below loads once it is downloaded.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download GPT-2, 548 MB")
    return (get_button,)


@app.cell(hide_code=True)
def gpt2_cell(get_button):
    gpt2 = C.gpt2_or_stop(get_button, {
        "ask": "*GPT-2 is not on your computer yet. The button downloads it once, 548 MB, from Hugging Face. The worked example and the cell with your own text need it; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading GPT-2** · {done} of {total} MB",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 548 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*GPT-2 small is on your computer: 124 million parameters, by OpenAI, license MIT.*",
    })
    return (gpt2,)


@app.cell
def example_once(gpt2):
    import torch
    import tiktoken

    _enc = tiktoken.get_encoding("gpt2")
    # 1. the tokens of the text
    _ids = _enc.encode("Once upon a time")
    print(_ids)
    # 2. the logits: a row per position, a logit per token of the vocabulary
    with torch.no_grad():
        _logits = gpt2(torch.tensor(_ids))
    print(_logits.shape)
    # 3. the next-token distribution: the softmax of the last row
    _probs = torch.softmax(_logits[-1], dim=0)
    print(round(_probs.sum().item(), 3))
    # 4. the 5 most likely next tokens
    _top = torch.topk(_probs, 5)
    # repr shows the piece of text in quotes, so that you see a space at its start
    for _j in range(5):
        _i = _top.indices[_j].item()
        print(repr(_enc.decode([_i])), round(_top.values[_j].item(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    _once = C.REFERENCE["Once upon a time"]
    mo.md(f"""
    The text has 4 tokens, so the logits have 4 rows of 50,257. The probabilities of the last row add up to 1. After "Once upon a time", GPT-2 expects a comma most, with {C.share(_once[0][1], 'en')}: in the stories it read, "Once upon a time, …" is how it goes on. Then come \" the\", \" I\" and \" he\", each with less than 0.07.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The next-token distribution

        Complete `next_probs(logits)`: step 3 of the recipe, for logits of any shape $(t, V)$. The check calls `next_probs(logits)` with small logits of 2 positions and a vocabulary of 3 tokens, `[[2, 0, 1], [0, 3, 1]]`, so that it works without GPT-2, and compares the probabilities, rounded to 3 decimals.
        """),
        L.task("Make `next_probs` give the softmax of the last row of `logits`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("next-probs", lines=6)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("next-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "next-probs",
        [
            "The token after the whole text comes from the last row of `logits`.",
            "The last row is `logits[-1]`, a tensor of $V$ logits.",
            "`torch.softmax(___, dim=___)`: the last row, along its only dimension.",
        ],
        why="`logits[-1]` is the last row: the logits of the token after the whole text. Softmax along its only dimension, `dim=0`, turns them into probabilities that add up to 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_top():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The most likely tokens

        Complete `top_tokens(probs, k)`: step 4 of the recipe. It gives the `k` most likely tokens as a list of pairs: the piece of text and its probability, the most likely first. The check calls `top_tokens(probs, 2)` with made-up probabilities over GPT-2's 50,257 tokens: 0.5 for token ID 262, 0.3 for 257, 0.15 for 1169 and 0.05 for 13, and 0 for every other.
        """),
        L.task("Make `top_tokens` give the `k` most likely tokens as pairs of a piece of text and its probability."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top():
    box_top = L.code_box("top-tokens", lines=14)
    box_top
    return (box_top,)


@app.cell(hide_code=True)
def check_top(box_top):
    result_top = L.check("top-tokens", box_top.value)
    result_top
    return (result_top,)


@app.cell(hide_code=True)
def hints_top():
    L.hints(
        "top-tokens",
        [
            "`torch.topk(probs, k)` gives the `k` largest probabilities and their token IDs.",
            "In the loop, `enc.decode([i])` is the piece of text of token ID `i`, and `top.values[j].item()` its probability.",
            "`top = torch.topk(___, ___)`, and in the loop `pairs.append((enc.decode([___]), top.values[j].item()))`.",
        ],
        why="`torch.topk` gives the largest values first, with their places in `probs`: those places are token IDs. `enc.decode` takes a list of token IDs, so one ID goes in as `[i]`. `.item()` turns a tensor of one number into a plain number.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your own text

    Type a text below, once GPT-2 is downloaded. The cell gives it to GPT-2, with your `next_probs` and `top_tokens` once both are right, else the course's, and shows the 10 most likely next tokens. GPT-2 learned from English web pages: an English text works best.
    """)
    return


@app.cell(hide_code=True)
def prompt_cell():
    prompt_box = mo.ui.text(value="The cat sat on the", label="Your text", full_width=True)
    prompt_box
    return (prompt_box,)


@app.cell(hide_code=True)
def demo_text(gpt2, prompt_box, box_probs, result_probs, box_top, result_top):
    _mine = result_probs.state == "correct" and result_top.state == "correct"
    _text = prompt_box.value or "The cat sat on the"
    _d = C.demo(gpt2, _text, box_probs.value if _mine else None, box_top.value if _mine else None)
    _who = "your `next_probs` and `top_tokens`" if _d["yours"] else "the course's `next_probs` and `top_tokens`"
    mo.vstack([
        mo.md(f"Your text has {C.count(len(_d['pieces']), 'en')} tokens: " + ("… · " if len(_d["pieces"]) > 40 else "")
              + " · ".join(f"`{C.piece(_p)}`" for _p in _d["pieces"][-40:]) + "."
              + (" GPT-2 reads at most 1,024 tokens: it got the last 1,024." if _d["cut"] else "") + f" With {_who}:"),
        mo.Html(C.token_bars(_text, _d["top"], {"title": "GPT-2, after your text",
                             "alt": "Ten bars, the most likely first: " + ", ".join(f"{C.piece(_t)} {C.share(_p, 'en')}" for _t, _p in _d["top"]) + "."}, "en")),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_context():
    mo.md(r"""
    ### Exercise 3. The same last token
    """)
    return


@app.cell(hide_code=True)
def choice_context_cell():
    _cat, _i = dict(C.REFERENCE["The cat sat on the"]), dict(C.REFERENCE["I sat on the"])
    choice_context = L.choice(
        "context",
        question="\"The cat sat on the\" and \"I sat on the\" both end in the token \" the\". A bigram table of tokens gives the same row after \" the\" for both. And GPT-2?",
        options={
            "same": "The same row: the next token depends on the last token",
            "random": "Another row every time you run it, even for the same text",
            "right": "Another row for each: it reads every token before \" the\"",
        },
        correct="right",
        why={
            "same": f"GPT-2 reads the whole text before. After \"The cat sat on the\", \" bed\" gets {C.share(_cat[' bed'], 'en')}; after \"I sat on the\", {C.share(_i[' bed'], 'en')}.",
            "random": "For the same text, GPT-2 gives the same logits every time: there is no chance in it. Choosing a token at random from the distribution is what [Sampling: let the model write](/course/open/lm-sampling?lang=en) does.",
            "right": f"After \"The cat sat on the\": \" floor\" {C.share(_cat[' floor'], 'en')}, \" bed\" {C.share(_cat[' bed'], 'en')}. After \"I sat on the\": \" couch\" {C.share(_i[' couch'], 'en')}, \" bench\" {C.share(_i[' bench'], 'en')}, and \" bed\" only {C.share(_i[' bed'], 'en')}. In the texts GPT-2 read, cats sit on beds more often than people do.",
        },
    )
    choice_context
    return (choice_context,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The row of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en), now computed for every text; the logits and softmax of [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en), from 124 million parameters instead of one table; the tokens of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en); and a model from a hub, as in [Using a pretrained model](/course/open/torch-pretrained?lang=en). The transformer blocks inside GPT-2 come in later lessons, where you build a GPT yourself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_probs, result_top, choice_context):
    L.wrap_up(result_probs, result_top, choice_context)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
