# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-tokens-prices. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tokens-prices", "en")
    C = L.checks   # the course's counts and answers of Qwen3, and the demo that counts before it sends


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en), and one about [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="With a tokenizer `enc`, what does `len(enc.encode(text))` give?",
        options={
            "chars": "The number of characters of `text`",
            "words": "The number of words of `text`",
            "tokens": "The number of tokens of `text`",
        },
        correct="tokens",
        why={
            "tokens": "`enc.encode(text)` gives the list of token IDs, one per piece the tokenizer cut, and `len` counts them.",
            "chars": "That is `len(text)`. What does `encode` give back? See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "words": "That is `len(text.split())`; a tokenizer cuts some words in more pieces. See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_role():
    quick_role = L.choice(
        "quick-role",
        question="What goes in a message with the role `\"system\"`?",
        options={
            "answer": "The model's answer",
            "instruction": "Instructions for the whole conversation, such as how the model should answer",
            "question": "The user's question",
        },
        correct="instruction",
        why={
            "instruction": "The system prompt comes first and holds for every turn: who the model is, and how it answers.",
            "answer": "The model's answer has the role `\"assistant\"`. See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "question": "The question has the role `\"user\"`. See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
        },
    )
    quick_role
    return (quick_role,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A hosted API sends its bill per token, not per question. An answer of 500 tokens costs more than one of 50, and the tokens that come out usually cost several times as much as the ones that go in. Whoever counts before sending knows what a feature costs, and where to save: a shorter prompt, a shorter answer, or a cheaper model.

    **An everyday example.** At a print shop you pay per page, and a page in color costs more than one in black and white. Count the pages before you print, and you know the bill. That is only a comparison; below you count real tokens.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **input token** | a token of what you send: the system prompt, the conversation so far, the question; the answer counts them in `usage` as `prompt_tokens` |
    | **output token** | a token the model writes; counted as `completion_tokens`, and usually dearer, because the model writes them one at a time |
    | **token price** | what an API asks per million tokens, one price for input tokens and one for output tokens; in this lesson, made-up prices of 3 and 15 dollars |
    | **chat template** | the few tokens of its own that the server puts around every message, so the model sees who says what; they count as input tokens too |

    ### Every model counts in its own tokens

    The course counted five texts with GPT-2's tokenizer, from [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en), and with Qwen3's:
    """)
    return


@app.cell(hide_code=True)
def picture_counts():
    mo.Html(C.counts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_counts():
    _n = C.numbers(L.lang)
    mo.md(rf"""
    The English sentence is {_n['en_qwen']} tokens for both. The same sentence in Dutch is {_n['nl_gpt2']} tokens for GPT-2 and {_n['nl_qwen']} for Qwen3, with fewer words: both tokenizers learned more English than Dutch, and GPT-2's hardly anything else. For the number it is the other way round: {_n['number_qwen']} tokens for Qwen3 against {_n['number_gpt2']} for GPT-2, because Qwen3 cuts digits one by one. So you count with the tokenizer of the model you call; another one is only an estimate.
    """)
    return


@app.cell(hide_code=True)
def text_answers():
    mo.md(r"""
    ### What comes out costs the most

    The course asked Qwen3 two questions through its own server, one that wants a short answer and one that wants a long one, and read the tokens in `usage`:
    """)
    return


@app.cell(hide_code=True)
def picture_answers():
    mo.Html(C.answers_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_bill():
    _n = C.numbers(L.lang)
    mo.vstack([
        mo.md(rf"""
        The long answer has {_n['long_out']} tokens out against {_n['short_out']}, {_n['times']} times as many, and it cost more than ten times as much. It also took {_n['long_seconds']} seconds instead of {_n['short_seconds']}: the model writes the tokens that come out one at a time. The question itself is {_n['short_text']} tokens for Qwen3's tokenizer, but the server counted {_n['short_in']} in: the chat template added {_n['template']}.

        A call costs `tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000`; `1_000_000` is a million, and Python skips the underscores. Before you send, you know the tokens in, by counting, but not the tokens out. You do know the most: the answer stops at `max_tokens`.
        """),
        L.watch_out("Every message of the conversation goes along with every call, and counts again as input tokens. A long conversation gets dearer with every turn."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: what a call costs, before you send it

    1. **Count the prompt:** `len(enc.encode(prompt))`, with the tokenizer of the model you call; the server adds a few tokens per message.
    2. **Limit the answer:** with `max_tokens`, the answer has at most that many tokens.
    3. **Price both:** tokens in times the price in, plus tokens out times the price out, divided by a million.
    4. **Afterwards, read the real count:** `usage` in the answer has `prompt_tokens` and `completion_tokens`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The most a call with the question *Name three colors of the rainbow.* can cost, with at most 50 tokens out, for a model with GPT-2's tokenizer:

    ```python
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    prompt = "Name three colors of the rainbow."
    tokens_in = len(enc.encode(prompt))
    tokens_in * 3 / 1_000_000 + 50 * 15 / 1_000_000
    ```

    1. **The tokenizer:** `enc` cuts text the way GPT-2 does, from the course's own copy.
    2. **The prompt:** `enc.encode(prompt)` gives 7 token IDs, so `tokens_in` is 7.
    3. **The tokens in:** 7 times 3 dollars per million is 21 millionths of a dollar.
    4. **The most for the answer:** 50 tokens at 15 dollars per million is 750 millionths; together `0.000771` dollars. The real answer may be shorter, and cost less.
    """)
    return


@app.cell(hide_code=True)
def text_cost():
    _f = C.findings()
    mo.vstack([
        mo.md(rf"""
        ## Your turn

        ### Exercise 1. What a call costs

        Write `cost(tokens_in, tokens_out, price_in, price_out)`: what one call costs, in dollars, with the prices per million tokens. The course's short answer above had {_f['short_in']} tokens in and {_f['short_out']} out; at 3 and 15 dollars per million, that is `{_f['short_cost']:.6f}` dollars:

        ```python
        cost({_f['short_in']}, {_f['short_out']}, 3, 15)
        ```
        """),
        L.task("Finish `cost`: the tokens in at `price_in` plus the tokens out at `price_out`, per million."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=3)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "Two parts, each with its own price: what goes in, and what comes out.",
            "Each part is its tokens times its price; the prices are per million tokens, so divide each part by `1_000_000`.",
            "`return tokens_in * price_in / 1_000_000 + ___`, with the part for the tokens out in the gap.",
        ],
        why="""```python
def cost(tokens_in, tokens_out, price_in, price_out):
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000
```

Each kind of token at its own price, and the prices are per million tokens.""",
    )
    return


@app.cell(hide_code=True)
def text_most():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The most, before you send

        Write `most(prompt, max_tokens, price_in, price_out)`: the most a call with `prompt` can cost, when the answer may have up to `max_tokens` tokens. Count the prompt with `enc`, GPT-2's tokenizer; the chat template you may leave out. For the worked example, `most("Name three colors of the rainbow.", 50, 3, 15)` gives `0.000771`.
        """),
        L.task("Finish `most`: count the prompt's tokens with `enc`, take `max_tokens` for the answer, and price both."),
    ])
    return


@app.cell(hide_code=True)
def exercise_most():
    box_most = L.code_box("most", lines=9)
    box_most
    return (box_most,)


@app.cell(hide_code=True)
def check_most(box_most):
    result_most = L.check("most", box_most.value)
    result_most
    return (result_most,)


@app.cell(hide_code=True)
def hints_most():
    L.hints(
        "most",
        [
            "Steps 1 to 3 of the recipe: count, limit, price.",
            "`enc.encode(prompt)` gives the prompt's tokens as a list, and `len` counts them; the answer has at most `max_tokens` tokens.",
            "`tokens_in = ___`; then `tokens_out = max_tokens` and the same `return` as in `cost` of exercise 1.",
        ],
        why="""```python
def most(prompt, max_tokens, price_in, price_out):
    tokens_in = len(enc.encode(prompt))
    tokens_out = max_tokens
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000
```

The prompt you can count before you send it; the answer you can only limit, so you price its most.""",
    )
    return


@app.cell(hide_code=True)
def text_cheaper():
    mo.md(r"""
    ### Exercise 3. Where to save
    """)
    return


@app.cell(hide_code=True)
def choice_cheaper_cell():
    choice_cheaper = L.choice(
        "cheaper",
        question="A feature sends 1,000 tokens in and gets 500 out, at 3 and 15 dollars per million. Which change saves the most?",
        options={
            "same": "Both save the same: 200 tokens are 200 tokens",
            "prompt": "A prompt that is 200 tokens shorter",
            "answer": "An answer that is 200 tokens shorter",
        },
        correct="answer",
        why={
            "answer": "200 tokens out at 15 dollars per million save $0.003; 200 tokens in at 3 save only $0.0006. Asking for a short answer, or a lower `max_tokens`, saves the most.",
            "same": "A token in and a token out have different prices. Which one is dearer?",
            "prompt": "That saves 200 tokens at 3 dollars per million. What do 200 tokens at the other price save?",
        },
    )
    choice_cheaper
    return (choice_cheaper,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Count with Qwen3's own tokenizer, then send

    Write a prompt: the course counts it with Qwen3's tokenizer and with GPT-2's, sends it to Qwen3-0.6B through its own server with at most 200 tokens out, and shows what the server counted, and what that costs at the made-up prices.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value="Explain in two sentences why the sky is blue.", label="Your prompt", full_width=True)
    send_button = mo.ui.run_button(label="Count it, and send it")
    mo.vstack([prompt_box, send_button])
    return prompt_box, send_button


@app.cell(hide_code=True)
def demo_answer(prompt_box, send_button):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to count your prompt and send it.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Write a prompt in the box first.*")
    else:
        mo.output.replace(mo.md("*Counted; the server loads Qwen3, which writes the answer …*"))
        _words = {"title": "Your prompt, counted and sent",
                  "done": "Before sending: {counted} tokens for Qwen3's tokenizer ({gpt2} for GPT-2's), and {template} for the chat template, so at most ${worst} with {most} tokens out. The server counted {sent} in and {out} out: ${real}, after {seconds} seconds.",
                  "failed": "The server did not answer ({why}). Press the button again.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(prompt_box.value, 200, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** You count with the tokenizers of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en), and every message of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), the system prompt too, counts as input tokens. In the next lesson, [Calling a model over HTTP](/course/open/build-call-api?lang=en), you send the request yourself, and read `usage` in the answer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cheaper, result_cost, result_most):
    L.wrap_up(result_cost, result_most, choice_cheaper)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
