# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-ai-services. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-ai-services", "en")
    C = L.checks   # the course's run of Qwen3 on 120 questions of customers, the stand-in of the checks, and the demo with your label


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Calling a model over HTTP](/course/open/build-call-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_endpoint():
    quick_endpoint = L.choice(
        "quick-endpoint",
        question="To which endpoint of an OpenAI-compatible server do you send the messages of a chat?",
        options={
            "chat": "`/v1/chat/completions`",
            "models": "`/v1/models`",
            "answers": "`/v1/answers`",
        },
        correct="chat",
        why={
            "chat": "That endpoint takes the messages and gives the answer; the same address at every OpenAI-compatible server.",
            "models": "That one lists the models a server has, with a GET. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "answers": "There is no such endpoint in OpenAI's API. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_endpoint
    return (quick_endpoint,)


@app.cell(hide_code=True)
def quick_check_finish():
    quick_finish = L.choice(
        "quick-finish",
        question="You ask with `\"max_tokens\": 20`, and the answer has `\"finish_reason\": \"stop\"`. What does that say?",
        options={
            "cut": "The server cut the answer off at 20 tokens",
            "error": "Something went wrong at the server",
            "ended": "The model ended the answer itself, before 20 tokens",
        },
        correct="ended",
        why={
            "ended": "`\"stop\"` means the model wrote its own end; `\"length\"` would mean it reached `max_tokens`.",
            "cut": "Then it would say `\"length\"`. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "error": "An error comes back as an error status, without an answer. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_finish
    return (quick_finish,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of the work an app asks of AI is one fixed task, done again and again: turn a recorded call into text, translate a manual, read the amount on a receipt, sort the messages of customers into the right queue. For many of those tasks there are ready-made services behind an API, built for that one task. And an LLM can do most of them too, with a prompt. This lesson shows both, and measures one: how well Qwen3-0.6B sorts 120 questions of customers of a bank.

    **An everyday example.** A sorting machine at the post office reads the postcode on each envelope and puts it in the right bin: one task, very fast, and nothing else. A clerk who reads the whole letter can do much more, also sort it, but slower. That is only a comparison; below you see the ready-made services, and an LLM doing the sorting.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **AI service** | a ready-made model behind an API for one task, such as reading text in a photo or translating: you send a photo, sound or text, and get the result back, without a prompt |
    | **speech to text** | sound in, the words that were said out; *text to speech* is the other way round |
    | **OCR** | *optical character recognition*: reading the text in an image, such as a scanned page or a receipt |
    | **classic NLP task** | a task on language that existed long before LLMs: *sentiment* (is a text positive or negative), *named entities* (the names of people, places, companies and dates in a text), *classification* (sorting a text into labels), summarizing, translating |
    | **label** | one of a fixed set of answers a classification can give; here, what a customer's message is about, such as `pin_blocked` |
    | `"temperature": 0` | in the body of a request: the model takes its most likely next token every time, so the same message gets the same answer |
    | `text.lower()` | the text in small letters: `"Exchange_Rate".lower()` is `"exchange_rate"` |

    ### Ready-made services

    | Task | What goes in, and what comes out | Examples (November 2026) |
    |---|---|---|
    | speech to text | sound; the words | OpenAI's transcription endpoint, Google Cloud Speech-to-Text, Amazon Transcribe |
    | text to speech | text; sound | Google Cloud Text-to-Speech, Amazon Polly, ElevenLabs |
    | translation | text; the same text in another language | DeepL, Google Cloud Translation, Azure AI Translator |
    | seeing and OCR | an image; the objects on it, or the text in it | Google Cloud Vision, Azure AI Vision, Amazon Rekognition and Textract |
    | classic NLP | text; its sentiment, its named entities, its labels | Google Cloud Natural Language, Azure AI Language, Amazon Comprehend |

    Each has its own endpoint and JSON, but the way of calling is the one you know: an HTTP request with the input, and JSON back. A service gives the same kind of answer every time, which is easy for a program to read, and it is usually fast and cheap per request. But it does only its own task, with its own labels.

    ### Or an LLM with a prompt

    An LLM does these tasks with a prompt: "Translate this into Dutch", "What is the sentiment of this review?", or a list of your own labels. The course tried it: BANKING77 is a set of questions of customers of an online bank, each with its label (Casanueva and others, 2020). The course took 20 questions for each of six labels, 120 in all, and sent each to its server with this system prompt, with `"temperature": 0`:

    ```text
    Sort the customer's message into one of these labels, and answer with the label only.
    card_arrival: the new card has not arrived yet
    lost_or_stolen_card: the card is lost or stolen
    exchange_rate: a question about exchange rates
    card_payment_wrong_exchange_rate: a card payment used the wrong exchange rate
    pin_blocked: the PIN is blocked
    request_refund: the customer wants money back for something
    ```

    This is a prompt without examples, *zero-shot*, as in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en). The table shows what Qwen3 answered: each row is a label the questions are about, each column a label Qwen3 gave. The cells on the diagonal, where the two are the same, are the right answers.
    """)
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_table():
    mo.vstack([
        mo.md(r"""
        Qwen3 got 99 of the 120 right, 82.5%, and every answer it wrote was one of the six labels, exactly. Its mistakes sit mostly in one place: of the 20 questions about a card payment with the wrong exchange rate, it put 9 under `exchange_rate`, the label that is a piece of the longer one. Each answer took about 1.0 second on the course's computer, with about 116 tokens in, mostly the system prompt, and 3 tokens out.
        """),
        L.watch_out("A model is not bound to your list: another model, or another prompt, writes `Label: pin_blocked.` or `Pin_Blocked`, or a label that is not on the list. Read the label out of the answer with care, and count it as no label when there is none."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: sort a text into labels with an LLM

    1. **Write a system prompt** that lists the labels, each with one line about what it means, and asks for the label only.
    2. **Send it** as the first message, with the text as the user's message, and `"temperature": 0`.
    3. **Read the label out of the answer:** of the labels that are in the answer, in small letters, take the longest; `None` when there is none.
    4. **Measure it** on texts whose label you know, before you trust it, as the course did with its 120 questions.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One of the 120 questions: *I believe my card payment exchange rate is incorrect.* Its label is `card_payment_wrong_exchange_rate`.

    1. **The request:** the system prompt, then the question as the user's message: 113 tokens went in.
    2. **The answer:** Qwen3 wrote `card_payment_wrong_exchange_rate`, in 1.3 seconds.
    3. **The label:** two labels are in that answer, `exchange_rate` and `card_payment_wrong_exchange_rate`, because the first is a piece of the second. A function that stops at the first label of the list that it finds would say `exchange_rate`, which is wrong. The longest one is right.
    """)
    return


@app.cell(hide_code=True)
def text_parse():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The label in an answer

        Finish `parse_label(answer, labels)`: it gives back the longest label of `labels` that is in `answer`, also when the answer has capitals or words around it, and `None` when no label is in it. The check tries answers with capitals, with words around the label, and with no label. The table under the check reads Qwen3's 120 answers with your function.
        """),
        L.task("Fill in the loop: when `label` is in the answer in small letters, and `best` is still `None` or shorter, `label` becomes the new `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parse():
    box_parse = L.code_box("parse_label", lines=6)
    box_parse
    return (box_parse,)


@app.cell(hide_code=True)
def check_parse(box_parse):
    result_parse = L.check("parse_label", box_parse.value)
    result_parse
    return (result_parse,)


@app.cell(hide_code=True)
def hints_parse():
    L.hints(
        "parse_label",
        [
            "For each label, two questions: is it in the answer, and is it longer than the best one so far?",
            "`label in answer.lower()` says whether the label is in the answer, whatever its capitals; `len(label) > len(best)` whether it is longer, which only works once `best` is not `None` any more.",
            "`if label in answer.lower() and (___ or len(label) > len(best)):`, with `best = label` under it, further in.",
        ],
        why="""```python
def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            best = label
    return best
```

`best is None or ...` lets the first label that is found in, and after that only a longer one; Python does not even look at `len(best)` while `best` is still `None`.""",
    )
    return


@app.cell(hide_code=True)
def text_label():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Sort a message through the API

        `label(url, message)` sends a request and reads the answer already. Fill in the messages, the system prompt `SYSTEM` first and then the customer's message, and give back the label, read with your `parse_label` from exercise 1 out of the text of the answer. `SYSTEM` and `LABELS` are there to use. The check calls your function with a stand-in of the server, which answers once with a label alone and once with words around it.
        """),
        L.task("Make `messages` the system prompt and the message, and return the label that your `parse_label` reads out of the answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label", lines=12)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label, box_parse):
    result_label = L.check("label", box_label.value, C.label_inputs(box_parse.value))
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label",
        [
            "Steps 2 and 3 of the recipe: two messages, and then the label out of the text of the answer.",
            "`messages` is a list of two dictionaries: one with the role `\"system\"` and `SYSTEM`, one with the role `\"user\"` and `message`. The text of the answer sits under `\"choices\"`: in the first choice, its `\"message\"`, and there its `\"content\"`.",
            "The last line: `return parse_label(___, LABELS)`, with the text of the answer in the gap.",
        ],
        why="""```python
def label(url, message):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 20, "temperature": 0}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return parse_label(answer["choices"][0]["message"]["content"], LABELS)
```

The system prompt tells the model the labels; your `parse_label` turns whatever it writes into one of them, or `None`.""",
    )
    return


@app.cell(hide_code=True)
def text_own_labels():
    mo.md(r"""
    ### Exercise 3. A service or an LLM?
    """)
    return


@app.cell(hide_code=True)
def choice_own_labels_cell():
    choice_own_labels = L.choice(
        "own-labels",
        question="A bank wants every message of a customer sorted into six labels it made up itself, such as `pin_blocked`. It has no sorted examples yet. What fits?",
        options={
            "sentiment": "A ready-made sentiment service, which says whether a text is positive or negative",
            "llm": "An LLM with a system prompt that lists the six labels",
            "translate": "A ready-made translation service",
        },
        correct="llm",
        why={
            "llm": "A prompt can name any labels, also ones nobody made before; a ready-made service knows only its own. With sorted examples, a small model of its own also becomes possible, as a later lesson shows.",
            "sentiment": "That service gives its own labels, positive and negative. Does it know `pin_blocked`?",
            "translate": "That turns a message into another language, but does it sort it?",
        },
    )
    choice_own_labels
    return (choice_own_labels,)


@app.cell(hide_code=True)
def text_mixed():
    mo.md(r"""
    ### Exercise 4. Read the table
    """)
    return


@app.cell(hide_code=True)
def choice_mixed_cell():
    choice_mixed = L.choice(
        "mixed",
        question="In the table of the course's run, which mistake did Qwen3 make most often?",
        options={
            "rate": "It answered `exchange_rate` for a card payment with the wrong exchange rate",
            "refund": "It answered `request_refund` for a lost or stolen card",
            "pin": "It answered `pin_blocked` for a card that has not arrived",
        },
        correct="rate",
        why={
            "rate": "9 of those 20 questions got the shorter label; the two are close in meaning, and one is a piece of the other.",
            "refund": "Look at the row of `lost_or_stolen_card`: how many questions are in the column of `request_refund`?",
            "pin": "Look at the row of `card_arrival`: how many questions are in the column of `pin_blocked`?",
        },
    )
    choice_mixed
    return (choice_mixed,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Sort your own message with your label

    Once exercises 1 and 2 are right, your `label` can talk to the real server: the course starts it on your computer, with Qwen3-0.6B behind it. Write a message as a customer of the bank would, and see which label comes back. Try one that fits none of the six, too.
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
def message_box_cell():
    message_box = mo.ui.text(value="I paid in euros in Spain, and the rate on my card was way off.", label="A customer's message", full_width=True)
    send_button = mo.ui.run_button(label="Sort it with your label")
    mo.vstack([message_box, send_button])
    return message_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_label, box_parse, message_box, send_button):
    _label = C.learner_label(box_label.value, box_parse.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _label is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo sorts with your own `label`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to sort the message.*")
    elif not message_box.value.strip():
        _out = mo.md("*Write a message in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `label` has sent the message; the server loads Qwen3, which writes its answer …*"))
        _words = {"title": "The label your label read",
                  "label": "{label}", "no_label": "None: no label in the answer",
                  "wrote": "Qwen3 wrote: {text} ({seconds} seconds)",
                  "failed": "Your label stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_label, message_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The request is the one of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with a system prompt as in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en). In [A large model or a small one?](/course/open/build-llm-or-small-model?lang=en), a small model trained on 50 sorted questions per label takes on Qwen3 on these same 120 questions, in quality, speed and price.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_mixed, choice_own_labels, result_label, result_parse):
    L.wrap_up(result_parse, result_label, choice_own_labels, choice_mixed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
