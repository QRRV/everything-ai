# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-system-prompt. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-system-prompt", "en")
    C = L.checks   # the course's recorded runs, the stand-in of the checks, and the demo with your with_system


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Calling a model over HTTP](/course/open/build-call-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_message():
    quick_message = L.choice(
        "quick-message",
        question="In a request to `/v1/chat/completions`, what is one message of the conversation?",
        options={
            "text": "A text that starts with the role and a colon, such as `user: Hi`",
            "dict": "A dictionary with `\"role\"` and `\"content\"`",
            "ids": "A list of the token IDs of the message",
        },
        correct="dict",
        why={
            "dict": "Such as `{\"role\": \"user\", \"content\": \"Hi\"}`; the conversation is a list of them, in `\"messages\"`.",
            "text": "In the end the model gets the conversation as one text, but the server writes it out; the request holds the messages apart. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "ids": "The server makes the tokens itself; the request holds text. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_message
    return (quick_message,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A question alone rarely makes an app. A shop's assistant should answer briefly and stay on its subject, in every conversation, whatever the customer types first. And when a program reads the answer, the answer has to start the way the program expects. Neither belongs in what the user types: the app adds it. The system prompt holds the instructions for the whole conversation, and a response prefill writes the first words of the answer.

    **An everyday example.** Before a shift, someone new at the counter of a shop gets a briefing: answer only questions about bikes, and keep it short. Every customer asks something else; the briefing holds for all of them, and the customers never hear it. And on a form where *Name:* is already printed, what follows is a name, not a story. That is only a comparison; below you see what a system prompt and a prefill did to Qwen3's answers, and where they fell short.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **system prompt** | the instructions in the message with the role `system`, at the start of the conversation; the builder of the app writes them, the user does not see them, and they hold for the whole conversation; they go in with every request, as input tokens |
    | **response prefill** | the first words of the model's answer, written by you: a last message with the role `assistant`, which the model goes on from; the answer holds only what the model writes after them |

    Some APIs say *system instructions* or *developer message* for the system prompt (examples in November 2026); the idea is the same.

    ### A short system prompt

    The course asked Qwen3 through its own server *What is a neural network?*, four times without a system prompt and four times with one. The server draws each next token from the most likely ones, so four answers to the same question differ a little:
    """)
    return


@app.cell(hide_code=True)
def picture_brief():
    mo.Html(C.brief_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_brief():
    _f = C.findings()
    mo.md(r"""
    Without a system prompt, all four answers ran on to `max_tokens` 200 and were cut off, in the middle of headings and lists: by itself, Qwen3 explains at length. With *Answer in one sentence of at most 20 words.* they had {lo} to {hi} tokens, and Qwen3 ended all four itself. It did not count well: {within} of the four stayed within 20 words, and one had {most}. The system prompt is not free either: it goes in with every request, so the question was {plain} input tokens without it, and {brief} with it.

    ### A system prompt that sets limits

    A system prompt can also say what the assistant should not do. The course gave Qwen3 the system prompt of a bike shop, and asked two questions four times each: one about bikes, and one for a poem:
    """.format(lo=min(_f["brief_tokens"]), hi=max(_f["brief_tokens"]), within=sum(w <= 20 for w in _f["brief_words"]),
               most=max(_f["brief_words"]), plain=_f["prompt_tokens"][0], brief=_f["prompt_tokens"][1]))
    return


@app.cell(hide_code=True)
def picture_shop():
    mo.Html(C.shop_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_shop():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Qwen3 wrote the poem in {poem} of the {runs} runs, and refused {bike} of the {runs} questions about bikes, which it should have answered: the rule did not hold, in either direction. What it said about tires is not much good either; a model this small knows little.
        """.format(poem=_f["runs"] - _f["poem_refused"], bike=_f["bike_refused"], runs=_f["runs"])),
        L.watch_out("A system prompt is not a lock, and not a secret. A user can type something that makes a model ignore it, and can often get the model to repeat it word for word. What must hold, such as who may see what, belongs in the code of your app, not in a prompt."),
    ])
    return


@app.cell(hide_code=True)
def text_prefill():
    mo.md(r"""
    ### Response prefill

    In [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en) you saw that the model gets the conversation as one text, which ends with the assistant's marker, so that what comes next is the answer. When the last message of your conversation has the role `assistant`, the course's server writes that message after the marker and leaves the turn open: for the model, its answer has already begun with your words, and it goes on from there.
    """)
    return


@app.cell(hide_code=True)
def picture_prompt():
    mo.Html(C.prompt_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_prefill_runs():
    mo.md(r"""
    The answer that comes back holds only what the model wrote after your words, `Paris"}` here, so your code puts the two together. The course asked Qwen3 for the capital of France as JSON, four times without and four times with the prefill `{"city": "`:
    """)
    return


@app.cell(hide_code=True)
def picture_json():
    mo.Html(C.json_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_json_after():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Without the prefill, {fenced} of the four answers were in a code block with backticks, which `json.loads` cannot read, and {capital} used the key `capital` instead of `city`: none of the four could be read as asked. With the prefill, all four could: the opening brace and the key were already written, so the model only had to fill in the rest. A prefill decides how the answer starts, not what it says; the next lesson checks the whole of a JSON answer.
        """.format(fenced=_f["fenced"], capital=_f["capital"])),
        L.watch_out("Not every API takes a response prefill. The server of llama.cpp reads a last message of the assistant the same way as the course's server. OpenAI's API has no prefill. Anthropic's API took one for its older Claude models, and refuses a request that ends with a message of the assistant for its newer ones (examples in November 2026). Without a prefill, a clear system prompt and a check of the answer in code have to do it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: system prompt and response prefill

    1. **For the whole conversation:** put the instructions in a first message with the role `system`. It goes in with every request, so keep it short and concrete: what to do, and in what form.
    2. **One system prompt:** when the conversation already has one, put the new one in its place, not next to it.
    3. **For how the answer starts:** add a last message with the role `assistant` that holds those first words, and put them in front of the answer that comes back.
    4. **Check in code what must hold:** a system prompt is no lock, and a prefill fixes only the start.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An app gives every text that a user sends a short title, as JSON that a program reads: `{"title": ...}`. Which messages?

    ```python
    messages = [
        {"role": "system", "content": "Make a title of at most five words for the text the user sends."},
        {"role": "user", "content": "Yesterday it rained all day, so we stayed in, baked bread and played cards."},
        {"role": "assistant", "content": '{"title": "'},
    ]
    ```

    1. **For the whole conversation:** the system prompt says what every answer must be, whatever text comes.
    2. **One system prompt:** this conversation has no other.
    3. **How the answer starts:** the prefill `{"title": "`. Qwen3 answered `Rainy Day"}`, with `finish_reason` `"stop"`, and the app puts the two together: `'{"title": "' + 'Rainy Day"}'` is `{"title": "Rainy Day"}`.
    4. **Check in code:** `json.loads` reads it, and the title has 2 words, within the five.
    """)
    return


@app.cell(hide_code=True)
def text_with():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A system prompt at the start

        An app often keeps the conversation as a list, and adds the system prompt only when it sends it: then it can change the system prompt without touching the conversation. Write `with_system(messages, system)`, which gives back a new list: first a message with the role `system` and the text `system`, then the messages of `messages`, without a system prompt that the conversation already had. The box starts with a function that copies the list.
        """),
        L.task("Start `new` with the system prompt as a message, and let the loop copy only the messages whose role is not `system`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_with():
    box_with = L.code_box("with_system", lines=8)
    box_with
    return (box_with,)


@app.cell(hide_code=True)
def check_with(box_with):
    result_with = L.check("with_system", box_with.value)
    result_with
    return (result_with,)


@app.cell(hide_code=True)
def hints_with():
    L.hints(
        "with_system",
        [
            "Two changes to the box: what `new` starts with, and which messages the loop copies.",
            "`new` starts as a list with one dictionary in it, with the keys `\"role\"` and `\"content\"`. In the loop, an `if` on `message[\"role\"]` decides whether a message goes into `new`.",
            "`new = [{\"role\": \"system\", \"content\": system}]`, and in the loop `if message[\"role\"] ___:` above `new.append(message)`, which moves one step further in.",
        ],
        why="""```python
def with_system(messages, system):
    new = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] != "system":
            new.append(message)
    return new
```

The function makes a new list and leaves `messages` as it was, so the app can send the same conversation again with another system prompt.""",
    )
    return


@app.cell(hide_code=True)
def text_json_exercise():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. JSON with a response prefill

        `ask_json(url, question, start)` asks a question whose answer must be JSON. Send `start` as a response prefill, and give back the whole JSON text: `start`, then what the model wrote after it. The check calls your `ask_json` against a stand-in of the server, which goes on after a prefill as the course's server does.
        """),
        L.task("Add the prefill as the last message, and put `start` in front of the text that comes back."),
    ])
    return


@app.cell(hide_code=True)
def exercise_json():
    box_json = L.code_box("ask_json", lines=13)
    box_json
    return (box_json,)


@app.cell(hide_code=True)
def check_json(box_json):
    result_json = L.check("ask_json", box_json.value)
    result_json
    return (result_json,)


@app.cell(hide_code=True)
def hints_json():
    L.hints(
        "ask_json",
        [
            "The prefill is one more message in `messages`, after the question. The answer that comes back holds only what comes after it.",
            "The prefill is a message with the role `assistant` and `start` as its content, the second item of the list. The last line puts `start +` in front of the text.",
            "`messages = [{\"role\": \"user\", \"content\": question}, ___]`, and `start +` at the front of the last line.",
        ],
        why="""```python
def ask_json(url, question, start):
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": start}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 40}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return start + answer["choices"][0]["message"]["content"]
```

The model goes on from `start`, and the answer holds only the rest; together they are the JSON text.""",
    )
    return


@app.cell(hide_code=True)
def text_secret():
    mo.md(r"""
    ### Exercise 3. A code for regular customers
    """)
    return


@app.cell(hide_code=True)
def choice_secret_cell():
    choice_secret = L.choice(
        "secret",
        question="The assistant of your shop may give a discount code only to customers who have ordered before. Where does the code go?",
        options={
            "system": "In the system prompt, with the rule *Give it only to customers who ordered before*",
            "prefill": "In a response prefill, where the user who types never sees it",
            "code": "Not in a prompt: your app's code checks the orders, and gives the code",
        },
        correct="code",
        why={
            "code": "A system prompt is no lock: the bike shop's model wrote the poem anyway, and a user can often get a model to repeat its system prompt. What must hold, your own code does.",
            "system": "The bike shop's system prompt said *only bikes*, and Qwen3 wrote the poem in every run. A rule in a prompt asks the model something; it locks nothing. Where can no prompt reach the code?",
            "prefill": "The model reads a prefill like the rest of the prompt, so it can simply repeat the code in its answer; and the prefill is the start of the answer, which your app may show. Where can no prompt reach the code?",
        },
    )
    choice_secret
    return (choice_secret,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 1 is right, you can try system prompts and prefills on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo puts your system prompt at the start with your own `with_system`, and adds the start of the answer as a prefill when you write one. Try the bike shop with other questions, or a question for JSON with a prefill.
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
def question_box_cell():
    system_box = mo.ui.text_area(value=C.SHOP, label="System prompt (may stay empty)", full_width=True, rows=3)
    question_box = mo.ui.text(value=C.POEM, label="Your question", full_width=True)
    start_box = mo.ui.text(value="", label="Start of the answer, the prefill (may stay empty)", full_width=True)
    send_button = mo.ui.run_button(label="Send with your with_system")
    mo.vstack([system_box, question_box, start_box, send_button])
    return question_box, send_button, start_box, system_box


@app.cell(hide_code=True)
def demo_answer(box_with, question_box, send_button, start_box, system_box):
    _with = C.learner_with_system(box_with.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _with is None:
        _out = mo.md("*Finish exercise 1 first: the demo puts the system prompt in with your own `with_system`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to send.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The messages have gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "What went to Qwen3, and what came back", "sent": "The messages:",
                  "answer": "The answer, with your prefill in bold:",
                  "done": "{given} tokens in, {written} out, finish_reason \"{finish}\", in {seconds} seconds.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_with, system_box.value, question_box.value, start_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The role `system` of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en) is now a message that your app adds to every request of [Calling a model over HTTP](/course/open/build-call-api?lang=en), and the one text of [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en) explains why a prefill works: the model goes on from wherever the text stops. In the next lesson, [JSON you can rely on](/course/open/build-structured-output?lang=en), you check a JSON answer against what it must hold, and ask again when it does not fit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_secret, result_json, result_with):
    L.wrap_up(result_with, result_json, choice_secret)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
