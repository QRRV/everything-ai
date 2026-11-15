# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-project-call. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-project-call", "en")
    C = L.checks   # the course's runs with and without the conversation, the stand-ins, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Calling a model over HTTP](/course/open/build-call-api?lang=en), and one about [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_content():
    quick_content = L.choice(
        "quick-content",
        question="`answer = json.loads(response.read())` holds the JSON a model's API sent back. Where is the text of the answer?",
        options={
            "usage": "`answer[\"usage\"]`",
            "content": "`answer[\"choices\"][0][\"message\"][\"content\"]`",
            "text": "`answer[\"text\"]`",
        },
        correct="content",
        why={
            "content": "`choices` is a list of answers; the first has the `message`, and its `content` is the text.",
            "usage": "`usage` counts the tokens that went in and came out. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "text": "The answer has no key `\"text\"` at the top. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_content
    return (quick_content,)


@app.cell(hide_code=True)
def quick_check_module():
    quick_module = L.choice(
        "quick-module",
        question="In the course, where does `from tools import stats` find its module?",
        options={
            "projects": "In `work/projects/tools/stats.py`",
            "lessons": "In the folder of the lesson that made it",
            "python": "Among the modules that come with Python",
        },
        correct="projects",
        why={
            "projects": "Your packages live in `work/projects/`, and Python looks there when you import, in every lesson.",
            "lessons": "Lessons can save code, but not next to themselves. See [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
            "python": "`tools` is a package of your own, not one of Python's. See [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
        },
    )
    quick_module
    return (quick_module,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        An assistant does not answer one question; it holds a conversation. But a model keeps nothing between two requests, as you saw in [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en): what it knows of the conversation is what the request carries. So your program keeps the conversation, and sends all of it, every turn.

        This lesson starts your project: an assistant of your own, in the package `app`. Today it gets its first module, the client: one function that talks to the API. Later steps give it a good prompt, and documents to read.

        **An everyday example.** You call a help desk, and get someone new each time you call. If you start with "it is about my order of last week, the one that came broken", they can help you with it. If you only say "so, what now?", they cannot. That is only a comparison; below you see it happen with Qwen3.
        """),
        draw.project_steps(["Your assistant answers", "Your assistant, prompted well", "Your assistant reads your documents"], 0,
                           here_label="← today", title="Where your project stands: your assistant"),
    ])
    return


@app.cell(hide_code=True)
def predict_name():
    guess_name = L.guess(
        "name",
        question="The course told Qwen3-0.6B *My name is Sam* and asked for a dinner idea. Then it asked *What is my name?* in a new request, once with the first question and answer in it, and once alone. Eight runs each. How often did the answer name Sam?",
        options={
            "both": "Almost always, both ways",
            "with": "8 of 8 with the conversation, 0 of 8 alone",
            "neither": "Rarely, both ways: the model is too small",
        },
        reveal=f"**8 of 8 with the conversation, 0 of 8 alone.** Alone, Qwen3 said it is not a person, or a language model without a name: the request held no name. With the conversation, it named Sam every time, but {C.findings()['own']} of the 8 answered *I am Sam.* or *My name is Sam.*, as if the model were Sam, and none said *Your name is Sam.*: a small model mixes up who said what.",
    )
    guess_name
    return (guess_name,)


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **API client** | the part of a program that talks to an API: a function that takes what to ask and gives back the answer, so the rest of the program never sees a URL, HTTP or JSON |

    ### One function between your program and the model

    Your `ask` from [Calling a model over HTTP](/course/open/build-call-api?lang=en) sent one question. The client sends a whole conversation: the list `messages` that the rest of your program keeps. Everything about HTTP stays inside it: switch to another server, or to a hosted API with a key, and you change the client, not the rest.

    The conversation grows with every turn. In the course's runs, the first request had {C.findings()['tokens_first']} input tokens, and the second, with the first question and its answer in it, {C.findings()['tokens_second']}. Every turn sends everything before it again, and a hosted API counts those tokens every time.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a client, and a conversation

    1. **The client:** `chat(url, messages, max_tokens=200)` puts `messages` and `max_tokens` in the body, sends it as in your `ask`, and gives back the text of the answer. The `=200` gives `max_tokens` a default: `chat(url, messages)` asks for 200 tokens, and `chat(url, messages, 4)` for 4.
    2. **The conversation:** one list `messages`, made once.
    3. **Each turn:** add the user's message, `{"role": "user", "content": text}`, and call `chat(url, messages)`.
    4. **After each answer:** add it as `{"role": "assistant", "content": answer}`, so the next request carries it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course's first run, seed 0, by hand:

    ```python
    messages = [{"role": "user", "content": "My name is Sam. Suggest a dinner for tonight, in one sentence."}]
    answer = chat(url, messages)
    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": "What is my name?"})
    chat(url, messages)
    ```

    1. **First turn:** one message goes out; the answer is *Sam's dinner could be a cozy meal with a classic blend of flavors, such as a classic steak and a side salad.*
    2. **Keep it:** the answer goes into `messages` as the assistant's.
    3. **Second turn:** the question goes in too, so the second request carries three messages.
    4. **The answer:** *I am Sam.* The name is there, because the request carried it; who is who, a model this small can still mix up.
    """)
    return


@app.cell(hide_code=True)
def text_chat():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The client

        Write `chat(url, messages, max_tokens=200)`: it sends the whole list `messages`, with `max_tokens`, to `url + "/chat/completions"`, and gives back the text of the answer. Start from your `ask`. The check calls it with a conversation of three messages and with a small `max_tokens`, against a stand-in of the server. When it is right, the check saves it in your project, as `work/projects/app/client.py`.
        """),
        L.task("Write `chat`: the whole `messages` and `max_tokens` in the body, and the text of the answer back."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chat():
    box_chat = L.code_box("chat", lines=11)
    box_chat
    return (box_chat,)


@app.cell(hide_code=True)
def check_chat(box_chat):
    result_chat = L.check("chat", box_chat.value)
    result_chat
    return (result_chat,)


@app.cell(hide_code=True)
def hints_chat():
    L.hints(
        "chat",
        [
            "It is your `ask`, with two changes in the body: what goes in `\"messages\"`, and in `\"max_tokens\"`.",
            "`body = {\"model\": \"qwen3-0.6b\", \"messages\": messages, \"max_tokens\": max_tokens}`; the rest is as in `ask`.",
            "After the body: `data = json.dumps(body).encode()`, a `Request` to `url + \"/chat/completions\"` with `data=data` and the header, `urlopen`, `json.loads`, and `return answer[\"choices\"][0][\"message\"][\"content\"]`.",
        ],
        why="""```python
def chat(url, messages, max_tokens=200):
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": max_tokens}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

The list goes into the body as it is: the whole conversation, every time.""",
    )
    return


@app.cell(hide_code=True)
def text_talk():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A conversation of several turns

        `talk(url, said)` holds a conversation: for each `text` in `said`, the user's turn, it asks your `chat`, and it gives back the list of the answers. It imports `chat` from your project. The check gives it three turns; the last one, *What do I like?*, can only be answered with the earlier turns in the request. The picture shows every request your `talk` sent.
        """),
        L.task("Fill in the loop: add the user's turn, ask `chat`, and add the answer to `messages` and to `answers`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_talk():
    box_talk = L.code_box("talk", lines=10)
    box_talk
    return (box_talk,)


@app.cell(hide_code=True)
def check_talk(box_talk):
    result_talk = L.check("talk", box_talk.value)
    result_talk
    return (result_talk,)


@app.cell(hide_code=True)
def hints_talk():
    L.hints(
        "talk",
        [
            "Steps 3 and 4 of the recipe, inside the loop: four lines.",
            "`messages.append({\"role\": \"user\", \"content\": text})`, then `answer = chat(url, messages)`.",
            "Then `messages.append({\"role\": \"assistant\", \"content\": answer})` and `answers.append(answer)`, in place of the `...`.",
        ],
        why="""```python
def talk(url, said):
    messages = []
    answers = []
    for text in said:
        messages.append({"role": "user", "content": text})
        answer = chat(url, messages)
        messages.append({"role": "assistant", "content": answer})
        answers.append(answer)
    return answers
```

The list is made once, before the loop, and grows by two messages each turn: the third request carries five.""",
    )
    return


@app.cell(hide_code=True)
def text_why_all():
    mo.md(r"""
    ### Exercise 3. Why send it all?
    """)
    return


@app.cell(hide_code=True)
def choice_why_all_cell():
    choice_why_all = L.choice(
        "why-all",
        question="Your assistant sends the first question and its answer along with the second question. Why?",
        options={
            "faster": "So that the model answers faster",
            "memory": "Because the model keeps nothing between requests: it only knows the conversation from what the request carries",
            "server": "Because the server throws the conversation away after an hour",
        },
        correct="memory",
        why={
            "memory": "Each request starts from nothing; the conversation exists only in your program, which sends it along. That is also why every turn costs more tokens than the one before.",
            "faster": "The other way round: more tokens to read take longer. What would the model know of the first turn without them?",
            "server": "The server never kept it in the first place: the model reads each request on its own. Where does the conversation live, then?",
        },
    )
    choice_why_all
    return (choice_why_all,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Talk to Qwen3 through your client

    The cell takes `chat` from your project, `work/projects/app/client.py`, and holds a conversation with Qwen3-0.6B on the course's server. Each turn sends everything before it. *Start over* empties the conversation.
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
def conversation_state():
    get_talk, set_talk = mo.state([])
    return get_talk, set_talk


@app.cell(hide_code=True)
def turn_box_cell():
    turn_box = mo.ui.text(value="My name is Sam. Suggest a dinner for tonight, in one sentence.", label="Your turn", full_width=True)
    send_button = mo.ui.run_button(label="Send")
    again_button = mo.ui.run_button(label="Start over")
    mo.vstack([turn_box, mo.hstack([send_button, again_button], justify="start")])
    return again_button, send_button, turn_box


@app.cell(hide_code=True)
def demo_turn(again_button, get_talk, result_chat, send_button, set_talk, turn_box):
    _chat = C.learner_chat() if result_chat.state == "correct" else None
    if again_button.value:
        set_talk([])
        _out = mo.md("*The conversation is empty again.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _chat is None:
        _out = mo.md("*Finish exercise 1 first: then your `chat` is in your project, and this cell uses it.*")
    elif not send_button.value:
        _out = mo.md("*Press Send to start the conversation.*")
    elif not turn_box.value.strip():
        _out = mo.md("*Write your turn in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `chat` sends the conversation; Qwen3 writes …*"))
        _words = {"title": "Your conversation with Qwen3", "done": "{seconds} seconds; the request carried {n} messages.",
                  "failed": "Your chat stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _after, _html = C.demo_turn(_chat, get_talk(), turn_box.value, _words, L.lang)
        set_talk(_after)
        _out = mo.Html(_html)
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The client is your `ask` of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with the conversation of [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en) in it, kept in your project as in [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en). Every turn sends more tokens, which is what [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en) counted. Your project comes back in *Project: your assistant, prompted well*, which gives it a system prompt and answers as JSON. First, the next lessons let the answer come in while it is written, and deal with a server that says no.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why_all, result_chat, result_talk):
    L.wrap_up(result_chat, result_talk, choice_why_all)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
