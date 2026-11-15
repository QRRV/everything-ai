# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-call-api. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-call-api", "en")
    C = L.checks   # the course's recorded call, the stand-in of the checks, and the demo with your ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Data as text: JSON](/course/open/py-json?lang=en), and one about [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loads():
    quick_loads = L.choice(
        "quick-loads",
        question="What does `json.loads('{\"n\": 3}')` give?",
        options={
            "text": "The text `'{\"n\": 3}'`, unchanged",
            "dict": "The dictionary `{'n': 3}`",
            "error": "A `JSONDecodeError`",
        },
        correct="dict",
        why={
            "dict": "`json.loads` turns JSON text into Python values: a JSON object becomes a dictionary.",
            "text": "That is what `json.dumps` would do the other way round. What does `loads` turn JSON text into? See [Data as text: JSON](/course/open/py-json?lang=en).",
            "error": "The text is valid JSON: double quotes around the key, a number as the value. See [Data as text: JSON](/course/open/py-json?lang=en).",
        },
    )
    quick_loads
    return (quick_loads,)


@app.cell(hide_code=True)
def quick_check_messages():
    quick_messages = L.choice(
        "quick-messages",
        question="How do you write a conversation for a chat model in code?",
        options={
            "string": "As one string with the whole conversation",
            "people": "As a dictionary with a key for each person",
            "list": "As a list of dictionaries, each with a `\"role\"` and a `\"content\"`",
        },
        correct="list",
        why={
            "list": "Each message is a dictionary with its role, `\"user\"`, `\"assistant\"` or `\"system\"`, and its content, in the order they were said.",
            "string": "The model does get one text in the end, but your code does not glue it together. See [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
            "people": "A dictionary has no order of turns, and one person says many things. See [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
        },
    )
    quick_messages
    return (quick_messages,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An app with a model in it, a chat on a website or a tool in your editor, rarely runs the model itself. It sends the question over the network to a server that runs the model, and gets the answer back. The way to ask is fixed, and almost every server for language models speaks the same one: the form of OpenAI's API. So the same ten lines reach a model in a data center on the other side of the world, or Qwen3-0.6B on your own computer.

    **An everyday example.** At the counter of a canteen, you fill in an order form: what you want, and how much. The kitchen makes it, and you get a tray back with a receipt on it. You never go into the kitchen, and any kitchen that takes the same form can serve you. That is only a comparison; below you see the real form, and what comes back.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **API** | *application programming interface*: a fixed way for one program to ask another for something; for a model, you send messages and get an answer |
    | **HTTP request** | a message to a server over the network: a *method* (GET to fetch something, POST to send something), an address, *headers* with facts about the request, such as what the body is, and a *body* |
    | **endpoint** | one address of an API, for one job: `/v1/chat/completions` for answers in a chat, `/v1/models` for the models a server has |
    | **local server** | a server on your own computer, at the address `127.0.0.1`, also called *localhost*, and a port number such as `8765`; nothing goes over the internet |
    | **OpenAI-compatible API** | an API with the endpoints and the JSON of OpenAI's API; Ollama, llama.cpp, vLLM, many hosted APIs and the course's own server speak it (examples in November 2026) |
    | `urllib.request` | Python's own module for HTTP: `Request(url, data=..., headers=...)` makes a request, `urlopen(request)` sends it and gives the response |

    ### What goes over the line

    The course started its own server on its computer, with Qwen3-0.6B behind it, and sent it one question. This is the request, and the answer the server sent back:
    """)
    return


@app.cell(hide_code=True)
def picture_call():
    mo.Html(C.call_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_parts():
    mo.vstack([
        mo.md(r"""
        The request is a POST to the endpoint `/v1/chat/completions`, with one header: the body is JSON. The body says which model, which messages, and at most how many tokens to write. In the answer, `choices` is a list, because a request can ask for more than one answer; the first one, `[0]`, has the `message` with its `content`, the text. `finish_reason` says why the model stopped: `"stop"` when it ended the answer itself, `"length"` when it reached `max_tokens`. And `usage` counts the tokens: 19 went in, 13 came out, what you pay for at a hosted API, as in [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en).

        A request goes over the network as bytes, not as Python text. `.encode()` turns a string into bytes; `response.read()` gives the answer as bytes, and `json.loads` reads those directly.
        """),
        L.watch_out("The course's server listens only at `127.0.0.1`: no other computer can reach it, so it needs no key. A hosted API is open to the whole internet, and wants a key with every request; a later lesson shows how to keep it out of your code."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: call a model's API with urllib

    1. **Make the body:** a dictionary with `"model"`, `"messages"` and `"max_tokens"`.
    2. **Turn it into JSON bytes:** `json.dumps(body).encode()`.
    3. **Make the request:** `urllib.request.Request(url + "/chat/completions", data=..., headers={"Content-Type": "application/json"})`. With `data=` it is a POST.
    4. **Send it, and read the answer:** `with urllib.request.urlopen(request) as response:` and then `answer = json.loads(response.read())`.
    5. **Take the text out:** `answer["choices"][0]["message"]["content"]`.

    Outside the course, you often use OpenAI's package `openai` for this (an example in November 2026), which does these steps for you: with `OpenAI(base_url=url, api_key=...)` it talks to any OpenAI-compatible server. Underneath, it sends this same request.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Which models does the course's server have? That is the endpoint `/v1/models`, and fetching is a GET: a request without `data=`.

    ```python
    import json
    import urllib.request

    url = "http://127.0.0.1:8765/v1"
    with urllib.request.urlopen(url + "/models") as response:
        models = json.loads(response.read())
    models["data"][0]["id"]
    ```

    1. **The address:** `url` is where the server listens, ending in `/v1`; the endpoint goes after it.
    2. **The request:** `urlopen` takes an address too, and then makes a GET of it.
    3. **The answer:** `{"object": "list", "data": [{"id": "qwen3-0.6b", "object": "model", "owned_by": "everything-ai"}, {"id": "granite-embedding-107m-multilingual", "object": "model", "owned_by": "everything-ai"}]}`, as bytes, which `json.loads` turns into a dictionary.
    4. **The value:** the list under `"data"` has two models; the first `"id"` is `'qwen3-0.6b'`, the chat model, the name you put in the body of a request. The other one turns texts into embeddings, as a later lesson shows.
    """)
    return


@app.cell(hide_code=True)
def text_body():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The body of a request

        Fill in `body`: the request for `question`, to the model `"qwen3-0.6b"`, with at most 50 tokens. The check also tries another question.
        """),
        L.task("Give `body` the keys `\"model\"`, `\"messages\"` and `\"max_tokens\"`, with `question` as the user's message."),
    ])
    return


@app.cell(hide_code=True)
def exercise_body():
    box_body = L.code_box("body", lines=3)
    box_body
    return (box_body,)


@app.cell(hide_code=True)
def check_body(box_body):
    result_body = L.check("body", box_body.value)
    result_body
    return (result_body,)


@app.cell(hide_code=True)
def hints_body():
    L.hints(
        "body",
        [
            "Look at the body in the picture of the course's call: it has three keys.",
            "`\"messages\"` is a list with one dictionary in it: `{\"role\": \"user\", \"content\": question}`.",
            "`body = {\"model\": \"qwen3-0.6b\", \"messages\": [...], \"max_tokens\": 50}`, with the message from hint 2 in the list.",
        ],
        why="`body = {\"model\": \"qwen3-0.6b\", \"messages\": [{\"role\": \"user\", \"content\": question}], \"max_tokens\": 50}`: the model, the conversation as a list of messages, and the most tokens the answer may have.",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Send it, and read the answer

        `ask(url, question)` makes the body already. Send it to `url + "/chat/completions"` and give back the text of the answer, as in the recipe. The check calls your `ask` with a stand-in of the server, which has the same API and answers at once; the picture under the check shows what your function sent, and what came back.
        """),
        L.task("Finish `ask`: send the body as a POST, and return only the text of the answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=10)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask",
        [
            "Steps 2 to 5 of the recipe: bytes, a request, send and read, the text.",
            "`data = json.dumps(body).encode()`, then `request = urllib.request.Request(url + \"/chat/completions\", data=data, headers={\"Content-Type\": \"application/json\"})`.",
            "`with urllib.request.urlopen(request) as response:` and on the next line, further in, `answer = json.loads(response.read())`; end with `return answer[\"choices\"][0][\"message\"][\"content\"]`.",
        ],
        why="""```python
def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

With `data=` the request is a POST; the server sends JSON back, and the text is in the first choice, in its message.""",
    )
    return


@app.cell(hide_code=True)
def text_finish():
    mo.md(r"""
    ### Exercise 3. Why did it stop?
    """)
    return


@app.cell(hide_code=True)
def choice_finish_cell():
    choice_finish = L.choice(
        "finish",
        question="You ask for a story with `\"max_tokens\": 50`. The answer ends in the middle of a sentence, and its `finish_reason` is `\"length\"`. What happened?",
        options={
            "done": "The model wrote a long answer, and finished it",
            "error": "The server stopped with an error",
            "limit": "The model reached 50 tokens, and the server stopped it there",
        },
        correct="limit",
        why={
            "limit": "`\"length\"` means the answer reached `max_tokens`; `\"stop\"` would mean the model ended it itself. Ask for more tokens, or for a shorter story.",
            "done": "Then it would say `\"stop\"`: the model ended the answer itself. What does `\"length\"` point to?",
            "error": "An error comes back as an error status, with no answer at all. Here there is an answer, cut off. Which number in the body limits it?",
        },
    )
    choice_finish
    return (choice_finish,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Ask Qwen3 with your own ask

    Once exercise 2 is right, your `ask` can talk to the real thing: the course starts its own server on your computer, with Qwen3-0.6B behind it, and your function sends it your question. Watch what you see while you wait.
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
    question_box = mo.ui.text(value="Name three colors of the rainbow.", label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Send it with your ask")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_ask, question_box, send_button):
    _ask = C.learner_ask(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _ask is None:
        _out = mo.md("*Finish exercise 2 first: the demo sends your question with your own `ask`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to send your question.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `ask` has sent the question; the server loads Qwen3, which writes the whole answer before anything comes back …*"))
        _words = {"title": "What came back to your ask",
                  "done": "{seconds} seconds, through {url}. Nothing showed until the whole answer was there: the lesson on streaming changes that.",
                  "failed": "Your ask stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_ask, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The `usage` in every answer counts the tokens of [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en), and the messages are the conversation of [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en), in JSON as in [Data as text: JSON](/course/open/py-json?lang=en). In the next lesson, [Project: your assistant answers](/course/open/build-project-call?lang=en), your `ask` becomes the client of your own assistant, which sends a whole conversation. Later lessons let the answer come in while it is written, and deal with a server that says no.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_finish, result_ask, result_body):
    L.wrap_up(result_body, result_ask, choice_finish)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
