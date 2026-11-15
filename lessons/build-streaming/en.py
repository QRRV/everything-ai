# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-streaming. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-streaming", "en")
    C = L.checks   # the course's recorded streams, the stand-in of the checks, and the demo with your piece


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
def quick_check_text():
    quick_text = L.choice(
        "quick-text",
        question="Where is the text of the answer, in the JSON `answer` a server sends back?",
        options={
            "text": "`answer[\"text\"]`",
            "choices": "`answer[\"choices\"][0][\"message\"][\"content\"]`",
            "usage": "`answer[\"usage\"][\"content\"]`",
        },
        correct="choices",
        why={
            "choices": "`choices` is a list, because a request can ask for more than one answer; the first one has the message, and its `content` is the text.",
            "text": "The answer has no key `\"text\"` at the top. Which list holds the answers? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "usage": "`usage` only counts the tokens in and out. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_text
    return (quick_text,)


@app.cell(hide_code=True)
def quick_check_wait():
    quick_wait = L.choice(
        "quick-wait",
        question="Your `ask` from that lesson sends a question. When does it get the first word of the answer?",
        options={
            "end": "Only when the whole answer is written",
            "start": "As soon as the model writes it",
            "before": "Before the model starts writing",
        },
        correct="end",
        why={
            "end": "`response.read()` reads the whole answer, and the server sends it only when the model is done: one JSON with all of it.",
            "start": "Then the server would have to send the answer in pieces, and `ask` would have to read them one at a time. What does `response.read()` read? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "before": "The model has to write the words first. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_wait
    return (quick_wait,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model writes its answer one token at a time, and a long answer takes seconds, on a small computer more than ten. Your `ask` waits all that time and shows nothing, and then everything at once. Every chat app does it differently: the first words appear almost at once, and the rest follows while you read. That is not a faster model; the server sends each piece as soon as it is written, and the app shows it.

    **An everyday example.** A friend tells you a story on the phone: you hear each sentence when it is said. With a voice message, you hear nothing until your friend is done, and then the whole story. The story takes just as long; on the phone you only start sooner. That is only a comparison; below you see when the pieces of a real answer came in.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **streaming** | getting a model's answer piece by piece while it is still being written, instead of waiting for the whole answer |
    | **server-sent events** | *SSE*: a way for a server to keep sending: the response stays open, and every *event* is a line that starts with `data: `, with an empty line after it |
    | `"stream": True` | in the body of a request: send the answer as server-sent events |
    | **delta** | in an event, the new piece only: `{"content": " keeper"}`, not the whole message so far |
    | `data: [DONE]` | the last line of the stream: the answer is complete |
    | `.decode()` | turns bytes back into text, the other way round from `.encode()` |

    ### When the pieces came in

    The course asked Qwen3-0.6B through its own server for a short story, with `"stream": True`, three times. Each tick is a piece of the first answer, at the second it came in:
    """)
    return


@app.cell(hide_code=True)
def picture_timeline():
    mo.Html(C.timeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_timeline():
    _n = C.numbers(L.lang)
    mo.md(rf"""
    The first piece came after {_n['first']} seconds, the end after {_n['seconds']}: {_n['pieces']} pieces, for the {_n['tokens']} tokens of the answer. Without a stream, the same answer shows nothing for those {_n['seconds']} seconds. Over the course's {_n['runs']} runs, the first piece always came within {_n['first_max']} seconds, and no answer was complete in less than {_n['seconds_min']}. That is the gain: the wait until the first text, not the whole answer.
    """)
    return


@app.cell(hide_code=True)
def text_lines():
    mo.md(r"""
    ### What comes over the line

    This is the start and the end of that stream, as the server sent it. Every event is one line: `data: `, and then a piece of JSON.
    """)
    return


@app.cell(hide_code=True)
def picture_lines():
    mo.Html(C.lines_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_parts():
    mo.vstack([
        mo.md(r"""
        Each event has the form of an answer, with `choices`, but instead of a `message` it has a `delta`: only what is new. The first delta has the role and no text yet; then come the pieces, each in `"content"`; the last delta is empty, with the `finish_reason` beside it. With `"stream_options": {"include_usage": True}` in the body, one more event counts the tokens. And `data: [DONE]` closes the stream.

        Your code reads a stream the way it reads a file: a `for` loop over the response gives one line at a time, each ending in `"\n"`, and an empty line after every event. Each line comes in as bytes, so `.decode()` makes it text first.
        """),
        L.watch_out("Only the part after `data: ` is JSON. `json.loads` on the whole line, on an empty line or on `data: [DONE]` stops with a `JSONDecodeError`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: read an answer as a stream

    1. **Ask for a stream:** add `"stream": True` to the body.
    2. **Send it as before:** the same request, the same `urlopen`.
    3. **Go through the lines:** `for line in response:`, and `line.decode()` for the text.
    4. **Skip what is not a piece:** a line that does not start with `data: `, and `data: [DONE]`.
    5. **Read the event:** `json.loads(text[6:])`, the part after the six characters of `data: `, and then `["choices"][0]["delta"]`.
    6. **Use the piece at once:** if `"content"` is in the delta, show it, or keep it.

    In November 2026, the package `openai` does steps 3 to 5 for you: with `stream=True`, it gives the events one at a time in a `for` loop.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One line of a stream, as it comes out of the response, read by hand. The line is shortened: a real one also has the id of the answer, the model and the time.

    ```python
    import json

    line = b'data: {"choices": [{"index": 0, "delta": {"content": " Every"}, "finish_reason": null}]}\n'
    text = line.decode()
    chunk = json.loads(text[6:])
    chunk["choices"][0]["delta"]
    ```

    1. **The bytes:** the `b` in front means the line is bytes, as it comes from the network; it ends in `\n`.
    2. **The text:** `line.decode()` gives the same characters as a string; `text[:6]` is `'data: '`.
    3. **The JSON:** `text[6:]` is everything after those six characters; `json.loads` reads it, and the `\n` at the end does not bother it.
    4. **The piece:** the first choice has the delta `{'content': ' Every'}`: the space in front belongs to the piece, so the pieces glued together give the text.
    """)
    return


@app.cell(hide_code=True)
def text_piece():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One line of the stream

        Write `piece(line)`: it gets one line of a stream as text, and gives back the piece of the answer in it, or `""` when the line has no piece. The lines come as they come out of a response: with `"\n"` at the end, empty lines, the first delta with the role and an empty `content`, the last with only `finish_reason`, and `data: [DONE]`. For the line of the worked example, `piece` gives `' keeper'`.

        ```python
        piece('data: {"choices": [{"index": 0, "delta": {"content": " keeper"}, "finish_reason": null}]}')
        ```
        """),
        L.task("Finish `piece`: give back `\"\"` for a line that is no event or is `data: [DONE]`, and else the `\"content\"` of the delta, if it has one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_piece():
    box_piece = L.code_box("piece", lines=10)
    box_piece
    return (box_piece,)


@app.cell(hide_code=True)
def check_piece(box_piece):
    result_piece = L.check("piece", box_piece.value)
    result_piece
    return (result_piece,)


@app.cell(hide_code=True)
def hints_piece():
    L.hints(
        "piece",
        [
            "Steps 4 and 5 of the recipe: first the lines without a piece, then the JSON.",
            "`line[:6]` is the start of the line: if that is not `\"data: \"`, or `line.strip()` is `\"data: [DONE]\"`, return `\"\"`. Then `chunk = json.loads(line[6:])`.",
            "`delta = chunk[\"choices\"][0][\"delta\"]`; then `if \"content\" in delta:` return `delta[\"content\"]`, and after it `return \"\"`.",
        ],
        why="""```python
def piece(line):
    if line[:6] != "data: " or line.strip() == "data: [DONE]":
        return ""
    chunk = json.loads(line[6:])
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        return delta["content"]
    return ""
```

Only the part after `data: ` is JSON, and not every delta has a `"content"`: the first has the role, the last only says why it stopped.""",
    )
    return


@app.cell(hide_code=True)
def text_stream():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The whole stream

        `stream(url, question)` sends a question, and gives back the list of the pieces of the answer, in the order they came, without the empty ones. It makes the request already; ask for a stream, and go through the lines of the response with your `piece` from exercise 1. The check calls your `stream` with a stand-in of the server, which sends its answer one word at a time.
        """),
        L.task("Ask for a stream in the body, and fill in the loop: each line through `piece`, and keep every piece with text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stream():
    box_stream = L.code_box("stream", lines=14)
    box_stream
    return (box_stream,)


@app.cell(hide_code=True)
def check_stream(box_piece, box_stream):
    result_stream = L.check("stream", box_stream.value, C.stream_inputs(box_piece.value))
    result_stream
    return (result_stream,)


@app.cell(hide_code=True)
def hints_stream():
    L.hints(
        "stream",
        [
            "Steps 1 and 3 of the recipe: the body asks for the stream, the loop reads it.",
            "In the body: `\"stream\": True`. In the `with`: `for line in response:`, and in that loop `text = piece(line.decode())`.",
            "After `text = piece(line.decode())`, still in the loop: `if text:` then `pieces.append(text)`. The `return pieces` is there already.",
        ],
        why="""```python
def stream(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 200,
            "stream": True}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    pieces = []
    with urllib.request.urlopen(request) as response:
        for line in response:
            text = piece(line.decode())
            if text:
                pieces.append(text)
    return pieces
```

The loop gets each line the moment it comes in; an app would show `text` there at once, instead of keeping it.""",
    )
    return


@app.cell(hide_code=True)
def text_when():
    mo.md(r"""
    ### Exercise 3. When does a stream not help?
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when",
        question="In which of these does streaming help least?",
        options={
            "chat": "A chat window, where a person reads the answer",
            "json": "A program that needs the whole answer, as JSON, before it can do anything with it",
            "story": "A tool that writes a long story for the user",
        },
        correct="json",
        why={
            "json": "Half a JSON cannot be read: the program waits for the end anyway, and the whole answer takes just as long with a stream.",
            "chat": "There a person starts reading after the first piece, instead of waiting for the end. Who can use half an answer?",
            "story": "A long answer is just where the wait is longest; with a stream the user reads along. Who can use half an answer?",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Watch Qwen3 write, through your piece

    Once exercise 1 is right, the course streams an answer of Qwen3-0.6B from its own server on your computer, and puts every line through your `piece`. What it gives appears at once.
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
    question_box = mo.ui.text(value="Write a short story of five sentences about a lighthouse keeper.", label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Stream it through your piece")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_piece, question_box, send_button):
    _piece = C.learner_piece(box_piece.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _piece is None:
        _out = mo.md("*Finish exercise 1 first: the demo reads the stream with your own `piece`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to stream the answer.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question is sent; the server loads Qwen3, and the first piece follows soon after …*"))
        _words = {"title": "What your piece read from the stream",
                  "writing": "Qwen3 is writing; every piece shows the moment your piece has read it …",
                  "done": "The first piece came after {first} seconds, the end after {seconds}.",
                  "failed": "Your piece stopped with an error on a line of the real stream: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_piece, question_box.value, lambda _h: mo.output.replace(mo.Html(_h)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A stream is the same request as in [Calling a model over HTTP](/course/open/build-call-api?lang=en), with one key more, and each event is JSON as in [Data as text: JSON](/course/open/py-json?lang=en), read line by line as in [Reading a text file](/course/open/py-files?lang=en). The pieces are the tokens the model writes one at a time. In the next lesson, [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en), the server says no, or does not answer at all, and your code deals with it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_piece, result_stream):
    L.wrap_up(result_piece, result_stream, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
