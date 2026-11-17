# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-multimodal-apis. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-multimodal-apis", "en")
    C = L.checks   # the course's recorded questions about photos, the stand-in of the checks, and the demo with your ask_photo


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
def quick_check_post():
    quick_post = L.choice(
        "quick-post",
        question="What makes `urllib.request.Request(url, data=body_bytes, headers={...})` a POST?",
        options={
            "headers": "The header `Content-Type: application/json`",
            "data": "The `data=` with the body",
            "url": "The address, which ends in `/chat/completions`",
        },
        correct="data",
        why={
            "data": "A request with `data=` sends a body, so urllib makes it a POST; without `data=` it is a GET.",
            "headers": "The header says what the body is, JSON, but not whether there is one. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "url": "The address says where the request goes, not how. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_post
    return (quick_post,)


@app.cell(hide_code=True)
def quick_check_choices():
    quick_choices = L.choice(
        "quick-choices",
        question="Why is the text of an answer in `answer[\"choices\"][0][\"message\"][\"content\"]`, with a `[0]`?",
        options={
            "token": "`[0]` is the first token of the answer",
            "sent": "`[0]` is the first message you sent",
            "list": "`choices` is a list, because a request can ask for more than one answer",
        },
        correct="list",
        why={
            "list": "A request can ask for several answers to the same question; usually there is one, the first, `[0]`.",
            "token": "The whole text is in `\"content\"`, as one string. What kind of value is `choices`? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "sent": "The messages you sent are in the request, not in the answer. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_choices
    return (quick_choices,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Many questions are not only words. A photo of a receipt to read, a screenshot of an error, a voice message, a chart in a report. The APIs of large models take those too, in the same request you already know: a message's content can be a list of parts, words and images side by side. And for a conversation by voice there is a kind of API that stays open while you speak.

    **An everyday example.** You send a friend a photo of a plant, and under it: "What plant is this?". One message, with two parts: a picture and a question. Your friend looks at both and answers. That is only a comparison; below you see how an image and a question go to a model in one message.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **multimodal API** | an API whose messages carry more than text: images, sound, sometimes video; a *multimodal model* reads them itself |
    | **content part** | one piece of a message's content: instead of one string, the content is a list of parts, each with its `"type"`: `"text"` for words, `"image_url"` for an image, `"input_audio"` for sound |
    | **base64** | a way to write any bytes as text, with 64 signs (`A` to `Z`, `a` to `z`, `0` to `9`, `+` and `/`): every 3 bytes become 4 signs, so the text is a third longer than the file |
    | **data URL** | an address that holds the file itself: `data:image/png;base64,` and then the base64 of the file; a browser shows it as an image, an API reads the image from it |
    | `open(path, "rb")` | opens a file to read its bytes as they are: `"r"` to read, `"b"` for bytes; without `"b"`, Python reads letters, and the bytes of an image are not letters |
    | `base64.b64encode(data)` | from Python's own module `base64`: gives the base64 of the bytes `data`, as bytes; `.decode()` turns those into a string |
    | **hosted tool** | a tool that runs at the provider, not in your program: you name it in the request's `"tools"`, the server runs it, and the answer says what it found |
    | **realtime API** | an API with a connection that stays open both ways: your program sends sound while you speak, and the spoken answer comes back while it is made |

    ### An image in a message

    The course's server runs Qwen3-0.6B, which reads text only. To still answer about an image, the server has a hosted tool: an object detector, DETR, a model that finds the objects in a photo and names them (the domain on computer vision builds such detectors). The server runs it on each image, and puts its words where the image was. The request is the same one a multimodal model gets:
    """)
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_three():
    mo.md(r"""
    The course asked the server three questions this way, each about one of its photos. Under each question: what the detector found, and what Qwen3 answered.
    """)
    return


@app.cell(hide_code=True)
def picture_recorded():
    mo.Html(C.recorded_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_more():
    mo.vstack([
        mo.md(r"""
        A large multimodal model at a hosted API does not need a detector in between: it turns the image into tokens of its own, and reads those with the question. So it can also read the text on a receipt, or say which of two cups is fuller. The request you send it is the same, with the same parts.

        ### Realtime: talking while it listens

        A request and its answer go one after the other: first the whole question, then the answer. For a conversation by voice that is slow and stiff. A realtime API keeps one connection open in both directions, through a *WebSocket* or *WebRTC*: your program sends sound while you speak, the server hears when you stop, and the spoken answer comes back while it is made; when you start talking again, it stops. OpenAI's Realtime API and Google's Live API work this way (examples in November 2026). The course's server has none: Qwen3-0.6B hears nothing.

        ### Hosted tools: the provider runs them

        A tool is something a model can ask for, such as a search on the web. Usually your own program runs it, as a later lesson shows. A hosted tool runs at the provider instead: you only name it in the request, and the answer comes back with the result in it. Searching the web and running code are hosted tools at OpenAI and at Anthropic, for example (examples in November 2026). The course's server has one: its detector.
        """),
        L.watch_out("An image costs tokens too, often more than the question with it: a multimodal model turns it into tokens, and the provider's documentation says how many. And base64 makes the image a third larger on its way."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: ask a question about an image

    1. **Read the bytes of the file:** `with open(path, "rb") as f:` and then `data = f.read()`.
    2. **Make a data URL:** `"data:image/png;base64," + base64.b64encode(data).decode()`.
    3. **Make the content a list of parts:** `[{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": the_data_url}}]`.
    4. **Send it as before**, as the content of the user's message; with the course's server, ask for its hosted tool: `"tools": [{"type": "detect_objects"}]`.
    5. **Read the answer as before:** the text is in `answer["choices"][0]["message"]["content"]`; the course's server also says what its tool found, under `"hosted_tools"`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course's question about its photo of a cup of coffee, `data/photos/coffee.png`: the request its code sent, and what came back.
    """)
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_worked_steps():
    mo.md(r"""
    1. **The content:** a list with two parts, the question as a text part and the photo as an image part.
    2. **The image:** the file has 466,706 bytes; as a data URL that is 622,298 characters, a third more, and the picture cuts it short.
    3. **The tool:** `"tools": [{"type": "detect_objects"}]` asks the server to run its detector on each image first.
    4. **What the model read:** the detector found a cup, a dining table and a spoon, and the server put the line `[Image 1. The objects the server's detector found on it: cup, dining table, spoon.]` after the question. That is what the 40 tokens under `"prompt_tokens"` are: the photo itself never reached Qwen3.
    5. **The answer:** Qwen3 wrote 24 tokens, from those words.
    """)
    return


@app.cell(hide_code=True)
def text_data_url():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. An image as a data URL

        `data_url(path)` reads the bytes of the PNG file at `path` already. Make the data URL of it, as in step 2 of the recipe. The check calls it with `"data/photos/fruit-sticker.png"`, a drawing of four fruits, and with another image; the picture under the check shows your data URL as a browser shows it.
        """),
        L.task("Finish `data_url`: turn `data` into base64 text, and give back the data URL of the PNG image."),
    ])
    return


@app.cell(hide_code=True)
def exercise_data_url():
    box_url = L.code_box("data_url", lines=7)
    box_url
    return (box_url,)


@app.cell(hide_code=True)
def check_data_url(box_url):
    result_url = L.check("data_url", box_url.value)
    result_url
    return (result_url,)


@app.cell(hide_code=True)
def hints_data_url():
    L.hints(
        "data_url",
        [
            "Two steps: first the base64 of the bytes, as a string; then the start of a data URL in front of it.",
            "`base64.b64encode(data)` gives bytes; `.decode()` after it makes a string of them.",
            "`return ___ + text`, with the start of the data URL from step 2 of the recipe, as a string, in the gap.",
        ],
        why="""```python
def data_url(path):
    with open(path, "rb") as f:
        data = f.read()
    text = base64.b64encode(data).decode()
    return "data:image/png;base64," + text
```

The start says what kind of file it is, `image/png`, and that base64 follows; after the comma comes the image itself, as text.""",
    )
    return


@app.cell(hide_code=True)
def text_ask_photo():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A question about an image

        `ask_photo(url, path, question)` sends a request and reads the answer already, as your `ask` did. Fill in the content, with the question and the image at `path` as two parts, and ask for the server's hosted tool. Your `data_url` from exercise 1 is there to use. The check calls your function with a stand-in of the server, which answers at once; the picture under the check shows what your function sent, with the data URL cut short.
        """),
        L.task("Make `content` a list with a text part and an image part, and make `\"tools\"` a list with the hosted tool `detect_objects`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask_photo():
    box_photo = L.code_box("ask_photo", lines=13)
    box_photo
    return (box_photo,)


@app.cell(hide_code=True)
def check_ask_photo(box_photo, box_url):
    result_photo = L.check("ask_photo", box_photo.value, C.photo_inputs(box_url.value))
    result_photo
    return (result_photo,)


@app.cell(hide_code=True)
def hints_ask_photo():
    L.hints(
        "ask_photo",
        [
            "Steps 3 and 4 of the recipe: the content becomes a list of two parts, and `\"tools\"` a list with one tool in it.",
            "The text part is `{\"type\": \"text\", \"text\": question}`. The image part has `\"type\": \"image_url\"`, and under `\"image_url\"` a dictionary with the key `\"url\"` and the data URL of `path`.",
            "The second part of `content`: `{\"type\": \"image_url\", \"image_url\": ___}`; and `\"tools\"` a list with one dictionary in it, whose `\"type\"` is the name of the hosted tool.",
        ],
        why="""```python
def ask_photo(url, path, question):
    content = [{"type": "text", "text": question},
               {"type": "image_url", "image_url": {"url": data_url(path)}}]
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": content}],
            "max_tokens": 60, "tools": [{"type": "detect_objects"}]}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

The question and the image go in one message, as two parts; the tool asks the server to name what is on the image before the model reads it.""",
    )
    return


@app.cell(hide_code=True)
def text_fruit():
    mo.md(r"""
    ### Exercise 3. Where does the apple come from?

    Look again at the course's third question, about the drawing of fruit: four red apples, an orange, a lemon and a lime.
    """)
    return


@app.cell(hide_code=True)
def choice_fruit_cell():
    choice_fruit = L.choice(
        "fruit",
        question="The detector found `sports ball (7)` on the drawing, and Qwen3 answered that the fruit is an apple. Where does that apple come from?",
        options={
            "guess": "Qwen3 got only the words `sports ball (7)` and the word fruit in the question, and guessed a round fruit",
            "saw": "Qwen3 saw the red apples on the image",
            "detector": "The detector told Qwen3 that the balls are apples",
        },
        correct="guess",
        why={
            "guess": "The model never sees the image, only the words of the detector, which knows no apples: a lucky guess, not a look. A multimodal model would read the image itself.",
            "saw": "Look at the picture with the arrows: what reaches Qwen3, the image or something in its place?",
            "detector": "Look at what the detector found, under the question: which word did it use for each of the seven?",
        },
    )
    choice_fruit
    return (choice_fruit,)


@app.cell(hide_code=True)
def text_realtime():
    mo.md(r"""
    ### Exercise 4. Which API for a voice assistant?
    """)
    return


@app.cell(hide_code=True)
def choice_realtime_cell():
    choice_realtime = L.choice(
        "realtime",
        question="You build a voice assistant for in the car: the driver speaks, hears the answer while it is made, and can cut in. Which API fits?",
        options={
            "chat": "The chat endpoint, with the sound as an `input_audio` part",
            "realtime": "A realtime API, with a connection that stays open both ways",
            "hosted": "A hosted tool that listens to the sound",
        },
        correct="realtime",
        why={
            "realtime": "Only an open connection lets the sound go in while the driver speaks, and the answer come out while it is made, and stop when the driver cuts in.",
            "chat": "A request goes once the whole question is in, and its answer comes back to that one request. Can the driver cut in?",
            "hosted": "A hosted tool runs during one request, at the provider. Does that make the conversation go both ways at once?",
        },
    )
    choice_realtime
    return (choice_realtime,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Ask about a photo with your own ask_photo

    Once exercises 1 and 2 are right, your `ask_photo` can talk to the real server: the course starts it on your computer, with DETR as its hosted tool and Qwen3-0.6B behind it. Pick a photo, ask your question, and compare what the detector found with the answer.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B and DETR (1.7 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The demo needs two models that are not both on your computer yet: the chat model Qwen3-0.6B, and DETR, the detector of the server's hosted tool. The button downloads what is missing once from Hugging Face, at fixed versions: `Qwen/Qwen3-0.6B`, 1.5 GB, with the tokenizer of Qwen3-0.6B-Base, and `facebook/detr-resnet-50`, 167 MB, both under the Apache-2.0 license.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B and DETR are downloaded.*")
    return


@app.cell(hide_code=True)
def photo_box_cell():
    photo_box = mo.ui.dropdown(options=C.DEMO_PHOTOS["en"], value="a cup of coffee", label="Photo")
    question_box = mo.ui.text(value="What is in this picture?", label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Ask it with your ask_photo")
    mo.vstack([photo_box, question_box, send_button])
    return photo_box, question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_photo, box_url, photo_box, question_box, send_button):
    _ask = C.learner_ask_photo(box_photo.value, box_url.value)
    if not C.downloaded():
        _out = mo.md("*Download the two models first, with the button above.*")
    elif _ask is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo asks with your own `ask_photo`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask your question.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `ask_photo` has sent the photo and the question; the server loads DETR and Qwen3, names what is on the photo, and Qwen3 writes the answer …*"))
        _words = {"title": "What came back to your ask_photo",
                  "done": "{seconds} seconds, the detector included.",
                  "failed": "Your ask_photo stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_ask, photo_box.value, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The request is the one of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with a list of parts as the content; the image tokens count in the price as in [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en). The next lesson, [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en), turns texts into vectors through an API, and sends documents along with a request. The tools your own program runs come in a later lesson on tool calling, and the domain on computer vision builds detectors like DETR.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fruit, choice_realtime, result_photo, result_url):
    L.wrap_up(result_url, result_photo, choice_fruit, choice_realtime)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
