# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-generation-hubs. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-generation-hubs", "en")
    C = L.checks   # the drawing server of the checks, the digit reader, the pictures, and the demo with your image_bytes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en), and one about [Using a pretrained model](/course/open/torch-pretrained?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_base64():
    quick_base64 = L.choice(
        "quick-base64",
        question="An image goes along with a request as `data:image/png;base64,iVBORw0KGgo…`. What is the text after the comma?",
        options={
            "link": "An address where the server finds the image",
            "bytes": "The bytes of the PNG, written as text",
            "words": "A description of the image, in code",
        },
        correct="bytes",
        why={
            "bytes": "Base64 writes every 3 bytes as 4 signs, so any file fits in the text of a JSON request. In this lesson an image comes back to you the same way.",
            "link": "A data URL holds the file itself, not where it is. See [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en).",
            "words": "It is the file itself, every byte of it, written with 64 signs. See [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en).",
        },
    )
    quick_base64
    return (quick_base64,)


@app.cell(hide_code=True)
def quick_check_hub():
    quick_hub = L.choice(
        "quick-hub",
        question="You found a model on Hugging Face. Where do you read which data it learned from, its license and its limits?",
        options={
            "name": "In the name of its repository",
            "weights": "In its state_dict, the file model.pt",
            "card": "In its model card, README.md",
        },
        correct="card",
        why={
            "card": "The model card says what the model does, its class, its training data and their license, the license of the model, how well it does, and what it does not cover. You read it before you use the model.",
            "name": "A name says little, and anyone can choose it. See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "weights": "The state_dict holds only the numbers of the layers. See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
        },
    )
    quick_hub
    return (quick_hub,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A web shop wants a picture for each of its 300 products; a teacher wants a drawing for every story in a reader. One request with a description gives an image. And when no service does your task, someone may have trained a model for it and shared it: on a model hub, thousands of models wait, each with its model card. To use one, you also need the steps around it: what goes in, and what its output means.

    **An everyday example.** You can order a cake by describing it to a baker: two layers, red berries on top. You can also bake it from a recipe out of a cookbook: then you do every step yourself, from weighing to the oven. An image API is the baker; a model from a hub with the steps around it is the recipe. That is only a comparison; below you see the requests.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **image generation API** | an endpoint that makes images from a description: in OpenAI's format a POST to `/v1/images/generations` with `"prompt"`; each image comes back under `"data"`, as base64 under `"b64_json"` |
    | **video generation API** | an endpoint that makes a video from a description, as a job: the POST starts it and gives an id, you ask for its status until it is done, and then you download the video |
    | **pipeline** | here: one function around a model from a hub; it prepares the input as the model learned it, runs the model, and turns its output into an answer |
    | `base64.b64decode(text)` | the way back of `base64.b64encode`: gives the bytes that the base64 text `text` holds |

    ### An image from a description

    You ask for an image as OpenAI's API does: a POST to `/v1/images/generations` with a `"model"`, the description as `"prompt"`, and a `"size"`, such as `"1024x1024"`. With `"n"` you ask for more than one image. The answer holds under `"data"` one item per image, with the image as base64 under `"b64_json"`: the way of [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en), now back to you. `base64.b64decode` turns the text into the bytes of a PNG again, ready to save or to show.

    The course's server answers at that endpoint too, but it has no image model: that would be a download of gigabytes. Its `course-shapes` is a small drawing program. It reads circles, squares and triangles in the prompt, in red, blue, green, yellow, purple or teal, up to 8 shapes, in English or in Dutch, and draws them; under `"revised_prompt"` it says what it drew. Each image of one request has the shapes in other places. A real image model draws anything you describe; how it does that comes in the domain on generative AI.

    Real image APIs work this way, such as OpenAI's, with its GPT Image models (examples in November 2026); they send no `"revised_prompt"`. You pay per image, more for a larger size or a higher quality. And a provider refuses a description that its rules forbid, with status 400.

    ### A video: a job, not an answer

    Making a video takes minutes, longer than a request can wait. So a video API gives you a job. The POST with the description answers at once, with an id and a status such as `queued`. Then you ask for the status of that id every few seconds, until it says `succeeded`, or `failed`; then you download the video. Runway's API works this way: `POST /v1/text_to_video` gives the id of a task, and `GET /v1/tasks/{id}` its status, until it is `SUCCEEDED` or `FAILED` (an example in November 2026). The course's server makes no videos; the picture shows the requests of such a job, in a made-up form:
    """)
    return


@app.cell(hide_code=True)
def picture_video():
    mo.Html(C.video_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_hubs():
    mo.md(r"""
    ### Model hubs and pipelines

    On a model hub such as Hugging Face, every model has a task, which the hub shows; a model card can name it in its block at the top, under `pipeline_tag`. The card of the course's digit reader, from [Using a pretrained model](/course/open/torch-pretrained?lang=en), starts like this:

    ```yaml
    ---
    license: cc-by-4.0
    pipeline_tag: image-classification
    ```

    The hub lets you search by task: `text-generation` for chat models, `image-classification` for models that give an image a label, `text-to-image` for image models, and many more. Behind most tasks are thousands of models.

    A model alone takes a tensor and gives logits. A **pipeline** does the steps around it: it prepares the input as the model learned it, runs the model, and turns the output into an answer. For the digit reader, its card says what goes in: pixels from 0 to 1. The softmax of [Many classes: softmax](/course/open/ml-softmax?lang=en) turns its 10 logits into probabilities:
    """)
    return


@app.cell(hide_code=True)
def picture_pipeline():
    mo.Html(C.pipeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_pipeline():
    mo.vstack([
        mo.md(r"""
        With Hugging Face's library transformers, one line makes such a pipeline for a model on the hub; the library reads on the hub how to prepare the input, and what the labels are:

        ```python
        from transformers import pipeline

        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        classifier("leaf.jpg")   # a list of labels, each with its score
        ```

        The course does not install transformers. In exercise 2 you build the pipeline of the digit reader yourself.
        """),
        L.watch_out("A pipeline must prepare the input exactly as the model learned it, and the card tells you how. The course gave its digit reader the pixels from 0 to 255, without dividing by 255. It still reads 927 of the 1,000 test digits right, against 929. But now it gives every digit a probability of at least 0.9997, also the 73 it reads wrong: its probabilities say nothing any more. With the right preparation, 64 of the 1,000 digits get less than 0.9. The word pipeline has other meanings too: in scikit-learn, a Pipeline is a chain of steps that you fit together, and in daily life a pipeline carries oil or gas."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: an image through an API

    1. **Write the request:** `{"model": ..., "prompt": ..., "size": "256x256"}`, with the description as the prompt.
    2. **Send it:** a POST to `url + "/images/generations"`, the way you send a chat request.
    3. **Take the image out:** the first item under `"data"` holds it under `"b64_json"`, as base64 text.
    4. **Turn it into bytes:** `base64.b64decode` gives the bytes of the PNG; save them with `open(path, "wb")`, or show them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course asked for "two red circles and a blue square", from a stand-in of its server. The request its code sent, what came back, and the image:
    """)
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_worked_steps():
    mo.md(r"""
    1. **The request:** the description under `"prompt"`, and the size. Without `"n"`, the server draws one image.
    2. **The answer:** under `"data"`, one item. Its `"b64_json"` has 2,492 characters; the picture shows the first 24. Under `"revised_prompt"`, the server says what it drew: 2 red circles and 1 blue square.
    3. **The bytes:** `base64.b64decode` gives 1,868 bytes, 3 for every 4 signs. They start with the 8 bytes `\x89PNG\r\n\x1a\n`, the start of every PNG file.
    4. **The image:** 256 by 256 pixels, as the request asked.
    """)
    return


@app.cell(hide_code=True)
def text_image():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. An image through the API

        `image_bytes(url, prompt)` sends the request and reads the answer already. Fill in the three `...`: the description, the endpoint for images, and the bytes of the first image. The check calls it with a stand-in of the server, which draws as the course's server does. The picture under the check shows what your function sent, what came back, and the image your function gave back.
        """),
        L.task("Finish `image_bytes`: send `prompt` to the endpoint for images, and give back the bytes of the first image in the answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_image():
    box_image = L.code_box("image", lines=12)
    box_image
    return (box_image,)


@app.cell(hide_code=True)
def check_image(box_image):
    result_image = L.check("image", box_image.value)
    result_image
    return (result_image,)


@app.cell(hide_code=True)
def hints_image():
    L.hints(
        "image",
        [
            "Steps 1 to 4 of the recipe: what goes under `\"prompt\"`, where the request goes, and what you take out of the answer.",
            "The description is the `prompt` the function gets; the endpoint is `/images/generations`, after `url`. The image is in the first item under `\"data\"`, under `\"b64_json\"`, as base64 text.",
            "The last `...`: `return base64.b64decode(___)`, with the base64 text of the first image in the gap.",
        ],
        why="""```python
def image_bytes(url, prompt):
    body = {"model": "course-shapes", "prompt": prompt, "size": "256x256"}
    request = urllib.request.Request(url + "/images/generations", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return base64.b64decode(answer["data"][0]["b64_json"])
```

`answer["data"]` is the list of images, `[0]` the first, and `["b64_json"]` its base64 text; `b64decode` turns that into the bytes of the PNG.""",
    )
    return


@app.cell(hide_code=True)
def text_digit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A pipeline of your own

        Now a model from a hub, with your own pipeline around it: the course's digit reader. `read_digit(model, pixels)` gets the reader and the 784 pixels of a test digit, as a list of whole numbers from 0 to 255, row after row. It gives back the digit and its probability. Fill in the two `...`: the pixels prepared as the model card says, and the 10 probabilities of the digit. Two things help:

        - `torch.tensor(pixels, dtype=torch.float32)` makes a tensor of floats from the list. The reader itself turns the 784 pixels into an image of 28 by 28, so it gives one row of 10 logits: shape `(1, 10)`.
        - `torch.softmax(t, dim=1)` is the softmax of [Many classes: softmax](/course/open/ml-softmax?lang=en) in PyTorch, along axis 1 of `t`: over the 10 logits of each row.

        The check calls it with `reader`, the course's digit reader, and `digits[36]`, the pixels of test digit 36, and then with three more digits. The picture under the check shows each digit with what your `read_digit` says.
        """),
        L.task("Finish `read_digit`: prepare the pixels as the model learned them, and turn the reader's 10 logits into probabilities."),
    ])
    return


@app.cell(hide_code=True)
def exercise_digit():
    box_digit = L.code_box("read_digit", lines=10)
    box_digit
    return (box_digit,)


@app.cell(hide_code=True)
def check_digit(box_digit):
    result_digit = L.check("read_digit", box_digit.value)
    result_digit
    return (result_digit,)


@app.cell(hide_code=True)
def hints_digit():
    L.hints(
        "read_digit",
        [
            "The card says what the model takes: a tensor with pixels from 0 to 1. And the model gives logits: which function turns 10 logits into probabilities?",
            "Dividing by 255 brings pixels from 0 to 255 to 0 to 1. What the model gives for `x` has shape `(1, 10)`: take the softmax along axis 1, and then row 0.",
            "The second `...`: `p = torch.softmax(___, dim=1)[0]`.",
        ],
        why="""```python
def read_digit(model, pixels):
    x = torch.tensor(pixels, dtype=torch.float32) / 255
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    digit = int(p.argmax())
    return digit, float(p[digit])
```

Preparing, the model, and the answer: the three steps of a pipeline. For test digit 36, a 7, it gives `(7, 0.897)`; for test digit 62, a 9, it gives `(1, 0.693)`: wrong, and less sure.""",
    )
    return


@app.cell(hide_code=True)
def text_video_job():
    mo.md(r"""
    ### Exercise 3. A video is on its way
    """)
    return


@app.cell(hide_code=True)
def choice_video_cell():
    choice_video = L.choice(
        "video-job",
        question="A video API answers your POST with `{\"id\": \"video_abc\", \"status\": \"queued\"}`. What does your program do next?",
        options={
            "poll": "Ask its status now and then, until it is done",
            "decode": "Decode the video out of this answer, as with an image",
            "again": "Send the POST again until the video itself comes back",
        },
        correct="poll",
        why={
            "poll": "The answer only says that the job has started: `queued`, waiting in line. Making a video takes minutes, so you ask for the status of `video_abc` every few seconds. It goes from `queued` to `running` to `succeeded`, or `failed`; then you download the video. Asking ten times a second costs requests and gains nothing.",
            "decode": "Is there a video in this answer? Look at what it holds: an id and a status.",
            "again": "Each POST starts a new job, and you pay for each video. How do you hear how the job you already started is doing?",
        },
    )
    choice_video
    return (choice_video,)


@app.cell(hide_code=True)
def text_hub_task():
    mo.md(r"""
    ### Exercise 4. Which task?
    """)
    return


@app.cell(hide_code=True)
def choice_task_cell():
    choice_task = L.choice(
        "hub-task",
        question="You want a model that says which of 20 plant diseases a photo of a leaf shows. Which task do you search for on the hub?",
        options={
            "t2i": "text-to-image: a description in, an image out",
            "cls": "image-classification: an image in, a label out",
            "i2t": "image-to-text: an image in, a text about it out",
        },
        correct="cls",
        why={
            "cls": "Twenty diseases are twenty labels, and a photo goes in: image classification, the task of the course's digit reader, whose card says `pipeline_tag: image-classification`. On the hub you then look for a model trained on leaves, and read its card.",
            "t2i": "That task makes images from a description. What goes in here, and what should come out?",
            "i2t": "Such a model writes about an image. Do you need a text, or one of 20 labels?",
        },
    )
    choice_task
    return (choice_task,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Draw with your own image_bytes

    Once exercise 1 is right, your `image_bytes` can ask the course's server itself, on your computer. Describe up to 8 circles, squares and triangles, in red, blue, green, yellow, purple or teal, in English or in Dutch. Nothing is downloaded: the server draws.
    """)
    return


@app.cell(hide_code=True)
def prompt_box_cell():
    prompt_box = mo.ui.text(value="three yellow triangles and a purple square", label="Your description", full_width=True)
    draw_button = mo.ui.run_button(label="Draw with your image_bytes")
    mo.vstack([prompt_box, draw_button])
    return draw_button, prompt_box


@app.cell(hide_code=True)
def demo_image(box_image, draw_button, prompt_box):
    _draw = C.learner_image(box_image.value)
    if _draw is None:
        _out = mo.md("*Finish exercise 1 first: the demo asks with your own `image_bytes`.*")
    elif not draw_button.value:
        _out = mo.md("*Press the button to draw.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Write a description in the box first.*")
    else:
        _words = {"title": "What your image_bytes got back",
                  "drawn": "The server read: {drawn}.",
                  "done": "{seconds} seconds.",
                  "failed": "Your image_bytes stopped with an error when it asked the course's server: {why}",
                  "not_png": "Your image_bytes gave back something other than the bytes of a PNG."}
        _out = mo.Html(C.demo(_draw, prompt_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The request is the one of [Calling a model over HTTP](/course/open/build-call-api?lang=en), at another endpoint, and base64 is the one of [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en), now on the way back. The digit reader and its model card come from [Using a pretrained model](/course/open/torch-pretrained?lang=en). The next lesson, [A large model or a small one?](/course/open/build-llm-or-small-model?lang=en), puts a small model of its own against an LLM, for one fixed task. How an image model makes an image from a description comes in the domain on generative AI.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_task, choice_video, result_digit, result_image):
    L.wrap_up(result_image, result_digit, choice_video, choice_task)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
