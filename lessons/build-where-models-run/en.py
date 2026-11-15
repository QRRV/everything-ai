# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-where-models-run. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-where-models-run", "en")
    C = L.checks   # the stand-ins at other places, the course's measured speeds, and the demo with your ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_choose():
    quick_choose = L.choice(
        "quick-choose",
        question="How do you find out which model is best for your app?",
        options={
            "list": "Take the model at the top of a public ranking",
            "own": "Try the candidates on examples from your own task, and weigh quality, speed and price",
            "big": "Take the biggest model you can pay for",
        },
        correct="own",
        why={
            "own": "A ranking measures other tasks; only your own examples say how well a model does yours, and how fast and dear it is there.",
            "list": "A ranking measures the tasks of the ranking. What does it say about yours? See [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en).",
            "big": "A bigger model is slower and dearer, and for easy questions often no better. See [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en).",
        },
    )
    quick_choose
    return (quick_choose,)


@app.cell(hide_code=True)
def quick_check_router():
    quick_router = L.choice(
        "quick-router",
        question="What does a router do?",
        options={
            "send": "It sends each question to the model that fits it, such as easy ones to a fast model and hard ones to a slow one",
            "all": "It asks every question of all models, and keeps the best answer",
            "network": "It connects your computer to the internet",
        },
        correct="send",
        why={
            "send": "A router looks at each question first, and picks one model for it, so easy questions do not pay the price of the slow way.",
            "all": "That would cost the price of every model for every question. See [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en).",
            "network": "That is a router for networks. Which router does that lesson mean? See [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en).",
        },
    )
    quick_router
    return (quick_router,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Behind every API is a computer that runs the model. That can be your own laptop, a server with a GPU in your company, or the data center of a company that sells the answers. Your code hardly changes: the request is the same, only the address differs. But the place decides who sees your data, what each answer costs, and how fast it comes.

    **An everyday example.** You can cook at home, have a cook come to your kitchen, or go to a restaurant. The recipe is the same; what changes is who has your kitchen, what you pay, and how long it takes. That is only a comparison; below you see the real places.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **open weights** | weights that anyone may download and run, under a license such as Apache-2.0, as Qwen3-0.6B's; the weights of a closed model you only reach through its maker's API |
    | **local runtime** | a program that loads open weights on your own computer and answers through an API on it, often OpenAI-compatible: Ollama, llama.cpp, LM Studio, and the course's own server (examples in November 2026) |
    | **inference provider** | a company that runs models on its GPUs for you, open-weight ones too, behind an API with a key, paid per token; running a model that way is also called *hosted inference* |
    | **base URL** | the part of the address before the endpoint, such as `http://127.0.0.1:8765/v1`; changing it changes where the model runs |

    ### The same call, other addresses

    | Where | Base URL (examples in November 2026) | Who sees your data | What you pay |
    |---|---|---|---|
    | the course's server, on your computer | `http://127.0.0.1:8765/v1` | only you | your electricity |
    | Ollama, on your computer | `http://localhost:11434/v1` | only you | your electricity |
    | llama.cpp's `llama-server`, on your computer | `http://localhost:8080/v1` | only you | your electricity |
    | vLLM, on your own server with a GPU | `http://<your server>:8000/v1` | your organization | the server |
    | an inference provider, or a maker's API | the company's address, with a key | the company | per token |

    `localhost` is another name for `127.0.0.1`. What runs at each address can be the same open model: Qwen3 runs in all of them.

    ### Will it fit?

    A model is its weights, and each weight takes memory: 32 bits in `float32`, 16 in `bfloat16`, and about 4 when the weights are stored rounded, as local runtimes often do. Eight bits make a byte.
    """)
    return


@app.cell(hide_code=True)
def picture_sizes():
    mo.Html(C.sizes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_speed():
    _n = C.numbers(L.lang)
    mo.vstack([
        mo.md(rf"""
        Qwen3-0.6B fits on any laptop. A model of 70 billion weights does not fit on a laptop with 16 GB, not even at 4 bits: it needs one or more large GPUs, in your own server or at a company. And memory is not all: on one thread of the course's computer, Qwen3-0.6B wrote {_n['float32']} tokens per second in `float32`, and {_n['bfloat16']} in `bfloat16`. A GPU writes many times as fast, which is what you pay a host for.
        """),
        L.watch_out("A local runtime keeps your data on your computer, but a small model also answers like one. Measure on your own examples, as in the lesson before, wherever the model runs."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: one call, any place

    1. **Pick the place:** your computer for privacy and no bill per token, a host for big models and speed.
    2. **Start or find the server:** a local runtime, such as `ollama serve`, or the address and key of a host.
    3. **Take its base URL and its model's name:** each place names its models in its own way, such as `qwen3-0.6b` or `llama3.2:1b`.
    4. **Call it as always:** `base_url + "/chat/completions"`, with the name in `"model"`.

    In November 2026, the package `openai` does the same with `OpenAI(base_url=..., api_key=...)`: for a local runtime, any key will do.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _n = C.numbers(L.lang)
    mo.md(rf"""
    ### Worked example

    How much memory do the weights of Qwen3-0.6B take in `bfloat16`?

    ```python
    params = 600_000_000
    bits = 16
    params * bits / 8 / 1_000_000_000
    ```

    1. **The weights:** about 600 million; `600_000_000` is that number, and Python skips the underscores.
    2. **The bits:** in `bfloat16`, every weight takes 16 bits.
    3. **The bytes:** divided by 8, because a byte has 8 bits: 1,200,000,000 bytes.
    4. **The gigabytes:** a gigabyte is a billion bytes, so `1.2`: the weights take 1.2 GB, half of what they take in `float32`.

    The download of Qwen3-0.6B is 1.5 GB all the same. Its file stores one table twice: the {_n['table']} weights that turn each token into numbers on the way in, and the same table again as the last layer, which picks the next token. The model uses one table for both, so it has {_n['weights']} weights, but the file holds {_n['file']} numbers: {_n['file_gb']} GB in `bfloat16`.
    """)
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One function, any place

        This `ask` only reaches the course's server, and only asks for `"qwen3-0.6b"`. Make it reach any OpenAI-compatible server: use the `base_url` and the `model` it gets. The check calls it at two places, a stand-in with `qwen3-0.6b` and one with `llama3.2:1b`, each at its own address.
        """),
        L.task("Use the `model` and the `base_url` that `ask` gets, in place of the fixed name and address."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=12)
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
            "Two things are fixed in the code that should come from the arguments: the model's name, and the address.",
            "In the body, `\"model\": model`, without quotes around `model`. The address is the `base_url` with the endpoint after it.",
            "`urllib.request.Request(base_url + \"/chat/completions\", data=data, ...)`, with the rest as it is.",
        ],
        why="""```python
def ask(base_url, model, question):
    body = {"model": model, "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(base_url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

Where the model runs is only the address and the name; everything else is the same request.""",
    )
    return


@app.cell(hide_code=True)
def text_gb():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The memory of the weights

        Write `weights_gb(params, bits)`: how many gigabytes the weights of a model take, with `params` weights of `bits` bits each. For Qwen3-0.6B in `float32`, `weights_gb(600_000_000, 32)` gives `2.4`.
        """),
        L.task("Finish `weights_gb`: weights times bits, then bytes, then gigabytes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gb():
    box_gb = L.code_box("weights_gb", lines=3)
    box_gb
    return (box_gb,)


@app.cell(hide_code=True)
def check_gb(box_gb):
    result_gb = L.check("weights_gb", box_gb.value)
    result_gb
    return (result_gb,)


@app.cell(hide_code=True)
def hints_gb():
    L.hints(
        "weights_gb",
        [
            "Follow the worked example, with `params` and `bits` in place of the numbers.",
            "`params * bits` is the number of bits; divided by 8 the bytes; divided by a billion the gigabytes.",
            "`return params * bits / 8 / ...`, with a billion in place of the dots, written as `1_000_000_000`.",
        ],
        why="""```python
def weights_gb(params, bits):
    return params * bits / 8 / 1_000_000_000
```

Bits to bytes is divided by 8; bytes to gigabytes is divided by a billion.""",
    )
    return


@app.cell(hide_code=True)
def text_where():
    mo.md(r"""
    ### Exercise 3. Where should it run?
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where",
        question="A hospital wants a model to summarize letters about its patients, and the letters may not leave the hospital's own computers. Where should the model run?",
        options={
            "host": "At a company that hosts open models, behind an API",
            "maker": "At the API of a big model maker",
            "own": "An open-weight model in a local runtime, on the hospital's own server",
        },
        correct="own",
        why={
            "own": "Open weights in a runtime of its own: the letters never leave the building. The price is the hardware, and a model small enough for it.",
            "host": "Then every letter goes to the host's computers. Which place keeps them inside?",
            "maker": "Then every letter goes to the maker's computers, and a closed model cannot run anywhere else. Which place keeps them inside?",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Your ask, at a local runtime

    Once exercise 1 is right, the course points your `ask` at its own server on your computer: a local runtime with Qwen3-0.6B. Nothing leaves your computer.
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
        _out = mo.md("*Finish exercise 1 first: the demo sends your question with your own `ask`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to send your question.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `ask` has sent the question to the server on your computer, which loads Qwen3 …*"))
        _words = {"title": "What the local runtime answered",
                  "done": "{seconds} seconds, through {url}, on your own computer.",
                  "failed": "Your ask stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_ask, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The request is the one of [Calling a model over HTTP](/course/open/build-call-api?lang=en), and the model you pick, by measuring, as in [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en), can run in any of these places. A host wants an API key with every request, and sends a bill per token. The next lessons go beyond text: images and sound, through APIs of the same kind.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, result_ask, result_gb):
    L.wrap_up(result_ask, result_gb, choice_where)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
