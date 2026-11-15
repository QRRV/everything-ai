# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-on-device. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-on-device", "en")
    C = L.checks   # the two functions' checks, the pictures, and the demo for this computer


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Where a model runs](/course/open/build-where-models-run?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gb():
    quick_gb = L.choice(
        "quick-gb",
        question="How much memory do the weights of a model with 8 billion weights take at 4 bits per weight?",
        options={
            "32": "32 GB",
            "4": "4 GB",
            "8": "8 GB",
        },
        correct="4",
        why={
            "4": "8 billion times 4 bits is 32 billion bits; divided by 8, that is 4 billion bytes: 4 GB.",
            "32": "32 billion is the number of bits. How many bits make a byte? See [Where a model runs](/course/open/build-where-models-run?lang=en).",
            "8": "That would be 8 bits per weight, one byte each. How many bytes do 4 bits make? See [Where a model runs](/course/open/build-where-models-run?lang=en).",
        },
    )
    quick_gb
    return (quick_gb,)


@app.cell(hide_code=True)
def quick_check_move():
    quick_move = L.choice(
        "quick-move",
        question="You move your app from Ollama on your computer to an inference provider. What do you change in your call?",
        options={
            "all": "The whole request: each place has its own format",
            "nothing": "Nothing: the same address works everywhere",
            "url": "The base URL, the model's name and a key",
        },
        correct="url",
        why={
            "url": "The request stays the same; the address and the model's name change, and a host wants a key with every request.",
            "all": "Both speak the same OpenAI-compatible API. What differs between the places? See [Where a model runs](/course/open/build-where-models-run?lang=en).",
            "nothing": "Ollama's address is on your own computer; the provider has its own. See [Where a model runs](/course/open/build-where-models-run?lang=en).",
        },
    )
    quick_move
    return (quick_move,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far every model ran on a computer that gets your requests: a company's server, or the course's server on your own computer. A model can also run inside the app itself: in the browser of the person who opens your page, or on their phone. Then it works on a train without a signal, no question costs you money, and a photo or a letter never leaves the device. The price: the model must be small enough for that device, and every user downloads it first.

    **An everyday example.** A map on your phone can come from the internet each time you look, or you download it once before a trip: then it works in the mountains too, but it takes room on your phone, and you download it again when the roads change. That is only a comparison; below you see the real trade.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **on-device AI** | a model that runs on the device of the person who uses it, such as a phone, a laptop or the browser on it, instead of on a server; it works without a network, and the data stays on the device |
    | **device** | here: the phone, laptop or other computer of the person who uses your app; in PyTorch code, later in the course, `device` means where a tensor is computed, the CPU or the GPU |
    | **Mbit/s** | megabits per second: how fast a connection is; the size of a file is in megabytes, MB, and a byte has 8 bits |
    | **TensorFlow.js** | a library from Google that trains and runs models in the browser, in JavaScript, the language of web pages |
    | **Pyodide** | Python made to run inside a web browser, with packages such as NumPy and pandas, so that Python code runs in a page without a server |

    ### What goes over the network
    """)
    return


@app.cell(hide_code=True)
def picture_network():
    mo.Html(C.network_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_trade():
    mo.vstack([
        mo.md(r"""
        With the model on a server, every question travels with its data, and every answer comes back: that needs a network each time, and whoever runs the server can see the data. With the model on the device, only the model comes over, once; after that, nothing leaves.

        | | On a server | On the device |
        |---|---|---|
        | without a network | does not work | works |
        | what a question costs | per token, or the server | nothing: the user's device does the work |
        | who sees the data | whoever runs the server | only the user |
        | how big the model can be | as big as you can pay for | small enough for the device |
        | before the first question | nothing | the download |
        | the hardware | you choose it | the user has it: from an old phone to a new laptop |

        On-device AI is all around already (examples in November 2026): the keyboard of a phone suggests the next word, dictation turns speech into text, and translation apps translate without a network once you have downloaded a language. Browsers get models of their own built in; a later lesson of this module calls one.

        ### Will it fit, and how long does it take?

        The weights take params × bits / 8 bytes, as in [Where a model runs](/course/open/build-where-models-run?lang=en). On a device you choose the bits: fewer bits make the weights smaller, at a small cost in precision. More bits than the model was made with add nothing: Qwen3-0.6B comes in `bfloat16`, so 16 bits is the most it can use. And a device needs memory for everything else it does too.

        | Model | Weights | Size of the weights |
        |---|---|---|
        | a small model that reads handwritten digits | 50,890 | 0.2 MB in `float32` |
        | Qwen3-0.6B | 0.6 billion | 1,200 MB in `bfloat16` |
        | a model of 8 billion | 8 billion | 4,000 MB at 4 bits |

        Every user downloads the model once, and again for every new version. Connections are given in megabits per second, sizes in megabytes, so the calculation has a factor 8 in it.

        ### The tools (examples in November 2026)

        | Where it runs | Tool | What it does |
        |---|---|---|
        | in the browser | TensorFlow.js | trains and runs models in JavaScript, on the computer's GPU when the browser allows it |
        | in the browser | Pyodide | runs Python with NumPy and pandas in the page; marimo, which runs this course, can run notebooks in the browser with it |
        | in the browser | Transformers.js, WebLLM | run open models from Hugging Face, where the course gets Qwen3, in the page, large language models too; a later lesson of this module uses them |
        | on a phone | Core ML (Apple), LiteRT (Google), ExecuTorch (Meta) | run a model that was converted for the phone, in an app |
        | on the user's own laptop | Ollama, llama.cpp | the local runtimes of [Where a model runs](/course/open/build-where-models-run?lang=en): the server then runs on the user's own device |
        """),
        L.watch_out("A small model on the device also answers like a small model. When some questions need a big one, a router can send those to a server and keep the rest on the device, as in [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en)."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: can it run on the device?

    1. **Memory:** work out the weights at 32, 16, 8 and 4 bits, and take the most bits that fit in what the device can spare, but no more than the model was made with.
    2. **Download:** the size in megabytes, times 8, divided by your users' megabits per second: the seconds, once per user and per version.
    3. **Speed:** try it on the slowest device your users have, not on yours.
    4. **Tool:** TensorFlow.js or Transformers.js in a web page, Pyodide for Python in a web page, Core ML or LiteRT in a phone app.

    If a step fails, run the model on a server, or let a router split the questions.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    How long does Qwen3-0.6B take to download over a connection of 100 Mbit/s?

    ```python
    megabytes = 1506
    megabits_per_second = 100
    megabytes * 8 / megabits_per_second
    ```

    1. **The size:** the course downloads 1,506 MB for Qwen3-0.6B: its file stores one table twice, as in [Where a model runs](/course/open/build-where-models-run?lang=en), and the tokenizer comes with it.
    2. **The megabits:** times 8, because a byte has 8 bits: 12,048 megabits.
    3. **The seconds:** divided by 100 megabits per second: `120.48`, two minutes, once.
    4. **The digit model:** 0.2 MB is about 1.6 megabits, which takes 0.016 seconds: no one notices.
    """)
    return


@app.cell(hide_code=True)
def text_download():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How long the download takes

        Write `download_seconds(megabytes, megabits_per_second)`: how many seconds a file of `megabytes` MB takes over a connection of `megabits_per_second` Mbit/s. For Qwen3-0.6B over 50 Mbit/s, `download_seconds(1506, 50)` gives `240.96`.
        """),
        L.task("Finish `download_seconds`: megabytes to megabits, then divided by the speed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_download():
    box_download = L.code_box("download_seconds", lines=3)
    box_download
    return (box_download,)


@app.cell(hide_code=True)
def check_download(box_download):
    result_download = L.check("download_seconds", box_download.value)
    result_download
    return (result_download,)


@app.cell(hide_code=True)
def hints_download():
    L.hints(
        "download_seconds",
        [
            "Follow the worked example, with the two names in place of the numbers.",
            "`megabytes * 8` is the number of megabits. How many seconds do they take, at `megabits_per_second` megabits each second?",
            "`return megabytes * 8 / ...`, with the speed in place of the dots.",
        ],
        why="""```python
def download_seconds(megabytes, megabits_per_second):
    return megabytes * 8 / megabits_per_second
```

A byte has 8 bits, so the megabits are 8 times the megabytes; divided by the megabits per second, they give the seconds.""",
    )
    return


@app.cell(hide_code=True)
def text_bits():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The most bits that fit

        Write `best_bits(params, memory_gb)`: the most bits per weight, of 32, 16, 8 and 4, at which the weights of a model with `params` weights fit in `memory_gb` GB, and `None` when they do not fit even at 4 bits. For Qwen3-0.6B in 1 GB, `best_bits(600_000_000, 1)` gives `8`: at 16 bits the weights would take 1.2 GB.
        """),
        L.task("Finish `best_bits`: try the numbers of bits from most to fewest, and give back the first at which the weights fit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bits():
    box_bits = L.code_box("best_bits", lines=6)
    box_bits
    return (box_bits,)


@app.cell(hide_code=True)
def check_bits(box_bits):
    result_bits = L.check("best_bits", box_bits.value)
    result_bits
    return (result_bits,)


@app.cell(hide_code=True)
def hints_bits():
    L.hints(
        "best_bits",
        [
            "A `for` loop over the list `[32, 16, 8, 4]` tries the most bits first. When none of them fits, the loop ends.",
            "In the loop: the gigabytes are `params * bits / 8 / 1_000_000_000`, as in [Where a model runs](/course/open/build-where-models-run?lang=en). If they are at most `memory_gb`, give back `bits` at once.",
            "In the loop: `if params * bits / 8 / 1_000_000_000 <= memory_gb:` with `return bits` under it. After the loop, not in it: `return None`.",
        ],
        why="""```python
def best_bits(params, memory_gb):
    for bits in [32, 16, 8, 4]:
        if params * bits / 8 / 1_000_000_000 <= memory_gb:
            return bits
    return None
```

The loop tries the most bits first, so the first that fits is the most precise one. `return` stops the function at once; only when no number of bits fits does Python get past the loop, to `return None`.""",
    )
    return


@app.cell(hide_code=True)
def text_device():
    mo.md(r"""
    ### Exercise 3. On the device or on a server?
    """)
    return


@app.cell(hide_code=True)
def choice_device_cell():
    choice_device = L.choice(
        "device",
        question="An app for hikers identifies the plant in a photo, even high in the mountains, where there is no signal. Where should its model run?",
        options={
            "server": "At an inference provider, with a large model",
            "device": "On the phone, with a model small enough for it",
            "home": "On a server at the maker's office, which the app calls",
        },
        correct="device",
        why={
            "device": "Only a model on the phone works without a signal. The photo stays on the phone too; the price is a model that fits in it, and a download before the trip.",
            "server": "Then every photo needs the network, and there is none in the mountains. Which place works without it?",
            "home": "That is a server as well: without a signal, the app cannot reach it. Which place works without a network?",
        },
    )
    choice_device
    return (choice_device,)


@app.cell(hide_code=True)
def text_tool():
    mo.md(r"""
    ### Exercise 4. Which tool?
    """)
    return


@app.cell(hide_code=True)
def choice_tool_cell():
    choice_tool = L.choice(
        "tool",
        question="You have Python code with NumPy that works out a score from a few numbers, and you want it to run in a web page, on the visitor's own computer, without a server. Which tool runs it?",
        options={
            "ollama": "Ollama",
            "tfjs": "TensorFlow.js",
            "pyodide": "Pyodide",
            "provider": "An inference provider",
        },
        correct="pyodide",
        why={
            "pyodide": "Pyodide is Python made to run in the browser, with NumPy: your code runs on the visitor's computer as it is.",
            "tfjs": "TensorFlow.js runs models in JavaScript, not Python code. Which tool runs Python in a page?",
            "ollama": "Ollama runs language models on your own computer, not in the browser of a visitor. Which tool runs Python in a page?",
            "provider": "That is a server: the visitor's numbers would go over the network to it. Which tool runs Python in the page itself?",
        },
    )
    choice_tool
    return (choice_tool,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### What fits on your computer

    Once exercises 1 and 2 are right, the course uses your two functions for this computer: which bits fit in half of its memory, keeping the other half for everything else, how big the weights are then, and how long they take to download at the speed you choose. The digit model was made in `float32`, the others in `bfloat16`.
    """)
    return


@app.cell(hide_code=True)
def speed_cell():
    speed = mo.ui.slider(steps=[10, 20, 50, 100, 200, 500, 1000], value=50, label="Your connection, in Mbit/s", show_value=True)
    speed
    return (speed,)


@app.cell(hide_code=True)
def demo_fit(box_bits, box_download, result_bits, result_download, speed):
    _memory = C.memory_gb()
    if result_download.state != "correct" or result_bits.state != "correct":
        _out = mo.md("*Finish exercises 1 and 2 first: the table uses your own `download_seconds` and `best_bits`.*")
    elif _memory is None:
        _out = mo.md("*Python cannot see how much memory this computer has, so the table cannot be made here.*")
    else:
        _words = {"memory": "Python sees {memory} GB of memory on this computer, in billions of bytes; the table gives a model half of it, and no more bits than the model was made with.",
                  "head": ["Model", "Bits in {room} GB", "Weights", "Download at {speed} Mbit/s"],
                  "none": "does not fit", "failed": "your function stopped with {why}"}
        _out = mo.Html(C.demo(C.learner(box_download.value, "download_seconds"), C.learner(box_bits.value, "best_bits"),
                              _memory, speed.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Where a model runs](/course/open/build-where-models-run?lang=en) put a model on your computer or at a company; this lesson puts it on the device of the person who uses it, with the same formula for its weights. The router of [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en) can split the work: easy questions on the device, hard ones on a server. In the next lesson, *Just enough JavaScript*, you learn to read the language of the browser, as a step toward running a model there yourself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_device, choice_tool, result_bits, result_download):
    L.wrap_up(result_download, result_bits, choice_device, choice_tool)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
