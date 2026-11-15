# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-model-in-browser. Route: course/route/lessen/bouwen.yaml. Checks, the picture and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-model-in-browser", "en")
    C = L.checks   # the digit network, the check of as_json, the two boxes with JavaScript, the picture and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question each about [Just enough JavaScript](/course/open/build-js-basics?lang=en), [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) and [Data as text: JSON](/course/open/py-json?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="Which positions of the array `a` does `for (let i = 0; i < a.length; i++)` go through?",
        options={
            "one": "From 1 up to and including `a.length`",
            "all": "Every position, from 0 to the last",
            "values": "None: it goes through the values",
        },
        correct="all",
        why={
            "all": "`i` starts at 0 and goes on while `i < a.length`, so it stops after the last position, `a.length - 1`.",
            "one": "Where does `i` start, and what does `i < a.length` do at the end? See [Just enough JavaScript](/course/open/build-js-basics?lang=en).",
            "values": "That is `for (const x of a)`. What is `i` here? See [Just enough JavaScript](/course/open/build-js-basics?lang=en).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_mlp():
    quick_mlp = L.choice(
        "quick-mlp",
        question="What does your `DigitsMLP` give for one image of a digit?",
        options={
            "digit": "The digit itself, one number from 0 to 9",
            "pixels": "784 numbers, one for each pixel",
            "logits": "Ten logits, one for each digit",
        },
        correct="logits",
        why={
            "logits": "Its output layer has ten neurons; the digit it reads is the one with the largest logit.",
            "digit": "Which number comes out of the output layer, and how many? See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "pixels": "Those go in. What comes out of the output layer? See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
        },
    )
    quick_mlp
    return (quick_mlp,)


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question='What does `json.dumps({"w": [1, 2]})` give?',
        options={
            "text": 'The text `{"w": [1, 2]}`',
            "dict": 'The dict `{"w": [1, 2]}` again',
            "list": "The list `[1, 2]`",
        },
        correct="text",
        why={
            "text": "`json.dumps` writes a dict as text, which a program in another language can read, such as a page in JavaScript.",
            "dict": "What kind of value does `json.dumps` make? See [Data as text: JSON](/course/open/py-json?lang=en).",
            "list": "`json.dumps` keeps the whole dict. What kind of value does it make of it? See [Data as text: JSON](/course/open/py-json?lang=en).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The network that reads digits, from the project lesson, is 50,890 numbers. A web page can hold those numbers, and do the sums of the network itself, in JavaScript: then the page reads a digit without a server, without an API key, and the drawing never leaves the computer. It is the on-device AI of [AI on your own device](/course/open/build-on-device?lang=en), with a network you know inside out. Libraries do the same with large open models, and Chrome even has a model built in.

    **An everyday example.** A recipe is the same on paper, on a phone or on a fridge door: whoever has it can cook without asking the chef. The weights of a network are its recipe. That is only a comparison; below you see how it works.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **Transformers.js** | a JavaScript library from Hugging Face that runs trained models from its model hub in the browser, with pipelines as in Python's `transformers` |
    | **WebLLM** | a library that runs large language models entirely in the browser, on the computer's GPU, with an interface like the OpenAI API |
    | **built-in browser AI** | a model the browser itself brings along, which every page can use without downloading a model of its own, such as Gemini Nano in Chrome through its Prompt API |
    | **WebGPU** | the way a page uses the computer's GPU for calculations, which makes a model in the browser many times faster than on the CPU |
    | **ONNX** | an open file format for trained models, so that a model from PyTorch can run in other software, such as a browser |
    | `.tolist()` | makes a tensor a list of Python numbers, a list of lists for a tensor with two axes, which `json.dumps` can write |

    ### From PyTorch to the page
    """)
    return


@app.cell(hide_code=True)
def picture_way():
    mo.Html(C.way_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_layers():
    mo.vstack([
        mo.md(r"""
        A layer `nn.Linear` holds a table of weights and a list of biases. Row `j` of `model.hidden.weight` holds the 784 weights of hidden neuron `j`, so the table has 64 rows of 784 numbers; `model.hidden.bias` has 64 numbers. For an input `x`, neuron `j` gives `dot(weight[j], x) + bias[j]`: your `dot` from the previous lesson, plus a bias. `forward` of `DigitsMLP` does that for the hidden layer, puts the sums through ReLU, and does it again for the output layer, which gives ten logits.

        So the page needs four things: `w1` and `b1` of the hidden layer, `w2` and `b2` of the output layer. `json.dumps` cannot write a tensor, but `.tolist()` makes it a list first: `json.dumps({"b": torch.tensor([0.5, -1.0]).tolist()})` gives the text `{"b": [0.5, -1.0]}`. As JSON text, a page reads them with `await (await fetch(url)).json()`. In JavaScript, ReLU is `Math.max(0, v)`, as `max(0, v)` in Python, and `out.push(v)` adds `v` at the end of an array, as `append` does in Python.

        As text, the network is about 1.1 MB, five and a half times its 0.2 MB in `float32`: JSON writes every number with all its digits. For a network this small that does not matter. A large model goes to the browser in a binary format instead, such as ONNX.
        """),
        L.watch_out("A page that loads its weights with `fetch` downloads them for every user, and a browser keeps them only as long as its cache does. Transformers.js and WebLLM keep what they downloaded in the browser's own storage, so a second visit does not download them again."),
    ])
    return


@app.cell(hide_code=True)
def text_large():
    mo.md(r"""
    ### Large models in the browser

    The same idea, weights in the page and the sums in JavaScript, works for large models, but nobody writes those sums by hand. Three ways (examples in November 2026):

    | Way | What runs | Code |
    |---|---|---|
    | Transformers.js | an open model from Hugging Face, converted to ONNX, for a task: embeddings, speech to text, images | `const extractor = await pipeline("feature-extraction", "mixedbread-ai/mxbai-embed-xsmall-v1", { device: "webgpu" });` |
    | WebLLM | an open chat model at 4 bits, such as Qwen3-0.6B, on the GPU through WebGPU | `const engine = await CreateMLCEngine("Qwen3-0.6B-q4f16_1-MLC");` and then `await engine.chat.completions.create({ messages })` |
    | the Prompt API of Chrome | Gemini Nano, built into Chrome: the page brings no model | `const session = await LanguageModel.create();` and then `await session.prompt("...")` |

    With Transformers.js and WebLLM the page chooses the model, and every user downloads it once. With the Prompt API, Chrome downloads Gemini Nano once, for all pages; the page cannot choose the model, and it works only where Chrome offers it: in Chrome 148 and later, on a computer with Windows, macOS or Linux or on a Chromebook Plus, with at least 22 GB free on the disk and a GPU with at least 4 GB of memory of its own, or 16 GB of memory and 4 CPU cores. A page asks first with `await LanguageModel.availability()`, and has a plan for when the answer is `"unavailable"`.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: your own network in the browser

    1. **Weights as JSON:** each tensor `.tolist()`, together in a dict, `json.dumps`.
    2. **In the page:** load the JSON text with `fetch`, and read it with `.json()`.
    3. **Each layer:** for each neuron, `dot` of its row of weights with the input, plus its bias.
    4. **Between the layers:** ReLU, `Math.max(0, v)`, just as in `forward` of your network.
    5. **The answer:** the position of the largest logit. For a large model, use Transformers.js, WebLLM or the Prompt API instead.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A network with two inputs, two hidden neurons and two outputs, as the page reads it:

    ```javascript
    const net = {w1: [[1, 1], [-1, 2]], b1: [-1, 0], w2: [[2, -1], [0, 1]], b2: [0, 0.5]};
    ```

    What does it make of the input `[3, 0]`?

    1. **Hidden neuron 0:** its row is `[1, 1]`, its bias `-1`: 1 × 3 + 1 × 0 − 1 = `2`.
    2. **Hidden neuron 1:** its row is `[-1, 2]`, its bias `0`: −1 × 3 + 2 × 0 + 0 = `-3`.
    3. **ReLU:** `2` stays `2`, and `-3` becomes `0`; the hidden layer gives `[2, 0]`.
    4. **The output layer:** 2 × 2 − 1 × 0 + 0 = `4`, and 0 × 2 + 1 × 0 + 0.5 = `0.5`: the logits are `[4, 0.5]`.
    5. **The answer:** the largest logit is at position 0, so the network says 0.
    """)
    return


@app.cell(hide_code=True)
def text_json():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The weights as text

        Write `as_json(model)`: it gives the weights and biases of a `DigitsMLP` as JSON text, with the keys `"w1"` and `"b1"` for `model.hidden`, and `"w2"` and `"b2"` for `model.out`. For `DigitsMLP(3)`, `"w1"` holds 3 lists of 784 numbers. The check calls your function with networks of a few sizes, and compares the numbers with the tensors.
        """),
        L.task("Finish `as_json`: put the four tensors as lists in a dict, and give it back as JSON text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_json():
    box_json = L.code_box("as_json", lines=12)
    box_json
    return (box_json,)


@app.cell(hide_code=True)
def check_json(box_json):
    result_json = L.check("as_json", box_json.value)
    result_json
    return (result_json,)


@app.cell(hide_code=True)
def hints_json():
    L.hints(
        "as_json",
        [
            "`json.dumps` can write dicts, lists and numbers, but no tensors: make each tensor a list first.",
            "`model.hidden.weight.tolist()` gives the weights of the hidden layer as a list of lists. Put the four lists in a dict with the keys `\"w1\"`, `\"b1\"`, `\"w2\"` and `\"b2\"`.",
            "`weights = {\"w1\": model.hidden.weight.tolist(), \"b1\": model.hidden.bias.tolist(), ...}` with `model.out` for `\"w2\"` and `\"b2\"`, then `return json.dumps(weights)`.",
        ],
        why="""```python
import json


def as_json(model):
    weights = {
        "w1": model.hidden.weight.tolist(),
        "b1": model.hidden.bias.tolist(),
        "w2": model.out.weight.tolist(),
        "b2": model.out.bias.tolist(),
    }
    return json.dumps(weights)
```

`.tolist()` makes each tensor a list of numbers, the dict gives each its name, and `json.dumps` writes it all as text.""",
    )
    return


@app.cell(hide_code=True)
def text_layer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One layer in JavaScript

        Write `layer(w, b, x)` in JavaScript: for each row `w[i]` it gives `dot(w[i], x) + b[i]`, all in one array. The box starts with a `dot` like yours from the previous lesson. `layer([[1, 2], [3, -1]], [0.5, -2], [2, 1])` gives `[4.5, 3]`: 1 × 2 + 2 × 1 + 0.5, and 3 × 2 − 1 × 1 − 2. Your browser runs your function, in a web worker.
        """),
        L.task("Finish `layer`: one number per row of `w`, each pushed onto an array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("layer", lines=16)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def run_layer_cell(box_layer):
    run_layer = L.js_run("layer", box_layer.value)
    run_layer
    return (run_layer,)


@app.cell(hide_code=True)
def check_layer(run_layer):
    result_layer = L.check("layer", run_layer)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "layer",
        [
            "Start with an empty array, and go through the rows of `w` with a `for` loop that counts, as in `dot`.",
            "`for (let i = 0; i < w.length; i++) { ... }` goes through the neurons; inside, push `dot(w[i], x) + b[i]` onto the array.",
            "`const out = [];` before the loop, `out.push(dot(w[i], x) + b[i]);` in it, and `return out;` after it.",
        ],
        why="""```javascript
function layer(w, b, x) {
  const out = [];
  for (let i = 0; i < w.length; i++) {
    out.push(dot(w[i], x) + b[i]);
  }
  return out;
}
```

Each row of `w` is one neuron: `dot` with the input, plus its bias, gives what that neuron gives.""",
    )
    return


@app.cell(hide_code=True)
def text_forward():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The whole network

        Write `forward(net, pixels)` in JavaScript, as `forward` of your `DigitsMLP`: the hidden layer with `net.w1` and `net.b1`, ReLU on what it gives, then the output layer with `net.w2` and `net.b2`, whose numbers come back as they are. The check puts the course's `dot` and `layer` before your code, so you can call them here; in the demo below, your own `layer` from exercise 2 does the work. With

        ```javascript
        const net = {w1: [[1, -1], [0.5, 2]], b1: [0, -3], w2: [[1, 1], [2, -1], [-1, 0]], b2: [0, 0.5, -1]};
        ```

        `forward(net, [1, 2])` gives `[1.5, -1, -1]`.
        """),
        L.task("Finish `forward`: the hidden layer, ReLU, the output layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_forward():
    box_forward = L.code_box("forward", lines=10)
    box_forward
    return (box_forward,)


@app.cell(hide_code=True)
def run_forward_cell(box_forward):
    run_forward = L.js_run("forward", box_forward.value)
    run_forward
    return (run_forward,)


@app.cell(hide_code=True)
def check_forward(run_forward):
    result_forward = L.check("forward", run_forward)
    result_forward
    return (result_forward,)


@app.cell(hide_code=True)
def hints_forward():
    L.hints(
        "forward",
        [
            "Follow the worked example: first what the hidden layer gives, then ReLU on each of those numbers, then the output layer on the result.",
            "`layer(net.w1, net.b1, pixels)` gives the sums of the hidden layer. Go through them with `for (const v of ...)`, and push `Math.max(0, v)` onto a new array.",
            "`const hidden = [];`, the loop with `hidden.push(Math.max(0, v));`, and then `return layer(net.w2, net.b2, hidden);`.",
        ],
        why="""```javascript
function forward(net, pixels) {
  const hidden = [];
  for (const v of layer(net.w1, net.b1, pixels)) {
    hidden.push(Math.max(0, v));
  }
  return layer(net.w2, net.b2, hidden);
}
```

The same three steps as `self.out(torch.relu(self.hidden(x)))` in your `DigitsMLP`: the hidden layer, ReLU, the output layer.""",
    )
    return


@app.cell(hide_code=True)
def text_builtin():
    mo.md(r"""
    ### Exercise 4. Built into the browser
    """)
    return


@app.cell(hide_code=True)
def choice_builtin_cell():
    choice_builtin = L.choice(
        "builtin",
        question="A page answers questions with the Prompt API of Chrome. Where does the model that answers come from?",
        options={
            "hub": "The page downloads it from Hugging Face",
            "google": "Google runs it on its servers",
            "chrome": "Chrome brings it along, for every page",
        },
        correct="chrome",
        why={
            "chrome": "Gemini Nano is built into Chrome: Chrome downloads it once, and runs it on the computer itself, for every page that asks.",
            "hub": "That is what a page does with Transformers.js or WebLLM. What does the page bring along with the Prompt API?",
            "google": "Then every question would go over the network. Where does a built-in model run?",
        },
    )
    choice_builtin
    return (choice_builtin,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Draw a digit

    Draw a digit in the square. After every stroke the drawing becomes 28 × 28 pixels, the way the digits of QMNIST were made, and your own code reads it, in your browser: the weights as your `as_json` wrote them, your `layer` and your `forward`. The network is the one from the project lesson, trained the same way by the course: 64 hidden neurons, 10 epochs, a learning rate of 0.5. It reads 929 of the 1,000 test digits right. Under the square, PyTorch reads the same pixels, here in Python.
    """)
    return


@app.cell(hide_code=True)
def demo_draw(box_forward, box_json, box_layer, result_forward, result_json, result_layer):
    mo.stop(not (result_json.state == result_layer.state == result_forward.state == "correct"),
            mo.md("*Finish exercises 1, 2 and 3 first: then the demo uses your `as_json`, your `layer` and your `forward`.*"))
    _net = C.learner_json(box_json.value)
    _words = {"clear": "Start over", "draw": "Draw a digit in the square.", "reads": "Your code in your browser reads a {digit}.",
              "stopped": "Your code was not done after 3 seconds, so your browser stopped it.", "error": "Your code stopped: {error}"}
    pad = mo.ui.anywidget(C.draw(box_layer.value + "\n" + box_forward.value, _net, _words, L.lang))
    pad
    return (pad,)


@app.cell(hide_code=True)
def demo_torch(pad):
    _v = pad.widget.value
    mo.stop(not _v.get("pixels"), None)
    _digit, _logits = C.torch_reads(_v["pixels"])
    _gap = max(abs(a - b) for a, b in zip(_logits, _v["scores"]))
    mo.md("PyTorch, here in Python, reads a **{digit}** in the same pixels. Its ten logits differ from those of your code in your browser by less than {gap}: the same network, in two languages.".format(digit=_digit, gap=L.number(*C.less_than(_gap))))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The network is the `DigitsMLP` of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), its weights travel as the JSON of [Data as text: JSON](/course/open/py-json?lang=en), and its sums are your `dot` from [Just enough JavaScript](/course/open/build-js-basics?lang=en), in your browser as in [AI on your own device](/course/open/build-on-device?lang=en). Transformers.js and WebLLM do the same with models too large to write by hand, and the Prompt API with a model built into Chrome. This ends the module *Calling models*; prompts for developers come next.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_builtin, result_forward, result_json, result_layer):
    L.wrap_up(result_json, result_layer, result_forward, choice_builtin)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
