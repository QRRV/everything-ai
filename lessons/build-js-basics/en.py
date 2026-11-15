# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-js-basics. Route: course/route/lessen/bouwen.yaml. Checks, the picture and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-js-basics", "en")
    C = L.checks   # the two JavaScript boxes, which your browser runs, the picture of await, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [AI on your own device](/course/open/build-on-device?lang=en), and one about [Functions](/course/open/py-functions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_device():
    quick_device = L.choice(
        "quick-device",
        question="Why does a model in the browser keep working without a network?",
        options={
            "device": "It runs on the device, after one download",
            "copy": "The browser keeps a copy of the whole internet",
            "phone": "The phone company keeps the connection open",
        },
        correct="device",
        why={
            "device": "The model came over once; after that, every question is answered on the device, and nothing has to go over the network.",
            "copy": "No browser has room for that. Where does the model compute its answer? See [AI on your own device](/course/open/build-on-device?lang=en).",
            "phone": "Without a signal there is no connection to keep open. Where does the model compute? See [AI on your own device](/course/open/build-on-device?lang=en).",
        },
    )
    quick_device
    return (quick_device,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="What does `return` do in a Python function?",
        options={
            "show": "It shows a value on the screen",
            "again": "It starts the function over again from its first line",
            "back": "It gives a value back to whoever called it",
        },
        correct="back",
        why={
            "back": "`return` hands the value to the code that called the function, and stops the function there.",
            "show": "That is what `print` does. What does the code that called the function get? See [Functions](/course/open/py-functions?lang=en).",
            "again": "A function starts from the top only when it is called again. What does `return` hand over? See [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A browser runs one language: JavaScript. Every web page with a model in it calls that model in JavaScript, whether the model comes from a library or is built into the browser. You do not need to become a web developer for it. Three things are enough to read and change that code: how a file loads a library, how `await` waits for an answer while the page keeps responding, and how a web worker computes beside the page.

    **An everyday example.** In a restaurant, the waiter takes an order to the kitchen and does not stand at the kitchen door until the dish is ready: the waiter serves other tables, and comes back when the kitchen calls. The page is the waiter, a long job is the kitchen. That is only a comparison; below you see how JavaScript does it.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **JavaScript** | the programming language every web browser runs; with Node.js it also runs outside the browser. It has nothing to do with Java |
    | **module** | a file of JavaScript that loads what it needs from other modules with `import`, and offers its own functions with `export` |
    | **promise** | a value that comes later, such as the answer to a request: JavaScript gives you the promise at once, and the value when it is there |
    | **await** | in a function marked `async`, waits for a promise to give its value; while it waits, the page goes on answering clicks and typing |
    | **web worker** | a piece of JavaScript that a page runs beside itself; heavy work there, such as a model computing, does not freeze the page. Page and worker talk by sending messages |

    ### Python and JavaScript side by side

    | Python | JavaScript |
    |---|---|
    | `x = 3` | `let x = 3;`, or `const x = 3;` for a name that keeps its value |
    | `def twice(x):` with `return 2 * x` indented under it | `function twice(x) { return 2 * x; }` |
    | `[1, 2, 3]` and `len(xs)` | `[1, 2, 3]` and `xs.length` |
    | `{"model": "qwen3-0.6b"}` and `d["model"]` | `{model: "qwen3-0.6b"}` and `d.model` |
    | `for x in xs:` | `for (const x of xs) { ... }` |
    | `for i in range(len(xs)):` | `for (let i = 0; i < xs.length; i++) { ... }` |
    | `if x > 0:` and `else:` | `if (x > 0) { ... } else { ... }` |
    | `total += x` | `total += x;` |
    | `print(x)` | `console.log(x);` |
    | `None` | `null`, and `undefined` for a value that was never given |
    | `json.loads(text)` | `JSON.parse(text)` |

    Braces `{ }` hold a block where Python indents, and a statement ends with `;`. `i++` adds 1 to `i`.

    ### Waiting without freezing
    """)
    return


@app.cell(hide_code=True)
def picture_await():
    mo.Html(C.await_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_await():
    mo.vstack([
        mo.md(r"""
        `fetch(url)` sends a request, as `urllib.request.urlopen` does in Python, but it gives back a promise at once. `await` waits for its value; meanwhile the page is free. Python's `urlopen` waits too, but nothing else runs while it does. And a module loads a library by its address:

        ```javascript
        import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";
        ```

        That line loads `pipeline` from Transformers.js, a library that the next lesson uses (the address as in November 2026); a real page adds a fixed version number after the name, so that a new version of the library does not change the page.

        `await` frees the page while your code waits, but not while your code computes. A model that computes for two seconds on the page freezes it for two seconds. That work goes to a web worker: the page sends it a message with the input, and the worker sends back a message with the answer.
        """),
        L.watch_out("JavaScript has a `for ... in` loop too, and it goes over the positions of an array as text, `\"0\"`, `\"1\"`, not over its values. For the values, use `for ... of`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: reading AI code in JavaScript

    1. **The imports at the top:** which library does the page load, and from where?
    2. **Every `await`:** each is a wait, for a download, a model that loads, or an answer.
    3. **`new Worker` and `postMessage`:** work that runs beside the page, and the messages to and from it.
    4. **The rest:** functions, arrays, objects and loops, as in Python, with braces.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A function that fetches a list of texts as JSON, and gives back how many characters they have on average:

    ```javascript
    async function averageLength(url) {
      const response = await fetch(url);
      const texts = await response.json();
      let total = 0;
      for (const text of texts) {
        total += text.length;
      }
      return total / texts.length;
    }
    ```

    1. **`async function`:** a function that may wait; it gives back a promise itself, so whoever calls it uses `await` too.
    2. **`await fetch(url)`:** the request; `response` is the answer, as soon as it has come.
    3. **`await response.json()`:** reading the body as JSON takes a moment too, so it gives a promise as well, and `texts` gets the list.
    4. **The loop:** `for (const text of texts)` goes over the texts, and `text.length` is the number of characters of each.
    5. **The average:** the total divided by the number of texts. For `["red", "blue"]` that is (3 + 4) / 2: `3.5`.
    """)
    return


@app.cell(hide_code=True)
def text_dot():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Pair by pair

        Write `dot(a, b)` in JavaScript: multiply the numbers of two arrays of the same length pair by pair, and add it all up. `dot([1, 2, 3], [4, 5, 6])` gives `32`: 1 × 4 + 2 × 5 + 3 × 6. Networks that read digits or text compute this sum many times over; the next lesson uses it to run a small network in the browser. Your browser runs your function, in a web worker, with the cases of the check.
        """),
        L.task("Finish `dot`: go through the positions, and add up the products."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dot():
    box_dot = L.code_box("dot", lines=8)
    box_dot
    return (box_dot,)


@app.cell(hide_code=True)
def run_dot_cell(box_dot):
    run_dot = L.js_run("dot", box_dot.value)
    run_dot
    return (run_dot,)


@app.cell(hide_code=True)
def check_dot(run_dot):
    result_dot = L.check("dot", run_dot)
    result_dot
    return (result_dot,)


@app.cell(hide_code=True)
def hints_dot():
    L.hints(
        "dot",
        [
            "Start with a total of 0, and go through the positions of `a` with a `for` loop that counts, as in the table above.",
            "`for (let i = 0; i < a.length; i++) { ... }` goes through every position; inside the braces, add `a[i] * b[i]` to the total.",
            "`let total = 0;` before the loop, `total += a[i] * b[i];` in it, and `return total;` after it.",
        ],
        why="""```javascript
function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}
```

`let` makes the total, the loop counts `i` from 0 to the last position, and `+=` adds every product to it.""",
    )
    return


@app.cell(hide_code=True)
def text_sum():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Wait for it

        Write `async function sumFrom(url)`: fetch the JSON at `url`, a list of numbers, and give back their sum. The check gives your function addresses that start with `data:`, such as `"data:application/json,[1,2,3]"`: the JSON is in the address itself, so `fetch` needs no network. For that one, `sumFrom` gives `6`.
        """),
        L.task("Finish `sumFrom`: fetch, then the JSON, each with `await`, then add up the list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sum():
    box_sum = L.code_box("sumFrom", lines=10)
    box_sum
    return (box_sum,)


@app.cell(hide_code=True)
def run_sum_cell(box_sum):
    run_sum = L.js_run("sumFrom", box_sum.value)
    run_sum
    return (run_sum,)


@app.cell(hide_code=True)
def check_sum(run_sum):
    result_sum = L.check("sumFrom", run_sum)
    result_sum
    return (result_sum,)


@app.cell(hide_code=True)
def hints_sum():
    L.hints(
        "sumFrom",
        [
            "Follow the worked example: first the answer of `fetch`, then its JSON, then a loop over the list.",
            "`const numbers = await response.json();` gives the list. Then add up its numbers with `for (const number of numbers)`.",
            "`const response = await fetch(url);` and `const numbers = await response.json();`, then `let total = 0;`, the loop with `total += number;`, and `return total;`.",
        ],
        why="""```javascript
async function sumFrom(url) {
  const response = await fetch(url);
  const numbers = await response.json();
  let total = 0;
  for (const number of numbers) {
    total += number;
  }
  return total;
}
```

Both `fetch` and `.json()` give a promise; `await` waits for each value before the next line uses it.""",
    )
    return


@app.cell(hide_code=True)
def text_worker():
    mo.md(r"""
    ### Exercise 3. Beside the page
    """)
    return


@app.cell(hide_code=True)
def choice_worker_cell():
    choice_worker = L.choice(
        "worker",
        question="A page runs a model that computes for two seconds for every answer. Why does it run the model in a web worker?",
        options={
            "faster": "A worker computes faster than the page itself",
            "free": "So the page keeps answering clicks",
            "download": "Only a worker is allowed to download a model",
        },
        correct="free",
        why={
            "free": "The worker computes beside the page, so the page stays free for the user; it gets the answer as a message.",
            "faster": "The work is the same, on the same computer. What changes for the page while the worker computes?",
            "download": "A page can download a model itself, with `fetch`. What would two seconds of computing do to the page?",
        },
    )
    choice_worker
    return (choice_worker,)


@app.cell(hide_code=True)
def text_import():
    mo.md(r"""
    ### Exercise 4. Load a library
    """)
    return


@app.cell(hide_code=True)
def choice_import_cell():
    choice_import = L.choice(
        "import",
        question="A web page needs the function `pipeline` from the module at `https://cdn.jsdelivr.net/npm/@huggingface/transformers`. Which line loads it?",
        options={
            "module": "`import { pipeline } from \"https://cdn.jsdelivr.net/npm/@huggingface/transformers\";`",
            "python": "`from \"https://cdn.jsdelivr.net/npm/@huggingface/transformers\" import { pipeline };`",
            "require": "`const { pipeline } = require(\"https://cdn.jsdelivr.net/npm/@huggingface/transformers\");`",
        },
        correct="module",
        why={
            "module": "A module imports a name, between braces, from another module, by its address.",
            "python": "That is the order of Python's `from ... import`. In JavaScript, what comes first?",
            "require": "`require` is how Node.js loaded code before JavaScript had modules, and a browser does not know it. How does a module load a name?",
        },
    )
    choice_import
    return (choice_import,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### The page, or a web worker

    The same work twice: once on the page itself, once in a web worker. After a press the dot moves for a second, then the work starts, and the dot moves on for as long as the page can draw it; the demo measures the longest pause between two drawings, on your own computer.
    """)
    return


@app.cell(hide_code=True)
def demo_freeze():
    _words = {"page": "Compute on the page", "worker": "Compute in a web worker", "busy": "Computing …",
              "result": "{where}: the work took {work} ms, and the dot stood still for at most {gap} ms.",
              "where_page": "On the page", "where_worker": "In a web worker", "frames": "{n} drawings so far"}
    mo.ui.anywidget(C.freeze(_words, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A JavaScript function works like the functions of [Functions](/course/open/py-functions?lang=en): it takes values and gives one back with `return`. `fetch` and `await` make the same kind of request as the calls to a model in this module, and a web worker keeps the page free while a model computes on the device, as in [AI on your own device](/course/open/build-on-device?lang=en). In the next lesson, *Your model in the browser*, a small network that reads digits goes to the browser as JSON, and a function like your `dot` does its sums there.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_import, choice_worker, result_dot, result_sum):
    L.wrap_up(result_dot, result_sum, choice_worker, choice_import)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
