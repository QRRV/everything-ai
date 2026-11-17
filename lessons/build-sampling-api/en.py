# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-sampling-api. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-sampling-api", "en")
    C = L.checks   # the course's recorded runs, the stand-in of the checks, and the demo with your ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Calling a model over HTTP](/course/open/build-call-api?lang=en), and one about [Top-k and top-p sampling](/course/open/lm-top-k-top-p?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_finish():
    quick_finish = L.choice(
        "quick-finish",
        question="An answer from the API has `\"finish_reason\": \"length\"`. What does that say?",
        options={
            "limit": "It reached `max_tokens` and was cut off",
            "done": "The model ended the answer itself",
            "long": "The question was too long for the model to read",
        },
        correct="limit",
        why={
            "limit": "`\"length\"` means the answer reached the most tokens the request allowed; `\"stop\"` means it ended another way.",
            "done": "Then it would say `\"stop\"`. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "long": "`finish_reason` is about the answer, not the question. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_finish
    return (quick_finish,)


@app.cell(hide_code=True)
def quick_check_top_p():
    quick_top_p = L.choice(
        "quick-top-p",
        question="The probabilities of the next token are `[0.5, 0.3, 0.15, 0.05]`. Which tokens does top-p with p = 0.75 keep?",
        options={
            "all": "All four, each with a smaller share",
            "two": "The first two",
            "three": "The first three",
        },
        correct="two",
        why={
            "two": "0.5 alone is less than 0.75; 0.5 + 0.3 = 0.8 reaches it. Top-p keeps as few tokens as it can that together hold at least p.",
            "all": "Top-p cuts: the tokens after the ones it needs get probability 0. See [Top-k and top-p sampling](/course/open/lm-top-k-top-p?lang=en).",
            "three": "Count from the top: after how many tokens is the sum at least 0.75? See [Top-k and top-p sampling](/course/open/lm-top-k-top-p?lang=en).",
        },
    )
    quick_top_p
    return (quick_top_p,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You know how a model picks its next token: at a temperature, from the tokens top-p keeps. Through most APIs you choose that per request. A test of your app needs the same answer every time it runs; a button that suggests names needs a new one each time; and an answer that only has to give one line should stop after that line, because every token it writes takes time and, at a hosted API, money. All of that goes in the body of the request, next to the messages.

    **An everyday example.** A coffee machine at work has buttons: stronger or milder, a small cup or a large one. The coffee is the same machine's every time, but you set how it comes out, cup by cup. That is only a comparison; below you see the settings of a model's API, and what each one did to Qwen3's answers.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | `"temperature"` | in the body: the temperature the server samples at; `0` is greedy decoding, the same answer every time; the course's server takes a number from 0 to 2 |
    | `"top_p"` | in the body: the p of top-p sampling, from 0 to 1 |
    | `"seed"` | in the body: a whole number where the server's random draw starts; the same seed with the same request gives the same answer again |
    | **max tokens** | `"max_tokens"`: the most tokens the answer may have; an answer that reaches it is cut off, with `finish_reason` `"length"` |
    | **stop sequence** | `"stop"`: a text, or a list of at most four, at which the server ends the answer; the stop sequence itself is not in the answer, and `finish_reason` is `"stop"` |

    ### The same question, three settings

    The course asked Qwen3 through its own server for a name for a bakery, eight times at each of three settings, with the seeds 0 to 7:
    """)
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        At temperature 0 every seed gave the same name: greedy decoding takes the most likely token each time, so there is no draw for a seed to change. At 0.7 with top_p 0.8, what Qwen3's makers advise for this model, the eight runs gave {advised} different names. At 1.5 with top_p 1.0 they gave {hot}, bolder ones, and {sentences} of them was a sentence instead of the name only, until `max_tokens` stopped it: bolder sampling gives more variety, and more room for an answer that leaves the instruction behind.

        The course asked the seeds 0, 1 and 2 at 0.7 once more, and got the same three names again: `Bakerbloom`, `Bakery` and `smallbakery`. So a seed makes a sampled answer repeatable: the answer still comes from a draw, but the same draw each time. That holds for the same model on the same computer and software: on yours, seed 0 may give another name than here.
        """.format(advised=_f["advised"], hot=_f["hot"], sentences=_f["sentences"])),
        L.watch_out("A request without `\"temperature\"` gets the server's own. The course's server then samples as Qwen3's makers advise, at 0.7 with top_p 0.8, and it always keeps only the 20 most likely tokens as well (top-k), for which OpenAI's API has no setting. A hosted API has its own defaults, and not every one takes a seed, or even a temperature, for every model; the ones that take a seed promise only that the same seed mostly gives the same answer (examples in November 2026)."),
    ])
    return


@app.cell(hide_code=True)
def text_ends():
    mo.md(r"""
    ### Where an answer ends

    An answer ends in one of three ways: the model ends it itself, it reaches `max_tokens`, or it reaches a stop sequence. The course asked Qwen3 for three fruits, one per line, three ways:
    """)
    return


@app.cell(hide_code=True)
def picture_ends():
    mo.Html(C.ends_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_ends_after():
    _f = C.findings()
    mo.md(r"""
    With `"stop": ["\n"]` the answer ends at the first line break, which is not in the answer, and `finish_reason` is `"stop"`, just as when the model ends an answer itself. The model stopped writing there: {first} tokens instead of {lines}. With `"max_tokens": 3` the answer is cut after {short} tokens, in the middle of what the model meant to write, and `finish_reason` says `"length"`. Both are shorter; only the stop sequence ends where you want. So read `finish_reason` before you use an answer: `"length"` means it may be missing its end.
    """.format(first=_f["first"]["tokens"], lines=_f["lines"]["tokens"], short=_f["short"]["tokens"]))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: set how the model samples

    1. **How much it varies:** `"temperature"`: `0` for the same answer every time, about `0.7` for what the model's makers advise, higher for bolder answers that keep to the instruction less often. `"top_p"` cuts away the unlikely tokens.
    2. **The same answer again:** a fixed `"seed"`, with the same request.
    3. **Where it ends:** `"max_tokens"` as a limit on length and cost, and `"stop"` for a text where the answer should end.
    4. **Check how it ended:** `answer["choices"][0]["finish_reason"]`: `"stop"`, or `"length"` for an answer that was cut off.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A test of an app checks that its client reads an answer right. The test must get the same answer every time it runs, but from the sampling the app really uses. Which body?

    ```python
    body = {"model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Suggest a name for a small bakery. Answer with the name only."}],
            "max_tokens": 12, "temperature": 0.7, "top_p": 0.8, "seed": 0}
    ```

    1. **How much it varies:** `0.7` and `0.8`, the settings of the app itself.
    2. **The same answer again:** `"seed": 0`. In the course's runs, seed 0 gave `Bakerbloom`, the first time and the second time.
    3. **Where it ends:** `"max_tokens": 12` is plenty for a name.
    4. **How it ended:** `finish_reason` was `"stop"`: the model ended the name itself.

    With `"temperature": 0` the test would get the same answer too, but from greedy decoding, which is not what the app does.
    """)
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Temperature and seed

        `ask(url, question, temperature, seed)` sends a question already. Make it send the `temperature` and the `seed` it gets. The check calls your `ask` against a stand-in of the server, once with a temperature of 0, and looks at what your request carried.
        """),
        L.task("Add `\"temperature\"` and `\"seed\"` to the body, with the values `ask` gets."),
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
            "The settings go in the body, next to `\"model\"`, `\"messages\"` and `\"max_tokens\"`, as in the worked example.",
            "Each one is a key with the argument as its value: `\"temperature\": temperature`.",
            "The body ends in `\"max_tokens\": 12, \"temperature\": temperature, \"seed\": ___}`.",
        ],
        why="""```python
body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 12,
        "temperature": temperature, "seed": seed}
```

The arguments themselves go in the body, so a temperature of 0 goes along as well.""",
    )
    return


@app.cell(hide_code=True)
def text_first():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One line, and why it ended

        `first_line(url, question)` should give back only the first line of the answer, and why the answer ended. Let the server end the answer at the first line break, and give back the text and the `finish_reason` together. The check asks one question whose answer has three lines, and one whose answer has no line break and runs to 20 tokens.
        """),
        L.task("Add a stop sequence with `\"\\n\"` to the body, and return the text and the `finish_reason`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first():
    box_first = L.code_box("first_line", lines=12)
    box_first
    return (box_first,)


@app.cell(hide_code=True)
def check_first(box_first):
    result_first = L.check("first_line", box_first.value)
    result_first
    return (result_first,)


@app.cell(hide_code=True)
def hints_first():
    L.hints(
        "first_line",
        [
            "Two changes: one key in the body, and a last line that gives back two things.",
            "In the body: `\"stop\": [\"\\n\"]`. The `finish_reason` sits in the choice, next to the message: `answer[\"choices\"][0][\"finish_reason\"]`.",
            "Take `choice = answer[\"choices\"][0]`, and make the last line `return ___, choice[\"finish_reason\"]`.",
        ],
        why="""```python
def first_line(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 20,
            "stop": ["\\n"]}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    choice = answer["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]
```

The server ends the answer at the line break; `"length"` would mean it reached 20 tokens before one came.""",
    )
    return


@app.cell(hide_code=True)
def text_names():
    mo.md(r"""
    ### Exercise 3. Ten names
    """)
    return


@app.cell(hide_code=True)
def choice_names_cell():
    choice_names = L.choice(
        "names",
        question="A button in your app asks the API for a name for a new café, ten times with the same question. You want ten different names. Which settings?",
        options={
            "same-seed": "`\"temperature\": 0.9`, and the same `\"seed\"` every time",
            "greedy": "`\"temperature\": 0`, and a new `\"seed\"` every time",
            "new-seed": "`\"temperature\": 0.9`, and a new `\"seed\"` every time",
        },
        correct="new-seed",
        why={
            "new-seed": "Sampling lets the names differ, and a new seed gives a new draw each time.",
            "same-seed": "The same seed with the same request gives the same answer: ten times the same name. What should change from one request to the next?",
            "greedy": "At temperature 0 there is no draw, so the seed changes nothing: at 0 every seed gave the bakery the same name.",
        },
    )
    choice_names
    return (choice_names,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Ask Qwen3 three times

    Once exercise 1 is right, your `ask` can talk to the real thing: the course starts its own server on your computer, with Qwen3-0.6B behind it, and your function asks your question three times, with the seeds 0, 1 and 2, at the temperature you choose.
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
    question_box = mo.ui.text(value=C.NAME, label="Your question", full_width=True)
    temperature_slider = mo.ui.slider(0, 2, step=0.1, value=0.7, label="temperature", show_value=True)
    send_button = mo.ui.run_button(label="Ask three times with your ask")
    mo.vstack([question_box, temperature_slider, send_button])
    return question_box, send_button, temperature_slider


@app.cell(hide_code=True)
def demo_answer(box_ask, question_box, send_button, temperature_slider):
    _ask = C.learner_ask(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _ask is None:
        _out = mo.md("*Finish exercise 1 first: the demo asks with your own `ask`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `ask` has sent the question; the server loads Qwen3 and answers three times …*"))
        _words = {"title": "What came back to your ask, three times", "seed": "seed",
                  "done": "Temperature {t}: {different} different answers, in {seconds} seconds together.",
                  "failed": "Your ask stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_ask, question_box.value, temperature_slider.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The temperature of [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en) and the p of [Top-k and top-p sampling](/course/open/lm-top-k-top-p?lang=en) are now two keys in the body of [Calling a model over HTTP](/course/open/build-call-api?lang=en), and a seed makes a sampled answer repeatable, as it made the dice of [Using code from modules](/course/open/py-imports?lang=en) repeatable. In the next lesson, [System prompts and response prefill](/course/open/build-system-prompt?lang=en), you tell the model how to behave for a whole conversation, and write the first words of its answer yourself. A later lesson asks the same question several times on purpose, and takes the answer that comes back most often.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_names, result_ask, result_first):
    L.wrap_up(result_ask, result_first, choice_names)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
