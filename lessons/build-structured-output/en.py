# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-structured-output. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-structured-output", "en")
    C = L.checks   # the course's recorded runs, the stand-in of the checks, and the demo with your ask_json


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [System prompts and response prefill](/course/open/build-system-prompt?lang=en) and [Catching and raising errors](/course/open/py-exceptions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_prefill():
    quick_prefill = L.choice(
        "quick-prefill",
        question="Your request ends with the prefill `{\"city\": \"`, and the answer that comes back holds `Paris\"}`. Which text can `json.loads` read?",
        options={
            "rest": "`Paris\"}`, the text of the answer just as it came back",
            "prefill": "`{\"city\": \"`, the prefill on its own",
            "joined": "`{\"city\": \"Paris\"}`, prefill and answer joined",
        },
        correct="joined",
        why={
            "joined": "The answer holds only what the model wrote after the prefill; put together, they are the JSON text.",
            "rest": "That is only what the model wrote after the prefill, and `json.loads` cannot read it on its own. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
            "prefill": "That is the start you wrote; the model's part is missing. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
        },
    )
    quick_prefill
    return (quick_prefill,)


@app.cell(hide_code=True)
def quick_check_except():
    quick_except = L.choice(
        "quick-except",
        question="`json.loads` stops with a `ValueError` for a text that is not JSON. What does this program show?\n\n```python\nimport json\n\ntry:\n    value = json.loads(\"Sure! Here it is.\")\nexcept ValueError:\n    value = None\nprint(value)\n```",
        options={
            "none": "`None`",
            "text": "`Sure! Here it is.`",
            "stops": "An error message, and the program stops",
        },
        correct="none",
        why={
            "none": "`json.loads` stops with a `ValueError`, so Python jumps to `except ValueError:`, and `value` becomes `None`; the program goes on.",
            "text": "`json.loads` does not give back a text it cannot read: it stops with a `ValueError`. See [Catching and raising errors](/course/open/py-exceptions?lang=en).",
            "stops": "The `except ValueError:` catches exactly that error, so the program goes on. See [Catching and raising errors](/course/open/py-exceptions?lang=en).",
        },
    )
    quick_except
    return (quick_except,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An app that asks a model for data, such as the name and the price of a product in a text, does not want a story back. It wants JSON that it can read, with the keys it needs, and a number where it computes with a number. A model writes text, one token at a time, and gets that right often, but not always. One answer that is almost JSON stops your program; worse, a price that is a text can slip through. This lesson makes the answer something your code can rely on: it checks every answer against a description of what it must hold, and asks again with what was wrong.

    **An everyday example.** The order form of a web shop does not accept *about twenty* in the field for the number. It says what is wrong, *enter a whole number*, and you fill it in again. That is only a comparison; below you see how often Qwen3's JSON was not what an app needs, and what asking again did.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **structured output** | an answer in a fixed form that a program can read without guessing, such as JSON with given keys and types |
    | **JSON mode** | a setting in the request, `"response_format": {"type": "json_object"}`, with which the server makes sure the answer is valid JSON |
    | **JSON Schema** | a standard for describing, in JSON, what a piece of JSON must hold: which keys, which of them are required, and the type of each value; one such description is *a JSON schema* |
    | **isinstance(x, t)** | `True` when the value `x` is of the type `t`, such as `isinstance("649", str)`; with a tuple of types, such as `(int, float)`, when it is one of them |
    | **not** | turns `True` into `False` and `False` into `True`; `key not in value` is `True` when the key is missing |

    ### Almost JSON

    The course asked Qwen3 for the details of a bike as JSON, eight times, through its own server:

    ```
    Give the name, color and price of this bike as JSON, with the keys name, color and price.

    Our new Cityline is a grey city bike with seven gears, for 649 euros.
    ```

    Each answer below is on one line: ↵ is a line break, and a price that is a text is marked.
    """)
    return


@app.cell(hide_code=True)
def picture_plain():
    mo.Html(C.plain_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_plain():
    _f = C.findings()
    mo.md(r"""
    All {fenced} answers were in a code block that starts with three backticks, and `json.loads` could read {read} of them. An app could take the backticks off, but even inside them, the price was the text `"649"` in {text} of the {runs}, and not the number 649: your code cannot compute with that, and `"649" + 10` stops with an error. In [System prompts and response prefill](/course/open/build-system-prompt?lang=en), a prefill fixed how an answer starts; this lesson has two more ways: JSON mode, and a check of what the answer holds.

    ### JSON mode

    With `"response_format": {{"type": "json_object"}}` in the request, the course's server switches on JSON mode. The model writes as always, one token at a time, but before the server keeps a token, it checks whether the text so far can still become JSON. A token after which it cannot, such as the backticks at the start, is put aside, and the server draws again from the other tokens. As soon as the JSON object is whole, the answer ends. The same question, with JSON mode:
    """.format(fenced=_f["plain_fenced"], read=_f["plain_json"], text=_f["plain_text_price"], runs=_f["runs"]))
    return


@app.cell(hide_code=True)
def picture_mode():
    mo.Html(C.mode_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_mode():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        This time, `json.loads` could read all {read}. The answers looked the same as the JSON in the code blocks before, with the same keys and the same layout, and in {text} of the {runs} the price was again the text `"649"`. JSON mode makes the answer valid JSON; it does not know which keys and types your app needs.
        """.format(read=_f["mode_json"], text=_f["runs"] - _f["mode_fit"], runs=_f["runs"])),
        L.watch_out("JSON mode cannot finish an answer that `max_tokens` stops first: `finish_reason` is then `\"length\"`, and the JSON is not whole. Give the answer room enough, and check it anyway."),
    ])
    return


@app.cell(hide_code=True)
def text_schema():
    mo.vstack([
        mo.md(r"""
        ### A schema

        What your app needs, you write down once, in a JSON schema: here the bike's. `"properties"` says which keys there are and what type each value has, and `"required"` which of the keys must be there:

        ```python
        SCHEMA = {"type": "object",
                  "properties": {"name": {"type": "string"}, "color": {"type": "string"},
                                 "price": {"type": "number"}},
                  "required": ["name", "color", "price"]}
        ```

        JSON Schema has many more rules, such as a lowest value, or a list of allowed words. Your code checks every answer against the schema: can `json.loads` read it, is every required key there, and does each value have its type? The answers above with `"649"` do not fit, because price must be a number.
        """),
        L.watch_out("The check in this lesson is small on purpose: it knows two types and the required keys. In Python, `isinstance(True, int)` is also `True`, so it would let `true` through as a price. The library `jsonschema` checks every rule of the standard, that one too; Pydantic checks against a Python class instead, and refuses `\"649\"` and `true` as a number only in its strict mode (examples in November 2026)."),
    ])
    return


@app.cell(hide_code=True)
def text_again():
    mo.md(r"""
    ### Asking again with what was wrong

    When an answer does not fit, the app can ask again. Not with the same question: then the model does not know what was wrong, and may write the same again. Instead, the app adds the answer to the conversation as the assistant's, and a message of the user that names the problems. The course did that for every run in JSON mode, with at most three tries:
    """)
    return


@app.cell(hide_code=True)
def picture_again():
    mo.Html(C.again_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_again_after():
    _f = C.findings()
    _first = C.answers("json-mode")[0]["tries"]
    mo.vstack([
        mo.md(r"""
        {again} of the {runs} runs needed a second try, and all {again} fit after it: in the end, {fit} of the {runs} answers fit the schema. Each try is a whole new request, and the conversation grows each time: the first try was {one} input tokens, the second {two}. So keep the number of tries low, and give back `None` when nothing fits: then the rest of your app can say so, instead of going on with a wrong answer.
        """.format(again=_f["again"], runs=_f["runs"], fit=_f["final_fit"], one=_first[0]["prompt_tokens"], two=_first[1]["prompt_tokens"])),
        L.watch_out("Other APIs and servers go further, with *structured outputs*: you send the schema itself in the request, and the server makes sure the answer fits it while the model writes, as JSON mode does for valid JSON. OpenAI's API takes a schema in `response_format`, the server of llama.cpp takes one too, and Ollama takes one in `format` (examples in November 2026). The course's server has only JSON mode: ask it for a schema, and it answers with an error that says so. Even with structured outputs, check the answer in code: a server supports only part of JSON Schema, and an answer that `max_tokens` cut off is still not whole."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: JSON you can rely on

    1. **Write down what you need:** a JSON schema with the keys, the type of each value, and which keys are required.
    2. **Ask for JSON:** name the keys in the prompt, and switch on JSON mode, or structured outputs where the API has them.
    3. **Check every answer in code:** `json.loads`, the required keys, and the type of each value.
    4. **Ask again with what was wrong:** add the answer and a message with the problems, a few times at most; when nothing fits, give back `None`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.findings()["worked"]
    mo.md(r"""
    ### Worked example

    The first run of the course in JSON mode. Qwen3's first answer was:

    ```json
    {first}
    ```

    1. **What you need:** the bike's schema above.
    2. **Ask for JSON:** the question names the keys, and JSON mode is on.
    3. **Check:** `json.loads` reads it, and it is a dictionary; `name`, `color` and `price` are all there. `name` and `color` are strings, as the schema says, but `"649"` is a string too, where the schema says `number`. One problem: `{found}`.
    4. **Ask again:** the conversation gets two messages: the answer above as the assistant's, and the user's `{fix}` Qwen3's second answer had `"price": 649`, and it fits, so the function gives back `{value}`.
    """.format(first=_w["first"], found=_w["found"][0], fix=C.fix_message(_w["found"]),
               value=repr(_w["value"])))
    return


@app.cell(hide_code=True)
def text_problems():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Check an answer against the schema

        `problems(text, schema)` gives back a list of what is wrong with an answer, in words the model can act on, and an empty list when it fits. The box already checks whether `json.loads` can read the answer, whether it is a JSON object, and whether every required key is there. Add the check of the types: for each key in `schema["properties"]` that is in the answer, whether its value has the type the schema names. `TYPES` gives the Python type for each name, and `WORDS` how to say it. When it is right, the check also keeps it as a file, `work/projects/app/schema.py`: a later step, [Project: your assistant, prompted well](/course/open/build-project-prompts?lang=en), uses it.
        """),
        L.task("After the loop over the required keys, add a loop over the keys and rules in `schema[\"properties\"]` that puts each value of the wrong type in `found`, in words such as `price must be a number`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_problems():
    box_problems = L.code_box("problems", lines=22)
    box_problems
    return (box_problems,)


@app.cell(hide_code=True)
def check_problems(box_problems):
    result_problems = L.check("problems", box_problems.value)
    result_problems
    return (result_problems,)


@app.cell(hide_code=True)
def hints_problems():
    L.hints(
        "problems",
        [
            "Go through `schema[\"properties\"].items()`: each key with its rule, such as `\"price\"` with `{\"type\": \"number\"}`. A value is a problem when it is there and `isinstance` says it is not of its type.",
            "`TYPES[rule[\"type\"]]` is the Python type, such as `(int, float)`. The condition: the key is in `value`, and `isinstance` says its value is not of that type. The text: `key + \" must be \" + WORDS[rule[\"type\"]]`.",
            "`for key, rule in schema[\"properties\"].items():`, under it `if key in value and ___:`, and under that the `found.append(...)` with the text.",
        ],
        why="""```python
    for key, rule in schema["properties"].items():
        if key in value and not isinstance(value[key], TYPES[rule["type"]]):
            found.append(key + " must be " + WORDS[rule["type"]])
    return found
```

A key that is missing is already in `found`, so the type loop skips it with `key in value`. `(int, float)` lets both `649` and `649.5` through as a number.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Ask again with what was wrong

        `ask_json(url, messages, schema, tries)` asks in JSON mode, with `ask`, and checks each answer with `problems`: the course's own, as in exercise 1. When an answer fits, it gives back what `json.loads` makes of it. But after an answer that does not fit, it asks the same question again. Make it add two messages to the conversation before the next try: the answer, with the role `assistant`, and a message with the role `user` that names the problems, such as *Fix these problems: price must be a number. Answer with the whole JSON again.* Build a new list with `+`: `[1, 2] + [3]` gives the new list `[1, 2, 3]`, and leaves `[1, 2]` as it was. The check calls your `ask_json` against a stand-in of the server, whose first answer to the bike's question has the price as a text.
        """),
        L.task("Inside the loop, after the `if`, make `messages` the conversation with the answer and your message with the problems added."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask_json", lines=25)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask_json", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask_json",
        [
            "After the `if`, the answer did not fit. The next try sends `messages` again: make it the old conversation with two new messages after it.",
            "The two messages are dictionaries with `\"role\"` and `\"content\"`: the answer is `text`, with the role `assistant`; your message holds the problems joined with `\"; \".join(found)`, with the role `user`.",
            "First `fix = \"Fix these problems: \" + \"; \".join(found) + \". Answer with the whole JSON again.\"`; then `messages = messages + [___, {\"role\": \"user\", \"content\": fix}]`.",
        ],
        why="""```python
def ask_json(url, messages, schema, tries):
    for attempt in range(tries):
        text = ask(url, messages)
        found = problems(text, schema)
        if found == []:
            return json.loads(text)
        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": fix}]
    return None
```

With `+`, `messages` becomes a new list each time, and the list the function got stays as it was. After the last try, the loop ends, and the function gives back `None`.""",
    )
    return


@app.cell(hide_code=True)
def text_guarantee():
    mo.md(r"""
    ### Exercise 3. What JSON mode makes sure of
    """)
    return


@app.cell(hide_code=True)
def choice_guarantee_cell():
    choice_guarantee = L.choice(
        "guarantee",
        question="Your app asks the course's server for the bike's details, with JSON mode on, and the answer comes back with `finish_reason` `\"stop\"`. Which answer can that still be?",
        options={
            "fence": "The JSON in a code block with backticks",
            "keys": "`{\"price\": \"649\"}`, without name and color",
            "cut": "`{\"name\": \"Cityline\"` without its closing brace",
        },
        correct="keys",
        why={
            "keys": "JSON mode makes sure the answer is valid JSON, not that it fits your schema: missing keys and a price that is a text get through. That is what your `problems` is for.",
            "fence": "Backticks cannot start valid JSON, so JSON mode puts them aside, as in the eight answers above. What does JSON mode not know about?",
            "cut": "In JSON mode, the answer ends as soon as the object is whole, and not before; in practice, only `max_tokens` cuts it off, and then `finish_reason` is `\"length\"`. What does JSON mode not know about?",
        },
    )
    choice_guarantee
    return (choice_guarantee,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 2 is right, you can try it on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo asks for the name, the color and the price as JSON, about the text in the box, with your own `ask_json` and the bike's schema, at most three times. Change the text: a bike without a price, or with the price in words.
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
def text_box_cell():
    text_box = mo.ui.text_area(value=C.BIKE_TEXT, label="The text about the bike", full_width=True, rows=2)
    send_button = mo.ui.run_button(label="Ask with your ask_json")
    mo.vstack([text_box, send_button])
    return send_button, text_box


@app.cell(hide_code=True)
def demo_answer(box_ask, send_button, text_box):
    _ask = C.learner_ask_json(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _ask is None:
        _out = mo.md("*Finish exercise 2 first: the demo asks with your own `ask_json`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not text_box.value.strip():
        _out = mo.md("*Write a text about a bike in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "What Qwen3 answered, and what your ask_json gave back", "try": "Try {n}:",
                  "found": "problems: {p}", "fits": "fits the schema", "gave": "Your ask_json gave back:",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_ask, text_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The `json.loads` of [Data as text: JSON](/course/open/py-json?lang=en) now reads a model's answers, with the `try` and `except` of [Catching and raising errors](/course/open/py-exceptions?lang=en) around it. The response prefill of [System prompts and response prefill](/course/open/build-system-prompt?lang=en) fixed how an answer starts; the schema checks all of it. In the next lesson, [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en), your prompts move out of your code into templates with fields, so that you know which version gave which answers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_guarantee, result_ask, result_problems):
    L.wrap_up(result_problems, result_ask, choice_guarantee)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
