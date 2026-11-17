# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-prompt-templates. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-prompt-templates", "en")
    C = L.checks   # the course's recorded runs of two versions, and the demo with a version of your own


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [System prompts and response prefill](/course/open/build-system-prompt?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="Your app adds the same instructions to every conversation. Where do they go?",
        options={
            "first": "In a first message with the role `system`",
            "user": "At the end of each message the user types",
            "prefill": "In a response prefill, after the question",
        },
        correct="first",
        why={
            "first": "The system prompt holds for the whole conversation: the app adds it, and the user does not see it.",
            "user": "Then the user's words and your instructions get mixed up, in every message. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
            "prefill": "A prefill writes the first words of the answer; it does not give instructions for the whole conversation. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In an app, nobody types the prompt: code builds it for every request, around what the user sent. And the words of a prompt change. You add a sentence because an answer went wrong, and a week later the answers are worse, or better: then you want to know which words gave which answers. That works when a prompt is not a string somewhere in your code, but a template: fixed words with fields that code fills in, in a file of its own, with a version number that goes into the log with every answer.

    **An everyday example.** A letter from your bank has the same text for everyone, with your name and your amount filled in. When the bank changes a sentence, that is a new version of the letter, and the bank knows who got which version. That is only a comparison; below you see two versions of one prompt, and what each gave.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **prompt template** | a prompt with fields, such as `{text}`, that code fills in for each request, so that every call uses the same tested words |
    | **prompt version** | one fixed wording of a template, with a number of its own; every change to the words gets a new number, and each answer is logged with the version that made it |
    | **.format(...)** | fills in the fields of a text: `"Hi {name}".format(name="Sam")` gives `"Hi Sam"`; a brace that should stay in the text is written twice, `{{` or `}}` |

    ### A template with a field

    The course's question about a bike, as a template: the fixed words, and the field `{text}` for the text about the bike. `.format(text=...)` makes a prompt of it for each text:
    """)
    return


@app.cell(hide_code=True)
def picture_fill():
    mo.Html(C.fill_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_fill():
    mo.md(r"""
    In code:

    ```python
    TEMPLATE = "Give the name, color and price of this bike as JSON, with the keys name, color and price.\n\n{text}"
    prompt = TEMPLATE.format(text="The Trailhawk is a green mountain bike with 21 gears. Price: 1199 euros.")
    ```

    The text goes in as it is. If it holds braces itself, they stay: `.format` reads only the braces of the template.

    ### Two versions

    The course sent this template, version 1, eight times for each of the three texts, in **JSON mode**: the server then lets the model write only valid JSON, which says nothing yet about which keys are in it. Each answer counts as fitting when `json.loads` reads it, with `name`, `color` and `price` in it, and the price a number. Version 2 adds one sentence: *The price is a number, without quotes, such as {"price": 100}.* The log of both versions:
    """)
    return


@app.cell(hide_code=True)
def picture_log():
    mo.Html(C.log_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_log():
    _f = C.findings()
    mo.md(r"""
    Version 1: {v1} of {k} answers fit, and in each of the {wrong} others, the price was a text, such as `"649"`. Version 2: {v2} of {k}. Without the version in the log, these would be {all} answers of one prompt, and the difference would be gone. Eight runs of three texts is not much: version 2 is better here, for these texts and this model. Another model, or other texts, can need other words: that is why every change gets a number of its own, so that you can run the same examples again and compare.

    ### Braces in a template

    Version 2 has braces of its own, in the example. For `.format`, every pair of braces is a field. Written once, `{{"price": 100}}` would be a field with the name `"price"` (`.format` reads what comes after the colon as a format), and `.format` stops with a KeyError, Python's error for a key that has no value: no value came for that field. A brace that should stay in the prompt is written twice. The template below is written over two lines: two strings next to each other inside parentheses are one string to Python.

    ```python
    TEMPLATE_V2 = ("Give the name, color and price of this bike as JSON, with the keys name, color and price. "
                   'The price is a number, without quotes, such as {{{{"price": 100}}}}.\n\n{{text}}')
    ```

    From `{{{{"price": 100}}}}`, `.format` makes `{{"price": 100}}`, and it fills in `{{text}}`.
    """.format(v1=_f["total"][1], v2=_f["total"][2], k=_f["asked"][1], wrong=_f["wrong"], all=_f["asked"][1] + _f["asked"][2]))
    return


@app.cell(hide_code=True)
def text_file():
    mo.vstack([
        mo.md(r"""
        ### In a file, with a version

        A template belongs in a file of its own, not in the middle of your code: then you can read all its words at once, and version control, such as Git, keeps every version of the file. One way is a JSON file for each template:

        ```json
        {"name": "bike-details", "version": 2,
         "template": "Give the name, color and price of this bike as JSON, ... such as {{\"price\": 100}}.\n\n{text}"}
        ```

        Your code reads it with `json.load`, fills in the template, and puts the name and the version in the log, next to each answer. Change a word, and the version goes up: the words of a version that has given answers never change.
        """),
        L.watch_out("Libraries help with templates. Jinja2 has loops and conditions in a template, and tools such as Langfuse and MLflow keep the versions of a prompt, with labels such as *production* (examples in November 2026). The idea stays the same: fixed words, fields, and a version that goes with every answer."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a prompt as code

    1. **Fixed words, fields for the rest:** write the prompt as a template, with a field for each part that changes, such as `{text}`.
    2. **Double the template's own braces:** `{{` and `}}` for every brace that should stay in the prompt.
    3. **A file of its own, with a name and a version:** every change to the words is a new version, and version control keeps the old ones.
    4. **Log the version with every answer:** then you can count, for each version, what went right.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _a = C.answers(2, "pixie")[0]
    mo.md(r"""
    ### Worked example

    Version 2, filled with the Pixie's text, and its first run.

    1. **Fixed words and a field:** version 2 of the template, with `text` the Pixie's text: `{text}`
    2. **Braces:** `.format` turns `{{{{"price": 100}}}}` into `{{"price": 100}}`, and puts the text in the field, so the prompt ends with `such as {{"price": 100}}.`, an empty line, and the text.
    3. **Name and version:** `bike-details`, version 2.
    4. **The log:** with seed 0, Qwen3 answered `{answer}`. It fits, so the line in the log is `{{"prompt": "bike-details", "version": 2, "text": "pixie", "seed": 0, "fits": true}}`. The name in it is *{name}*, not *Pixie*: `fits` checks the keys and the types, not whether each value is right.
    """.format(text=C.TEXTS["pixie"], answer=" ".join(_a["text"].split()), name=C.value(_a["text"])["name"]))
    return


@app.cell(hide_code=True)
def text_prompt():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Braces in the template

        The box has version 2 of the bike template, but with the braces of the example written once, so `bike_prompt` stops with a KeyError. Make the template right, so that `bike_prompt(text)` gives the prompt with the example in it, and the text in the field.
        """),
        L.task("Write each brace of the example in `TEMPLATE` twice, and leave the field `{text}` as it is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompt():
    box_prompt = L.code_box("bike_prompt", lines=7)
    box_prompt
    return (box_prompt,)


@app.cell(hide_code=True)
def check_prompt(box_prompt):
    result_prompt = L.check("bike_prompt", box_prompt.value)
    result_prompt
    return (result_prompt,)


@app.cell(hide_code=True)
def hints_prompt():
    L.hints(
        "bike_prompt",
        [
            "`.format` reads every pair of braces as a field. Which braces in `TEMPLATE` are a field, and which should stay in the prompt?",
            "`{text}` is the field. The braces around `\"price\": 100` should stay, so each of them is written twice.",
            "The second line of `TEMPLATE` becomes `'The price is a number, without quotes, such as ___\"price\": 100___.\\n\\n{text}')`.",
        ],
        why="""```python
TEMPLATE = ("Give the name, color and price of this bike as JSON, with the keys name, color and price. "
            'The price is a number, without quotes, such as {{"price": 100}}.\\n\\n{text}')
```

`.format` makes one brace of each double one, and fills in the field with single braces.""",
    )
    return


@app.cell(hide_code=True)
def text_tally():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What fit, per version

        The log has a dictionary for each answer, with the template's `"version"`, and `"fits"`: `True` when the answer fits. Write `fits_per_version(log)`, which gives back how many answers of each version in the log fit, such as `{1: 1, 2: 2}`. A version none of whose answers fit gets 0, so that you can see it was tried. `version not in counts` is `True` while `counts` has no key `version` yet. The picture under the check shows what your function counts in the course's log.
        """),
        L.task("In the loop, give a version that is not in `counts` yet the count 0, and add 1 for each answer whose `entry[\"fits\"]` is `True`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tally():
    box_tally = L.code_box("fits_per_version", lines=10)
    box_tally
    return (box_tally,)


@app.cell(hide_code=True)
def check_tally(box_tally):
    result_tally = L.check("fits_per_version", box_tally.value)
    result_tally
    return (result_tally,)


@app.cell(hide_code=True)
def hints_tally():
    L.hints(
        "fits_per_version",
        [
            "Two steps for each entry: make sure its version is in `counts`, and add 1 when the answer fits.",
            "`if version not in counts:` gives the version its first count, 0. Then `if entry[\"fits\"]:` adds 1 to `counts[version]`.",
            "`if version not in counts:` with `counts[version] = 0` under it, then `if entry[\"fits\"]:` with `counts[version] = ___` under it.",
        ],
        why="""```python
def fits_per_version(log):
    counts = {}
    for entry in log:
        version = entry["version"]
        if version not in counts:
            counts[version] = 0
        if entry["fits"]:
            counts[version] = counts[version] + 1
    return counts
```

Every version gets its count the first time it comes by, also when none of its answers fit.""",
    )
    return


@app.cell(hide_code=True)
def text_new_version():
    mo.md(r"""
    ### Exercise 3. When a new version?
    """)
    return


@app.cell(hide_code=True)
def choice_new_version_cell():
    _t, _k = C.findings()["total"], C.findings()["asked"][1]
    choice_new_version = L.choice(
        "new-version",
        question="Which change to the template `bike-details` needs a new version number?",
        options={
            "longer": "Only a change that makes the prompt clearly longer or shorter",
            "comma": "Every change to its words, even one comma",
            "field": "Only a new field, such as `{shop}` next to `{text}`",
        },
        correct="comma",
        why={
            "comma": "One sentence took the course's answers from {v1} to {v2} of {k} that fit, and a comma can change answers too. With a new number for every change, the log says which words gave which answers.".format(v1=_t[1], v2=_t[2], k=_k),
            "longer": "Version 2 differs from version 1 by one sentence, and gave other answers: a small change can matter. Which changes could change the answers?",
            "field": "Versions 1 and 2 have the same field, and gave other answers: the words alone made the difference. Which changes could change the answers?",
        },
    )
    choice_new_version
    return (choice_new_version,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try a version of your own on Qwen3

    Write a version 3, and send it to Qwen3-0.6B on the course's own server. The demo fills your template with each of the three texts, with `.format`, and asks once for each, with seed 0, in JSON mode. Mind the braces.
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
def template_box_cell():
    template_box = mo.ui.text_area(value=C.TEMPLATES[2], label="Your template, with the field {text}", full_width=True, rows=4)
    send_button = mo.ui.run_button(label="Send for the three bikes")
    mo.vstack([template_box, send_button])
    return send_button, template_box


@app.cell(hide_code=True)
def demo_answer(send_button, template_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to send.*")
    else:
        mo.output.replace(mo.md("*The prompts have gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "What Qwen3 answered to your version", "fits": "fits", "not": "does not fit",
                  "format": ".format stopped: {why}. Write each brace that should stay in the prompt twice.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(template_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The system prompt of [System prompts and response prefill](/course/open/build-system-prompt?lang=en) is fixed text too, and belongs in a file next to the templates, with a version of its own. Counting what fits, for each version, on the same examples, is what an eval does on a larger scale, later in this domain. In the next lesson, [Project: your assistant, prompted well](/course/open/build-project-prompts?lang=en), your assistant gets a system prompt and templates of its own, and answers as JSON that it checks itself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_version, result_prompt, result_tally):
    L.wrap_up(result_prompt, result_tally, choice_new_version)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
