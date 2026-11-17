# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-project-prompts. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-project-prompts", "en")
    C = L.checks   # the course's runs with and without the rule about JSON, the stand-ins, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en), and one about [JSON you can rely on](/course/open/build-structured-output?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_version():
    quick_version = L.choice(
        "quick-version",
        question="Why does a template in a program get a version, written next to every answer in the log?",
        options={
            "answers": "To know which words gave which answers",
            "model": "So that the model knows which version it reads",
            "format": "Because `.format` only fills in a template that has a version",
        },
        correct="answers",
        why={
            "answers": "Change a word and the version goes up, so the log shows, for each version, what went right.",
            "model": "The model never sees the version: it is not in the prompt. See [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en).",
            "format": "`.format` fills in the fields of any string. See [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en).",
        },
    )
    quick_version
    return (quick_version,)


@app.cell(hide_code=True)
def quick_check_again():
    quick_again = L.choice(
        "quick-again",
        question="An answer does not fit the schema. What does the next request of `ask_json` carry?",
        options={
            "same": "The same messages as before, so the model gets another try",
            "both": "The messages so far, the answer, and the problems",
            "problems": "Only a message that names the problems",
        },
        correct="both",
        why={
            "both": "The model keeps nothing between requests: it needs its own answer to know what the problems are about.",
            "same": "Then the model hears nothing about what was wrong. See [JSON you can rely on](/course/open/build-structured-output?lang=en).",
            "problems": "Without its own answer, the model does not know what the problems are about. See [JSON you can rely on](/course/open/build-structured-output?lang=en).",
        },
    )
    quick_again
    return (quick_again,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your assistant from [Project: your assistant answers](/course/open/build-project-call?lang=en) sends what the user types, and gives back text. For a chat, that is enough. But an app does more with an answer than show it: a calendar needs the day and the hour of an event as values it can work with, not as a sentence. Since then you have learned what that takes: a system prompt with rules for every task, from [System prompts and response prefill](/course/open/build-system-prompt?lang=en), templates with a version, and JSON that your code checks. Today all of it goes into your project: your assistant gets its prompts in a file of their own, and a function that gives back JSON it has checked itself.

        **An everyday example.** A new colleague gets a handbook on the first day, with the rules for every letter. For each kind of letter there is a form with blanks to fill in. And before a letter goes out, someone checks it against a list. Your assistant gets the same three things: the system prompt is the handbook, the templates are the forms, and the schema is the list. That is only a comparison; below you see what each of them does with Qwen3.
        """),
        draw.project_steps(["Your assistant answers", "Your assistant, prompted well", "Your assistant reads your documents"], 1,
                           here_label="← today", title="Where your project stands: your assistant"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_rule():
    _f = C.findings()
    guess_rule = L.guess(
        "rule",
        question="The course asked Qwen3-0.6B for the event in a message as JSON, eight times for each of three messages, with the system prompt *You are a friendly assistant. Answer briefly.* Then 24 times more, with one sentence added to the system prompt: *When a task asks for JSON, answer with the JSON only, without a code block.* What did that sentence do?",
        options={
            "same": "Little: the template already asks for JSON",
            "more": "More answers that `json.loads` can read",
            "fewer": "Fewer: a longer system prompt confuses a small model",
        },
        reveal="**More answers that `json.loads` can read: {rule} of {k} at the first try, against {plain} of {k} without it.** Without the sentence, the {fenced} others were all in a code block, which `json.loads` cannot read. Asking again with the problem did not help them: in each of those {fenced} runs, the second and the third try were in a code block too. A rule in the system prompt worked where asking again did not.".format(
            rule=_f["rule"]["first_all"], plain=_f["plain"]["first_all"], k=_f["rule"]["runs"], fenced=_f["plain"]["fenced"]),
    )
    guess_rule
    return (guess_rule,)


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### Four files, each with one job

        After today, your package `app` has four files. Each does one thing, and only that:

        - `app/client.py`, from [Project: your assistant answers](/course/open/build-project-call?lang=en), talks to the server: `chat` sends a conversation and gives back the text.
        - `app/prompts.py`, new today, holds the words: `SYSTEM`, the system prompt with the rules for every task, and `TEMPLATES`, one entry for each task, with the template's `"text"`, its `"version"` and the `"schema"` of the answer it asks for. Its `messages_for(name, text)` makes the first messages of a task.
        - `app/schema.py` holds your `problems` from [JSON you can rely on](/course/open/build-structured-output?lang=en): the check of that lesson kept it as a file.
        - `app/answers.py`, new today, has `ask_json(url, name, text)`: it asks with the prompts through the client, checks the answer, and asks again with what was wrong.
        """),
        mo.Html(C.files_picture(L.lang)),
        mo.md(r"""
        The rest of an app calls only `ask_json`, and gets a dictionary, or `None` when no answer fit. Better words for a prompt change only `app/prompts.py`, and its version goes up; another server changes only `app/client.py`. The schema sits next to its template, because they belong together: a template that asks for other keys needs another schema. And the rule about JSON is in the system prompt, not in each template, because it holds for every task: the title asks for JSON too.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_meaning():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        ### A check of the form does not see the meaning

        The third message says the book club meets *on Sunday at 3 in the afternoon*: hour 15. With the rule, all {k} answers fit the schema at the first try, and all {k} gave hour {hour}:
        """.format(k=len(_f["rule"]["hours"]), hour=_f["rule"]["hours"][0])),
        mo.Html(C.book_picture(L.lang)),
        mo.md(r"""
        `3` is a number from 0 to 23, so `problems` finds nothing. It checks the form of an answer, not whether its values are right. A sentence in the template can help, such as one that says the hour is in 24-hour time; that is a new version of the template. Whether the values are right, you can only learn by comparing answers with answers you know are right, which a later lesson does.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: your assistant, prompted well

    1. **The words in `app/prompts.py`:** `SYSTEM` with the rules for every task, and `TEMPLATES` with, for each task, the template's `"version"`, its `"text"` with the field `{text}`, and the `"schema"` of the answer.
    2. **The first messages:** `messages_for(name, text)` gives back the system prompt as `{"role": "system", ...}`, and then the template of `name`, filled in with `.format(text=text)`, as the user's.
    3. **Asking in `app/answers.py`:** `ask_json(url, name, text, tries=3)` starts from `messages_for(name, text)` and the template's schema. Each try: `chat`, then `problems`. No problems: give back `json.loads(answer)`. Problems: add the answer as the assistant's and a message with the problems as the user's, and try again.
    4. **No answer fit:** after the last try, give back `None`, so the rest of your app can say that it could not read the message, instead of stopping with an error.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.findings()["worked"]
    mo.md(r"""
    ### Worked example

    The course's first run with the rule, seed 0, on the football message: `ask_json(url, "event", "{text}")`.

    1. **The words:** the template `"event"`, version 1, with its schema: `what` and `day` strings, `hour` a number.
    2. **The first messages:** the system prompt with the rule about JSON, and the template filled in with the message, which ends with an empty line and *{text}*
    3. **Asking:** `chat` sends the two messages; Qwen3 answered `{answer}`.
    4. **Checking:** `problems` finds nothing, so `ask_json` gives back `{value}` after {tries} try. A calendar can put it on Tuesday at 19:00.
    """.format(text=C.FOOTBALL, answer=_w["text"], value=repr(_w["value"]), tries=_w["tries"]))
    return


@app.cell(hide_code=True)
def text_prompts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your assistant's prompts

        The box holds the words of your assistant: `SYSTEM`, and `TEMPLATES` with two tasks, the event and a title. Long strings are written over two lines, as in [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en): two strings next to each other inside parentheses are one string to Python. Write `messages_for(name, text)`: it gives back a list of two messages, the system prompt, and the template of `name` filled in with `text`, as the user's. Leave `SYSTEM` and `TEMPLATES` as they are for now; the check compares with the course's. When it is right, the check saves the whole box in your project, as `work/projects/app/prompts.py`. After that, the file is yours to change; keep its two tasks, which exercise 2 asks for.
        """),
        L.task("Write `messages_for`: take the template that `name` says, and give back the system's message and the filled template as the user's."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompts():
    box_prompts = L.code_box("messages_for", lines=30)
    box_prompts
    return (box_prompts,)


@app.cell(hide_code=True)
def check_prompts(box_prompts):
    result_prompts = L.check("messages_for", box_prompts.value)
    result_prompts
    return (result_prompts,)


@app.cell(hide_code=True)
def hints_prompts():
    L.hints(
        "messages_for",
        [
            "Two steps: find the entry of the template by its name, and build a list of two dictionaries.",
            "`template = TEMPLATES[name]` finds the entry; the string with `{text}` in it is under one of its keys, and `.format(text=text)` fills it in.",
            "`return [{\"role\": \"system\", \"content\": SYSTEM},` and on the next line `{\"role\": \"user\", \"content\": ___}]`.",
        ],
        why="""```python
def messages_for(name, text):
    template = TEMPLATES[name]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": template["text"].format(text=text)}]
```

The system prompt comes first, so the model reads its rules before the task.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. JSON your assistant checks itself

        `ask_json(url, name, text, tries=3)` asks for the JSON of a task with your prompts, through your `chat`, and checks each answer with your `problems`. The start already makes the first messages, takes the template's schema, and asks in a loop. Fill in the rest: after each answer, find its problems. When there are none, give back what `json.loads` makes of it. Otherwise, add the answer as the assistant's and a message with the problems as the user's, such as *Fix these problems: the answer is not JSON. Answer with the whole JSON again.*, and try again. When no try fits, it gives back `None`. The check calls it against a stand-in whose first answer to the football message is in a code block. When it is right, the check saves it in your project, as `work/projects/app/answers.py`.
        """),
        L.task("After `chat`: find the problems, give back the JSON when there are none, and otherwise add the answer and the problems to `messages`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask_json", lines=17)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask, result_prompts):
    result_prompts  # run again when exercise 1 saves your prompts
    result_ask = L.check("ask_json", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask_json",
        [
            "It is the loop of your `ask_json` in [JSON you can rely on](/course/open/build-structured-output?lang=en), with `chat` in place of `ask`.",
            "`found = problems(answer, schema)`; when `found == []`, give back `json.loads(answer)`.",
            "Otherwise: `fix = \"Fix these problems: \" + ___ + \". Answer with the whole JSON again.\"`, and `messages = messages + [...]` with the answer as the assistant's and `fix` as the user's.",
        ],
        why="""```python
    for attempt in range(tries):
        answer = chat(url, messages)
        found = problems(answer, schema)
        if found == []:
            return json.loads(answer)
        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."
        messages = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]
    return None
```

`messages_for` makes a new list for every call, and `+` makes a new one for every try.""",
    )
    return


@app.cell(hide_code=True)
def text_new_version():
    mo.md(r"""
    ### Exercise 3. A better template
    """)
    return


@app.cell(hide_code=True)
def choice_new_version_cell():
    choice_new_version = L.choice(
        "new-version",
        question="To get hour 15 for the book club, you add a sentence about 24-hour time to the event's template, in `app/prompts.py`. What else do you change?",
        options={
            "nothing": "Nothing: the old words are gone, so nothing can mix them up",
            "client": "`app/client.py`, so that it sends the new template",
            "version": "The template's `\"version\"`, from 1 to 2",
        },
        correct="version",
        why={
            "version": "New words are a new version. Answers from version 1 came from the old words, and the next ones from version 2, so you can compare them; the demo below shows the version with every answer.",
            "nothing": "Answers to the old words are still around, in your app or its log. Without a new version, how would you tell them from the answers to the new ones?",
            "client": "The client sends whatever messages it gets: `messages_for` reads the template from `app/prompts.py` each time. Which file holds the words, and what goes with every change to them?",
        },
    )
    choice_new_version
    return (choice_new_version,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### What your project can do now

    The cell takes `ask_json` from your project, `work/projects/app/answers.py`, with your prompts and your client, and asks Qwen3-0.6B on the course's server for the event in a message of yours, as JSON. You see each try and its problems, and the event as a card, the way a calendar in your app could show it. Your `app/prompts.py` is used as it is: change a template there and raise its version, and the demo uses the new words.
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
def message_box_cell():
    message_box = mo.ui.text_area(value="Can we have lunch on Thursday at 12? Kim", label="A message with an event", full_width=True, rows=2)
    send_button = mo.ui.run_button(label="Ask with your ask_json")
    mo.vstack([message_box, send_button])
    return message_box, send_button


@app.cell(hide_code=True)
def demo_event(message_box, result_ask, result_prompts, send_button):
    result_prompts  # run again when an exercise saves your module
    _ask, _why = C.learner_ask_json() if result_ask.state == "correct" else (None, None)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif result_ask.state != "correct":
        _out = mo.md("*Finish exercise 2 first: then your `ask_json` is in your project, and this cell uses it.*")
    elif _ask is None:
        _out = mo.md("*Your `ask_json` does not run as your project has it now: {why}*".format(why=_why))
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not message_box.value.strip():
        _out = mo.md("*Write a message in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `ask_json` asks; Qwen3 writes …*"))
        _words = {"title": "What your assistant made of the message", "prompt": "Template event, version {version}; requests sent: {n}; {seconds} seconds.",
                  "try": "Try {n}:", "fits": "fits the schema", "found": "problems: {p}",
                  "gave": "Your ask_json gave back:", "none": "No try fit, so your ask_json gave back None: your app can say it could not read the message.",
                  "card": "In your calendar", "yours": "All modules were yours.", "course": "The course's version stood in for: {files}.", "failed": "Your ask_json stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_ask, message_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The system prompt of [System prompts and response prefill](/course/open/build-system-prompt?lang=en) now carries a rule for every task of your assistant; the templates with a version of [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en) are in a file of their own; and the check and the second try of [JSON you can rely on](/course/open/build-structured-output?lang=en) run through the client of [Project: your assistant answers](/course/open/build-project-call?lang=en). Your project comes back in *Project: your assistant reads your documents*, where it answers from your own documents and names its source. First, the next lesson shows why a prompt that works for one model can fail with another.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_version, result_ask, result_prompts):
    L.wrap_up(result_prompts, result_ask, choice_new_version)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
