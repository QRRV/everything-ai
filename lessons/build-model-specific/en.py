# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-model-specific. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-model-specific", "en")
    C = L.checks   # the course's recorded runs, the checks, and the demo with your run_examples and worse


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_versions():
    quick_versions = L.choice(
        "quick-versions",
        question="You changed one sentence in a prompt template. How do you find out whether the new version is better?",
        options={
            "examples": "Run the same examples with both versions, and compare what comes back",
            "longer": "The new version says more, so the model has more to go on and answers better",
            "ask": "Ask the model which version it likes best",
        },
        correct="examples",
        why={
            "examples": "Only the answers on your own examples show what a version does, with this model.",
            "longer": "More words can help or hurt; with the course's bike template one sentence more helped, but only the runs showed that. See [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en).",
            "ask": "What a model says about a prompt is no measurement of what it does with it. See [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en).",
        },
    )
    quick_versions
    return (quick_versions,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A prompt grows on one model: you try words until the answers on your examples are right. Then the model changes. A cheaper one comes out, the provider replaces the version behind its name, or you move the same model to another server. The prompt stays the same, and still answers that were right now go wrong, because each model has its own habits and its own settings. Moving to a new model is part of the work, and it goes the way any change to a prompt goes: with your examples.

    **An everyday example.** A recipe for a cake that always works in your own oven can come out burnt in a friend's oven: same recipe, same steps, another oven. An experienced baker reads the friend's oven first, and bakes one test cake. That is only a comparison; below you measure how one prompt fares with four ways of running the same model.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **prompt migration** | moving a prompt to another model, another version or other settings, and adjusting it until your examples are right again |
    | **prompting guide** | what the maker of a model advises for it: how to write prompts, which settings to send, and the switches it knows |
    | **/no_think** | a switch of Qwen3 at the end of a message: no thinking for this answer, also when the server has switched thinking on |

    ### One prompt, four ways of running the model

    Qwen3 has two modes, as you saw in [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en): it answers right away, or it thinks first. The course's server answers right away unless a request asks for thinking. Qwen3's own chat template, and the servers vLLM and SGLang, think unless told not to. So an app built against the course's server that moves to such a server gets the same model with the same prompt, and still another way of answering.

    The course asked Qwen3 for the event in three messages as JSON, with the system prompt and template below, {seeds} times each, in four ways: the normal way with `max_tokens` 80, with thinking switched on and the same 80 tokens, with thinking and 1000 tokens, and with thinking and `/no_think` at the end of each message.

    ```text
    {system}

    {event}
    ```
    """.format(seeds=_f["seeds"], system=C.SYSTEM, event=C.EVENT))
    return


@app.cell(hide_code=True)
def picture_modes():
    mo.Html(C.modes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_modes():
    _f = C.findings()
    _n = _f["seeds"] * _f["examples"]
    mo.vstack([
        mo.md(r"""
        The normal way, {normal} of the {n} answers were right. With thinking switched on and the same settings, none was: all {cut} ran out of their 80 tokens while Qwen3 was still thinking, so the answer itself stayed empty. The thought is not part of the answer, so an app that reads only the answer gets nothing at all:
        """.format(normal=_f["normal"]["all"], n=_n, cut=_f["think-short"]["cut"])),
        mo.Html(C.answer_picture(L.lang, "think-short")),
        mo.md(r"""
        With 1000 tokens there was room, and still only {think} of the {n} were right. For *7 pm* in the football message, every thinking run gave the hour {think_hour}, where the normal way gave {normal_hour} every time; and an answer took up to {think_tokens} tokens, thought included, against {normal_tokens} the normal way. Qwen3's prompting guide, its model card on Hugging Face, the page where its maker describes the model, names the other way out: `/no_think` at the end of a message. With it, {no_think} of the {n} were right, in up to {no_think_tokens} tokens: as many as before the move.

        The book club went wrong in every way: wherever an answer gave an hour, it was {book}, for *3 in the afternoon*. The move did not break that. It was wrong before, so it is not among what got worse; it asks for a change of its own to the template.
        """.format(think=_f["think"]["all"], n=_n, think_hour=_f["think"]["hours"]["football"][0], normal_hour=_f["normal"]["hours"]["football"][0],
                   think_tokens=_f["think"]["tokens"], normal_tokens=_f["normal"]["tokens"],
                   no_think=_f["no-think"]["all"], no_think_tokens=_f["no-think"]["tokens"], book=_f["normal"]["hours"]["book-club"][0])),
        L.watch_out("Every maker writes its own prompting guide (examples in November 2026). OpenAI and Anthropic publish one for their models, with advice that changes per generation, and the model card of Qwen3 lists the settings for sampling and the switches `/think` and `/no_think`. What worked for one model is a start for the next, not a promise."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: moving a prompt to another model

    1. **Keep your examples:** inputs with what a right answer is, the same ones you test each version of a template with.
    2. **Read the new model's guide:** which settings it wants, which switches it knows, and in which form it expects instructions.
    3. **Run every example with the old and the new,** and list the examples that were right before and are wrong now.
    4. **Adjust the prompt or the settings** for the new model, and run the examples again, until that list is empty.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _n = _f["seeds"] * _f["examples"]
    mo.md(r"""
    ### Worked example

    The app moves to a server that thinks unless told not to. What has to change?

    1. **The examples:** the three messages, each with its day and hour; a run counts as right when the JSON in the answer has that day and hour.
    2. **The guide:** in thinking mode Qwen3 first writes its reasoning, inside the answer's tokens, and `/no_think` at the end of a message switches that off.
    3. **Old and new:** the normal way {normal} of {n} right, with thinking and 80 tokens {short}: every example that was right got worse.
    4. **Adjust:** a larger `max_tokens` gave room, but only {think} of {n} right, for up to {think_tokens} tokens an answer. With `/no_think` in the template, {no_think} of {n} were right, for up to {no_think_tokens}: the list of what got worse is empty again, for one word in the template. The book club stays wrong, as it was before the move.
    """.format(normal=_f["normal"]["all"], n=_n, short=_f["think-short"]["all"], think=_f["think"]["all"],
               think_tokens=_f["think"]["tokens"], no_think=_f["no-think"]["all"], no_think_tokens=_f["no-think"]["tokens"]))
    return


@app.cell(hide_code=True)
def text_worse():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What got worse

        After a run with the old and with the new model you have two dictionaries with, for each example, whether its answer was right: `True` or `False`. Write `worse(old, new)`, which gives back a list of the names that were right in `old` and are wrong in `new`, in the order of `old`. The box gives back every name. Two words of Python help: `not x` is `True` when `x` is `False`, and the other way round; `a and b` is `True` only when `a` and `b` are both `True`.
        """),
        L.task("Keep only the names that were right in `old` and are wrong in `new`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_worse():
    box_worse = L.code_box("worse", lines=7)
    box_worse
    return (box_worse,)


@app.cell(hide_code=True)
def check_worse(box_worse):
    result_worse = L.check("worse", box_worse.value)
    result_worse
    return (result_worse,)


@app.cell(hide_code=True)
def hints_worse():
    L.hints(
        "worse",
        [
            "A name belongs in the list when two things hold at once: it was right before, and it is wrong now.",
            "Put an `if` around the `append`, with both tests joined by `and`: `old[name]` and `not new[name]`.",
            "In the loop: `if old[name] and not new[name]:` with `names.append(name)` under it.",
        ],
        why="""```python
def worse(old, new):
    names = []
    for name in old:
        if old[name] and not new[name]:
            names.append(name)
    return names
```

An example that was already wrong in `old` did not get worse, and one that went from wrong to right got better; neither belongs in the list.""",
    )
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Every example, with given settings

        To compare, every example has to go out the same way, with only the settings changed. The box sends each text in `texts`, a dictionary from a name to a message, with `messages_for(text)`, and keeps the answer itself. Change `run_examples(url, settings, texts)` in two places. First, put every key of `settings` with its value in `body`, after the line that makes the body, so that a setting such as `max_tokens` replaces the 80. Second, keep `ok(name, ...)` of the answer instead of the answer itself: `True` when its JSON has the right day and hour. `messages_for` and `ok` are given. The check runs your function against a stand-in, also with thinking switched on.
        """),
        L.task("Put each setting in the body, and keep `ok(name, ...)` of each answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run_examples", lines=18)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run_examples", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run_examples",
        [
            "A dictionary takes a new key, or a new value for a key it has, with `body[key] = value`. Do that for every key in `settings`.",
            "Right after the line with `body = ...`: a `for` loop over `settings`, with `body[key] = settings[key]` in it. At the end, `ok(name, answer[\"choices\"][0][\"message\"][\"content\"])`.",
            "Under `body = ...`: `for key in settings:` with `body[key] = settings[key]` under it. The last line in the loop: `results[name] = ok(name, answer[\"choices\"][0][\"message\"][\"content\"])`.",
        ],
        why="""```python
import json
import urllib.request


def run_examples(url, settings, texts):
    results = {}
    for name in texts:
        body = {"model": "qwen3-0.6b", "messages": messages_for(texts[name]), "max_tokens": 80}
        for key in settings:
            body[key] = settings[key]
        data = json.dumps(body).encode()
        request = urllib.request.Request(url + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            answer = json.loads(response.read())
        results[name] = ok(name, answer["choices"][0]["message"]["content"])
    return results
```

The settings go in after the body is made, so they win over what the body had: `{"max_tokens": 1000}` replaces the 80.""",
    )
    return


@app.cell(hide_code=True)
def text_migrate():
    mo.md(r"""
    ### Exercise 3. A new version behind the same name
    """)
    return


@app.cell(hide_code=True)
def choice_migrate_cell():
    choice_migrate = L.choice(
        "migrate",
        question="Your provider will put a newer version of the model behind the name your app uses, next month. What do you do first?",
        options={
            "nothing": "Nothing: a newer version of the same model is better at everything the old one could do",
            "longer": "Make every prompt longer, so no version can misread it",
            "run": "Run your examples on the new version, and fix what got worse",
        },
        correct="run",
        why={
            "run": "Only your examples show what the new version does with your prompts; the list of what got worse says what to adjust.",
            "nothing": "A new version can be better on average and still worse on your examples: the same prompt can get other answers. Run them first.",
            "longer": "More words can help or hurt, and without a run you do not know which. Run your examples against the new version first.",
        },
    )
    choice_migrate
    return (choice_migrate,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercises 1 and 2 are right, you can run the three examples on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo runs them with your `run_examples`, first the normal way and then in the way you choose, and shows with your `worse` what got worse. Your computer is not the course's, so your answers can differ.
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
def mode_cell():
    mode_box = mo.ui.dropdown(options={"thinking on, max_tokens 80": "think-short", "thinking on, with /no_think in each message": "no-think"},
                              value="thinking on, max_tokens 80", label="The new way")
    run_button = mo.ui.run_button(label="Run with your run_examples")
    mo.vstack([mode_box, run_button])
    return mode_box, run_button


@app.cell(hide_code=True)
def demo_answer(box_run, box_worse, mode_box, run_button):
    _fns = C.learner_functions(box_worse.value, box_run.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _fns is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo runs your own `run_examples` and `worse`.*")
    elif not run_button.value:
        _out = mo.md("*Press the button to run.*")
    else:
        mo.output.replace(mo.md("*The examples go out twice, the normal way and the new way; the server loads Qwen3 and answers …*"))
        _words = {"title": "Your examples, the normal way and the new way", "old": "normal", "new": "new",
                  "right": "right", "wrong": "wrong",
                  "lost": "Your worse: {names}, in {seconds} seconds.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fns[0], _fns[1], mode_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The examples and versions of [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en) are what you move a prompt with, and the thinking mode of [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en) is one of the settings that make a model answer in another way. [Prompts that hold](/course/open/build-defensive-prompts?lang=en) looks at text in a prompt that tries to give instructions of its own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_migrate, result_run, result_worse):
    L.wrap_up(result_worse, result_run, choice_migrate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
