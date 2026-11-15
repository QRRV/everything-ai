# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-context-engineering. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-context-engineering", "en")
    C = L.checks   # the course's recorded runs, the checks, and the demo with your compact


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
        question="Your app has a system prompt. How often does it go to the model?",
        options={
            "once": "Once, at the start of the conversation: after that the model remembers it",
            "first": "With the first request of each day, when the server starts again",
            "every": "With every request, at the start of the messages, before the conversation",
        },
        correct="every",
        why={
            "every": "The model keeps nothing between requests, so the system prompt goes in again each time, and costs input tokens each time.",
            "once": "The model keeps nothing between requests: the app sends the system prompt with every request. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
            "first": "The model keeps nothing between requests, not even within a day: the app sends the system prompt with every request. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model knows nothing about your conversation except what is in the request. So a chat app sends the whole conversation again with every new message, and the conversation only grows. Each request then costs more input tokens and takes longer to read; at some point it no longer fits in the context window, and long before that, models find a fact in the middle of a long text less well than at its start or end, as the study *Lost in the Middle* measured in 2023. The app has to choose what goes in, and that choice decides what the model can know.

    **An everyday example.** A doctor who sees you for the first time does not read every note from twenty years of visits. The file has a summary page, and an allergy stands on its cover in red, where nobody can miss it. That is only a comparison; below you measure what Qwen3 still knows when the start of a conversation is left out, and what brings it back.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **context engineering** | choosing what goes into the model's context window at every request: instructions, the messages that matter, notes, documents, and little else |
    | **compaction** | making a long conversation shorter when it gets too long: the old messages are replaced by a summary of them, so the work can go on |

    ### What the model gets to see

    The course wrote a dinner talk of {n} messages, five from the user and five answers. The first message holds a fact that matters at the end: *{first}* Then the talk goes on about the starter, the main course, the wine and the music. The last question was:

    > {last}

    The course asked it {seeds} times in each of four ways: with the whole talk; with a window of only the last {keep} messages; with that window and a note in the system prompt, *{note}*; and with that window and a summary of the messages that dropped out, which Qwen3 wrote itself, at the end of the system prompt. An answer counts when it names the nut allergy:
    """.format(n=len(C.TALK), first=C.TALK[0][1], last=C.LAST, seeds=C.SEEDS, keep=C.KEEP, note=C.NOTE))
    return


@app.cell(hide_code=True)
def picture_modes():
    mo.Html(C.modes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_modes():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        With the whole talk, {whole} of the {seeds} answers named the nut allergy; with the window, {window}: the first message was not in the request, so for the model the allergy did not exist, and the first run made up another rule instead: *{made_up}* The note brought the allergy back, {notes} of {seeds}, and so did the summary, {compact} of {seeds}. But the note held only the allergy, so no answer with it named the six friends; the summary held both, and {compact_six} of {seeds} answers with it named them, against {whole_six} with the whole talk. The window and the summary also cost fewer tokens: the request with the whole talk had {whole_tokens} prompt tokens, the one with the window {window_tokens}, and the one with the summary {compact_tokens}. The summary took a call of its own, {sp} prompt tokens and {so} written ones; that pays off once it is sent with the requests after it.
        """.format(made_up=C.answers("window")[0]["text"].strip(), seeds=C.SEEDS, whole=_f["whole"]["knows"], window=_f["window"]["knows"], notes=_f["notes"]["knows"],
                   compact=_f["compact"]["knows"], compact_six=_f["compact"]["six"], sp=_f["summary_cost"]["prompt_tokens"], so=_f["summary_cost"]["tokens"], whole_six=_f["whole"]["six"], whole_tokens=_f["whole"]["prompt_tokens"],
                   window_tokens=_f["window"]["prompt_tokens"], compact_tokens=_f["compact"]["prompt_tokens"])),
        L.watch_out("A summary keeps what the model that wrote it thought mattered, and nothing else. What must never be lost, such as an allergy, a budget or a deadline, belongs in a note that your code keeps and sends along every time, not only in a summary that a model writes again and again."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: what goes into the window

    1. **What always holds goes in the system prompt:** the instructions, and notes about the user that must never be lost.
    2. **The conversation goes in as a window:** the last messages, which the next answer builds on.
    3. **What drops out gets a summary** (compaction), in the system prompt, so that the gist of the start stays.
    4. **Everything else is fetched only when it is needed,** such as a document the question is about: the next module does that.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    mo.md(r"""
    ### Worked example

    How did the summary bring the allergy back?

    1. **What always holds:** the system prompt is *{system}*
    2. **The window:** the last {keep} messages, about the wine and the music, and then the question.
    3. **The summary:** the course asked Qwen3 to summarize the {dropped} messages that dropped out, with the prompt in the box below. It wrote: *{summary}* That text went at the end of the system prompt, after *Summary of the conversation so far:*.
    4. **Nothing else:** the request had {tokens} prompt tokens, against {whole_tokens} for the whole talk.

    ```text
    {summarize}
    ```
    """.format(system=C.SYSTEM, keep=C.KEEP, dropped=len(C.TALK) - C.KEEP, summary=_f["summary"].strip(),
               tokens=_f["compact"]["prompt_tokens"], whole_tokens=_f["whole"]["prompt_tokens"], summarize=C.SUMMARIZE))
    return


@app.cell(hide_code=True)
def text_window():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A window that keeps the system prompt

        Write `window(messages, keep)`, which gives back the system prompt, `messages[0]`, followed by the last `keep` messages of the conversation. The system prompt must come along exactly once, also when `keep` reaches back to the start of the conversation, and with `keep` 0 only the system prompt is left. The box keeps the last `keep` messages, but leaves out the system prompt. `max(a, b)` gives the larger of two numbers.
        """),
        L.task("Put `messages[0]` first, and start the slice at 1 or later."),
    ])
    return


@app.cell(hide_code=True)
def exercise_window():
    box_window = L.code_box("window", lines=6)
    box_window
    return (box_window,)


@app.cell(hide_code=True)
def check_window(box_window):
    result_window = L.check("window", box_window.value)
    result_window
    return (result_window,)


@app.cell(hide_code=True)
def hints_window():
    L.hints(
        "window",
        [
            "Two changes: the list starts with the system prompt, and the slice must never reach back to place 0, where the system prompt is.",
            "Start with `out = [messages[0]]`. The slice starts at `len(messages) - keep`, but never below 1: `max(1, len(messages) - keep)`.",
            "`out = [messages[0]]`, and `for message in messages[max(1, len(messages) - keep):]:` with the `append` as it was.",
        ],
        why="""```python
def window(messages, keep):
    out = [messages[0]]
    for message in messages[max(1, len(messages) - keep):]:
        out.append(message)
    return out
```

`messages[-keep:]` looks shorter, but with `keep` 0 it is `messages[-0:]`, which is `messages[0:]`: the whole list. Counting from the front has no such surprise.""",
    )
    return


@app.cell(hide_code=True)
def text_compact():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A summary of what drops out

        The box asks the server for a summary of the messages that drop out of the window, `old`, with the prompt `SUMMARIZE` around a `transcript` of them, and then gives back the window without the summary. Change the end of `compact(url, messages, keep)` so that it gives back the window with a new system message in front: the content of `messages[0]`, then `"\n\nSummary of the conversation so far: "`, then the summary. Make a new dictionary for it, so that the list of messages you got stays the same. `SUMMARIZE`, `transcript` and `window` are given. The check runs your function against a stand-in.
        """),
        L.task("Give back the window with a new system message: the system prompt, the line, and the summary."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compact():
    box_compact = L.code_box("compact", lines=18)
    box_compact
    return (box_compact,)


@app.cell(hide_code=True)
def check_compact(box_compact):
    result_compact = L.check("compact", box_compact.value)
    result_compact
    return (result_compact,)


@app.cell(hide_code=True)
def hints_compact():
    L.hints(
        "compact",
        [
            "Only the last line changes: the window keeps its messages, but its first one, the system prompt, gets the summary at the end.",
            "Make `out = window(messages, keep)`, and a new dictionary with the role `system` and as content `messages[0][\"content\"] + \"\\n\\nSummary of the conversation so far: \" + summary`. Give back that dictionary in front of `out[1:]`.",
            "`system = {\"role\": \"system\", \"content\": messages[0][\"content\"] + \"\\n\\nSummary of the conversation so far: \" + summary}`, then `return [system] + out[1:]`.",
        ],
        why="""```python
import json
import urllib.request


def compact(url, messages, keep):
    old = messages[1:max(1, len(messages) - keep)]
    body = {"model": "qwen3-0.6b", "max_tokens": 120,
            "messages": [{"role": "user", "content": SUMMARIZE.format(talk=transcript(old))}]}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    summary = answer["choices"][0]["message"]["content"]
    out = window(messages, keep)
    system = {"role": "system",
              "content": messages[0]["content"] + "\\n\\nSummary of the conversation so far: " + summary}
    return [system] + out[1:]
```

`out[1:]` is the window without its first message, the old system prompt; the new one takes its place. Changing `messages[0]["content"]` itself would change the conversation your app keeps, and the summary would pile up there with every call.""",
    )
    return


@app.cell(hide_code=True)
def text_notes():
    mo.md(r"""
    ### Exercise 3. A fact that must stay
    """)
    return


@app.cell(hide_code=True)
def choice_notes_cell():
    choice_notes = L.choice(
        "notes",
        question="A user of your cooking app says in the first message that they cannot eat gluten. Where does your app keep that, so that it is never lost?",
        options={
            "first": "In the first message of the conversation, where the user wrote it, so that it stays in the history",
            "model": "Nowhere: once the model has read it, it knows it",
            "note": "As a note your code keeps and puts in the system prompt of every request",
        },
        correct="note",
        why={
            "note": "The system prompt goes along with every request, whatever the window keeps, so the note is always in the context window.",
            "first": "The first message is the first to drop out of a window, and a summary may leave it out. Keep it as a note in the system prompt, sent every time.",
            "model": "The model keeps nothing between requests: what is not in the request does not exist for it, as the window showed. Keep it as a note in the system prompt.",
        },
    )
    choice_notes
    return (choice_notes,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercises 1 and 2 are right, you can run your `compact` on the dinner talk with the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo makes the window with the number of messages you choose, lets Qwen3 summarize the rest, asks the last question with what your `compact` gave back, and shows the summary and the answer. Your computer is not the course's, so your answers can differ.
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
def keep_cell():
    keep_slider = mo.ui.slider(start=0, stop=8, step=2, value=4, label="Messages in the window", show_value=True)
    compact_button = mo.ui.run_button(label="Ask with your compact")
    mo.vstack([keep_slider, compact_button])
    return compact_button, keep_slider


@app.cell(hide_code=True)
def demo_answer(box_compact, box_window, compact_button, keep_slider):
    _fn = C.learner_compact(box_window.value, box_compact.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo runs your own `compact` and `window`.*")
    elif not compact_button.value:
        _out = mo.md("*Press the button to ask.*")
    else:
        mo.output.replace(mo.md("*Qwen3 summarizes, then answers; the server loads it and writes …*"))
        _words = {"title": "The dinner talk with your compact, {keep} messages in the window".format(keep=keep_slider.value),
                  "summary": "The summary:", "answer": "The answer to the last question:",
                  "knows": "It named the nut allergy.", "forgot": "It did not name the nut allergy.",
                  "tokens": "{tokens} prompt tokens, in {seconds} seconds.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fn, keep_slider.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The system prompt of [System prompts and response prefill](/course/open/build-system-prompt?lang=en) is the place for what always holds, and now also for notes and a summary. Everything that goes into the window is read again with every request; [Pay once for the same start](/course/open/build-prompt-caching?lang=en) makes the part that stays the same cheap to read again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_notes, result_compact, result_window):
    L.wrap_up(result_window, result_compact, choice_notes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
