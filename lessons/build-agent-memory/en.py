# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-agent-memory. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-memory", "en")
    C = L.checks   # the tool remember, the course's runs of Qwen3 with and without memory, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [The agent loop](/course/open/build-agent-loop?lang=en), and one about [What goes into the window](/course/open/build-context-engineering?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_request():
    quick_request = L.choice(
        "quick-request",
        question="In `run_agent`, what does the model know of the conversation when it gets the fourth request?",
        options={
            "all": "What is in that request: the system prompt, the task, and each answer and result so far",
            "last": "Only the last result of a tool, since the server keeps the rest of the conversation for it",
            "server": "What the server kept from the three requests before, together with the new messages",
        },
        correct="all",
        why={
            "all": "That is why `run_agent` sends the whole conversation each time: the model knows only what the request holds.",
            "last": "The server keeps nothing between requests; `run_agent` sends every message again. See [The agent loop](/course/open/build-agent-loop?lang=en).",
            "server": "The server keeps nothing between requests: what the model knows is what the request holds. See [The agent loop](/course/open/build-agent-loop?lang=en).",
        },
    )
    quick_request
    return (quick_request,)


@app.cell(hide_code=True)
def quick_check_note():
    quick_note = L.choice(
        "quick-note",
        question="A fact must never be lost from a long conversation, such as an allergy. Where did it belong?",
        options={
            "summary": "In the summary of the old messages, which the model writes once the window is full",
            "window": "In the window of the last four messages, which always holds the most recent facts",
            "note": "In a note your code keeps and sends in the system prompt every time",
        },
        correct="note",
        why={
            "note": "A note your code keeps goes into every request, however long the conversation gets.",
            "summary": "A summary keeps only what the model that wrote it thought mattered. See [What goes into the window](/course/open/build-context-engineering?lang=en).",
            "window": "A fact from the first message drops out of a window of the last four. See [What goes into the window](/course/open/build-context-engineering?lang=en).",
        },
    )
    quick_note
    return (quick_note,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model remembers nothing between requests. In [The agent loop](/course/open/build-agent-loop?lang=en), each request sent the whole conversation again: that is how the agent knew what its tools had found. When the conversation ends, all of that is gone, and the next conversation starts empty. In [What goes into the window](/course/open/build-context-engineering?lang=en), your code kept a note and sent it every time. In this lesson, the agent decides itself what to keep, with a tool, and the notes go into a file that outlives the conversation.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **short-term memory** | what an agent knows within one conversation: the messages in its request |
        | **long-term memory** | what an agent knows across conversations: notes kept outside the model, such as in a file, which your code puts into the context of a new conversation |
        | **`open(path, "a")`** | opens a file to add to: what you write goes at the end, and a file that is not there yet is made |
        | **`f.write(text)`** | writes the text into the open file as it is, without a newline at the end |
        | **`FileNotFoundError`** | the error `open` gives when the file to read is not there |

        Long-term memory here has two halves. In a conversation, the agent has a tool `remember`, and the system prompt says what is worth saving:

        > {told}

        Your code runs each call of `remember` by adding the note to a file, and sends back `"Saved."`. At the start of a new conversation, your code reads the file and puts the notes in the system prompt, under the line *{head}*. The model then knows in the new conversation what the user told it in an earlier one.
        """.format(told=C.TOLD, head=C.NOTES_HEAD)),
        mo.accordion({"The tool remember as the request gives it, if you want to read it": mo.md(
            "```json\n" + C.REMEMBER_JSON + "\n```")}),
    ])
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course told Qwen3-0.6B {n} things about a user, each in its own conversation with the tool `remember`, at temperature 0, so that the same request always gets the same answer. Each went twice: with `tool_choice` on `"auto"`, so that the model chooses whether to call the tool, and with `tool_choice` naming `remember`, so that it has to. Then it asked {n} questions that need them: in a new conversation without memory, in a conversation in which the user had just said the thing, and in a new conversation whose system prompt held every note the {n} conversations saved.
    """.format(n=F["facts"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A note is what the model wrote, not what the user said: read the notes your agent saves. A note that no longer holds keeps steering every new conversation, and notes about a person are about that person: tell users what your agent keeps, and let them see and delete it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: long-term memory for an agent

    1. **Give the agent a tool to save a note,** and say in the system prompt what is worth saving.
    2. **Add each note at the end of a file,** on a line of its own: `open(path, "a")`.
    3. **Start a new conversation with the notes:** read the file, and put the notes in the system prompt under a line that says what they are. No file yet? Then there are no notes.
    4. **Keep the notes right:** a note that no longer holds is replaced, not added to.
    5. **With many notes, send only the ones that fit,** so that the system prompt does not grow with every note.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(C.WORKED, "en"))
    return


@app.cell(hide_code=True)
def text_remember():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The tool remember

        `remember(arguments)` is what your code runs for a call of the tool. The model's call gives its arguments as a dictionary, such as `{"note": "The user has no oven."}`. Add the note to the notes file, on a line of its own, and keep the notes that were already there. `NOTES` is the path of the file; the box knows it. The check calls your function twice on a new file, once on a file that already holds two notes, and once more on a new file; the picture shows the file before and after your two calls.
        """),
        L.task("Replace `...` with a `with` block that opens `NOTES` to add to, and writes the note with a newline after it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_remember():
    box_remember = L.code_box("remember", lines=7)
    box_remember
    return (box_remember,)


@app.cell(hide_code=True)
def check_remember(box_remember):
    result_remember = L.check("remember", box_remember.value)
    result_remember
    return (result_remember,)


@app.cell(hide_code=True)
def hints_remember():
    L.hints(
        "remember",
        [
            "Reading a file was `open(path)` in a `with` block. Adding to a file works the same way, with a second value that says how to open it.",
            "The second value of `open` says how: `\"r\"` to read, `\"w\"` to write a new file over the old one, `\"a\"` to add to its end. `f.write` writes exactly what you give it, so a note needs `\"\\n\"` after it to end its line.",
            "`with open(NOTES, ___, encoding=\"utf-8\") as f:`, and the `f.write` indented under it.",
        ],
        why="""```python
def remember(arguments):
    note = arguments["note"]
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(note + "\\n")
    return "Saved."
```

`"a"` keeps what is in the file and makes a file that is not there yet; `"w"` would empty it each time. `"Saved."` goes back to the model as the tool's result.""",
    )
    return


@app.cell(hide_code=True)
def text_first_messages():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A new conversation with the notes

        `first_messages(question)` builds the first messages of a new conversation. Read the notes file; when it holds notes, the system prompt becomes `SYSTEM`, an empty line, `NOTES_HEAD`, a newline and the notes. When there is no file yet, or an empty one, the system prompt stays `SYSTEM`. `f.read()` gives the whole file as one string, and `.strip()` takes the newline off its end. The box knows `NOTES`, `SYSTEM` and `NOTES_HEAD`. The check tries a file with two notes, no file, an empty file and a file with one note; the picture shows the messages your function builds for the first.
        """),
        L.task("Replace `...` with code that reads the notes, or no notes when the file is not there, and adds `NOTES_HEAD` and the notes to `system` when there are any."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first_messages():
    box_first_messages = L.code_box("first-messages", lines=12)
    box_first_messages
    return (box_first_messages,)


@app.cell(hide_code=True)
def check_first_messages(box_first_messages):
    result_first_messages = L.check("first-messages", box_first_messages.value)
    result_first_messages
    return (result_first_messages,)


@app.cell(hide_code=True)
def hints_first_messages():
    L.hints(
        "first-messages",
        [
            "Two parts: read the notes, where a missing file means no notes; then change `system` only when there are notes.",
            "Put the `with open(NOTES, encoding=\"utf-8\") as f:` block under `try:`, and give `notes` the empty text `\"\"` under `except FileNotFoundError:`. An empty text counts as false, so `if notes:` holds only when there are notes.",
            "Under `if notes:`: `system = SYSTEM + ___ + NOTES_HEAD + \"\\n\" + notes`.",
        ],
        why="""```python
def first_messages(question):
    system = SYSTEM
    try:
        with open(NOTES, encoding="utf-8") as f:
            notes = f.read().strip()
    except FileNotFoundError:
        notes = ""
    if notes:
        system = SYSTEM + "\\n\\n" + NOTES_HEAD + "\\n" + notes
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]
```

A new user has no notes file yet: `except FileNotFoundError` makes that an ordinary case instead of an error.""",
    )
    return


@app.cell(hide_code=True)
def text_stale_note():
    mo.md(r"""
    ### Exercise 3. A note that no longer holds
    """)
    return


@app.cell(hide_code=True)
def choice_stale_note_cell():
    choice_stale_note = L.choice(
        "stale-note",
        question="Your notes file says *The user has no oven.* Today the user says: *I bought an oven.* What should happen to the note?",
        options={
            "keep": "Keep it as it is, and add *The user has an oven.* below it: the model reads both notes anyway",
            "all": "Empty the whole notes file, so that nothing old can mislead the agent any more",
            "replace": "Replace it with the new fact, so that every note in the file still holds",
        },
        correct="replace",
        why={
            "replace": "Each new conversation believes the notes. That needs a way to change a note, such as a tool that replaces one, or code that rewrites the file.",
            "keep": "Then the system prompt says both *no oven* and *an oven*, and nothing tells the model which of the two holds now.",
            "all": "Then the agent also forgets every other note, and those still hold.",
        },
    )
    choice_stale_note
    return (choice_stale_note,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercises 1 and 2 are right, you can give Qwen3 a memory: the course starts its own server on your computer, with Qwen3-0.6B behind it. Tell it something about yourself: the request names `remember` in `tool_choice`, as in the course's second run, so the model calls it, and your function saves the note. Then ask a question in a new conversation, which starts with your `first_messages`. The notes stay in a file in your work folder until you forget them.
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
def tell_box_cell():
    tell_box = mo.ui.text(value="I am allergic to nuts.", label="Tell the agent something about yourself", full_width=True)
    tell_button = mo.ui.run_button(label="Send")
    mo.vstack([tell_box, tell_button])
    return tell_box, tell_button


@app.cell(hide_code=True)
def demo_tell(box_remember, tell_box, tell_button):
    _remember = C.learner_fn("remember", box_remember.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _remember is None:
        _out = mo.md("*Finish exercise 1 first: the demo saves notes with your own `remember`.*")
    elif not tell_button.value:
        _out = mo.md("*Press the button to send.*")
    elif not tell_box.value.strip():
        _out = mo.md("*Write something in the box first.*")
    else:
        mo.output.replace(mo.md("*The message has gone; the server loads Qwen3 …*"))
        _words = {"title": "Your message, and what your remember saved", "no_call": "The model called no tool: nothing was saved.",
                  "file": "Your notes file now:", "no_file": "(no file yet)",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.tell(_remember, tell_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def ask_box_cell():
    ask_box = mo.ui.text(value=C.FACTS[0][1], label="Ask in a new conversation", full_width=True)
    ask_button = mo.ui.run_button(label="Ask")
    forget_button = mo.ui.run_button(label="Forget all notes")
    mo.vstack([ask_box, mo.hstack([ask_button, forget_button], justify="start")])
    return ask_box, ask_button, forget_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_first_messages, forget_button):
    _first = C.learner_fn("first-messages", box_first_messages.value)
    if forget_button.value:
        C.forget()
        _out = mo.md("*The notes file is gone: the agent starts with no notes again.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _first is None:
        _out = mo.md("*Finish exercise 2 first: the new conversation starts with your own `first_messages`.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 …*"))
        _words = {"title": "A new conversation, started by your first_messages",
                  "error": "Your first_messages stopped with {e}.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.ask_new(_first, ask_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The agent loop](/course/open/build-agent-loop?lang=en), the agent's memory was the conversation it sent again with every request: short-term memory. [What goes into the window](/course/open/build-context-engineering?lang=en) put a note from your code in the system prompt; here the agent saves its own notes with a tool, in a file, and a new conversation starts with them: long-term memory. The next lesson runs code that an agent wrote, in a sandbox.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stale_note, result_first_messages, result_remember):
    L.wrap_up(result_remember, result_first_messages, choice_stale_note)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
