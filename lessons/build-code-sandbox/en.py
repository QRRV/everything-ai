# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-code-sandbox. Route: course/route/lessen/bouwen.yaml. Checks, the course's sandbox, the table and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-code-sandbox", "en")
    C = L.checks   # the course's sandbox and guard, eight pieces of an agent's code run on this computer, the checks
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [The agent loop](/course/open/build-agent-loop?lang=en), one about [What a tool may do](/course/open/build-tool-permissions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="In the agent loop, the model's answer holds two tool calls. What does the loop do next?",
        options={
            "user": "It shows both calls to the user, who runs them and types in what each one gave back",
            "run": "It runs both calls, adds each result with the id of its call, and then asks the model again",
            "first": "It runs only the first call, since the model can only take one step in each answer",
        },
        correct="run",
        why={
            "run": "Each result goes back with the role `tool` and the id of its call, and the next request holds the whole conversation so far.",
            "user": "The loop runs the tools itself: that is what makes it an agent. See [The agent loop](/course/open/build-agent-loop?lang=en).",
            "first": "An answer can hold several calls, and the loop runs each of them. See [The agent loop](/course/open/build-agent-loop?lang=en).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_allowlist():
    quick_allowlist = L.choice(
        "quick-allowlist",
        question="Why does a policy list what is allowed, and not what is forbidden?",
        options={
            "shorter": "A list of what is allowed is always shorter, so the model reads it faster",
            "model": "The model then knows beforehand which calls it must not make, and skips them",
            "forgot": "A tool or place that nobody thought of is then denied too, instead of let through",
        },
        correct="forgot",
        why={
            "forgot": "An allowlist denies everything it does not name; a blocklist always forgets something.",
            "shorter": "The policy is not for the model to read: your code checks it, before a call runs. See [What a tool may do](/course/open/build-tool-permissions?lang=en).",
            "model": "The model does not need to know the policy; your code stops what is not on the list. See [What a tool may do](/course/open/build-tool-permissions?lang=en).",
        },
    )
    quick_allowlist
    return (quick_allowlist,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Some things are easier to do in code than in words: a sum, sorting a list, counting the rows of a table. In [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en), the course's code worked out what was missing from the pantry, and the model only wrote the sentence. An agent can do the same on its own when it has a tool that runs code: the model writes a few lines of Python, your program runs them, and the model reads what came out. But that code was written by a model, which can misunderstand its task, or be fooled by a text it read, as in [What a tool may do](/course/open/build-tool-permissions?lang=en). It can loop for hours, write files where it should not, delete them, read your keys and send them away. So it does not run in your own program. It runs in a **sandbox**: a place closed off from the rest of your computer, where whatever it does wrong stays inside. In this lesson you build a small one, and measure what it stops and what it does not.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **sandbox** | a closed-off place to run code, such as a separate process, a container or a small virtual machine, without your files, your network or your keys |
    | **process** | a program while it runs, with its own memory; your Python program is one, and it can start others |
    | **`subprocess.run(command, ...)`** | starts a program as a separate process, waits until it is done, and gives back what came out: `returncode` (0 when all went well), `stdout` (what it printed) and `stderr` (its errors) |
    | **environment variable** | a name with a value that a program gets from your computer when it starts, outside its code; an **API key**, the secret that gives access to a paid model, often lives in one |
    | **`cwd=folder`, `env={}`** | keyword arguments of `subprocess.run`: the folder the process starts in, and its environment variables, here none |
    | **`capture_output=True`, `text=True`** | keep what the process prints, in `stdout` and `stderr`, as text and not as bytes |
    | **`timeout=seconds`** | stop the process after that many seconds; `subprocess.run` then raises `subprocess.TimeoutExpired` |
    | **`tempfile.TemporaryDirectory()`** | makes a new, empty folder, and throws it away with everything in it at the end of the `with` block |
    | **`text.split("\n")`** | the lines of a text, as a list: `"a\nb".split("\n")` is `["a", "b"]` |

    ### Four layers

    The course's sandbox has four layers. **A separate process:** when the code crashes or eats memory, that process stops, and your program goes on. **A time limit:** code that loops forever is stopped. **A new, empty folder,** and none of your environment variables: a file the code writes lands in a folder that is thrown away afterwards, and a key you keep in an environment variable is not there. **A guard:** a few lines of Python, from the course, that run before the agent's code and stop it when it opens the network, a file outside its folder, or another program.
    """)
    return


@app.cell(hide_code=True)
def picture_layers():
    mo.Html(C.layers_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course wrote {n} pieces of code of the kind an agent could write, and ran each of them on your computer, when this lesson opened: once in a sandbox with the first three layers, once with the guard as well, each time with a limit of {s} seconds.
    """.format(n=F["rows"], s=F["seconds"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("The guard is a check within Python, in the same process as the agent's code. Python's own documentation says that code made to get out can get around such a check. It stops what an agent does by mistake, not someone who sets out to break in. For code that strangers can steer, use a sandbox of the operating system: a container without network, such as one of Docker, a separate user that can open only one folder, or a service built to run code."),
        mo.accordion({"The course's guard, if you want to read it": mo.md("```python\n" + C.GUARD + "```")}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: run code an agent wrote

    1. **Never in your own program:** start a separate process for it.
    2. **Give it a time limit,** and stop it when the time is up.
    3. **Start it in a new, empty folder,** which is thrown away afterwards, and give it none of your environment variables: keys often live there.
    4. **Close off the network and the files outside the folder:** here with the course's guard, for code from strangers with a container or a separate user.
    5. **Send back what the model can use:** the output, cut off at a limit, or the last line of an error, or that the time ran out.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A process of its own

        `run(code, seconds)` runs code an agent wrote in a sandbox. The box already makes the command, a list: `PYTHON`, the Python the course runs on, with `-I`, which leaves out your own settings, and `-c` with the guard and the code. The `with` line makes a new, empty folder. Complete it. The check runs your function on six pieces of code: the shopping list, a typo, code that lists its folder, code that reads an environment variable, code that reads your key file, and code that takes longer than its limit. The picture under it shows what your `run` gave back for each.
        """),
        L.task("In the `with` block, run `command` in `folder` with `subprocess.run`, without your environment variables, with the output as text and a time limit of `seconds`; give back what it gives back, or `None` when the time limit stops it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run", lines=14)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run",
        [
            "`subprocess.run` takes the command as a list, and the rest as keyword arguments: one for the folder, one for the environment variables, two for the output, one for the time limit. When the time is up, it stops the process and raises `subprocess.TimeoutExpired`.",
            "The keyword arguments are `cwd`, `env`, `capture_output`, `text` and `timeout`. Put the call, after `return`, in a `try:`, and catch the time limit with `except subprocess.TimeoutExpired:`.",
            "In the `try:`: `return subprocess.run(command, cwd=folder, env=___, capture_output=True, text=True, timeout=seconds)`.",
        ],
        why="""```python
def run(code, seconds):
    command = [PYTHON, "-I", "-c", GUARD + code]
    with tempfile.TemporaryDirectory() as folder:
        try:
            return subprocess.run(command, cwd=folder, env={}, capture_output=True,
                                  text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            return None
```

The `return` is inside the `with` block, so the folder is thrown away right after: whatever the code wrote there is gone.""",
    )
    return


@app.cell(hide_code=True)
def text_result_text():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What the model reads

        As a tool of an agent, the sandbox gives its result back to the model as text, with the role `tool`. `result_text(done, seconds)` makes that text from what `run` gave back: `done` is `None` when the time limit stopped the code; otherwise it has `done.returncode`, `done.stdout` and `done.stderr`. The box already handles the time limit. Add the two other cases. When `done.returncode` is not 0, the code stopped with an error: the model gets `"Error: "` and the last line of `done.stderr`, the line that says what went wrong. When the output is longer than `LIMIT` characters, {n} here, the model gets the first `LIMIT`, and a line that says how many more there were: one `print` in a loop can write a million characters, and each of them would go to the model again in every next request. The check tries an output, an error with its traceback, a long output, code that printed nothing, and the time limit.
        """.format(n=C.number(C.LIMIT, L.lang))),
        L.task("Before the last line, give back a text with the last line of `done.stderr` when `done.returncode` is not 0, and cut `done.stdout` off at `LIMIT` characters, with a note of how many were cut, when it is longer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_result_text():
    box_result_text = L.code_box("result_text", lines=12)
    box_result_text
    return (box_result_text,)


@app.cell(hide_code=True)
def check_result_text(box_result_text):
    result_result_text = L.check("result_text", box_result_text.value)
    result_result_text
    return (result_result_text,)


@app.cell(hide_code=True)
def hints_result_text():
    L.hints(
        "result_text",
        [
            "Two tests before the last line, each with its own `return`: one on `done.returncode`, one on the length of `done.stdout`.",
            "`done.stderr.strip().split(\"\\n\")` gives the lines of the error, and `[-1]` the last of them. `done.stdout[:LIMIT]` is the first `LIMIT` characters, and `len(done.stdout) - LIMIT` is how many were cut.",
            "For a long output, for instance: `return done.stdout[:LIMIT] + \"\\n(cut off: \" + ___ + \" more characters)\"`.",
        ],
        why="""```python
def result_text(done, seconds):
    if done is None:
        return "Stopped: the code was still running after " + str(seconds) + " seconds."
    if done.returncode != 0:
        lines = done.stderr.strip().split("\\n")
        return "Error: " + lines[-1]
    if len(done.stdout) > LIMIT:
        cut = len(done.stdout) - LIMIT
        return done.stdout[:LIMIT] + "\\n(cut off: " + str(cut) + " more characters)"
    return done.stdout
```

The note matters: without it, the model takes the first part for the whole output.""",
    )
    return


@app.cell(hide_code=True)
def text_guard_wall():
    mo.md(r"""
    ### Exercise 3. Strangers' code
    """)
    return


@app.cell(hide_code=True)
def choice_guard_wall_cell():
    choice_guard_wall = L.choice(
        "guard-wall",
        question="A web shop lets its visitors ask an agent questions, and the agent runs the code it writes in the course's sandbox. Is that safe enough?",
        options={
            "yes": "Yes: the guard stops the network, files outside the folder and other programs, and that covers it",
            "no": "No: the guard is a check in the same Python as the code, and code made to get out gets past it",
            "short": "Yes, if the time limit is short: code that cannot run for long cannot do much harm",
        },
        correct="no",
        why={
            "no": "Visitors can write questions that make the agent write such code on purpose. For code that strangers can steer, use a sandbox of the operating system: a container without network, a separate user, or a service built to run code.",
            "yes": "The guard catches what an agent does by mistake. Python has ways around such a check, and someone who wants to get out looks for them.",
            "short": "Reading a key and sending it away takes less than a second.",
        },
    )
    choice_guard_wall
    return (choice_guard_wall,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: your sandbox on your own code

    Once exercise 1 is right, you can run code of your own in your sandbox, as an agent would: write it in the box, and read what the model would get back, through your `result_text` once exercise 2 is right, and through the course's until then. Try a `print`, an error, a loop that never ends, or `open` on a file outside the folder.
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.code_editor(value=C.SHOP, language="python", label="Code for your sandbox")
    try_button = mo.ui.run_button(label="Run in your sandbox")
    mo.vstack([try_box, try_button])
    return try_box, try_button


@app.cell(hide_code=True)
def demo_try(box_result_text, box_run, try_box, try_button):
    _run = C.learner_run(box_run.value) if try_button.value else None
    if not try_button.value:
        _out = mo.md("*Press the button to run.*")
    elif _run is None:
        _out = mo.md("*Finish exercise 1 first: the demo runs your own `run`.*")
    else:
        _words = {"title": "Your code in your sandbox, with a limit of 2 seconds", "reads": "What the model would read:",
                  "error": "Your run stopped with {e}."}
        _out = mo.Html(C.tried(_run, C.learner_text(box_result_text.value), try_box.value, 2, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What a tool may do](/course/open/build-tool-permissions?lang=en) checked each call in your code, before it ran. A tool that runs code needs more: the call itself can be fine, while the code in it does something else. So the code runs where it can do little harm, in a separate process with a time limit, in an empty folder, behind a guard, and the loop of [The agent loop](/course/open/build-agent-loop?lang=en) gets back a short text it can read. The next lesson lets an agent use a browser.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_guard_wall, result_result_text, result_run):
    L.wrap_up(result_run, result_result_text, choice_guard_wall)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
