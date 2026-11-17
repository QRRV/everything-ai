# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-agent-history. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-history", "en")
    C = L.checks   # the course's small BabyAGI, its runs of Qwen3, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Coding, research and personal agents](/course/open/build-kinds-of-agents?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_same():
    quick_same = L.choice(
        "quick-same",
        question="What made the coding agent different from the shopping agent?",
        options={
            "loop": "Another loop: a coding agent runs its tests in a loop of its own, and a shopping agent pays at the end",
            "tools": "Its system prompt, its tools and its limit; the loop was the same",
            "model": "Another model, one trained on code, since a chat model cannot fix a function",
        },
        correct="tools",
        why={
            "tools": "One loop, the same model and the same step limit; only what the agent can do, and what may go wrong with it, differed.",
            "loop": "Both ran with the same loop: send, run the call, add the result, again. See [Coding, research and personal agents](/course/open/build-kinds-of-agents?lang=en).",
            "model": "All four worlds ran with Qwen3-0.6B. See [Coding, research and personal agents](/course/open/build-kinds-of-agents?lang=en).",
        },
    )
    quick_same
    return (quick_same,)


@app.cell(hide_code=True)
def quick_check_mandate():
    quick_mandate = L.choice(
        "quick-mandate",
        question="The shopping agent's system prompt already says where it may pay, and how much. Why does `pay` still ask `may_pay`?",
        options={
            "code": "A line in the prompt is a request to the model; only the check in the code that pays holds",
            "speed": "So that the model does not have to work out the total of the cart itself, which saves it a request",
            "user": "So that the user sees the mandate once more before every payment",
        },
        correct="code",
        why={
            "code": "The model can pick a shop outside the mandate anyway; the code that runs `pay` stops it there.",
            "speed": "`may_pay` adds no request and does not replace one; it decides whether the payment may happen. See [Coding, research and personal agents](/course/open/build-kinds-of-agents?lang=en).",
            "user": "Within the mandate, the user is not asked at all. See [Coding, research and personal agents](/course/open/build-kinds-of-agents?lang=en).",
        },
    )
    quick_mandate
    return (quick_mandate,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The loop with tools and limits did not come first. In 2022 and 2023, several designs tried to make a model do more than answer, and in 2023 two of them drew a lot of attention: AutoGPT and BabyAGI promised that a model given only an objective would work toward it by itself. Most of the time they went round in circles or wandered off. Knowing what they did explains why today's agents get a task instead of an objective, and with it a step limit and a person who checks.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **autonomous agent** | an agent that sets its own tasks toward an objective and does them without a person in between |
    | **MRKL** | an early design (AI21 Labs, 2022), said as *miracle*: a router sends each question to the right module, such as a calculator |
    | **HuggingGPT** | a design from 2023 in which a chat model plans the steps of a task and gives each step to another model |
    | **`.lstrip(chars)`, `.rstrip(chars)`** | take every character that is in `chars` off the start of a text, or off its end: `"2. Buy milk.".lstrip("0123456789. ")` is `"Buy milk."`, and `.rstrip(".")` makes that `"Buy milk"` |
    | **`x not in items`** | `True` when `x` is not in the list `items`; `x in items` is its opposite |
    | **`a and b`** | `True` only when `a` and `b` are both `True` |

    The table below puts the designs in a row. Each left something that is still there, except the autonomous agents: what came of them is mostly what today's agents do to prevent their mistakes.
    """)
    return


@app.cell(hide_code=True)
def picture_history():
    mo.Html(C.history_picture("en"))
    return


@app.cell(hide_code=True)
def text_babyagi():
    mo.md(r"""
    BabyAGI fits in a few lines. It keeps a list of tasks toward an objective. Each round, the model does the first task in the list; a second request asks it, with the result, which new tasks are still needed, and those go at the end of the list; a third puts the list in order. Then the next round starts. Nothing in it says when the objective is reached; the original kept going until someone stopped it. The course wrote its own small version with the first two requests, and a line in the second that asks the model to write DONE when the objective is reached. The run ends when the list is empty, or after the last round.
    """)
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course ran its small BabyAGI with Qwen3-0.6B on {o} objectives, {r} rounds each, at temperature 0, so that the same request always gets the same answer. Each objective went twice: once with every new task added to the list, as BabyAGI did, and once with the list of exercise 2, which skips a task that is done or already in the list, and holds at most {l}.
    """.format(o=F["objectives"], r=F["rounds"], l=F["limit"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("An agent that sets its own tasks needs an end that is not up to the model: a step limit, a list that cannot grow without bounds, and a person who looks at what it did. Today's agents start from a task a person gave, and stop when it is done or the limit is reached."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a list of tasks an agent writes itself

    1. **Start from a task a person gave,** not only an objective.
    2. **Read what the model writes as tasks:** one per line, without numbers or dashes, and recognize when it says it is done.
    3. **Skip a task that is done or already in the list,** so the agent does not do it twice.
    4. **Cap the list and the rounds,** so that the agent stops even when the model keeps writing tasks.
    5. **Show a person the tasks and the results,** before anything happens that cannot be undone.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(*C.worked_run(), "en"))
    return


@app.cell(hide_code=True)
def text_parse():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The model's lines as tasks

        `parse_tasks(text)` turns what the model wrote into a list of tasks. Each line is one task. Take the spaces off both ends of a line, then, with `.lstrip("-*0123456789.) ")`, the dashes, stars and numbers a list may start with, and with `.rstrip(".")` a dot at the end. Skip a line that is empty after that, and the lines `DONE` and `Done`, with which the model says it has finished. The check tries three plain lines, a numbered list, a list with dashes and an empty line, a lone `DONE`, the text in the picture, and two texts in which the model wrote `DONE.` and `Done.` with a dot.
        """),
        L.task("Replace `...` with a line that cleans up `line`, and an `if` that adds it to `tasks` when it is not empty, not `DONE` and not `Done`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parse():
    box_parse = L.code_box("parse", lines=7)
    box_parse
    return (box_parse,)


@app.cell(hide_code=True)
def check_parse(box_parse):
    result_parse = L.check("parse", box_parse.value)
    result_parse
    return (result_parse,)


@app.cell(hide_code=True)
def hints_parse():
    L.hints(
        "parse",
        [
            "Clean the line first, then decide whether it is a task. Each of these methods gives back a new text, so you can put them one after the other.",
            "`line.strip().lstrip(\"-*0123456789.) \").rstrip(\".\")` takes the spaces off both ends, what a list puts in front, and the dot at the end. A line that is a task is not in the list `[\"\", \"DONE\", \"Done\"]`.",
            "`line = line.strip().lstrip(\"-*0123456789.) \").rstrip(\".\")`, then `if line not in [\"\", \"DONE\", \"Done\"]:` with `tasks.append(line)` under it.",
        ],
        why="""```python
def parse_tasks(text):
    tasks = []
    for line in text.split("\\n"):
        line = line.strip().lstrip("-*0123456789.) ").rstrip(".")
        if line not in ["", "DONE", "Done"]:
            tasks.append(line)
    return tasks
```

A model writes a list in its own way, with numbers, dashes or neither; the code that reads it must cope with each.""",
    )
    return


@app.cell(hide_code=True)
def text_add():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A list that stays in bounds

        `add_tasks(queue, new, done, limit)` adds the new tasks to the list `queue` and gives the list back. Skip a task that is already in `queue`, and one that is in `done`, the tasks the agent did. Add only while the list holds fewer than `limit` tasks. The check tries the case in the picture, a task that is already in the list, a list that fills up, the same new task twice, and a list with room for two.
        """),
        L.task("Replace `...` with an `if` that adds `task` to `queue` only when the list has room, and the task is neither in `queue` nor in `done`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_add():
    box_add = L.code_box("add", lines=6)
    box_add
    return (box_add,)


@app.cell(hide_code=True)
def check_add(box_add):
    result_add = L.check("add", box_add.value)
    result_add
    return (result_add,)


@app.cell(hide_code=True)
def hints_add():
    L.hints(
        "add",
        [
            "Three things must all hold before a task goes in: there is room, it is not in the list yet, and it was not done. Join them with `and`.",
            "Room means `len(queue) < limit`. The other two are `task not in queue` and `task not in done`.",
            "`if len(queue) < limit and task not in queue and task not in done:` with `queue.append(task)` under it.",
        ],
        why="""```python
def add_tasks(queue, new, done, limit):
    for task in new:
        if len(queue) < limit and task not in queue and task not in done:
            queue.append(task)
    return queue
```

The test for `queue` also catches the same task twice in `new`: the first one is in the list by the time the second comes.""",
    )
    return


@app.cell(hide_code=True)
def text_stayed():
    mo.md(r"""
    ### Exercise 3. What stayed
    """)
    return


@app.cell(hide_code=True)
def choice_stayed_cell():
    choice_stayed = L.choice(
        "stayed",
        question="Today's agents kept a part of AutoGPT and BabyAGI, and replaced another. Which is which?",
        options={
            "kept-loop": "They kept the loop of a model that acts and reads the result; the endless objective became a task from a person, with a step limit",
            "kept-list": "They kept the list of tasks the model writes for itself, without an end; the loop with tools was dropped, since it went round in circles",
            "kept-all": "They kept both, and only made the model larger, since a larger model no longer wanders off",
        },
        correct="kept-loop",
        why={
            "kept-loop": "The loop works; what went wrong was that nothing outside the model decided when to stop or what was allowed.",
            "kept-list": "It was the other way round: the loop with tools stayed, in every agent of this module; the list without end did not.",
            "kept-all": "A larger model makes fewer mistakes, but an agent still needs an end that is not up to the model.",
        },
    )
    choice_stayed
    return (choice_stayed,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Give the course's small BabyAGI an objective of your own, and it runs {r} rounds on Qwen3-0.6B, with your `parse_tasks` reading the model's tasks and your `add_tasks` keeping the list in bounds. The course starts its own server on your computer, with the model behind it.
    """.format(r=C.DEMO_ROUNDS))
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
def objective_cell():
    objective_box = mo.ui.text(value=C.OBJECTIVES[1], label="Objective", full_width=True)
    start_button = mo.ui.run_button(label="Start")
    mo.vstack([objective_box, start_button])
    return objective_box, start_button


@app.cell(hide_code=True)
def demo_run(box_add, box_parse, objective_box, start_button):
    _parse = C.learner_fn("parse", box_parse.value)
    _add = C.learner_fn("add", box_add.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _parse is None or _add is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo reads and keeps the tasks with your own functions.*")
    elif not start_button.value:
        _out = mo.md("*Press the button to start.*")
    elif not objective_box.value.strip():
        _out = mo.md("*Write an objective in the box first.*")
    else:
        mo.output.replace(mo.md("*The agent has started; the server loads Qwen3 …*"))
        _words = {"title": "Your small BabyAGI, round by round", "first": "the first tasks", "round": "round {k}: the model did",
                  "new": "new tasks it wrote", "left": "left in the list at the end ({n})", "empty": "(none)",
                  "stopped": "It stopped with {e}.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.baby_agi(objective_box.value.strip(), _parse, _add, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** ReAct is the loop of [The agent loop](/course/open/build-agent-loop?lang=en), MRKL's modules are the tools of this whole module, and the autonomous agents explain the limits of [Coding, research and personal agents](/course/open/build-kinds-of-agents?lang=en): a task from a person, a step limit, and a check in the code. The next lesson looks at agent frameworks, which give you the loop, the tools and the rest ready-made.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stayed, result_add, result_parse):
    L.wrap_up(result_parse, result_add, choice_stayed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
