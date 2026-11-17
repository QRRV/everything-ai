# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-plan-execute. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-plan-execute", "en")
    C = L.checks   # the kitchen's three tools, the course's runs of Qwen3 as a plan and as a loop, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [The agent loop](/course/open/build-agent-loop?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_messages():
    quick_messages = L.choice(
        "quick-messages",
        question="In `run_agent`, the model answers with two tool calls. What goes into `messages` before the next request?",
        options={
            "results": "Only the two results, put together in one message of the user, so that the request stays short",
            "both": "The model's message as it came, then a message with the role `tool` for each call",
            "last": "Only the result of the last call, since the model can always ask for the first one again later",
        },
        correct="both",
        why={
            "both": "The server checks that each call has its result, by the id of the call, and the model reads its own calls and their results in the next round.",
            "results": "Without its own message with the calls, the model does not know what the results belong to, and the server refuses calls without results. See [The agent loop](/course/open/build-agent-loop?lang=en).",
            "last": "Each call needs its result: the server refuses a request with a call whose result is missing. See [The agent loop](/course/open/build-agent-loop?lang=en).",
        },
    )
    quick_messages
    return (quick_messages,)


@app.cell(hide_code=True)
def quick_check_limit():
    quick_limit = L.choice(
        "quick-limit",
        question="What does the step limit in `run_agent` protect you from?",
        options={
            "wrong": "A wrong answer: once the limit is reached, the loop checks the answer against the results of the tools",
            "long": "An answer that is too long: the limit is the largest number of tokens the model may write",
            "loop": "A run that keeps calling tools and never answers, at a cost in time and tokens",
        },
        correct="loop",
        why={
            "loop": "After `max_steps` requests, the loop stops and gives back `None`, so that your code knows the agent did not finish.",
            "wrong": "The loop does not check answers; it only counts requests. See [The agent loop](/course/open/build-agent-loop?lang=en).",
            "long": "That is `max_tokens`, per request. The step limit counts requests. See [The agent loop](/course/open/build-agent-loop?lang=en).",
        },
    )
    quick_limit
    return (quick_limit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [The agent loop](/course/open/build-agent-loop?lang=en), the model chose one step at a time: after each tool call, the whole conversation went back to it, and it decided what to do next. For a task whose steps are clear from the start, such as working out what to buy for a dish, that is many requests for little choice, and the model has to copy each result into its next call. Another way is to have the model write the whole plan first, and let your code carry it out. You can then also read the plan before anything runs. In this lesson you run such a plan, find the steps that can run at the same time, and measure it against an agent loop.
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
        | **plan-and-execute** | an agent in which the model first writes all the steps as a plan, after which code carries them out; at the end, one more request writes the answer from the results |
        | **ReWOO** | *reasoning without observation*, from Xu and others (2023): a form of plan-and-execute in which a step can use the result of an earlier step by its number, so that the model writes the plan without seeing any result |
        | **`"#1"`** | in this lesson's plans: the result of step 1, as a value in a later step |
        | **LLMCompiler** | from Kim and others (2023): a form of plan-and-execute that runs the steps that do not need each other's results at the same time |
        | **wave** | steps that can run at the same time, because every result they use is already there |

        The kitchen is the one of [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en), with three tools for the plan. Each takes its arguments as a dictionary:

        ```python
        recipe({"dish": "pancakes"})       # {"flour": 250, "milk": 450, "eggs": 110, "butter": 25}
        pantry({"items": ["milk", "eggs"]})   # {"milk": 200, "eggs": 0}
        missing({"need": {"milk": 450, "eggs": 110}, "have": {"milk": 200, "eggs": 0}})   # {"milk": 250, "eggs": 110}
        ```

        `pantry` also takes a recipe as `items`: it then looks up each of its ingredients. A plan is JSON with a list of steps, each a tool and its arguments. Where a step needs an earlier result, the plan writes its number. The plan for pancakes:

        ```json
        {"steps": [{"tool": "recipe", "args": {"dish": "pancakes"}},
                   {"tool": "pantry", "args": {"items": "#1"}},
                   {"tool": "missing", "args": {"need": "#1", "have": "#2"}}]}
        ```

        Your code runs step 1, keeps its result as `"#1"`, gives it to step 2 as `items`, and so on. The model writes the plan in one request, in JSON mode, and never has to copy a result. A second request gets the question and every step's result, and writes the answer.
        """),
        mo.accordion({"The course's two prompts, if you want to read them": mo.md(
            "```text\n" + C.PLAN.format(question="...") + "\n```\n\n```text\n" + C.SOLVE + "\n```")}),
    ])
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course asked Qwen3-0.6B the {single} questions of [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en) and {double} more about two dishes, at temperature 0, so that the same request always gets the same answer. Each question went three ways. As Qwen3's own plan: the planning prompt in JSON mode, the course's code running the steps, and a second request for the answer. With the course's plan: the code ran the plan the course wrote for that question, and the model only wrote the answer from the results. And as an agent loop, with the same three tools as tools of the API and a limit of {steps} requests.
    """.format(single=F["single"], double=F["double"], steps=F["steps"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A plan written beforehand cannot react to what a step finds. If the recipe is missing, the steps after it still run, on an error. Use a plan where the steps are clear from the start, and a loop where the next step depends on what the last one found."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: plan first, then act

    1. **Describe the tools in the prompt,** with the form of a plan and how a step names an earlier result.
    2. **Ask for the plan in JSON mode,** and read it with `json.loads`. Can your code not read it? Then there is no plan to run.
    3. **Read the plan before it runs,** or have a person approve it, as with a tool that changes something.
    4. **Run the steps in order,** and give each step the result of every earlier step it names by its number.
    5. **Run the steps of a wave at the same time** when your tools are slow, such as calls over a network.
    6. **Ask once more for the answer,** with the question and every step's result.
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
def text_run_plan():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Run a plan

        `run_plan(plan)` is the executor: it runs the steps of a plan in order, and gives back the result of each step under its number, as `{"#1": ..., "#2": ..., ...}`. `FUNCTIONS["recipe"]` is the tool itself, and takes the arguments as a dictionary. In the inner loop, `value` is one argument of a step: a text, such as `"pancakes"` or `"#1"`, or a list, such as `["milk", "eggs"]`. A text that is the number of an earlier step becomes that step's result; every value then goes into `arguments` under its name. Complete the inner loop. The box knows `FUNCTIONS`. The check runs your function on the plan for pancakes and on six other plans; the picture under it shows each step of the plan for pancakes with what your function gave.
        """),
        L.task("In the inner loop, replace a value that is in `results` with that result, and put every value in `arguments` under its name."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_plan():
    box_run_plan = L.code_box("run-plan", lines=12)
    box_run_plan
    return (box_run_plan,)


@app.cell(hide_code=True)
def check_run_plan(box_run_plan):
    result_run_plan = L.check("run-plan", box_run_plan.value)
    result_run_plan
    return (result_run_plan,)


@app.cell(hide_code=True)
def hints_run_plan():
    L.hints(
        "run-plan",
        [
            "Two things happen to each value. First: is it the number of an earlier step? Then it becomes that step's result. Second: it goes into `arguments`.",
            "`results` holds the result of each earlier step under its number, such as `\"#1\"`. `value in results` tells you whether `value` is one of those numbers. A list cannot be a key of a dictionary, so `value in results` stops with an error on a list: look first whether `value` is a text, with `isinstance(value, str)`.",
            "`if isinstance(value, str) and value in results:` with `value = results[value]` under it; then, not indented under the `if`, `arguments[name] = value`.",
        ],
        why="""```python
def run_plan(plan):
    results = {}
    for step in plan["steps"]:
        arguments = {}
        for name, value in step["args"].items():
            if isinstance(value, str) and value in results:
                value = results[value]
            arguments[name] = value
        results["#" + str(len(results) + 1)] = FUNCTIONS[step["tool"]](arguments)
    return results
```

`"pancakes"` is not a key of `results`, so it stays as it is; `"#1"` is, and becomes the recipe. A list is not a text, so it goes into `arguments` as it is; `and` only looks in `results` when `value` is a text. The number of a new result is one more than the number of results so far.""",
    )
    return


@app.cell(hide_code=True)
def text_waves():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The steps that can run at the same time

        A step can start as soon as every result it uses is there. `waves(plan)` gives the wave of each step under its number: 1 for a step that uses no earlier result, and otherwise one more than the latest wave of the results it uses. In the plan for two dishes, both recipes are in wave 1, both lookups in the pantry in wave 2, and both steps with `missing` in wave 3: six steps in three waves. Complete the inner loop. The check runs your function on that plan and six others; the picture shows the plan for two dishes in the waves your function found.
        """),
        L.task("In the inner loop, when `value` is the number of an earlier step, make `w` at least one more than that step's wave."),
    ])
    return


@app.cell(hide_code=True)
def exercise_waves():
    box_waves = L.code_box("waves", lines=10)
    box_waves
    return (box_waves,)


@app.cell(hide_code=True)
def check_waves(box_waves):
    result_waves = L.check("waves", box_waves.value)
    result_waves
    return (result_waves,)


@app.cell(hide_code=True)
def hints_waves():
    L.hints(
        "waves",
        [
            "A step waits for every result it uses. So it can start only after the latest of them: its wave is one more than the largest wave among them.",
            "`wave` holds the wave of each earlier step under its number. When `value in wave`, that step's wave is `wave[value]`, and this step needs at least `wave[value] + 1`. `max(a, b)` gives the larger of two numbers. As in exercise 1, a value can be a list: look first whether it is a text.",
            "`if isinstance(value, str) and value in wave:` with `w = max(w, wave[value] + 1)` under it.",
        ],
        why="""```python
def waves(plan):
    wave = {}
    for step in plan["steps"]:
        w = 1
        for name, value in step["args"].items():
            if isinstance(value, str) and value in wave:
                w = max(w, wave[value] + 1)
        wave["#" + str(len(wave) + 1)] = w
    return wave
```

With `max`, the order of the arguments does not matter: a step with `"have": "#2"` before `"need": "#1"` still waits for step 2.""",
    )
    return


@app.cell(hide_code=True)
def text_plan_or_loop():
    mo.md(r"""
    ### Exercise 3. A plan or a loop
    """)
    return


@app.cell(hide_code=True)
def choice_plan_or_loop_cell():
    choice_plan_or_loop = L.choice(
        "plan-or-loop",
        question="Which task suits a plan written first better than an agent loop?",
        options={
            "search": "Finding a recipe your guests like, by reading pages until one of them fits what they eat",
            "known": "Shopping for two dishes: look up both recipes, then the pantry, then what is missing",
            "debug": "Fixing a test that fails, where each new run of the tests shows what to try next",
        },
        correct="known",
        why={
            "known": "The steps are known before any result comes in; only the values in them depend on earlier steps, and the numbers such as `\"#1\"` take care of that.",
            "search": "What to read next depends on what the last page said: the agent has to see each result before it chooses. That calls for a loop.",
            "debug": "Each run of the tests changes the next step. A plan written beforehand cannot know what the tests will say.",
        },
    )
    choice_plan_or_loop
    return (choice_plan_or_loop,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 1 is right, you can run a plan of the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it, asks it for a plan, runs the plan with your `run_plan`, and asks for the answer. When exercise 2 is right too, you also see your waves. In the demo, a tool that fails gives back its error as text, as in the course's runs, so that the steps after it still run. The question in the box is the one of the worked example. Ask about one or two dishes of the kitchen in your own words.
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
    question_box = mo.ui.text(value=C.TWO[0][0], label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Plan and run")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_plan(box_run_plan, box_waves, question_box, send_button):
    _run_plan = C.learner_fn("run-plan", box_run_plan.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _run_plan is None:
        _out = mo.md("*Finish exercise 1 first: the demo runs the plan with your own `run_plan`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to run.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3, and the model writes its plan …*"))
        _words = {"title": "A plan of Qwen3, run by your run_plan", "wrote": "1. The plan Qwen3 wrote",
                  "waves": "Your waves", "wave": "wave {k}", "ran": "2. What your run_plan gave for each step",
                  "answered": "3. The answer Qwen3 wrote from the results",
                  "unread": "Your code cannot read this as a plan: it is not JSON with a list of steps. There is nothing to run.",
                  "error": "Your run_plan stopped with {e}. A plan from a model can hold a value your code does not expect: read the plan above.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_run_plan, C.learner_fn("waves", box_waves.value), question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loop of [The agent loop](/course/open/build-agent-loop?lang=en) let the model choose each step after seeing the last result; here it writes all steps first, and your code carries them out, with the results passed on by their numbers. Running independent steps at the same time is what the workers of an orchestrator in [Five workflow patterns](/course/open/build-workflow-patterns?lang=en) could do too. The next lesson is about what an agent remembers, within one conversation and across conversations.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_plan_or_loop, result_run_plan, result_waves):
    L.wrap_up(result_run_plan, result_waves, choice_plan_or_loop)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
