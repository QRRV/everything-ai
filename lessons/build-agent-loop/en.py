# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-agent-loop. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-loop", "en")
    C = L.checks   # the kitchen's agent with and without a thought before each call, the course's runs of Qwen3, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Five workflow patterns](/course/open/build-workflow-patterns?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_orchestrator():
    quick_orchestrator = L.choice(
        "quick-orchestrator",
        question="What makes an orchestrator with workers different from parts side by side?",
        options={
            "decides": "The model decides which parts there are and how many, once it has read the input",
            "faster": "Its workers always run one after the other, so that each can read the last one's result",
            "model": "It needs a large model for every worker, while parts side by side can use a small one",
        },
        correct="decides",
        why={
            "decides": "With parts side by side, your code knows the parts beforehand; with an orchestrator, a menu can hold two dishes or five.",
            "faster": "The workers of an orchestrator do not need each other either, so they can run at the same time. See [Five workflow patterns](/course/open/build-workflow-patterns?lang=en).",
            "model": "Both patterns work with any model; in the exercise, the worker was plain code. See [Five workflow patterns](/course/open/build-workflow-patterns?lang=en).",
        },
    )
    quick_orchestrator
    return (quick_orchestrator,)


@app.cell(hide_code=True)
def quick_check_evaluator():
    quick_evaluator = L.choice(
        "quick-evaluator",
        question="In `improve`, the check finds the tip too long. What does the model read in the next round?",
        options={
            "task": "Only the task again, so that it starts fresh and does not repeat its mistake",
            "problem": "Only what was wrong, as a new message of the user, without the tip it wrote",
            "both": "The task, its own tip with the role `assistant`, and what was wrong with it",
        },
        correct="both",
        why={
            "both": "That is what `improve` sends back: the tip the model wrote, with the role `assistant`, and why it went back, so that the model knows what to change.",
            "task": "At temperature 0, the same messages give the same tip again. See [Five workflow patterns](/course/open/build-workflow-patterns?lang=en).",
            "problem": "Without its own tip, the model does not know what was too long. See [Five workflow patterns](/course/open/build-workflow-patterns?lang=en).",
        },
    )
    quick_evaluator
    return (quick_evaluator,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the five patterns, your code decided the paths. Some tasks have no fixed path: what to do next depends on what the last step found. Then the model has to choose, and you need an agent. In [What an agent is](/course/open/agent-idea?lang=en), Qwen3 chose badly: it looked up the recipe, and answered without looking in the pantry. One well-known way to help a model choose is to have it write down what it knows and what it needs next, before each tool call. In this lesson you build the whole loop of an agent through the API, add that, and measure whether it helps.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **ReAct** | *reason and act*: an agent in which the model takes turns between a thought, an action (a tool call) and an observation, until it can answer; from Yao and others (2022) |
    | **thought** | a line of text the model writes with its tool call: what it knows so far, and what it needs next |
    | **observation** | what the agent gets to see after an action; here, the result of the tool, which the loop sends back with the role `tool` |

    The loop is the one of [What an agent is](/course/open/agent-idea?lang=en), now through the API of [Tools as JSON](/course/open/build-tool-calling?lang=en). Each request holds the whole conversation so far, with the tools. When the model's answer has tool calls, your code adds the answer as it came, runs each call, and adds each result with the id of its call; then it asks again. An answer without tool calls is the agent's answer. After `max_steps` requests, the loop stops anyway.

    ReAct changes only the system prompt. The course's prompt is the one of the last lessons, with one more sentence:

    > {react}

    The thought then comes back as the text of an answer with tool calls, and the loop sends it back in the next request, so the model reads its own reasoning.
    """.format(react=C.REACT))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course asked Qwen3-0.6B the {n} questions of [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en) as an agent, at temperature 0, so that the same request always gets the same answer, with a limit of {steps} requests: once with the system prompt of that lesson, once with the sentence above added.
    """.format(n=F["questions"], steps=F["steps"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A thought is text the model writes, not proof that it reasons well. It can write *I have all the ingredients* and be wrong, and every thought costs tokens in each next request. Read the traces of your own runs, and measure on your own questions whether the thoughts help."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: an agent loop through the API

    1. **Start the messages** with the system prompt and the task.
    2. **Ask the model,** with the tools in the request.
    3. **No tool calls in the answer?** Then it is the answer: give back its content.
    4. **Otherwise, add the answer as it came,** thought and calls together, run each call, and add each result with the role `tool` and the id of its call.
    5. **Stop after `max_steps` requests,** and give back `None`, so that your code knows the agent did not finish.
    6. **Write each run down as a trace,** and read it: the trace shows where the agent went wrong.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def picture_run():
    mo.Html(C.run_picture(C.WORKED, "en"))
    return


@app.cell(hide_code=True)
def text_run_agent():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The whole loop

        `run_agent(url, task, max_steps)` is the agent of this lesson. `ask(url, messages)` sends the messages with `TOOLS` to the server, and gives back the model's message as a dictionary: its `"content"`, and, when it calls tools, its `"tool_calls"`, each with an `"id"` and a `"function"` with a `"name"` and the `"arguments"` as JSON text. `FUNCTIONS["pantry"]` is the tool itself: it takes the arguments as a dictionary and gives back the result as text. Complete the loop. The box knows `TOOLS`, `FUNCTIONS` and `REACT`. The check runs your loop against a stand-in of the server that plays three runs: one on pancakes that calls four tools at once, one that never stops calling tools, and one that answers at once. The picture under it shows your run on pancakes as a trace.
        """),
        L.task("In the loop, add the model's message to `messages`; give back its content when it has no tool calls; otherwise run each call and add its result as a message with the role `tool`, the call's id and the result."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_agent():
    box_run_agent = L.code_box("run-agent", lines=26)
    box_run_agent
    return (box_run_agent,)


@app.cell(hide_code=True)
def check_run_agent(box_run_agent):
    result_run_agent = L.check("run-agent", box_run_agent.value)
    result_run_agent
    return (result_run_agent,)


@app.cell(hide_code=True)
def hints_run_agent():
    L.hints(
        "run-agent",
        [
            "Each round has two endings. An answer without tool calls ends the run. An answer with tool calls means: run them, and ask again with their results in the messages.",
            "`not` turns a test around: `not x` holds when `x` is `None` or an empty list. Add `message` to `messages` first, as it came. For each call in `message[\"tool_calls\"]`, read its arguments with `json.loads`, run `FUNCTIONS[...]` with the call's name, and add a message with the role `tool`.",
            "For each call, once `arguments` is read with `json.loads`: `result = FUNCTIONS[___](arguments)`.",
        ],
        why="""```python
def run_agent(url, task, max_steps):
    messages = [{"role": "system", "content": REACT}, {"role": "user", "content": task}]
    for step in range(max_steps):
        message = ask(url, messages)
        messages = messages + [message]
        if not message.get("tool_calls"):
            return message["content"]
        for call in message["tool_calls"]:
            arguments = json.loads(call["function"]["arguments"])
            result = FUNCTIONS[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
    return None
```

The model's message goes back as it came, with its thought and its calls: the server checks that each call has its result, and the model reads its own reasoning in the next round.""",
    )
    return


@app.cell(hide_code=True)
def text_trace():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The trace

        A run is easier to read as lines than as messages. `trace(messages)` goes through the messages of a run and gives back a list of lines: for a tool's message, `"Observation: "` and its content; for an answer with tool calls, its thought as the model wrote it, when it wrote one, and then for each call `"Action: "`, its name and its arguments in brackets, such as `Action: pantry({"item": "milk"})`; for an answer without tool calls, `"Answer: "` and its content. The system prompt and the task get no line. Complete the loop. The check tries a run with thoughts, a run without any, and an answer without a tool call.
        """),
        L.task("Under each `if`, add the lines that the text above gives for that kind of message."),
    ])
    return


@app.cell(hide_code=True)
def exercise_trace():
    box_trace = L.code_box("trace", lines=16)
    box_trace
    return (box_trace,)


@app.cell(hide_code=True)
def check_trace(box_trace):
    result_trace = L.check("trace", box_trace.value)
    result_trace
    return (result_trace,)


@app.cell(hide_code=True)
def hints_trace():
    L.hints(
        "trace",
        [
            "A tool's message is simple: one line. An assistant's message is one of two kinds: with tool calls, or without. Tell them apart with `m.get(\"tool_calls\")`.",
            "With tool calls: when `m.get(\"content\")` has text, that is the thought, its own line; then one line per call. Without: one `\"Answer: \"` line.",
            "For each call `c`: `lines.append(\"Action: \" + c[\"function\"][\"name\"] + \"(\" + ___ + \")\")`.",
        ],
        why="""```python
def trace(messages):
    lines = []
    for m in messages:
        if m["role"] == "tool":
            lines.append("Observation: " + m["content"])
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                if m.get("content"):
                    lines.append(m["content"])
                for c in m["tool_calls"]:
                    lines.append("Action: " + c["function"]["name"] + "(" + c["function"]["arguments"] + ")")
            else:
                lines.append("Answer: " + m["content"])
    return lines
```

`if m.get("content"):` does not hold for `None`, nor for the empty text `""`: a model without a thought can send either.""",
    )
    return


@app.cell(hide_code=True)
def text_step_limit():
    mo.md(r"""
    ### Exercise 3. At the step limit
    """)
    return


@app.cell(hide_code=True)
def choice_step_limit_cell():
    choice_step_limit = L.choice(
        "step-limit",
        question="Your agent reaches the step limit while it still calls tools, and `run_agent` gives back `None`. What should your app do then?",
        options={
            "raise": "Raise the limit to a thousand requests, so that every run finishes in the end",
            "say": "Say that the agent did not finish, and show the trace of what it did so far",
            "last": "Show the text of the last step as the answer, since the model wrote it too",
        },
        correct="say",
        why={
            "say": "The user then knows there is no answer, and the trace shows where the agent went round in circles.",
            "raise": "An agent that goes round in circles keeps doing so, and costs time and tokens for each request. The limit is there to stop it.",
            "last": "The last step is a thought with a tool call, not an answer. Showing it as one hides that the agent did not finish.",
        },
    )
    choice_step_limit
    return (choice_step_limit,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 1 is right, you can run your agent on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it, and your loop runs the kitchen's tools. Ask about a dish in your own words, and read the trace.
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
    question_box = mo.ui.text(value=C.QUESTIONS[4][0], label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Run your agent")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_run_agent(box_run_agent, question_box, send_button):
    _run_agent = C.learner_run_agent(box_run_agent.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _run_agent is None:
        _out = mo.md("*Finish exercise 1 first: the demo runs your own loop.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to run.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3, and your agent runs …*"))
        _words = {"title": "Your agent with Qwen3, as a trace", "gave": "Your run_agent gave back: {v}",
                  "error": "Your run_agent stopped with {e}.", "none": "Your run_agent sent nothing to the server.",
                  "requests": "{k} requests",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_run_agent, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loop of [What an agent is](/course/open/agent-idea?lang=en) now runs through the API of [Tools as JSON](/course/open/build-tool-calling?lang=en), with the results sent back by the ids of their calls. Where the patterns of [Five workflow patterns](/course/open/build-workflow-patterns?lang=en) fixed the path in code, this loop lets the model choose it, step by step, with a thought before each call and a limit on the steps. The next lesson lets the model write its whole plan first, before any step runs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_step_limit, result_run_agent, result_trace):
    L.wrap_up(result_run_agent, result_trace, choice_step_limit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
