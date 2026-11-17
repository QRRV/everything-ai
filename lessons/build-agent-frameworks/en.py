# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-agent-frameworks. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-frameworks", "en")
    C = L.checks   # the tool convert three ways, the course's runs of Qwen3, the graph of the agent loop, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Tools through a protocol: MCP](/course/open/build-mcp?lang=en), and one about [What an agent remembers](/course/open/build-agent-memory?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mcp():
    quick_mcp = L.choice(
        "quick-mcp",
        question="A tool comes from an MCP server instead of from your own code. What changes for the model?",
        options={
            "protocol": "The model talks JSON-RPC with the server itself, so it needs no tool definitions",
            "nothing": "Nothing: your code asks the server for its tools and puts them in the request as before",
            "faster": "The tool is called without a request to the model, since the server decides when it is needed",
        },
        correct="nothing",
        why={
            "nothing": "MCP fixes how your program finds and calls the tools; the model sees the same tool definitions and writes the same calls.",
            "protocol": "Your code is the MCP client; the model only sees the tools in the request. See [Tools through a protocol: MCP](/course/open/build-mcp?lang=en).",
            "faster": "The model still decides when to call a tool; the server only runs the call. See [Tools through a protocol: MCP](/course/open/build-mcp?lang=en).",
        },
    )
    quick_mcp
    return (quick_mcp,)


@app.cell(hide_code=True)
def quick_check_memory():
    quick_memory = L.choice(
        "quick-memory",
        question="How does a new conversation know what the agent saved with `remember` last week?",
        options={
            "model": "The model learned it: each call of `remember` trains the model a little",
            "server": "The server keeps every earlier conversation, and sends it along with each request by itself",
            "notes": "Your code reads the notes from the file and puts them in the system prompt",
        },
        correct="notes",
        why={
            "notes": "The model remembers nothing between requests; the notes are kept outside it, and your code gives them back.",
            "model": "Nothing trains the model; `remember` writes a note in a file. See [What an agent remembers](/course/open/build-agent-memory?lang=en).",
            "server": "The server answers each request on its own and keeps nothing. See [What an agent remembers](/course/open/build-agent-memory?lang=en).",
        },
    )
    quick_memory
    return (quick_memory,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In this module you wrote the agent loop, tool definitions, an MCP client, a memory, a trace and limits yourself. An agent framework gives you all of that ready-made: LangGraph, LlamaIndex, AutoGen, PydanticAI, and the agent SDKs of the providers of models. That saves work, but a framework also does things you do not see, such as adding its own instructions to every request. Once you know which of your own parts sits under each name, you can choose a framework, or none, and find out what it does when your agent behaves differently than you expected.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **agent framework** | a library that gives the parts of agents ready-made, such as the agent loop, tool definitions, memory and traces |
    | **docstring** | the text in triple quotes right under the `def` line of a function, which says what the function does |
    | **graph** | nodes joined by edges; here a node is a step that changes the state, and the edge after it says which node comes next |
    | **state** | everything the steps share, such as the messages so far; each node gets it and gives back a new one |
    | **`while ...:`** | runs the lines under it again and again, as long as what comes after `while` holds |
    | **`a and b`** | `True` only when `a` and `b` are both `True` |

    The parts of a framework have their own names, but under each is something you built:

    | In a framework | What you built |
    |---|---|
    | the agent, its run | the loop of [The agent loop](/course/open/build-agent-loop?lang=en): send, run the calls, add the results, until there are no calls or the step limit |
    | a tool made from a function | the tool definition of [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en), written by hand |
    | MCP tools | your client of [Tools through a protocol: MCP](/course/open/build-mcp?lang=en) |
    | memory | the notes of [What an agent remembers](/course/open/build-agent-memory?lang=en) |
    | tracing | the trace of [The agent loop](/course/open/build-agent-loop?lang=en) |
    | guardrails, approvals | the limits of [What a tool may do](/course/open/build-tool-permissions?lang=en) |

    Some frameworks write the loop as a graph. LangGraph is the best known: you give the nodes, the edges and a start, and the framework runs them. The agent loop is then two nodes, `model` and `tools`, and the edge after `model` looks at the last answer:
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    A framework makes a tool from a function: from its name, its docstring and the types of its parameters. It can also put instructions before every request without showing them. The course asked Qwen3-0.6B the {n} questions of [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en) three ways, at temperature 0, so that the same request always gets the same answer, with at most two requests per question: with `convert` written by hand and no system prompt, as in that lesson; with the same tool and a system prompt of the kind a framework adds by itself, written by the course; and with `convert` made from the function, as your `tool_from` of exercise 1 makes it.
    """.format(n=F["questions"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A framework's hidden instructions and its made tool definitions go to the model like your own. Before you build on a framework, look at the requests it really sends: most can show them, in their tracing or in a log."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: judging an agent framework

    1. **Name the parts you need:** the loop, tools, MCP, memory, traces, limits.
    2. **Find each part under the framework's names,** and what it does differently from your own.
    3. **Look at the requests it sends:** the system prompt it adds, and the tool definitions it makes from your functions.
    4. **Run the same tasks with and without it,** with the same model, and compare the answers and the tokens.
    5. **Keep the limits in your own code,** in the tools and in the step limit, whichever framework runs the loop.
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
def text_tool():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A tool made from a function

        `tool_from(name, doc, params)` makes a tool definition from a function, as a framework does: `name` is the function's name, `doc` its docstring, and `params` a dictionary of each parameter with its type in a JSON schema, such as `{"amount": "number", "from_unit": "string"}`. In the definition, each parameter is a dictionary of its own, with its type under `"type"`. Frameworks differ in how much of the docstring they take; here the description is its first line, without the spaces and new lines around the docstring. The check tries `convert` from the picture, a tool with one parameter, a docstring with spaces around it and no parameters, and a tool with a number of whole pieces, of type `"integer"`.
        """),
        L.task("Replace the two `...`: the line that puts each parameter in `properties`, and the description."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tool():
    box_tool = L.code_box("tool-from", lines=9)
    box_tool
    return (box_tool,)


@app.cell(hide_code=True)
def check_tool(box_tool):
    result_tool = L.check("tool-from", box_tool.value)
    result_tool
    return (result_tool,)


@app.cell(hide_code=True)
def hints_tool():
    L.hints(
        "tool-from",
        [
            "Look at how `convert` was written by hand in Tools as JSON: each parameter under `\"properties\"` is a small dictionary, and the description is one sentence.",
            "For a parameter, that dictionary is `{\"type\": kind}`, stored under its name in `properties`. For the description, take the spaces off with `.strip()`, cut the text at each `\"\\n\"` with `.split`, and keep the first piece.",
            "`properties[param] = {\"type\": kind}` in the loop, and `\"description\": doc.strip().split(\"\\n\")[0],` in the definition.",
        ],
        why="""```python
def tool_from(name, doc, params):
    properties = {}
    for param, kind in params.items():
        properties[param] = {"type": kind}
    return {"type": "function", "function": {
        "name": name,
        "description": doc.strip().split("\\n")[0],
        "parameters": {"type": "object", "properties": properties, "required": list(params)},
    }}
```

A framework reads `params` from the types you write after each parameter in the `def` line, and turns `float` into `"number"` and `str` into `"string"`; the rest is the same.""",
    )
    return


@app.cell(hide_code=True)
def text_graph():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The agent loop as a graph

        `run_graph(nodes, edges, state, start, limit)` runs a graph of steps. `nodes` is a dictionary of each node's name with a function that gets the state and gives back a new state. `edges` is a dictionary of each node's name with a function that gets the new state and gives back the name of the next node, or `"END"`. Start at `start`. While the node is not `"END"` and fewer than `limit` steps have run: run the node, ask its edge for the next node, and count the step. Give back the state. The check tries the agent loop with a model that calls `convert` once, one that answers at once, one that calls a tool every time, and a small graph of numbers that adds 1 and doubles, with a limit of 10 it does not reach and with a limit of 3.
        """),
        L.task("Replace `...` with a `while` loop of three lines, one for each step above."),
    ])
    return


@app.cell(hide_code=True)
def exercise_graph():
    box_graph = L.code_box("graph", lines=8)
    box_graph
    return (box_graph,)


@app.cell(hide_code=True)
def check_graph(box_graph):
    result_graph = L.check("graph", box_graph.value)
    result_graph
    return (result_graph,)


@app.cell(hide_code=True)
def hints_graph():
    L.hints(
        "graph",
        [
            "Two things stop the loop: the node `\"END\"`, and the limit. Put both in the condition of the `while`, joined with `and`.",
            "In each round, the node's function gives the new state: `nodes[node](state)`. Then the edge of that node gives the name of the next one: `edges[node](state)`, with the new state.",
            "`while node != \"END\" and steps < limit:`, then under it `state = nodes[node](state)`, `node = edges[node](state)` and `steps = steps + 1`.",
        ],
        why="""```python
def run_graph(nodes, edges, state, start, limit):
    node = start
    steps = 0
    while node != "END" and steps < limit:
        state = nodes[node](state)
        node = edges[node](state)
        steps = steps + 1
    return state
```

With the nodes `model` and `tools`, these lines are the agent loop of the picture; with other nodes, they run any workflow of steps.""",
    )
    return


@app.cell(hide_code=True)
def text_look():
    mo.md(r"""
    ### Exercise 3. Where to look first
    """)
    return


@app.cell(hide_code=True)
def choice_look_cell():
    choice_look = L.choice(
        "look",
        question="With a framework, your agent calls its tool less often than with your own loop, with the same model and the same function. Where do you look first?",
        options={
            "temperature": "At the temperature, since a framework sends a higher one so that the answers vary more",
            "requests": "At the requests the framework sends: the system prompt it adds, and the tool definition it made",
            "tool": "At the code of the tool, since a framework runs a tool differently from your own loop, and can break it",
        },
        correct="requests",
        why={
            "requests": "The model sees only the requests; what the framework adds to them, or writes differently, is what changed.",
            "temperature": "The temperature can differ, but first look at what the framework adds to each request, such as a system prompt.",
            "tool": "Whether the tool runs well matters only once it is called; the model decides that, from what is in the request.",
        },
    )
    choice_look
    return (choice_look,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Ask a question about kilometers, miles, kilograms or pounds, and your `run_graph` runs the agent loop of the picture on Qwen3-0.6B, with `convert` made by your `tool_from`, at most {k} steps. The table shows each node your graph went through, as a framework's trace shows it. The course starts its own server on your computer, with the model behind it.
    """.format(k=C.DEMO_LIMIT))
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
def question_cell():
    question_box = mo.ui.text(value=C.QUESTIONS[4][0], label="Question", full_width=True)
    run_button = mo.ui.run_button(label="Run the graph")
    mo.vstack([question_box, run_button])
    return question_box, run_button


@app.cell(hide_code=True)
def demo_run(box_graph, box_tool, question_box, run_button):
    _graph = C.learner_fn("graph", box_graph.value)
    _tool = C.learner_fn("tool-from", box_tool.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _graph is None or _tool is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo runs your own graph, with the tool your own function makes.*")
    elif not run_button.value:
        _out = mo.md("*Press the button to run the graph.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your graph has started; the server loads Qwen3 …*"))
        _words = {"title": "Your run_graph on the agent loop, node by node", "lang": "en",
                  "head": ["step", "node", "prompt tokens", "what it gave"],
                  "stopped": "It stopped with {e}.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.run_demo(_graph, _tool, question_box.value.strip(), _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Each part of a framework is one you built in this module: the loop of [The agent loop](/course/open/build-agent-loop?lang=en), the tools of [Tools through a protocol: MCP](/course/open/build-mcp?lang=en), and the notes of [What an agent remembers](/course/open/build-agent-memory?lang=en). With that, the agents module is complete. The next module is about what comes after building: finding out where an LLM application goes wrong, measuring it, and keeping it running.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_look, result_graph, result_tool):
    L.wrap_up(result_tool, result_graph, choice_look)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
