# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-mcp. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-mcp", "en")
    C = L.checks   # the course's MCP server with convert, the runs of Qwen3 with six servers, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Tools an agent can use well](/course/open/build-tool-design?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_runs():
    quick_runs = L.choice(
        "quick-runs",
        question="Your program sends Qwen3 the tool `convert`, and the model calls it. Where does `convert` run?",
        options={
            "provider": "At the provider of the model, which runs each tool it gets in a request",
            "program": "In your program: the model only writes the call, and your code runs it",
            "model": "In the model itself, which works out the result while it writes its answer",
        },
        correct="program",
        why={
            "program": "That is a client-side tool: the model writes a name and arguments, and your code does the work and sends the result back.",
            "provider": "Only a hosted tool runs at the provider. `convert` is a client-side tool. See [Tools an agent can use well](/course/open/build-tool-design?lang=en).",
            "model": "The model only writes the call; the result comes from code. See [Tools an agent can use well](/course/open/build-tool-design?lang=en).",
        },
    )
    quick_runs
    return (quick_runs,)


@app.cell(hide_code=True)
def quick_check_description():
    quick_description = L.choice(
        "quick-description",
        question="Where does a model learn which values an argument of a tool takes, such as the units of `convert`?",
        options={
            "code": "From the tool's code, which the API sends along with the tool",
            "first": "From the first error the tool gives, after a call with a value it cannot use",
            "description": "From the tool's description: an enum and a description of each argument",
        },
        correct="description",
        why={
            "description": "A model knows a tool only from its description, so that is where the units belong.",
            "code": "The code of a tool never reaches the model. See [Tools an agent can use well](/course/open/build-tool-design?lang=en).",
            "first": "A good error helps, but only after a wrong call: the description comes first. See [Tools an agent can use well](/course/open/build-tool-design?lang=en).",
        },
    )
    quick_description
    return (quick_description,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en), the tool `convert` lived inside your program: its description in the request, its code next to the loop that ran it. To give the same tool to another assistant, such as a coding agent or a chat app, you would write it again for each of them. The Model Context Protocol, MCP for short, is one fixed way for an assistant to find the tools of a program and call them. You write the tool once, as an MCP server, and every assistant that speaks MCP can use it. Anthropic published MCP in November 2024; Claude, ChatGPT, Cursor and VS Code are among the assistants that speak it now.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **MCP** | the Model Context Protocol: fixed messages with which an assistant asks a program for its tools and calls them |
    | **MCP server** | a program that offers tools over MCP, such as `convert` |
    | **MCP client** | the part of an assistant that talks to a server: it gets the tools, gives them to the model, and sends the model's calls on |
    | **JSON-RPC** | a way to send requests and answers as JSON: a request has a `method`, `params` and an `id`; its answer has the same `id`, and a `result` or an `error` |
    | **notification** | a JSON-RPC message without an `id`, which gets no answer |
    | **stdio** | a program's standard input and output: the client starts the server as a program, writes each message as a line to its input, and reads each answer as a line from its output |
    | **`_meta`** | in MCP version 2026-07-28, the part of each request's params that says which version of MCP the client speaks, and what it can do |

    An MCP server answers a few methods. `tools/list` gives each tool: its `name`, its `description` and its `inputSchema`, the JSON Schema of its arguments. `tools/call` runs a tool, and its result holds `content`: a list of parts, each with a type, such as `{"type": "text", "text": "22.99 mi"}`, and `isError` when the tool ran but failed. `server/discover` says which versions of MCP the server speaks. The version this lesson uses, 2026-07-28, is the newest: each request carries everything the server needs in its `_meta`, so the server keeps nothing between requests.
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

    The course's server lists `convert` with the same name, description and arguments as in [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en): the client turns `inputSchema` into `parameters`, and the tool the model gets is {same}. So MCP changes how a tool reaches the model, not what the model reads. What does change is what a client gives the model once it has several servers. The course asked Qwen3-0.6B the eight questions of [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en) again, at temperature 0, so that the same request always gets the same answer, now with the tools of {s} servers: a kitchen, a calendar, the units server, notes, a shop and the weather, {t} tools in all.
    """.format(same="exactly the same, character for character" if F["same_tool"] else "not the same", s=F["servers"], t=F["tools"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("An MCP server is a program you run, and its tools act with your rights: a server that can read your files can send them on. Every tool description also goes into the model's context, and a description can carry instructions that the model follows, a prompt injection. Add servers you trust, and only the tools you need."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a tool as an MCP server

    1. **List each tool** with its name, its description and its `inputSchema`, as you would describe it for a model.
    2. **Answer `tools/list`** with the tools, and say how long a client may keep the list: `ttlMs` and `cacheScope`.
    3. **Answer `tools/call`** by running the tool, and give back its text as `content`; set `isError` when the tool ran and failed.
    4. **Answer what you cannot follow with an error:** a tool you do not have, a version of MCP you do not speak, a method you do not know.
    5. **In the client,** turn each listed tool into the chat API's form, send each call of the model as `tools/call`, and give the text back to the model as the tool's result.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture("en"))
    return


@app.cell(hide_code=True)
def text_server():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The server's answers

        `handle(message)` is the heart of the server: it gets one JSON-RPC message, as a dictionary, and gives back the answer, or `None` for a notification. The checks of the version and of `clientCapabilities` are there, and so are `server/discover` and the helpers `result` and `error`. Fill in the two methods. For `tools/list`, give back `TOOLS`, with `"ttlMs": 300000` (five minutes) and `"cacheScope": "public"`. For `tools/call`, the params hold the tool's `name` and its `arguments`, which a tool without arguments may leave out; `FUNCTIONS` maps each name to its function, which gives back a text. A name that is not in `FUNCTIONS` gets the error `-32602`; a text that starts with `Error` gets `"isError": True`. The box knows `TOOLS` and `FUNCTIONS`; around your code, the course adds the tool and the loop that reads and writes the lines. The check sends the list request of the picture and eight more messages.
        """),
        L.task("Replace each `...` with the line or lines that answer that method."),
    ])
    return


@app.cell(hide_code=True)
def exercise_server():
    box_server = L.code_box("server", lines=34)
    box_server
    return (box_server,)


@app.cell(hide_code=True)
def check_server(box_server):
    result_server = L.check("server", box_server.value)
    result_server
    return (result_server,)


@app.cell(hide_code=True)
def hints_server():
    L.hints(
        "server",
        [
            "Both methods end with `return result(message, {...})`, with what goes into the result between the braces. `result` adds `resultType` and the server's `_meta` itself.",
            "For `tools/call`: take `name = params.get(\"name\")`; when it is not in `FUNCTIONS`, `return error(message, -32602, \"Unknown tool: \" + str(name))`; else run `FUNCTIONS[name](params.get(\"arguments\") or {})`. `text.startswith(\"Error\")` is `True` when the text begins with `Error`.",
            "`return result(message, {\"tools\": TOOLS, \"ttlMs\": 300000, \"cacheScope\": \"public\"})`, and after running the tool: `content = [{\"type\": \"text\", \"text\": text}]` and `return result(message, {\"content\": content, \"isError\": text.startswith(\"Error\")})`.",
        ],
        why="""```python
    if message["method"] == "tools/list":
        return result(message, {"tools": TOOLS, "ttlMs": 300000, "cacheScope": "public"})
    if message["method"] == "tools/call":
        name = params.get("name")
        if name not in FUNCTIONS:
            return error(message, -32602, "Unknown tool: " + str(name))
        text = FUNCTIONS[name](params.get("arguments") or {})
        content = [{"type": "text", "text": text}]
        return result(message, {"content": content, "isError": text.startswith("Error")})
```

A tool that ran and failed is a result with `isError`: the model reads the text and can try again. A tool that is not there is an error of the request, which the client gets, not the model.""",
    )
    return


@app.cell(hide_code=True)
def text_call():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The client's call

        `call_tool(send, number, name, arguments)` is the client's half. Build the JSON-RPC request for `tools/call`: `"jsonrpc": "2.0"`, `"id": number`, `"method": "tools/call"`, and `params` with the tool's `name`, its `arguments` and `"_meta": META`. `send(request)` writes it to the server and gives back the answer. When the answer holds an `error`, give back `"Error: "` and the error's `message`, so that the model can read what went wrong; else join the `text` of every part of the result's `content` whose type is `"text"`, with a newline between them. The check calls a server with `convert`, and one whose tool gives back an image and two texts.
        """),
        L.task("Replace `request = ...` with the request, and the last `...` with the lines that turn the answer into a text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_call():
    box_call = L.code_box("call", lines=14)
    box_call
    return (box_call,)


@app.cell(hide_code=True)
def check_call(box_call):
    result_call = L.check("call", box_call.value)
    result_call
    return (result_call,)


@app.cell(hide_code=True)
def hints_call():
    L.hints(
        "call",
        [
            "The request is a dictionary with four keys; the tool's name and arguments go into `params`, next to `_meta`. The answer is a dictionary too: look whether it has the key `\"error\"`.",
            "`if \"error\" in answer:` then `return \"Error: \" + answer[\"error\"][\"message\"]`. The parts are in `answer[\"result\"][\"content\"]`; keep `part[\"text\"]` of each part whose `part[\"type\"]` is `\"text\"`.",
            "`request = {\"jsonrpc\": \"2.0\", \"id\": number, \"method\": \"tools/call\", \"params\": {\"name\": name, \"arguments\": arguments, \"_meta\": META}}`, and at the end `texts = [part[\"text\"] for part in answer[\"result\"][\"content\"] if part[\"type\"] == \"text\"]` and `return \"\\n\".join(texts)`.",
        ],
        why="""```python
def call_tool(send, number, name, arguments):
    request = {"jsonrpc": "2.0", "id": number, "method": "tools/call",
               "params": {"name": name, "arguments": arguments, "_meta": META}}
    answer = send(request)
    if "error" in answer:
        return "Error: " + answer["error"]["message"]
    texts = [part["text"] for part in answer["result"]["content"] if part["type"] == "text"]
    return "\\n".join(texts)
```

The model reads only text here; an image part would need a model that sees, as in [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en).""",
    )
    return


@app.cell(hide_code=True)
def text_trust():
    mo.md(r"""
    ### Exercise 3. A server from somewhere else
    """)
    return


@app.cell(hide_code=True)
def choice_trust_cell():
    choice_trust = L.choice(
        "trust-server",
        question="You find an MCP server online with a tool that reads your e-mail. You want your agent to use it. What do you check first?",
        options={
            "protocol": "Whether it speaks version 2026-07-28 of MCP, so that the protocol keeps it safe",
            "who": "Who made it, what each tool can do, and what its descriptions say to the model",
            "nothing": "Nothing yet: the agent will notice a harmful tool from its description",
        },
        correct="who",
        why={
            "who": "The server runs as a program with your rights, and its descriptions go into the model's context. Read them, and keep only the tools you need.",
            "protocol": "The protocol fixes the messages, not what a server does with your e-mail: a server that speaks it can still send it on.",
            "nothing": "A description can say anything, and a model tends to follow what it reads: a description with instructions hidden in it is a prompt injection.",
        },
    )
    choice_trust
    return (choice_trust,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercises 1 and 2 are right, your server runs as a program of its own: the course saves it as a file, `units_server.py` in your work folder, starts it, and talks to it over stdio with your `call_tool`. Qwen3-0.6B, on the course's server on your computer, gets the tools your server lists. Ask a question about units; under the answer, you can open the lines that went to and from your server.
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
def ask_box_cell():
    ask_box = mo.ui.text(value=C.QUESTIONS[0][0], label="Your question", full_width=True)
    ask_button = mo.ui.run_button(label="Ask")
    mo.vstack([ask_box, ask_button])
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_call, box_server):
    _server = C.learner_fn("server", box_server.value)
    _call = C.learner_fn("call", box_call.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _server is None or _call is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo runs your own server, and calls it with your own `call_tool`.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your server is starting, and the server of the course loads Qwen3 …*"))
        _words = {"title": "Qwen3 with the tool of your server", "no_call": "The model called no tool: it answered by itself.",
                  "client": "client → server", "server": "server → client", "lines": "The lines over stdio",
                  "saved": "Your server is saved as {path}.", "stopped": "Your server stopped: {e}", "failed": "The request to Qwen3 stopped: {e}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.ask_server(_server, _call, ask_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The tool is still the `convert` of [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en), with the description of [Tools an agent can use well](/course/open/build-tool-design?lang=en); MCP only fixes how a client finds it and calls it, in JSON-RPC messages over stdio. Your server is a program of its own, which any assistant that speaks MCP can start. The next lesson, [AGENTS.md, skills and other extensions](/course/open/build-agent-extensions?lang=en), gives an agent more than tools: notes about a project, and instructions it loads only when it needs them.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_trust, result_call, result_server):
    L.wrap_up(result_server, result_call, choice_trust)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
