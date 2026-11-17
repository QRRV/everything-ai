# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-agent-extensions. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-extensions", "en")
    C = L.checks   # the course's six skills, the runs of Qwen3 with and without them, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Tools through a protocol: MCP](/course/open/build-mcp?lang=en), and one about [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_list():
    quick_list = L.choice(
        "quick-list",
        question="An MCP client has just started a server. How does it learn which tools the server has?",
        options={
            "list": "It sends tools/list, and the server answers with each tool and its input schema",
            "start": "The server writes all its tools to its output by itself, as soon as the client starts it",
            "call": "It sends tools/call with an empty name, and the error lists every name",
        },
        correct="list",
        why={
            "list": "The client then turns each tool into the chat API's form, and gives the tools to the model.",
            "start": "A server answers requests; it sends nothing before it is asked. See [Tools through a protocol: MCP](/course/open/build-mcp?lang=en).",
            "call": "A call of a tool the server does not have is an error, and that error lists nothing. See [Tools through a protocol: MCP](/course/open/build-mcp?lang=en).",
        },
    )
    quick_list
    return (quick_list,)


@app.cell(hide_code=True)
def quick_check_agents():
    quick_agents = L.choice(
        "quick-agents",
        question="What belongs in AGENTS.md?",
        options={
            "spec": "The examples of what this one task must give, so that the agent can check its work",
            "plan": "The steps of this task, in the order in which the agent will take them",
            "always": "What holds for every task in the project, such as how to run the tests",
        },
        correct="always",
        why={
            "always": "An agent reads AGENTS.md at the start of every task, so it holds what every task needs.",
            "spec": "That is the spec of one task. See [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en).",
            "plan": "That is the plan of one task. See [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en).",
        },
    )
    quick_agents
    return (quick_agents,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An agent knows what is in its context and can do what its tools can do; the model itself stays the same. So you teach an agent something new by what you put in front of it. [Tools through a protocol: MCP](/course/open/build-mcp?lang=en) gave it tools from a program. This lesson adds three more ways: notes about a project that it reads at the start of every task, procedures that it reads only when a task needs them, and the description of a web API, from which it gets a tool for each thing the API can do. A plugin puts such extensions together, so that you can install them in one go.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **AGENTS.md** | a file at the top of a project with what holds for every task, which a coding agent reads before your request |
    | **skill** | a folder with a file SKILL.md: at the top a name and a description, under them the instructions; it may hold scripts and other files too |
    | **frontmatter** | the lines at the top of a file between two `---` lines, each a key, `: ` and a value |
    | **progressive disclosure** | showing the short version of something always, and the whole only when it is needed |
    | **OpenAPI specification** | a description of a web API as JSON or YAML: for each path and method an operation, with its `operationId`, a `summary` and its parameters with their schemas |
    | **plugin** | a package of extensions for an agent, such as skills, MCP servers and commands, which you install together |

    AGENTS.md costs room in every request, so it should hold only what every task needs. A procedure that matters only now and then belongs in a skill. The agent always sees each skill as one line: its name and its description, which says what the skill is for and when to use it. The instructions stay in the file until a question matches; then the agent loads them, here with a tool `load_skill`. That is progressive disclosure: with a hundred skills, the agent reads a hundred short lines, and the instructions of one. Anthropic introduced skills in October 2025 and published the format as an open standard; other coding agents read it too.
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

    The course wrote {k} skills for a kitchen assistant, such as cups and spoons in grams, how long leftovers keep, and the allergens in its dishes; each has numbers a model cannot know. Then it asked Qwen3-0.6B {n} questions, two per skill, at temperature 0, so that the same request always gets the same answer, in three ways: without the skills; with the list of skills in the system prompt and the tool `load_skill`; and with the instructions of all {k} skills in the system prompt. Then it ran the second way once more, with a more helpful error, as in [Tools an agent can use well](/course/open/build-tool-design?lang=en): when the model called a skill as if it were a tool, the error said to load it with `load_skill`.
    """.format(k=F["skills"], n=F["questions"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A skill is instructions the agent follows, and it may come with scripts that run on your computer: install skills and plugins only from people you trust, and read them first. A skill from somewhere else can carry instructions you did not ask for, which the agent then follows: a prompt injection."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: teach an agent something without changing the model

    1. **What holds for every task** in a project goes in AGENTS.md: how to run the tests, the rules, what not to touch.
    2. **A procedure that matters now and then** becomes a skill: a folder with a SKILL.md, whose description says what it is for and when to use it.
    3. **Give the agent the list of skills,** a line per skill, and a way to load one; it then reads the instructions of a skill only when a question needs them.
    4. **A web API with an OpenAPI description** becomes tools: an operation becomes a tool, its parameters the tool's parameters, and each call an HTTP request your code sends.
    5. **To share a set of extensions,** put them in a plugin, and install plugins only from people you trust.
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
def text_front():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The name and description of a skill

        `front(text)` reads the frontmatter of a SKILL.md: the lines between the first `---` line and the next one. The first lines are there: they split the text into lines and find the line that closes the frontmatter. For each line in between that has `: ` in it, put the part before the first `: ` in `found` as the key, and the part after it, without spaces at its ends, as the value. The value itself may hold `: ` too. (Agents read real frontmatter as YAML, where such a value stands in quotes; your reader keeps it simple.) The check tries the SKILL.md of cup-measures, which the picture shows, and two skills of its own.
        """),
        L.task("Replace `...` with the lines that put each key and its value in `found`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_front():
    box_front = L.code_box("front", lines=10)
    box_front
    return (box_front,)


@app.cell(hide_code=True)
def check_front(box_front):
    result_front = L.check("front", box_front.value)
    result_front
    return (result_front,)


@app.cell(hide_code=True)
def hints_front():
    L.hints(
        "front",
        [
            "Look at each line: when it holds `: `, split it into two parts, at the first `: ` only.",
            "`line.split(\": \")` splits at every `: `; with a second value, as in `split(\": \", n)`, it splits at most `n` times. `.strip()` takes the spaces off the ends of a text.",
            "Split each line that holds `: ` at the first `: ` only: `key, value = line.split(\": \", ___)`.",
        ],
        why="""```python
def front(text):
    lines = text.split("\\n")
    end = lines.index("---", 1)   # the line that closes the frontmatter
    found = {}
    for line in lines[1:end]:
        if ": " in line:
            key, value = line.split(": ", 1)
            found[key] = value.strip()
    return found
```

`lines.index("---", 1)` looks for `---` from the second line on, so it finds the line that closes the frontmatter, not the one that opens it.""",
    )
    return


@app.cell(hide_code=True)
def text_openapi():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Tools from the description of a web API

        `openapi_tools(spec)` makes a tool for each operation of an OpenAPI specification. The loops over the paths and their methods are there, and so is each tool's name and description. Fill in its parameters: for each parameter in the operation's `"parameters"`, copy its `"schema"`, give the copy the parameter's `"description"`, and put it in `properties` under the parameter's `"name"`. When the parameter has `"required": true`, add its name to `required`. An operation without parameters has no key `"parameters"`, and a parameter that is not required has no key `"required"`. The check reads the corner shop's API below, and the API of a weather service.
        """),
        mo.accordion({"The corner shop's API, as its OpenAPI specification gives it": mo.md(
            "```json\n" + C.SHOP_API_JSON + "\n```")}),
        L.task("Replace `...` with a loop over the operation's parameters that fills `properties` and `required`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_openapi():
    box_openapi = L.code_box("openapi", lines=20)
    box_openapi
    return (box_openapi,)


@app.cell(hide_code=True)
def check_openapi(box_openapi):
    result_openapi = L.check("openapi", box_openapi.value)
    result_openapi
    return (result_openapi,)


@app.cell(hide_code=True)
def hints_openapi():
    L.hints(
        "openapi",
        [
            "Loop over `operation.get(\"parameters\", [])`: `.get` with a second value gives an empty list when the key is not there.",
            "`schema = dict(p[\"schema\"])` makes a copy you can change; `schema[\"description\"] = p[\"description\"]` gives it the description. `p.get(\"required\")` is `None` when the key is not there.",
            "After the two lines with `schema`, put the copy in its place: `properties[___] = schema`.",
        ],
        why="""```python
            for p in operation.get("parameters", []):
                schema = dict(p["schema"])
                schema["description"] = p["description"]
                properties[p["name"]] = schema
                if p.get("required"):
                    required.append(p["name"])
```

A tool made this way only describes the operation. To run a call, your code sends the HTTP request: the method, the path with `{product}` filled in, and the other arguments as the query.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.md(r"""
    ### Exercise 3. Which extension?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-extension",
        question="Your team's release takes a list of twelve steps, two pages long, and happens once a month. Where do the steps go, so that your coding agent follows them?",
        options={
            "agents": "In AGENTS.md, which the agent reads in full at the start of every single task",
            "skill": "In a skill: the agent sees one line about it, and loads the steps when needed",
            "server": "In an MCP server, with a tool of its own for each of the twelve steps of the release",
        },
        correct="skill",
        why={
            "skill": "The steps matter once a month, so they cost one line in other tasks, and the agent loads them for a release.",
            "agents": "Then two pages go into every task, also the ones that have nothing to do with a release: AGENTS.md is for what every task needs.",
            "server": "A tool does one thing when it is called; steps the agent must follow in an order are instructions, which is what a skill holds.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Write a skill of your own: the box below holds one to start from. Once exercise 1 is right, your `front` reads its name and description, and the demo adds it to the list of the course's six skills, with the tool `load_skill`. Qwen3-0.6B, on the course's server on your computer, then answers your question. Does it load your skill?
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
def skill_box_cell():
    skill_box = mo.ui.text_area(value=C.DEMO_SKILL, label="Your SKILL.md", rows=14, full_width=True)
    ask_box = mo.ui.text(value=C.DEMO_QUESTION, label="Your question", full_width=True)
    ask_button = mo.ui.run_button(label="Ask")
    mo.vstack([skill_box, ask_box, ask_button])
    return ask_box, ask_button, skill_box


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_front, skill_box):
    _front = C.learner_fn("front", box_front.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _front is None:
        _out = mo.md("*Finish exercise 1 first: the demo reads your skill with your own `front`.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 …*"))
        _words = {"title": "Qwen3 with the course's skills and yours", "line": "The line the agent always sees of your skill:",
                  "no_load": "The model loaded no skill: it answered by itself.",
                  "unreadable": "Your front could not read this text as a SKILL.md: it should start with a line `---`, and have a second `---` line under the frontmatter.",
                  "no_name": "Your skill needs a name and a description in its frontmatter.",
                  "bad_name": "The name of a skill is lower case letters, digits and hyphens, such as bread-rising.",
                  "taken": "The course already has a skill with that name. Give yours another one.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.ask_with_skill(_front, skill_box.value, ask_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en) put what holds for every task in AGENTS.md; a skill holds what holds only now and then, and costs one line until it is needed. [Tools through a protocol: MCP](/course/open/build-mcp?lang=en) gave an agent tools from a program; the description of a web API gives it tools too, made by your `openapi_tools`. A plugin puts skills, servers and commands together. The next lesson looks at three kinds of agents built from all these parts: for code, for research and for your own tasks.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_front, result_openapi):
    L.wrap_up(result_front, result_openapi, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
