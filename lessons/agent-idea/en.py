# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson agent-idea. Route: course/route/lessen/bouwen.yaml. Checks, the stand-ins and the course's runs: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("agent-idea", "en")
    C = L.checks   # the pantry, its two tools, the stand-ins for the model, and the course's runs of Qwen3
    R = C.course()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Learning from rewards](/course/open/rl-what-is-rl?lang=en), [Prompts that check themselves](/course/open/start-prompt-chains?lang=en) and [Data as text: JSON](/course/open/py-json?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rl():
    quick_rl = L.choice(
        "quick-rl",
        question="A robot vacuum cleaner learns with reinforcement learning. After each move, what does the environment give back to the agent?",
        options={
            "policy": "Its policy: which move to make next",
            "both": "The new situation, and a reward",
            "return": "The return of the whole run",
        },
        correct="both",
        why={
            "policy": "The policy belongs to the agent: it is how the agent chooses. See the new words in [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
            "both": "After every action the environment gives back where the agent is now, and a number for how good that step was.",
            "return": "The return adds up the rewards of many steps; after one move there is one reward. See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
        },
    )
    quick_rl
    return (quick_rl,)


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="You let a model write a summary of a long report in three prompts: first the main points, then a draft from those points, then a shorter version of the draft. What is that called?",
        options={
            "few-shot": "Few-shot prompting",
            "chaining": "Prompt chaining",
            "refine": "Self-Refine",
        },
        correct="chaining",
        why={
            "few-shot": "Few-shot prompting puts examples in one prompt, as in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en). Here the answer of each prompt goes into the next. See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
            "chaining": "Small prompts in a row, where each answer goes into the next prompt.",
            "refine": "With Self-Refine the model judges its own answer and improves it. Here each prompt does a different step. See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question="What does this give?\n\n```python\nimport json\njson.loads('{\"item\": \"milk\", \"grams\": 200}')[\"grams\"]\n```",
        options={
            "number": "`200`, a number",
            "text": "`\"200\"`, as text",
            "error": "An error: `\"grams\"` is not a key",
        },
        correct="number",
        why={
            "text": "In the JSON text, 200 has no quotes, so it is a number, and `json.loads` gives the number. See [Data as text: JSON](/course/open/py-json?lang=en).",
            "number": "`json.loads` turns the text into a dict, and the value at its key `\"grams\"` is the number 200.",
            "error": "`json.loads` gives a dict with the keys `\"item\"` and `\"grams\"`. See [Data as text: JSON](/course/open/py-json?lang=en).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Ask a chat model what you have to buy to make pancakes, and it cannot know: it cannot look in your cupboard. A language model only writes text. Give the same model two functions it may use, one that looks up a recipe and one that says how much of an ingredient is in your pantry, and a small program that runs them for it, and it can find out. That program with the model is an **agent**. The coding assistants that read your files and run your tests, and the assistants that search the web before they answer, are this same loop around a language model. In this lesson you write that loop yourself, in a dozen lines.
    """)
    return


@app.cell(hide_code=True)
def predict_no_tools():
    guess_no_tools = L.guess(
        "no-tools",
        question=f"Qwen3-0.6B, the course's chat model, gets the question *{R['task']}*, and no tools. The course asked it {len(R['without_tools'])} times, each with another seed. What did it do?",
        options={
            "asks": "Every time, it said that it cannot see your pantry, and asked what you have",
            "both": "Some times it asked what you have, other times it made up an answer",
            "invents": "Every time, it made up an answer",
        },
        reveal=C.no_tools_reveal(L.lang),
    )
    guess_no_tools
    return (guess_no_tools,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    The model never runs anything itself: it can only write text. So your program tells it, in the prompt, which tools there are: the name of each, what it does, and its arguments. To use one, the model writes a **tool call** in a fixed form, as JSON: the name of the tool and the arguments, such as `{"name": "pantry", "arguments": {"item": "milk"}}`. Your program reads that, runs the function, and puts what it gave back into the conversation, as a message of its own. Then it asks the model again. The model reads the result, and writes the next tool call, or its answer. Round and round: that is the **agent loop**. Because a model can keep calling tools, the loop asks it at most a fixed number of times, its **step limit**.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **LLM agent**, or agent | a language model that works towards a goal in a loop: it chooses a tool, sees the result, and chooses again, until it is done | `run_agent(ask, tools, task, max_steps)` |
    | **tool** | a function your program lets the model use, with a name, a description and arguments; it gives back text | `tools["pantry"]({"item": "milk"})` |
    | **tool call** | what the model writes to use a tool: the tool's name and its arguments, as JSON | `{"name": "pantry", "arguments": {"item": "milk"}}` |
    | **agent loop** | ask the model, run the tools it calls, add their results to the messages, and ask again, until it answers without a tool call | `for step in range(max_steps):` |
    | **step limit** | the most times the loop asks the model, also when it keeps calling tools | `max_steps` |
    | **autonomy spectrum** | from a chatbot that only answers, to an agent that does a whole task on its own: how much the model decides without you | |
    """)
    return


@app.cell(hide_code=True)
def watch_out_words():
    mo.vstack([
        L.watch_out("*Agent* means something else here than in [Learning from rewards](/course/open/rl-what-is-rl?lang=en). There, an agent learns a policy from rewards, by trying. Here, an LLM agent is a language model with tools: nobody rewards it while it works, and what it does next comes from the model, which read the messages. The idea they share: something that acts, sees what happened, and acts again, towards a goal. In this lesson, *agent* is the LLM agent, except where it says *RL agent*."),
        L.watch_out("*Tool* means something else here than in [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en). There, an AI tool was a whole program, such as a chat assistant. Here, a tool is one function that the model may ask your program to run."),
    ])
    return


@app.cell(hide_code=True)
def picture_loop():
    mo.vstack([
        draw.flow(
            [("The language model", ["reads the messages", "writes a tool call", "or the answer"]),
             ("Your program", ["the agent loop:", "keeps the messages", "runs the tools"]),
             ("The tools", ["recipe(dish)", "pantry(item)"])],
            ["a tool call,\nor the answer", "the call,\nwith its arguments"], ["all messages\nso far", "the result,\nas text"],
            title="An agent: a model, tools and a loop"),
        mo.md(r"""
        The model and the tools never meet: your program sits between them. It sends the model all messages so far, every time, because the model remembers nothing between two questions. When the model writes a tool call, your program runs the tool and adds the result to the messages. When it writes an answer without a tool call, the loop stops.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_prompt():
    mo.vstack([
        mo.md(r"""
        **What the model really reads.** The tools go into the prompt as text, in the first message, with the role `system`: the message with the instructions of whoever runs the model. For the pantry, this is the text Qwen3 reads before its first answer, the way the course sends it. `<|im_start|>` and `<|im_end|>` mark where each message starts and ends, with its role right after the start:
        """),
        mo.md("```text\n" + C.prompt_example() + "\n```"),
        mo.md(r"""
        Everything the model knows about the tools is in that text: their names, what they do, their arguments, and the form of a tool call, between `<tool_call>` tags. Qwen3 learned from many conversations in exactly this form, so it writes its tool calls in it too. The empty `<think>` and `</think>` at the very end tell Qwen3 to answer without first writing out its reasoning, which keeps it fast on a laptop. The course's `ask` finds the tool calls in the answer, reads their JSON with `json.loads`, and gives your loop each call as a dict, with its `"name"` and its `"arguments"`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_spectrum():
    mo.md(r"""
    **From chatbot to agent.** How much does the model decide by itself? That is the autonomy spectrum:

    | | Who chooses the next step | Example |
    |---|---|---|
    | a chatbot | you: it only answers | a chat assistant without tools |
    | an assistant with tools | the model proposes a step, you say yes or no | a coding assistant that asks before it runs a command |
    | an agent | the model, within its step limit and its tools | a coding assistant that may edit files and run tests on its own until they pass |

    The further along, the more an agent gets done without you, and the more can go wrong before anyone sees it. A tool that only reads, such as `pantry`, can do little harm; a tool that sends an email or deletes a file can. That is why agents get only the tools they need, and ask before a step that cannot be undone: a lesson of the domain *Building with AI* is about that.

    **The RL agent and the LLM agent.** Both are the same loop: look at the situation, act, see what happened, act again. They differ in where the choice comes from.

    | | RL agent | LLM agent |
    |---|---|---|
    | sees | the situation in the environment | all messages so far: the task and every tool result |
    | does | an action | a tool call, or the answer |
    | chooses with | a policy it learned from rewards, by trying | a language model, trained on text, that reads the messages |
    | stops | when the task ends, such as a game that is won or lost | when it answers without a tool call, or at the step limit |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To let a language model work on a task with tools:

    1. Give each tool one argument, a dict with its arguments, and let it give back text.
    2. Start the messages with the task: `[{"role": "user", "content": task}]`.
    3. Ask the model for its next message, with all messages so far, and add its message to them.
    4. If it calls no tool, it has answered: you are done.
    5. Otherwise, run each tool it calls, with its arguments, and add each result as a message with the role `"tool"`.
    6. Go back to step 3, but at most `max_steps` times.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The model here is a stand-in, `ask`: a function that reads the messages and answers the way a model that uses its tools well would, always the same, so you can see each step. It returns a message as a dict: `"content"` is its text, and `"tool_calls"` is a list of the tools it calls, each with its `"name"` and its `"arguments"`. The tools are in the dict `tools`. The cell does steps 2 to 5 of the recipe by hand, round after round, until the model answers. Run it.
    """)
    return


@app.cell
def example_kitchen():
    ask, tools = C.kitchen()   # the kitchen: the stand-in for the model, and the two tools

    # 2. the task is the first message
    _messages = [{"role": "user", "content": "What do I have to buy to make pancakes?"}]
    # 3. ask the model, and add its message
    _reply = ask(_messages)
    _messages.append(_reply)
    print(_reply["tool_calls"])
    # 5. run each tool it calls, and add each result as a message
    for _call in _reply["tool_calls"]:
        _result = tools[_call["name"]](_call["arguments"])
        _messages.append({"role": "tool", "content": _result})
        print(_result)
    # 3. and 5. again: the model now sees the recipe
    _reply = ask(_messages)
    _messages.append(_reply)
    print(_reply["tool_calls"])
    for _call in _reply["tool_calls"]:
        _messages.append({"role": "tool", "content": tools[_call["name"]](_call["arguments"])})
    # 3. and 4. a third time: it calls no tool, so this is its answer
    _reply = ask(_messages)
    print(_reply["tool_calls"], _reply["content"])
    return ask, tools


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    Under the cell appear four lines. The model's first message calls one tool: `recipe`, with `{{"dish": "pancakes"}}`. The tool gives back the recipe as JSON text: `{{"flour": 250, "milk": 450, "eggs": 110, "butter": 25}}`, in grams. With that recipe in the messages, the model's next message calls `pantry` four times at once, once for each ingredient. The cell runs those four and adds their results, and asks a third time. Now the model calls no tool, `[]`, and its message is the answer: *Buy 250 grams of milk and 110 grams of eggs.* That is step 4: the task is done. Steps 3 and 5 came back, with other tool calls each time: that is what a loop is for.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The agent loop

        Write `run_agent(ask, tools, task, max_steps)`: it starts the messages with the task, asks `ask` for the model's next message at most `max_steps` times, runs the tools that message calls, and gives back the whole list of messages. Three lines are left for you, at the `...`. The check tries your loop with the stand-in of the worked example, and with three other stand-ins: one that only chats, one that calls two tools at once, and one that never stops calling tools.
        """),
        L.task("Make `run_agent(ask, tools, \"What do I have to buy to make pancakes?\", 5)` give the 9 messages of the pancakes: the task, then the model, a tool, the model, four tools, and the model's answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loop():
    box_loop = L.code_box("run-agent", lines=11)
    box_loop
    return (box_loop,)


@app.cell(hide_code=True)
def check_loop(box_loop):
    result_loop = L.check("run-agent", box_loop.value)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "run-agent",
        [
            "The worked example did the steps by hand, one round after the other; the loop does steps 3 to 5 again and again. Which line of the example asks the model, which runs a tool, and when did it stop?",
            "The model needs all messages so far, so it gets the list `messages`. When its message calls no tool, `reply[\"tool_calls\"]` is an empty list, and the task is done. A tool gets `call[\"arguments\"]`.",
            "`reply = ask(messages)`, then under the `if`: `return ___`, and in the inner loop `result = tools[___](call[\"arguments\"])`.",
        ],
        why="`ask(messages)` gives the model every message so far, its own earlier tool calls and every result included. An answer without a tool call ends the task, so `return messages` gives back the list right there. Otherwise the inner loop runs each tool by its name with its arguments. And `range(max_steps)` asks the model at most `max_steps` times: the stand-in that keeps asking the time is stopped after 3 questions.",
    )
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(rf"""
    ### Your loop with a real model

    Your `run_agent` works with any `ask` that reads messages and gives back a message. Here it gets Qwen3-0.6B, with the two tools of the pantry and the instructions of the course in a message with the role `system` at the start: *{R['system']}* The course ran this {len(R['runs'])} times, with the course's `run_agent`, each with another seed; below you go through those runs, step by step. {C.runs_summary(L.lang)}
    """)
    return


@app.cell(hide_code=True)
def course_runs_cell():
    L.explore("course-runs", "run", lambda _v: C.run_frame(R["runs"][int(_v) - 1], int(_v), len(R["runs"]), L.lang), start=C.SHOWN_RUN, low=1,
              high=len(R["runs"]), step=1, role="", task="Go through the runs. Which tool does the model never call, and what does its answer say about the pantry anyway?")
    return


@app.cell(hide_code=True)
def text_runs_after():
    mo.md(C.runs_after(L.lang))
    return


@app.cell(hide_code=True)
def text_download():
    mo.md(rf"""
    **Run it yourself.** With the button, your own `run_agent` runs with Qwen3 on the same task, with the seed of run {C.SHOWN_RUN}: on the course's computer that gives run {C.SHOWN_RUN} again, and on yours, rounding can change a token, and from there the run can go another way. On the course's computer, with one thread, it took {C.number(C.seconds_of(R['runs'][C.SHOWN_RUN - 1]), 0, L.lang)} seconds; with more threads it is faster. While Qwen3 writes, you see its message so far.
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
def run_button_cell():
    run_button = mo.ui.run_button(label="Run Qwen3 with your run_agent")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_loop, result_loop, run_button):
    if result_loop.state != "correct":
        _out = mo.md("*Make `run_agent` right in exercise 1 first: the demo uses it.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not run_button.value:
        _out = mo.md("*Press the button to run Qwen3.*")
    else:
        _words = {"title": "Your run_agent with Qwen3-0.6B", "writing": "Qwen3 is writing …", "loading": "Loading Qwen3 …",
                  "alt": "The messages of your run so far, one row each, and what Qwen3 is writing now."}
        _r = C.run_live(run_code(box_loop.value).names["run_agent"], lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang)
        if isinstance(_r, str):
            _out = mo.md({
                "agent": "*Your `run_agent` stops with an error on a message of Qwen3. Does it still pass its check?*",
                "load": "*Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.*",
                "memory": "*The memory of your device is full. Close other programs, and press the button again.*",
            }[_r])
        else:
            _out = mo.Html(C.live_picture(_r, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_who_runs():
    mo.md(r"""
    ### Exercise 2. Who runs the tool?
    """)
    return


@app.cell(hide_code=True)
def choice_who_runs_cell():
    choice_who_runs = L.choice(
        "who-runs",
        question="Qwen3 writes `<tool_call>{\"name\": \"pantry\", \"arguments\": {\"item\": \"milk\"}}</tool_call>`. Who looks in the pantry?",
        options={
            "model": "The model: writing the call runs the function",
            "provider": "Hugging Face, where the model came from",
            "program": "Your program: `run_agent` calls the function `pantry`",
        },
        correct="program",
        why={
            "model": "The model only writes text, also when that text is a tool call. Something has to read it and run the function. Look at the picture of the loop again.",
            "provider": "Hugging Face only gave you the file with the model. It runs nowhere else: on your computer, it can only write text. Look at the picture of the loop again.",
            "program": "The model only writes which tool and which arguments. Your loop reads that, runs `pantry` with `{\"item\": \"milk\"}`, and puts the result in the messages.",
        },
    )
    choice_who_runs
    return (choice_who_runs,)


@app.cell(hide_code=True)
def text_rl_agent():
    mo.md(r"""
    ### Exercise 3. Two kinds of agent
    """)
    return


@app.cell(hide_code=True)
def choice_rl_agent_cell():
    choice_rl_agent = L.choice(
        "rl-agent",
        question="What differs between the RL agent of [Learning from rewards](/course/open/rl-what-is-rl?lang=en) and your LLM agent?",
        options={
            "loop": "Only the RL agent works in a loop of acting and seeing what happened",
            "choice": "Where its choices come from: a policy learned from rewards, or a language model that reads the messages",
            "goal": "Only the LLM agent has a goal",
        },
        correct="choice",
        why={
            "loop": "Your `run_agent` is a loop of acting and seeing too: a tool call, then its result. Look at the table of the two agents again.",
            "choice": "The RL agent learned its policy by trying and getting rewards. Your agent learned nothing while it worked: its choices come from Qwen3, which reads the task and every result so far.",
            "goal": "The RL agent has a goal too: as much return as it can get. Look at the table of the two agents again.",
        },
    )
    choice_rl_agent
    return (choice_rl_agent,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from rewards](/course/open/rl-what-is-rl?lang=en) an agent acted in an environment and saw what happened; your LLM agent does the same, with a language model choosing instead of a learned policy. In [Prompts that check themselves](/course/open/start-prompt-chains?lang=en) you put prompts in a row yourself; in an agent, the model chooses the next step. And in [Data as text: JSON](/course/open/py-json?lang=en) you read JSON; every tool call is JSON, which the course's `ask` reads for your loop. The lessons of the domain *Building with AI* go further: how to describe a tool so a model uses it well, what a tool may do, and when a fixed series of steps beats an agent. In the next lesson, *Graphs: nodes, edges and paths*, a maze turns out to be one example of a graph, and the same search runs on any network.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rl_agent, choice_who_runs, result_loop):
    L.wrap_up(result_loop, choice_who_runs, choice_rl_agent)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
