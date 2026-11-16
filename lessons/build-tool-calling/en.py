# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-tool-calling. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tool-calling", "en")
    C = L.checks   # the tool convert, the course's runs of Qwen3 with and without it, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [What an agent is: a model, tools and a loop](/course/open/agent-idea?lang=en), one about [JSON you can rely on](/course/open/build-structured-output?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="In the agent loop, the model's message has a tool call. What does your program do next?",
        options={
            "show": "It shows the call to the user as the model's answer, since the model wrote it, and the loop ends",
            "model": "Nothing yet: the model runs the tool on its own and writes the result into its next message",
            "run": "It runs the tool, adds the result as a message with the role `tool`, and asks the model again",
        },
        correct="run",
        why={
            "run": "The model only writes text; your program runs the tool, and the model reads the result in its next turn.",
            "show": "A tool call is not an answer for the user: it asks your program to do something. The loop ends only at a message without a tool call. See [What an agent is](/course/open/agent-idea?lang=en).",
            "model": "A model only writes text, a tool call too: it cannot run anything. See [What an agent is](/course/open/agent-idea?lang=en).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_schema():
    quick_schema = L.choice(
        "quick-schema",
        question="Which answer fits the schema `{\"type\": \"object\", \"properties\": {\"price\": {\"type\": \"number\"}}, \"required\": [\"price\"]}`?",
        options={
            "fits": "`{\"price\": 649}`",
            "text": "`{\"price\": \"649\"}`",
            "key": "`{\"cost\": 649}`",
        },
        correct="fits",
        why={
            "fits": "It has the required key `price`, and its value is a number.",
            "text": "`\"649\"` is a text, between quotes, and the schema wants a number. See [JSON you can rely on](/course/open/build-structured-output?lang=en).",
            "key": "The schema requires the key `price`; `cost` is another key. See [JSON you can rely on](/course/open/build-structured-output?lang=en).",
        },
    )
    quick_schema
    return (quick_schema,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A language model writes the most likely next token; it does not calculate. Ask it how many miles 37 km is, and it writes something that looks like a calculation. For an app, *looks like* is not enough. A function in your program converts exactly, every time. So let the model ask your program to run it: the model reads the question and picks the numbers, your code does the arithmetic. In [What an agent is](/course/open/agent-idea?lang=en), the course's `ask` packed the tools into the prompt for you, and unpacked the calls from the answer. Every chat API has its own way for that, called function calling. In this lesson you send the tools yourself, and read the calls as the API gives them.
    """)
    return


@app.cell(hide_code=True)
def predict_own():
    guess_own = L.guess(
        "own",
        question="The course asked Qwen3-0.6B {n} conversions, such as *{q}*, without tools. How many answers ended on the right number, to two decimals?".format(
            n=F["questions"], q=C.QUESTIONS[0][0]),
        options={
            "all": "All {n}: converting units is simple arithmetic".format(n=F["questions"]),
            "some": "Some of them, not all",
            "none": "None: a model that small cannot convert at all",
        },
        reveal="**{r} of the {n}** ended on the right number. The model knew the factors, such as 0.621371 miles in a kilometer, but the arithmetic went wrong: for 37 km it wrote 37 × 0.621371 ≈ 23.39, and 37 × 0.621371 is 22.99. It also wrote out each calculation: {t} tokens and {s} seconds per answer, on the course's computer.".format(
            r=F["own_right"], n=F["questions"], t=round(F["own_tokens"]), s=round(F["own_seconds"])),
    )
    guess_own
    return (guess_own,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    With function calling, the tools go into the request in a field of their own, `"tools"`: for each tool its name, a description, and a JSON schema of its arguments, as in [JSON you can rely on](/course/open/build-structured-output?lang=en). The server puts them into the prompt in the form its model learned. When the model calls a tool, the answer's message has no text but a list `"tool_calls"`, and `"finish_reason"` is `"tool_calls"`. Each call has an `"id"`, the name of the function, and its arguments as a JSON text. Your program runs the functions, adds each result as a message with the role `tool` and the `id` of its call, and asks again with the same tools. Now the model reads the results and answers.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **function calling** | the way a chat API lets a model call your functions: you describe the tools in the request, and the answer gives the tool calls apart from the text | `"tools": [convert_tool]` |
    | **tool choice** | a field of the request that says whether the model may call a tool (`"auto"`, the default), may not (`"none"`), must call one (`"required"`), or must call the one you name | `"tool_choice": "auto"` |

    Two more things you meet below. **`enum`**, in a JSON schema, lists the only values an argument may have: `"enum": ["km", "mi", "kg", "lb"]`. And **`tool_call_id`** is the field in which a result names its call, with the call's `"id"`, so that the model knows which result belongs to which call.

    ### The tool convert

    This is how the course describes its function `convert` for the API. The function itself is ordinary Python: it gets the arguments as a dictionary, and gives back the amount in the other unit as text, such as `"22.99 mi"`.

    ```python
    convert_tool = {
        "type": "function",
        "function": {
            "name": "convert",
            "description": ("Convert an amount from one unit to another: "
                            "kilometres and miles, or kilograms and pounds."),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "the amount to convert, such as 37"},
                    "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                                  "description": "the unit of the amount"},
                    "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                                "description": "the unit to convert to"},
                },
                "required": ["amount", "from_unit", "to_unit"],
            },
        },
    }
    ```

    The model never sees the code of `convert`: it only reads this description. Every request with the tool carries it, as tokens: the question *{q}* is {own} prompt tokens alone, and {tools} with the description of `convert`.
    """.replace("{q}", C.QUESTIONS[0][0]).replace("{own}", str(F["prompt_own"])).replace("{tools}", str(F["prompt_tools"])))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course sent the same {n} questions with the tool `convert`, in two ways: with `tool_choice` `"auto"`, so the model decides itself whether to call it, and with `tool_choice` naming `convert`, so it must. Each time, the course's program ran the calls and asked again, as in the recipe below.
    """.format(n=F["questions"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(r"""
        With `"auto"`, Qwen3 called `convert` for {called} of the {n} questions, each time with the right arguments, and each of those answers was right. The other {missed} it answered itself, without a call, and none of those was right to two decimals: once a near miss, once rounded to one decimal, and once *{wrong}* With `tool_choice` naming `convert`, it called the tool every time, with the right arguments, and all {n} answers were right. Two requests sound slower than one, but here they were quicker: about {tool_s} seconds for both together, against {own_s} for a written-out calculation.

        A question that needs no tool, such as *{no_need}*, got no call with `"auto"`: in {k} questions of that kind, Qwen3 called nothing. That is what `"auto"` is for: the model decides. With `"none"`, the course's server leaves the tools out of the prompt altogether: {none_p} prompt tokens instead of {auto_p}.
        """.format(called=len(F["auto_called"]), n=F["questions"], missed=len(F["auto_missed"]),
                   wrong=C.run("auto-{}".format(C.WRONG))["steps"][-1]["text"], tool_s=L.number(F["tool_seconds"], 1),
                   own_s=L.number(F["own_seconds"], 1), no_need=C.NO_NEED[0], k=len(C.NO_NEED),
                   none_p=F["none_prompt"], auto_p=F["auto_prompt_no_need"])),
        L.watch_out("A tool call is a wish of the model, not a promise. The model may skip a tool it should use, as above, call one it should not, or write arguments the function cannot use, such as a unit it does not know. Your code checks what comes in, and the function gives back an error as text when it cannot do the work. The next lesson is about tools and errors a model can use well."),
    ])
    return


@app.cell(hide_code=True)
def text_choice():
    mo.md(r"""
    ### tool_choice

    | Value | The model … |
    |---|---|
    | `"auto"` | may call a tool, or answer with text: it decides. The default, when the request has tools. |
    | `"none"` | answers with text, and calls nothing. |
    | `"required"` | must call a tool, one of the list. |
    | `{"type": "function", "function": {"name": "convert"}}` | must call `convert`. |

    A forced call holds for one request. Send `"required"` or a named tool again with the results, and the model must call again: it never gets to answer. So force the call in the first request only, as the course did, and send the second with `"auto"`, or without `tool_choice`.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: one round of function calling

    1. **Describe each tool**: `"type": "function"`, and under `"function"` its name, a description, and its arguments as a JSON schema under `"parameters"`.
    2. **Ask** with the question and `"tools": [...]`.
    3. **No `tool_calls` in the answer?** Then its `"content"` is the answer: done.
    4. **Otherwise**, add the answer's message to the messages as it came back. For each call, read its arguments with `json.loads`, run the function with its name, and add `{"role": "tool", "tool_call_id": call["id"], "content": result}`.
    5. **Ask again** with all messages and the same tools. The model reads the results and answers.

    An agent repeats steps 3 to 5 in a loop, as in [What an agent is](/course/open/agent-idea?lang=en). Here, one round is enough: one question, one tool.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The question *{q}*, as the course sent it with `"auto"`.

    1. **The tool**: `convert_tool`, above.
    2. **Ask**: the messages hold only the question; the request has `"tools": [convert_tool]`.
    3. **The answer** has no text, `"finish_reason"` is `"tool_calls"`, and `"tool_calls"` holds one call: its `"id"` is `"call_0"`, its name `"convert"`, and its arguments the text `'{args}'`.
    4. **Run it**: `json.loads` makes a dictionary of that text, and `convert` gives back `"{result}"`. The messages now hold the question, the answer with its call, and `{{"role": "tool", "tool_call_id": "call_0", "content": "{result}"}}`.
    5. **Ask again**: the model answers *{answer}*, with `"finish_reason"` `"stop"`.
    """.format(q=C.QUESTIONS[C.WORKED][0], args=C.recorded_messages(C.WORKED)[0][1]["tool_calls"][0]["function"]["arguments"],
               result=C.run("auto-{}".format(C.WORKED))["steps"][0]["results"][0],
               answer=C.run("auto-{}".format(C.WORKED))["steps"][-1]["text"]))
    return


@app.cell(hide_code=True)
def picture_talk():
    mo.Html(C.talk_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_describe():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Describe convert

        The box holds the description of `convert` with only its first argument, `amount`. Add the two units, `from_unit` and `to_unit`: each a string that can only be `km`, `mi`, `kg` or `lb`, with a short description. Then make all three arguments required. The picture under the check shows your arguments, and the line of the prompt in which Qwen3 reads them.
        """),
        L.task("Add `from_unit` and `to_unit` to `\"properties\"`, each with `\"type\": \"string\"` and an `\"enum\"` of the four units, and list all three in `\"required\"`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_describe():
    box_describe = L.code_box("describe", lines=17)
    box_describe
    return (box_describe,)


@app.cell(hide_code=True)
def check_describe(box_describe):
    result_describe = L.check("describe", box_describe.value)
    result_describe
    return (result_describe,)


@app.cell(hide_code=True)
def hints_describe():
    L.hints(
        "describe",
        [
            "`\"properties\"` is a dictionary with one key per argument. Each value is a small schema of its own, such as the one of `amount`.",
            "A unit is a string with a fixed list of values: `{\"type\": \"string\", \"enum\": [\"km\", \"mi\", \"kg\", \"lb\"], \"description\": ...}`.",
            "Add `\"from_unit\": {...}` and `\"to_unit\": {...}` under `\"amount\"`, and make the last line of `\"parameters\"` `\"required\": [\"amount\", \"from_unit\", \"to_unit\"]`.",
        ],
        why="""```python
"properties": {
    "amount": {"type": "number", "description": "the amount to convert, such as 37"},
    "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                  "description": "the unit of the amount"},
    "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                "description": "the unit to convert to"},
},
"required": ["amount", "from_unit", "to_unit"],
```

With `\"enum\"`, the model reads which four values there are, and writes `\"mi\"`, not `\"miles\"`: a value `convert` knows. Your descriptions may be other words; the check looks at the arguments, their types and the allowed values.""",
    )
    return


@app.cell(hide_code=True)
def text_answer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Run the calls, and send the results back

        `answer(url, question, tools, functions)` asks the model with the tools, through `ask`, which sends a request as in [JSON you can rely on](/course/open/build-structured-output?lang=en) and gives back the answer's message. `functions` holds the functions themselves by name, such as `{"convert": convert}`. When the message has tool calls, `answer` adds it to the messages and asks again, but the loop over the calls does nothing yet. Make it run each call and add its result. The check calls your `answer` against a stand-in of the server, with a question that gets one call, one that gets two at once, and one that gets none; the picture under the check shows what your function sent for the two calls.
        """),
        L.task("In the loop, read the call's arguments with `json.loads`, run `functions[name]` with them, and add a message with the role `tool`, the call's `id` as `tool_call_id`, and the result as `content`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("answer", lines=24)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "answer",
        [
            "A call is a dictionary: `call[\"id\"]`, and under `call[\"function\"]` the `\"name\"` and the `\"arguments\"`. The arguments are a JSON text, not yet a dictionary.",
            "`json.loads(call[\"function\"][\"arguments\"])` gives the dictionary. `functions[call[\"function\"][\"name\"]]` is the function to run with it.",
            "Three lines: `arguments = json.loads(...)`, then `result = functions[...](arguments)`, then `messages = messages + [{\"role\": \"tool\", \"tool_call_id\": call[\"id\"], \"content\": result}]`.",
        ],
        why="""```python
def answer(url, question, tools, functions):
    messages = [{"role": "user", "content": question}]
    message = ask(url, messages, tools)
    if message.get("tool_calls"):
        messages = messages + [message]
        for call in message["tool_calls"]:
            arguments = json.loads(call["function"]["arguments"])
            result = functions[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
        message = ask(url, messages, tools)
    return message["content"]
```

The results come right after the assistant's message, one per call, in the order of the calls. `tool_call_id` ties each to its call: with two calls at once, the model reads which number is the distance and which the date.""",
    )
    return


@app.cell(hide_code=True)
def text_tool_choice_choice():
    mo.md(r"""
    ### Exercise 3. Make it call the tool
    """)
    return


@app.cell(hide_code=True)
def choice_tool_choice_cell():
    choice_tool_choice = L.choice(
        "tool-choice",
        question="With `\"auto\"`, Qwen3 answered *{wrong}* without calling `convert`. How do you make it call `convert`, and still answer after the result?".format(
            wrong=C.run("auto-{}".format(C.WRONG))["steps"][-1]["text"]),
        options={
            "every": "Send `\"tool_choice\": \"required\"` in every request of the conversation, the first one and all after it",
            "first": "Name `convert` in `tool_choice` in the first request, and leave it out of the second",
            "none": "Send `\"tool_choice\": \"none\"`, so that it answers the question itself, with more care",
        },
        correct="first",
        why={
            "first": "The first request must call `convert`; the second, with the result, may answer. That is how the course got all {n} answers right.".format(n=F["questions"]),
            "every": "Then the second request must call a tool again, and the third too: the model never gets to answer. Which request needs the call?",
            "none": "`\"none\"` switches the tools off: then the model calculates itself, as without tools. Which value makes it call?",
        },
    )
    choice_tool_choice
    return (choice_tool_choice,)


@app.cell(hide_code=True)
def text_other_apis():
    mo.vstack([
        mo.md(r"""
        ### The same idea, in other APIs

        The form above is that of OpenAI's API, which the course's server copies, as do most servers of open models. Other APIs have the same parts under other names. In Anthropic's API, for example, in November 2026, a tool has `input_schema` instead of `parameters`; a call is a block of the type `tool_use` in the answer's content, with its arguments as a dictionary already, under `input`; and the result goes back as a block of the type `tool_result` with the call's `tool_use_id`, in a message with the role `user`. A description, a call with an id, and a result with that id: those three parts you find everywhere.
        """),
        L.watch_out("Every tool you send costs tokens in every request, also when the model calls none: one tool with three arguments took Qwen3's prompt from {own} to {tools} tokens. Send the tools a request needs, not every tool you have.".format(
            own=F["prompt_own"], tools=F["prompt_tools"])),
    ])
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 2 is right, you can try your `answer` on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo asks the question in the box with the tool `convert`, as your description of exercise 1 has it, once that is right, and otherwise as the course's. Ask a conversion, a question that needs no tool, or one with a unit `convert` does not know, such as feet.
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
    question_box = mo.ui.text(value=C.QUESTIONS[0][0], label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Ask with your answer")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_answer, box_describe, question_box, send_button):
    _answer = C.learner_answer(box_answer.value)
    _tool, _mine = C.learner_tool(box_describe.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _answer is None:
        _out = mo.md("*Finish exercise 2 first: the demo asks with your own `answer`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "What your answer sent to Qwen3, with {}".format("your description of convert" if _mine else "the course's description of convert"),
                  "request": "Request {n}: your answer sent these messages", "answer": "The server answered:",
                  "id": "id {id}", "answers": "answers {id}:", "finish": "finish_reason: {f}",
                  "status": "The server answered with status {s}: {m}", "gave": "Your answer gave back: {v}",
                  "error": "Your answer stopped with {e}.", "none": "Your answer sent nothing to the server.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_answer, _tool, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The agent loop of [What an agent is: a model, tools and a loop](/course/open/agent-idea?lang=en) now speaks the API's own form: tools in the request, calls with an id in the answer, results with that id. The JSON schema of [JSON you can rely on](/course/open/build-structured-output?lang=en) now describes a tool's arguments. In the next lesson, [Tools an agent can use well](/course/open/build-tool-design?lang=en), you measure what a name, a description and an error message do for a small model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_tool_choice, result_answer, result_describe):
    L.wrap_up(result_describe, result_answer, choice_tool_choice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
