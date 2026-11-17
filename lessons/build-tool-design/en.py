# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-tool-design. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tool-design", "en")
    C = L.checks   # three descriptions of convert, two kinds of error, the course's runs of Qwen3, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en), one about [Beyond text: images, audio and realtime](/course/open/build-multimodal-apis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_arguments():
    quick_arguments = L.choice(
        "quick-arguments",
        question="A tool call in the answer has `\"arguments\": \"{\\\"amount\\\": 37, \\\"from_unit\\\": \\\"km\\\", \\\"to_unit\\\": \\\"mi\\\"}\"`. What does your program give the function `convert`?",
        options={
            "text": "That text itself, as it came: convert can read it as it is",
            "loads": "What `json.loads` makes of that text: a dictionary",
            "id": "The call's `id`, so that convert knows which call it answers",
        },
        correct="loads",
        why={
            "loads": "The arguments come as a JSON text; `json.loads` makes the dictionary your function reads.",
            "text": "It is a text, not yet a dictionary: `arguments[\"amount\"]` would fail on it. See [Tools as JSON](/course/open/build-tool-calling?lang=en).",
            "id": "The `id` goes back with the result, as `tool_call_id`; the function needs the arguments. See [Tools as JSON](/course/open/build-tool-calling?lang=en).",
        },
    )
    quick_arguments
    return (quick_arguments,)


@app.cell(hide_code=True)
def quick_check_hosted():
    quick_hosted = L.choice(
        "quick-hosted",
        question="Who runs `detect_objects`, the hosted tool of the course's server?",
        options={
            "server": "The server, on each image, before the model reads the question",
            "program": "Your program, once the model calls it in its answer, with the image as its argument",
            "model": "Qwen3 itself: it reads the image and runs the detector inside the model",
        },
        correct="server",
        why={
            "server": "A hosted tool runs at the provider: you name it in the request, and the server runs it.",
            "program": "That is how a tool your own program runs works. A hosted tool you only name. See [Beyond text](/course/open/build-multimodal-apis?lang=en).",
            "model": "Qwen3 reads text only; the server's detector, DETR, is a model of its own. See [Beyond text](/course/open/build-multimodal-apis?lang=en).",
        },
    )
    quick_hosted
    return (quick_hosted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Tools as JSON](/course/open/build-tool-calling?lang=en), Qwen3 called `convert` with the right arguments all {n} times it had to. That was with the description in full. A model knows a tool only from what it reads in the prompt: the name, the description, the arguments. It cannot read your code, try the tool out, or ask a colleague what `b` means. So the description is not paperwork: it decides whether the model can use the tool at all. The course measured what happens with less, and with errors that say less.
    """.format(n=F["questions"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Everything an agent sees of a tool is its interface: the name, the description and the arguments it reads before a call, and the result or the error it reads after. A screen for people gets designed, tested and improved; the interface for a model deserves the same care.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **agent-computer interface** | everything an agent sees of its tools: names, descriptions, arguments, and what comes back, results and errors; a user interface, but for a model |
    | **client-side tool** | a tool your own program runs, after the model calls it, such as `convert`; the opposite of a hosted tool, which the provider runs |

    ### Three descriptions of convert

    The course described the same function three ways. Bare: a short name, three letters as arguments, no descriptions.

    ```python
    {"name": "conv", "description": "",
     "parameters": {"type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "string"}, "c": {"type": "string"}},
                    "required": ["a", "b", "c"]}}
    ```

    Names only: good names, but no descriptions of the arguments, and no `enum` with the allowed units.

    ```python
    {"name": "convert", "description": "Convert an amount.",
     "parameters": {"type": "object",
                    "properties": {"amount": {"type": "number"}, "from_unit": {"type": "string"},
                                   "to_unit": {"type": "string"}},
                    "required": ["amount", "from_unit", "to_unit"]}}
    ```

    And in full, as in [Tools as JSON](/course/open/build-tool-calling?lang=en). With each, the course asked the same {n} questions, and named the tool in `tool_choice`, so that every first answer is a call.
    """.replace("{n}", str(F["questions"])))
    return


@app.cell(hide_code=True)
def picture_described():
    mo.Html(C.described_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_described():
    mo.vstack([
        mo.md(r"""
        Bare, {bare} of the {n} first calls were right; with names only, {names}; in full, all {full}. With names only, Qwen3 wrote the units as words: {words}. `convert` knows none of those. The bare tool did worse in a way that is harder to see: for *{q}* it called `conv` with `{call}`, the wrong way round. `convert` can do that, so it gave back `{result}`, without an error: a wrong number that looks just as right as a good one. With `from_unit` and `to_unit` as names, the model put the units the right way round every time.

        The full description costs tokens, in every request: {p_full} prompt tokens, against {p_names} with names only and {p_bare} bare. About eighty tokens for eight right calls instead of three: that is worth it.
        """.format(bare=len(F["right_bare"]), names=len(F["right_names"]), full=len(F["right_full"]), n=F["questions"],
                   words=", ".join("`{}`".format(x) for x in F["words"]), q=C.QUESTIONS[F["wrong_way"][0]][0],
                   call=C.short(C.as_convert(C.steps_of("bare", F["wrong_way"][0])[0]["tool_calls"][0])),
                   result=C.steps_of("bare", F["wrong_way"][0])[0]["results"][0],
                   p_full=F["prompt_full"], p_names=F["prompt_names"], p_bare=F["prompt_bare"])),
        L.watch_out("The worst call is not the one that fails, but a wrong one that the tool can run. Make such calls hard: names that say which way round, a fixed list of values, and every argument that is needed required."),
    ])
    return


@app.cell(hide_code=True)
def text_errors():
    mo.md(r"""
    ### Errors the model can use

    With names only, {k} first calls got an error: an unknown unit. The course let those {k} conversations go on, once with each of two kinds of error from `convert`: only `Error`, or one that says what went wrong and what to do instead, such as `Error: unknown unit 'miles'. Use km, mi, kg or lb.` Each run got at most three requests.
    """.format(k=len(F["failed"])))
    return


@app.cell(hide_code=True)
def picture_errors():
    mo.Html(C.errors_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_errors_after():
    mo.vstack([
        mo.md(r"""
        With only `Error`, Qwen3 gave up all {k} times, with an apology such as *{sorry}* With the error that says what to do, it called again in {again} of the {k}; in {fixed} of those it found the right call, and once it answered right within three requests. Twice it passed the error on as its answer. So a small model uses a good error only now and then; the description helped far more. Make both good: the description keeps most wrong calls from happening, and the error gives the model a way out of the rest.
        """.format(k=len(F["failed"]), sorry=C.steps_of("names-terse", F["failed"][0])[-1]["text"].split(". ")[0] + ".",
                   again=len(F["helpful_again"]), fixed=len(F["helpful_fixed"]))),
        L.watch_out("An error goes to the model as text, and costs tokens like any message. A whole Python traceback tells a model little: give back one or two sentences, with what it can do next."),
    ])
    return


@app.cell(hide_code=True)
def text_who_runs():
    mo.md(r"""
    ### Who runs the tool

    `convert` is a client-side tool: the model writes the call, your program runs it, and the result goes back in the next request. The `detect_objects` of [Beyond text](/course/open/build-multimodal-apis?lang=en) was hosted: you only named it, the server ran it, and your program never saw a call. Hosted tools, such as a web search or running code at a provider, save you work, but they run where the provider decides, with what it allows. A client-side tool runs in your program, on your data, under your rules: the next lesson is about those rules.

    The words *agent-computer interface* come from SWE-agent, a coding agent of 2024, whose makers found that a model did better with commands made for it than with an ordinary terminal. Anthropic, in *Building effective agents* (December 2024), advises spending as much care on that interface as on a screen for people.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a tool an agent can use well

    1. **A name that says what it does**: `convert`, not `conv`.
    2. **A description** of what the tool does, when to use it, and where it stops: *kilometres and miles, or kilograms and pounds*.
    3. **Arguments with clear names**, each with a description and an example, and an `enum` where only a few values are allowed.
    4. **Make wrong calls hard**: names that say which way round, required arguments, fixed values.
    5. **Errors that say what went wrong and what to do instead**, starting with `Error`.
    6. **Try it on the model**: send real questions, and read the calls it writes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _steps = C.steps_of("names-helpful", C.WORKED)
    mo.md(r"""
    ### Worked example

    *{q}*, with names only and the error that says what to do.

    1. **The first call**: `{c1}`. The unit `{u}` is not one `convert` knows.
    2. **The error** goes back as the call's result: `{r1}`. It names the unit, and the four the model can use.
    3. **The second call**: `{c2}`, and `convert` gives back `{r2}`.
    4. **The answer**: *{a}*

    With only `Error`, the same first call ended in an apology, and no number at all.
    """.format(q=C.QUESTIONS[C.WORKED][0], c1=C.short(C.as_convert(_steps[0]["tool_calls"][0])),
               u=C.as_convert(_steps[0]["tool_calls"][0])["from_unit"], r1=_steps[0]["results"][0],
               c2=C.short(C.as_convert(_steps[1]["tool_calls"][0])), r2=_steps[1]["results"][0], a=_steps[2]["text"]))
    return


@app.cell(hide_code=True)
def text_helpful():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Errors the model can use

        The box holds `convert` as a function your program runs: it gets the arguments as a dictionary, as `json.loads` makes them of a call. Every error still says only `"Error"`. Make each say what went wrong and what to do instead: for an unknown unit, the unit the model wrote and the four units `convert` knows; for two units that do not go together, which do. Start each error with `Error`. The check calls your `convert` with the arguments of {k} calls, as a model could write them; the picture under the check shows what each gets back.
        """.format(k=len(C.VISIBLE) + len(C.HIDDEN))),
        L.task("Replace each `\"Error\"` with a text that starts with `Error`, says what went wrong, and lists the units `km`, `mi`, `kg` and `lb`; with an unknown unit, put that unit in the text too."),
    ])
    return


@app.cell(hide_code=True)
def exercise_helpful():
    box_helpful = L.code_box("helpful", lines=15)
    box_helpful
    return (box_helpful,)


@app.cell(hide_code=True)
def check_helpful(box_helpful):
    result_helpful = L.check("helpful", box_helpful.value)
    result_helpful
    return (result_helpful,)


@app.cell(hide_code=True)
def hints_helpful():
    L.hints(
        "helpful",
        [
            "An error is read by a model that wants to try again. What does it need to know? Which word of its call was wrong, and what it may write instead.",
            "Glue the unit the model wrote into the text with `+`: `\"Error: unknown unit '\" + a + ...`. The four units can be in the text as they are.",
            "For the last `if`: `return \"Error: cannot convert \" + a + \" to \" + b + ___`, with which units go together in the gap. The first two end in `+ \"'. Use km, mi, kg or lb.\"`.",
        ],
        why="""```python
def convert(arguments):
    amount = arguments["amount"]
    a = arguments["from_unit"]
    b = arguments["to_unit"]
    if a not in FACTOR:
        return "Error: unknown unit '" + a + "'. Use km, mi, kg or lb."
    if b not in FACTOR:
        return "Error: unknown unit '" + b + "'. Use km, mi, kg or lb."
    if KIND[a] != KIND[b]:
        pairs = "Convert km and mi into each other, or kg and lb."
        return "Error: cannot convert " + a + " to " + b + ". " + pairs
    return str(round(amount * FACTOR[a] / FACTOR[b], 2)) + " " + b
```

These are the errors of the course's runs. Your words may differ: the check looks at what is in them, not at how you put it.""",
    )
    return


@app.cell(hide_code=True)
def text_describe_better():
    mo.md(r"""
    ### Exercise 2. A better description
    """)
    return


@app.cell(hide_code=True)
def choice_describe_better_cell():
    choice_describe_better = L.choice(
        "describe-better",
        question="With names only, Qwen3 wrote units such as `miles` and `pounds`. In the course's runs, what made all {n} first calls right?".format(n=F["questions"]),
        options={
            "name": "A longer name for the tool, `convert_between_kilometers_miles_kilograms_and_pounds`, that says it all",
            "enum": "The four units as an `enum`, and a description of each argument",
            "prompt": "A system prompt that tells the model to write every unit in short, such as mi and lb",
        },
        correct="enum",
        why={
            "enum": "With `\"enum\": [\"km\", \"mi\", \"kg\", \"lb\"]`, the model reads which four values there are, and writes those. That is the description in full of the table above.",
            "name": "A name says what a tool does, but not which values an argument may have. Where in a description does that go?",
            "prompt": "The course did not measure that; and the rule belongs with the tool, so that it goes wherever the tool goes. Where in the description can you list the allowed units?",
        },
    )
    choice_describe_better
    return (choice_describe_better,)


@app.cell(hide_code=True)
def text_client_side():
    mo.md(r"""
    ### Exercise 3. Who runs it
    """)
    return


@app.cell(hide_code=True)
def choice_client_side_cell():
    choice_client_side = L.choice(
        "client-side",
        question="An app gives a model three tools: the provider's hosted web search, `convert`, and `send_email`, which your program runs. For which tools does your program get tool calls to run?",
        options={
            "all": "All three: every tool call of the model comes back to your program to run",
            "search": "Only the web search, because that one reaches outside the app",
            "two": "`convert` and `send_email`",
        },
        correct="two",
        why={
            "two": "Those two are client-side tools: your program runs them. The provider runs the web search itself, and the answer comes back with what it found.",
            "all": "A hosted tool runs at the provider: you name it, and never get a call to run. Which of the three is hosted?",
            "search": "It is the other way round: the provider runs the hosted search itself. Which tools run in your program?",
        },
    )
    choice_client_side
    return (choice_client_side,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 1 is right, you can put your `convert` behind the description with names only, and see what Qwen3 does with your errors: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo asks the question in the box, makes the first answer a call, runs each call with your `convert`, and asks again with the result, three requests at most. Try a question with miles or pounds in words.
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
    question_box = mo.ui.text(value=C.QUESTIONS[1][0], label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Ask, with your convert")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_helpful, question_box, send_button):
    _convert = C.learner_convert(box_helpful.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _convert is None:
        _out = mo.md("*Finish exercise 1 first: the demo runs the calls with your own `convert`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "What Qwen3 called, and what your convert gave back", "request": "Request {n}:",
                  "open": "No answer after {n} requests.", "failed": "It stopped with an error: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_convert, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The description you completed in [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en) is the reason Qwen3 called `convert` right every time: with less, it called right two or three times in eight. The hosted tool of [Beyond text](/course/open/build-multimodal-apis?lang=en) now has a name for its opposite: a client-side tool, which your program runs. In the next lesson, [What a tool may do](/course/open/build-tool-permissions?lang=en), you decide which of those calls your program lets through.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_client_side, choice_describe_better, result_helpful):
    L.wrap_up(result_helpful, choice_describe_better, choice_client_side)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
