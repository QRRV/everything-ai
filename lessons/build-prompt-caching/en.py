# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-prompt-caching. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-prompt-caching", "en")
    C = L.checks   # the course's recorded runs, the checks, and the demo with your messages_for


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [What goes into the window](/course/open/build-context-engineering?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_window():
    quick_window = L.choice(
        "quick-window",
        question="What does a model know about your conversation when it answers a request?",
        options={
            "earlier": "What it heard in the earlier requests of your app",
            "window": "Only what is in that request, in its context window",
            "training": "What it learned in training, and every conversation since",
        },
        correct="window",
        why={
            "window": "The model keeps nothing between requests: the app sends everything it needs again, every time. That is why the same long start goes in again and again.",
            "earlier": "The model keeps nothing between requests; the app sends the conversation again each time. See [What goes into the window](/course/open/build-context-engineering?lang=en).",
            "training": "Training ended before your app ran, and the model learns nothing from your requests. See [What goes into the window](/course/open/build-context-engineering?lang=en).",
        },
    )
    quick_window
    return (quick_window,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An app sends its long instructions with every request: a system prompt with the rules of a shop, a document a user asks about, a list of tools. The model keeps nothing between requests, so the server reads that same long start again each time, before it can write a word. Most of the waiting, and on a hosted API much of the bill, is then the same text over and over. With prompt caching the server keeps its work on that start, and the next request that begins the same way goes on from there.

    **An everyday example.** A teacher who marks thirty tests with the same answer key reads the key once, and keeps it open on the desk; reading it again before every test would take most of the evening. But if the key changes on page one, everything after it has to be read again. That is only a comparison; below you measure what the course's server saves, and what made it save nothing.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **prompt caching** | the server keeps what it computed for the start of a prompt, and a later prompt that starts with exactly the same tokens goes on from there, instead of computing them again |
    | **KV cache** | what the model computes for each token it has read and needs again for every next token: a key and a value per token in every layer; a server keeps it while it writes an answer, and with prompt caching also after it |
    | **cached tokens** | the tokens of a prompt that came from the kept cache: in OpenAI's form, `usage.prompt_tokens_details.cached_tokens` |

    ### What the server keeps

    To write an answer, the model first reads the whole prompt: every token goes through every layer, and for each token every layer keeps a key and a value, which the tokens after it look back at. That is the KV cache. The course's server keeps the KV cache of its last request, as the server of llama.cpp does. When the next prompt starts with the same tokens, it takes their keys and values from the cache and computes only the rest. The tokens must be the same from the very first one: as soon as one differs, every token after it sees something else, so from there on nothing can be reused.

    The course gave Qwen3 the handbook of a bike shop as its system prompt, {handbook} tokens of rules, and sent seven requests one after the other. `max_tokens` was 1, so the time is almost all reading the prompt:
    """.format(handbook=C.number(C.findings()["handbook"], "en", 0)))
    return


@app.cell(hide_code=True)
def picture_sequence():
    mo.Html(C.sequence_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_sequence():
    _f = C.findings()
    _s = _f["steps"]
    mo.vstack([
        mo.md(r"""
        The first request read all {cold_tokens} tokens itself, in {cold} seconds. The second asked another question after the same handbook: {same_cached} of its {same_tokens} tokens came from the cache, and it took {same} seconds, {times} times as fast. Then the time went in front of the handbook, as an app might add it to the system prompt: the two prompts a minute apart had only {first_cached} tokens in common at the start, up to the minute, so the server read the whole handbook again, {first} seconds. With the time after the question, at the end, the first of the two still read almost everything itself, because the request before it began with the time; a minute later, the second gave {last_cached} cached tokens, and {last} seconds.

        The cache does not change what the model reads, only what the server computes again. The third question, asked once without the cache and once with {again_cached} of its tokens from it, got {same_answer}.
        """.format(cold_tokens=C.number(_s["cold"]["prompt"], "en", 0), cold=C.number(_s["cold"]["seconds"], "en", 2),
                   same_cached=C.number(_s["same-1"]["cached"], "en", 0), same_tokens=C.number(_s["same-1"]["prompt"], "en", 0),
                   same=C.number(_s["same-1"]["seconds"], "en", 2), times=C.number(_s["cold"]["seconds"] / _s["same-1"]["seconds"], "en", 0),
                   first_cached=_s["first-2"]["cached"], first=C.number(_s["first-2"]["seconds"], "en", 2),
                   last_cached=C.number(_s["last-2"]["cached"], "en", 0), last=C.number(_s["last-2"]["seconds"], "en", 2),
                   again_cached=C.number(_f["same"][1]["cached_tokens"], "en", 0),
                   same_answer="the same answer word for word" if _f["same"][0]["text"] == _f["same"][1]["text"] else "answers that differ")),
        L.watch_out("Hosted APIs cache too, each in its own way (examples in November 2026). OpenAI caches a prompt of at least 1,024 tokens by itself and bills the cached tokens at a fraction of the normal price. Anthropic caches where the request marks it, with `cache_control`, and bills writing to the cache a little more and reading from it much less. By default a cache lives a few minutes after its last use, so it helps most an app with many requests close together."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a prompt that can be cached

    1. **What stays the same goes first:** the system prompt, the documents, the tools; what changes per request, such as the question, the time or the user's name, goes last.
    2. **Keep the start exactly the same:** no time, no counter and no list in a random order in it; one token that differs early means the rest is computed again.
    3. **Count what came from the cache:** read `cached_tokens` in `usage`, and compare it with `prompt_tokens`.
    4. **Keep the requests close together:** by default a cache is kept for a few minutes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    mo.md(r"""
    ### Worked example

    The course asked the three questions after the handbook one after the other, with `max_tokens` 60, the first with the cache switched off: *{q0}*, *{q1}* and *{q2}*. How much did the cache save on the third?

    1. **What stays the same goes first:** the handbook is the system prompt, and the question is the message of the user after it.
    2. **The start exactly the same:** the three prompts differ only from the question on.
    3. **Count what came from the cache:** the third request had {tokens} prompt tokens, and `cached_tokens` was {cached}. The server read only the last {rest} tokens, from the question on: the handbook and the markers before the question came from the cache that the second request left. The answer was *{a2}*.
    4. **Close together:** the three requests came right after each other.
    """.format(q0=C.QUESTIONS[0], q1=C.QUESTIONS[1], q2=C.QUESTIONS[2], tokens=C.number(C.findings()["steps"]["same-2"]["prompt"], "en", 0),
               cached=C.number(_f["answer_cached"][2], "en", 0), rest=C.findings()["steps"]["same-2"]["prompt"] - _f["answer_cached"][2],
               a2=_f["answers"][2].strip()))
    return


@app.cell(hide_code=True)
def text_shared():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What two prompts have in common

        The server compares the tokens of a new prompt with the ones it kept, from the first token on, and reuses them up to the first one that differs. Write `shared_start(a, b)`, which gives back how many tokens the lists `a` and `b` have in common at the start. The box counts every place where the two are the same, also after a place where they differ.
        """),
        L.task("Stop at the first place where the tokens differ, and give back how many came before it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shared():
    box_shared = L.code_box("shared_start", lines=8)
    box_shared
    return (box_shared,)


@app.cell(hide_code=True)
def check_shared(box_shared):
    result_shared = L.check("shared_start", box_shared.value)
    result_shared
    return (result_shared,)


@app.cell(hide_code=True)
def hints_shared():
    L.hints(
        "shared_start",
        [
            "Counting goes on as long as the tokens are the same; at the first place where they differ, the function is done.",
            "Turn the test around: `if a[i] != b[i]:`, and give back `n` there with `return`. After the `if`, the loop counts one more.",
            "In the loop: `if a[i] != b[i]:` with `return n` under it, and then `n = n + 1`, not under the `if`.",
        ],
        why="""```python
def shared_start(a, b):
    n = 0
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return n
        n = n + 1
    return n
```

The last `return n` is for lists that never differ up to the end of the shorter one: then the whole shorter list is in common.""",
    )
    return


@app.cell(hide_code=True)
def text_order():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The time after the handbook

        The shop's assistant should know the time, for questions such as *Are you still open?*. The box puts it at the very start of the system prompt, so the prompt starts differently every minute. Change `messages_for(question, now)` so that the system prompt is the handbook alone, `HANDBOOK`, and the message of the user is the question, a blank line, and `Now: ` with the time. The check tries your function with three questions at three times, and compares the messages it makes.
        """),
        L.task("Make the system prompt `HANDBOOK` alone, and put the time after the question."),
    ])
    return


@app.cell(hide_code=True)
def exercise_order():
    box_order = L.code_box("messages_for", lines=6)
    box_order
    return (box_order,)


@app.cell(hide_code=True)
def check_order(box_order):
    result_order = L.check("messages_for", box_order.value)
    result_order
    return (result_order,)


@app.cell(hide_code=True)
def hints_order():
    L.hints(
        "messages_for",
        [
            "Only the place of the time changes: from the front of the system prompt to the back of the user's message.",
            "The system message gets `HANDBOOK` as its content, and the user message `question + \"\\n\\nNow: \" + now`.",
            "`return [{\"role\": \"system\", \"content\": HANDBOOK}, {\"role\": \"user\", \"content\": question + \"\\n\\nNow: \" + now}]`",
        ],
        why="""```python
def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK},
            {"role": "user", "content": question + "\\n\\nNow: " + now}]
```

Now every prompt starts with the same tokens of the handbook, whatever the time.""",
    )
    return


@app.cell(hide_code=True)
def text_breaks():
    mo.md(r"""
    ### Exercise 3. A greeting by name
    """)
    return


@app.cell(hide_code=True)
def choice_breaks_cell():
    choice_breaks = L.choice(
        "breaks",
        question="The shop wants its assistant to greet every customer by name. Where does the name go, so that the cache keeps working?",
        options={
            "after": "In the message of the user, after the whole handbook",
            "top": "At the top: *You are talking to Sam.*",
            "both": "It makes no difference: the server caches each customer's prompt apart",
        },
        correct="after",
        why={
            "after": "The handbook stays the same start for every customer, so the server reuses it; only the tokens from the name on are new.",
            "top": "Then the prompt starts differently for every customer, and the server reads the whole handbook again each time, as with the time in front of it. What changes goes last.",
            "both": "The course's server keeps only the cache of its last request, and a hosted API matches the start of the prompt too: with the name on the first line, no two customers share a start. What changes goes last.",
        },
    )
    choice_breaks
    return (choice_breaks,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 2 is right, you can send questions with your own `messages_for` to the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo sends each question with the time of your computer, one after the other, the first with the cache switched off, and shows for each how many tokens came from the cache, and how long it took. Your computer is not the course's, so your seconds differ.
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
def questions_cell():
    questions_box = mo.ui.text_area(value="\n".join(C.QUESTIONS), label="Your questions, one per line", full_width=True, rows=4)
    send_button = mo.ui.run_button(label="Send with your messages_for")
    mo.vstack([questions_box, send_button])
    return questions_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_order, questions_box, send_button):
    _fn = C.learner_messages_for(box_order.value)
    _questions = [q.strip() for q in questions_box.value.splitlines() if q.strip()]
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _fn is None:
        _out = mo.md("*Finish exercise 2 first: the demo makes the messages with your own `messages_for`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to send.*")
    elif not _questions:
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The questions go one after the other; the server loads Qwen3 and answers …*"))
        _words = {"title": "Your questions, one after the other",
                  "done": "{given} tokens in, {cached} of them from the cache, in {seconds} seconds:",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fn, _questions, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The context window of [What goes into the window](/course/open/build-context-engineering?lang=en) is read again with every request, and prompt caching makes the part that stays the same cheap to read again. The system prompt of [System prompts and response prefill](/course/open/build-system-prompt?lang=en) is usually that part, and the input tokens of [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en) are what the cache saves. With this lesson the module on prompts is complete; the next module looks up texts that fit a question and puts them in the prompt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_breaks, result_order, result_shared):
    L.wrap_up(result_shared, result_order, choice_breaks)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
