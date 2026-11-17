# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-errors-retries. Route: course/route/lessen/bouwen.yaml. Checks, the stand-ins and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-errors-retries", "en")
    C = L.checks   # the stand-ins that say no at first, and the pictures of the tries


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Calling a model over HTTP](/course/open/build-call-api?lang=en), and one about [Catching and raising errors](/course/open/py-exceptions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_urlopen():
    quick_urlopen = L.choice(
        "quick-urlopen",
        question="What does `urllib.request.urlopen(request)` do with a request that has `data=`?",
        options={
            "get": "It fetches the address with a GET, and leaves the data out",
            "file": "It writes the data to a file on your computer",
            "post": "It sends the data with a POST, and gives back the response",
        },
        correct="post",
        why={
            "post": "With `data=`, the request is a POST: the body goes to the server, and `response.read()` reads what comes back.",
            "get": "A request without `data=` is a GET. What changes with `data=`? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "file": "`urlopen` talks to a server over the network. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_urlopen
    return (quick_urlopen,)


@app.cell(hide_code=True)
def quick_check_try():
    quick_try = L.choice(
        "quick-try",
        question="What does this program show?\n\n```python\ntry:\n    print(int(\"7\"))\n    print(int(\"seven\"))\nexcept ValueError:\n    print(\"no number\")\nprint(\"on\")\n```",
        options={
            "traceback": "`7`, and then a traceback",
            "all": "`7`, `no number` and `on`",
            "skip": "`no number` and `on`",
        },
        correct="all",
        why={
            "all": "`int(\"7\")` works and shows 7; `int(\"seven\")` stops with a `ValueError`, so Python jumps to the `except`, and after it the program goes on.",
            "traceback": "The `ValueError` happens inside the `try`, and the `except` names that kind. See [Catching and raising errors](/course/open/py-exceptions?lang=en).",
            "skip": "The lines in the `try` run one by one, until one fails. Which one fails? See [Catching and raising errors](/course/open/py-exceptions?lang=en).",
        },
    )
    quick_try
    return (quick_try,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model's API says no now and then: too many requests in one minute, a server that is busy, a key that is wrong. Code that stops at the first error makes an app that breaks for no good reason. Code that tries again at once, again and again, makes the server's trouble worse. Good code reads why the call failed, tries again only what can work a moment later, and waits longer each time.

    **An everyday example.** You call a shop and the line is busy. You try again after a minute, then after a few minutes more, not every second. And if the number does not exist, calling again does not help at all. That is only a comparison; below you see what a server says, and how long to wait.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **HTTP status code** | a number in every response that says how it went: `200` is fine, a status in the 400s says the request is wrong, one from 500 up says the server has trouble |
    | **rate limit** | how many requests or tokens a key may send per minute; above it, the server answers `429`, often with a header `Retry-After`: the seconds to wait |
    | **timeout** | how long to wait for the server at most: `urlopen(request, timeout=30)` stops with a `TimeoutError` when the server stays silent for 30 seconds; the 30 seconds count for each wait, not for the whole answer, so a slow stream can take longer; without `timeout=`, it can wait forever |
    | **retry** | sending the same request again after it failed |
    | **exponential backoff** | retries with waits that double: `base`, then twice that, then four times, which is `base * 2 ** attempt` |
    | `urllib.error.HTTPError` | the error urllib raises for a status from 400 up: `e.code` is the status, `e.headers` the headers, `e.read()` the body |

    These are the statuses a model's API gives most:

    | Status | What it means |
    |---|---|
    | `400` | the request is not valid, such as a body without `"messages"` |
    | `401` | no valid key came with the request |
    | `403` | the key may not do this |
    | `404` | there is no such address, or no such model |
    | `429` | too many requests: the rate limit is reached |
    | `500` | something went wrong in the server |
    | `503` | the server is busy, or down for a moment |

    Which ones are worth another try? Ask: would the same request get another answer a moment later?

    ### Waits that double

    A server that said 429, then 503 twice, and then answered, with `base` 1 second:
    """)
    return


@app.cell(hide_code=True)
def picture_waits():
    mo.Html(C.waits_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_waits():
    _w = C.all_waits()
    mo.vstack([
        mo.md(rf"""
        The first wait is 1 second, then 2, then 4: `base * 2 ** attempt`, with `attempt` 0, 1 and 2. Why not try again at once? If thousands of programs get a 503 at the same moment, and all of them ask again straight away, the server gets the same flood again. Waits that double spread the retries out, and give the server room. They also grow fast: ten tries would wait 1 + 2 + … + {_w['last']:g} = {_w['total']:g} seconds in all, so you set a number of tries too, and after the last one you pass the error on.
        """),
        L.watch_out("A timeout stops with a `TimeoutError`, not an `HTTPError`; a timeout while connecting comes as a `urllib.error.URLError`, with the `TimeoutError` in it. Both are worth another try: catch them with `except TimeoutError:` and `except urllib.error.URLError:` under the first."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: try again, with backoff

    1. **Set a timeout:** `urlopen(request, timeout=30)`.
    2. **Try in a loop:** `for attempt in range(tries):`, with `try:` around the call, and `return` the text when it works.
    3. **Catch the error:** `except urllib.error.HTTPError as e:`. The `as e` gives the error a name, so you can read its status, `e.code`.
    4. **Decide:** if the status will not change, or this was the last try, pass the error on with `raise`. On its own, in an `except`, `raise` raises the same error again, for the code that called yours.
    5. **Wait, and go round again:** `time.sleep(base * 2 ** attempt)`; `time.sleep(s)` waits `s` seconds. When the server sends `Retry-After`, wait that long instead.

    In November 2026, the package `openai` does this for you: it tries again a few times on its own, with backoff, and `max_retries` sets how often.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    What does a busy server say? Here `url` is the address of the course's stand-in of a server, which answers the first request with 429.

    ```python
    import json
    import urllib.error
    import urllib.request

    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(e.code, e.headers["Retry-After"])
        print(json.loads(e.read())["error"]["message"])
    ```

    ```
    429 2
    Too many requests: wait a moment and try again.
    ```

    1. **The request:** the same as in [Calling a model over HTTP](/course/open/build-call-api?lang=en), now with a timeout of 30 seconds.
    2. **The error:** the server answers with status 429, so `urlopen` raises an `HTTPError`, and Python jumps to the `except`.
    3. **The status and the header:** `e.code` is `429`, and the header `Retry-After` says `2`: wait two seconds.
    4. **The body:** an error has a body too, JSON with the server's own words under `"error"`; `e.read()` gives it as bytes, as `response.read()` does.
    """)
    return


@app.cell(hide_code=True)
def text_should_retry():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Worth another try?

        Write `should_retry(status)`: `True` when the same request can get another answer a moment later, `False` when it will get the same error however often you send it. The check tries the statuses in the table, and a few more from 500 up.
        """),
        L.task("Finish `should_retry`: give `True` for a status that is worth another try, else `False`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_should_retry():
    box_retry = L.code_box("should_retry", lines=4)
    box_retry
    return (box_retry,)


@app.cell(hide_code=True)
def check_should_retry(box_retry):
    result_retry = L.check("should_retry", box_retry.value)
    result_retry
    return (result_retry,)


@app.cell(hide_code=True)
def hints_should_retry():
    L.hints(
        "should_retry",
        [
            "For each status in the table, ask: is the request wrong, or is the server too busy, or in trouble, right now?",
            "429 can be fine after waiting, and so can trouble in the server: every status from 500 up. The other statuses in the 400s stay as they are.",
            "One line: `return status == 429 or ___`.",
        ],
        why="""```python
def should_retry(status):
    return status == 429 or status >= 500
```

A wrong body, key or address stays wrong; too many requests and a busy server can be over in a moment.""",
    )
    return


@app.cell(hide_code=True)
def text_ask_again():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Try again, and wait longer each time

        `ask_again(url, question, tries, base)` sends a question, at most `tries` times. The loop and the `try` are there already; fill in what happens under `except`: pass the error on when `should_retry` says no, or when it was the last try, and else wait `base * 2 ** attempt` seconds. It uses your `should_retry` from exercise 1. The check runs your function against three stand-ins: one that says 429 and 503 and then answers, one that gets a wrong key, and one that stays down. The picture under the check shows your tries on a time line.
        """),
        L.task("Under `except`: pass the error on when it will not change or this was the last try; else wait, `base` seconds after the first try and twice as long after each next one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask_again():
    box_again = L.code_box("ask_again", lines=18)
    box_again
    return (box_again,)


@app.cell(hide_code=True)
def check_ask_again(box_again, box_retry):
    result_again = L.check("ask_again", box_again.value, C.again_inputs(box_retry.value))
    result_again
    return (result_again,)


@app.cell(hide_code=True)
def hints_ask_again():
    L.hints(
        "ask_again",
        [
            "Steps 4 and 5 of the recipe: first decide whether to stop, then wait.",
            "`should_retry(e.code)` says whether the status of the error is worth another try; `attempt` counts the tries from 0, so which value does it have on the last one? `raise` on a line of its own passes the error on, and `time.sleep(seconds)` waits.",
            "First `if not should_retry(e.code) or ___:`, with `raise` under it, further in; the wait comes after it.",
        ],
        why="""```python
def ask_again(url, question, tries, base):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.loads(response.read())
            return answer["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if not should_retry(e.code) or attempt == tries - 1:
                raise
            time.sleep(base * 2 ** attempt)
```

`raise` passes on an error that will not change, or the last one; otherwise the function waits, and the loop sends the request again, each time after a wait twice as long.""",
    )
    return


@app.cell(hide_code=True)
def text_crowd():
    mo.md(r"""
    ### Exercise 3. Everyone at once
    """)
    return


@app.cell(hide_code=True)
def choice_crowd_cell():
    choice_crowd = L.choice(
        "crowd",
        question="A busy server answers 503 to ten thousand apps at the same moment. Every app tries again at once, again and again, without waiting. What happens?",
        options={
            "worse": "The server gets even more requests than before, and stays overloaded",
            "fine": "The server catches up, because every app asks only once more",
            "faster": "The apps get their answers sooner, because they keep asking",
        },
        correct="worse",
        why={
            "worse": "Every 503 brings a new request at once, so the flood stays. With waits that double, the retries spread out over time, and the server gets room to catch up.",
            "fine": "Without waiting, every app asks again and again, as fast as it can. What does that do to the number of requests?",
            "faster": "An overloaded server gets slower with every extra request. What would give it room?",
        },
    )
    choice_crowd
    return (choice_crowd,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The errors are what the server of [Calling a model over HTTP](/course/open/build-call-api?lang=en) says when something is wrong, caught with `try` and `except` as in [Catching and raising errors](/course/open/py-exceptions?lang=en). A hosted API's rate limit counts requests and tokens per minute, per key: the next lesson, [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en), brings in the key, and many requests at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_crowd, result_again, result_retry):
    L.wrap_up(result_retry, result_again, choice_crowd)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
