# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-batch-keys-costs. Route: course/route/lessen/bouwen.yaml. Checks, the stand-ins and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-batch-keys-costs", "en")
    C = L.checks   # the stand-ins that want a key, the batch of the worked example, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_401():
    quick_401 = L.choice(
        "quick-401",
        question="A call gets status 401: no valid key came with the request. Is it worth another try?",
        options={
            "wait": "Yes, after a short wait",
            "no": "No: the same request gets the same answer",
            "now": "Yes, straight away",
        },
        correct="no",
        why={
            "no": "A missing or wrong key stays missing or wrong, however often you send the request: pass the error on, and fix the key.",
            "wait": "Waiting helps a busy server, not a wrong request. See [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en).",
            "now": "Would the same request, with the same key, get another answer? See [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en).",
        },
    )
    quick_401
    return (quick_401,)


@app.cell(hide_code=True)
def quick_check_wait():
    quick_wait = L.choice(
        "quick-wait",
        question="With `base` 2, how many seconds does `time.sleep(base * 2 ** attempt)` wait when `attempt` is 2?",
        options={
            "four": "4",
            "six": "6",
            "eight": "8",
        },
        correct="eight",
        why={
            "eight": "`2 ** 2` is 4, and 2 times 4 is 8: the waits are 2, 4 and 8 seconds.",
            "four": "That is `2 ** 2` alone. Times what? See [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en).",
            "six": "`**` is a power, not a times. See [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en).",
        },
    )
    quick_wait
    return (quick_wait,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The course's server runs on your own computer: no key, no bill. A hosted API is open to the whole internet. It wants a key with every request, to know who asks and who pays, and it charges per token. Three habits keep that under control: the key stays out of your code, many requests that may wait go as one batch, and you count what every call costs.

    **An everyday example.** The key of a shared car: whoever has it can drive, and you pay. You do not tape it to the window, and you do not lend it to strangers. That is only a comparison; below you see where a real key goes.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **API key** | a secret text, such as `sk-…`, that says who sends a request, and so who pays; it goes along in the header `Authorization: Bearer` and the key |
    | **environment variable** | a name with a value that a program gets from the computer when it starts, outside its code; in a terminal, `export MODEL_API_KEY=sk-…` sets one |
    | `os.environ` | the environment variables in Python, as a dictionary: `os.environ["MODEL_API_KEY"]` gives the value, after `import os` |
    | **batch API** | a way to send many requests at once, in one file, for answers later, often at a lower price |

    ### Where the key goes
    """)
    return


@app.cell(hide_code=True)
def picture_key():
    draw.flow(
        [("The terminal", ["before the course starts:", "export MODEL_API_KEY=sk-…"]),
         ("Your code", ["import os", "os.environ[\"MODEL_API_KEY\"]"]),
         ("The server", ["checks the key,", "counts the tokens"])],
        ["the environment", "Authorization:\nBearer sk-…"], ["", "the answer,\nwith usage"],
        title="The key: from the environment, through your code, to the server")
    return


@app.cell(hide_code=True)
def text_key():
    mo.vstack([
        mo.md(r"""
        The key is never in your code: the code asks the environment for it, at the moment it runs. So you can share your code, and put it on GitHub, without your key; whoever runs it sets their own. The server reads the header, and refuses a request without a valid key with 401.
        """),
        L.watch_out("Never write a key in your code or in a notebook. Code ends up on GitHub, notebooks get shared, and programs search public code for keys all day. A key that leaked: delete it at the provider straight away, and make a new one."),
    ])
    return


@app.cell(hide_code=True)
def text_batch():
    mo.md(r"""
    ### Many requests at once: a batch

    Summaries of 5,000 reviews, one by one, are 5,000 calls: the rate limit stops you, and every call costs the full price. A batch API takes them all in one file, works through them when it suits the provider, and puts the answers in a file for you, often within hours instead of seconds. In return, many providers ask a lower price. The file has one line of JSON per request: an id of your own, `custom_id`, and the request itself, the same body you already send. In the file with the answers, each answer has the same `custom_id`, so you know which question it belongs to.

    ### The bill

    Each answer counts its tokens in `usage`. With a price per million tokens, a call costs `prompt_tokens * price_in / 1_000_000 + completion_tokens * price_out / 1_000_000`; `1_000_000` is a million, and Python skips the underscores. The course's call in [Calling a model over HTTP](/course/open/build-call-api?lang=en) had 19 tokens in and 13 out. At made-up prices of 3 dollars per million in and 15 out, that is 19 × 3 + 13 × 15 = 252 millionths of a dollar: almost nothing. But an app with ten thousand users makes such calls all day, and the tokens that come out cost the most. Adding up `usage` after every call keeps the bill in sight.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: keys, batches and the bill

    1. **Keep the key out of the code:** set it in the environment, with `export MODEL_API_KEY=sk-…` in the terminal before you start the course.
    2. **Read it when the code runs:** `import os`, then `os.environ["MODEL_API_KEY"]`.
    3. **Send it along:** the header `"Authorization": "Bearer " + key`, with a space after `Bearer`.
    4. **Count every call:** add `usage` times the price per million to a total.
    5. **Many requests that may wait:** write one line of JSON per request, with a `custom_id`, and send them as one batch.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The lines of a batch file, for two questions:

    ```python
    import json

    questions = ["Name a color.", "Name a fruit."]
    lines = []
    for i in range(len(questions)):
        request = {"custom_id": f"q{i}", "method": "POST", "url": "/v1/chat/completions",
                   "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": questions[i]}], "max_tokens": 50}}
        lines.append(json.dumps(request))
    for line in lines:
        print(line)
    ```

    ```
    {"custom_id": "q0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Name a color."}], "max_tokens": 50}}
    {"custom_id": "q1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Name a fruit."}], "max_tokens": 50}}
    ```

    1. **One request per question:** `custom_id` is `q0` and `q1`, made from the place in the list, so each answer can find its question again.
    2. **Where it goes:** `"method"` and `"url"` say which endpoint each request is for; `"body"` is the body you would send yourself.
    3. **One line each:** `json.dumps` writes the dictionary as JSON on one line, without line breaks.
    4. **The file:** these lines, saved as a file ending in `.jsonl`, *JSON lines*, are what you upload to a batch API. They follow OpenAI's batch format, an example in November 2026; other providers use much the same lines.
    """)
    return


@app.cell(hide_code=True)
def text_key_exercise():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The key from the environment

        This `ask` sends a question without a key, and the server wants one. Send the key along, taken from the environment variable `MODEL_API_KEY`. The check calls your `ask` against two stand-ins, each with its own key, and puts each key in `MODEL_API_KEY` during the call; the picture under the check shows the request your code sent, with the key hidden.
        """),
        L.task("Read the key from the environment variable `MODEL_API_KEY`, and send it in the header `Authorization`, after `Bearer ` and a space."),
    ])
    return


@app.cell(hide_code=True)
def exercise_key():
    box_key = L.code_box("key", lines=13)
    box_key
    return (box_key,)


@app.cell(hide_code=True)
def check_key(box_key):
    result_key = L.check("key", box_key.value)
    result_key
    return (result_key,)


@app.cell(hide_code=True)
def hints_key():
    L.hints(
        "key",
        [
            "Steps 2 and 3 of the recipe: read the key, and send it in a header.",
            "`import os` at the top. The key is `os.environ[\"MODEL_API_KEY\"]`; the header is called `\"Authorization\"`, and its value starts with `\"Bearer \"`.",
            "`headers = {\"Content-Type\": \"application/json\", \"Authorization\": \"Bearer \" + ...}`, with the key from hint 2 in place of the dots.",
        ],
        why="""```python
import json
import os
import urllib.request


def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}
    request = urllib.request.Request(url + "/chat/completions", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

The code only names the variable; the key itself comes from the environment when the code runs.""",
    )
    return


@app.cell(hide_code=True)
def text_bill():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The bill

        Write `bill(usages, price_in, price_out)`: `usages` is a list of the `usage` of calls, each a dictionary with `"prompt_tokens"` and `"completion_tokens"`; the prices are in dollars per million tokens. Give back what all calls cost together. For the two calls in the example, at 3 and 15 dollars per million, that is 0.0135 dollars:

        ```python
        bill([{"prompt_tokens": 1200, "completion_tokens": 300}, {"prompt_tokens": 800, "completion_tokens": 200}], 3, 15)
        ```
        """),
        L.task("Finish `bill`: for each call, add its tokens in at `price_in` and its tokens out at `price_out` to `total`, per million."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bill():
    box_bill = L.code_box("bill", lines=7)
    box_bill
    return (box_bill,)


@app.cell(hide_code=True)
def check_bill(box_bill):
    result_bill = L.check("bill", box_bill.value)
    result_bill
    return (result_bill,)


@app.cell(hide_code=True)
def hints_bill():
    L.hints(
        "bill",
        [
            "A `for` loop over `usages`; each call adds to `total`.",
            "One call costs `usage[\"prompt_tokens\"] * price_in / 1_000_000`, plus the same for `\"completion_tokens\"` at `price_out`.",
            "In the loop: `total = total + usage[\"prompt_tokens\"] * price_in / 1_000_000`, and a second line like it for the tokens that come out.",
        ],
        why="""```python
def bill(usages, price_in, price_out):
    total = 0
    for usage in usages:
        total = total + usage["prompt_tokens"] * price_in / 1_000_000
        total = total + usage["completion_tokens"] * price_out / 1_000_000
    return total
```

Two prices, because tokens in and tokens out cost differently, and a sum over every call.""",
    )
    return


@app.cell(hide_code=True)
def text_batch_choice():
    mo.md(r"""
    ### Exercise 3. A job for a batch
    """)
    return


@app.cell(hide_code=True)
def choice_batch_cell():
    choice_batch = L.choice(
        "batch",
        question="Which of these jobs fits a batch best?",
        options={
            "reviews": "Summaries of 5,000 product reviews, for a report next week",
            "chat": "The answers of a chat, while a person waits for them",
            "now": "One question that you want answered right now",
        },
        correct="reviews",
        why={
            "reviews": "Many requests, and nobody waits for them: they can come in hours later, maybe at a lower price, and the rate limit does not get in the way.",
            "chat": "A batch answers within hours. Can the person in the chat wait that long?",
            "now": "One request gains nothing from a batch, and the answer would come hours later. Which job has many requests, and time?",
        },
    )
    choice_batch
    return (choice_batch,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The key goes along with the same request as in [Calling a model over HTTP](/course/open/build-call-api?lang=en), and the rate limit of [When a call fails: errors, limits and retries](/course/open/build-errors-retries?lang=en) counts per key. The tokens that `usage` counts are the ones of [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en). The bill also depends on which model you ask: the next lesson, [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en), sends easy questions to a cheaper model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_batch, result_bill, result_key):
    L.wrap_up(result_key, result_bill, choice_batch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
