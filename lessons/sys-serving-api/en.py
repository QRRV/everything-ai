# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-serving-api. Route: course/route/lessen/systemen.yaml. Checks, the course's demand model, the server and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-serving-api", "en")
    C = L.checks   # the course's demand model, the requests of the lesson, the server and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Batch, online, streaming or on the device](/course/open/sys-deployment-patterns?lang=en) and [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_online():
    quick_online = L.choice(
        "quick-online",
        question="With online inference, when does the model make its prediction?",
        options={
            "night": "Once a night, for every member at once, stored for the app",
            "right": "The moment a request comes in, from the input of that moment",
            "event": "After every event of a stream, without anyone asking for it",
            "phone": "Only on the user's own phone, so that no network is needed at all",
        },
        correct="right",
        why={
            "right": "Online inference makes the prediction when a request comes in, while someone waits for it. This lesson builds the part that waits: a server that takes each request and answers it.",
            "night": "Everything in advance, once a night, is batch inference. When does online inference make its prediction? See [Batch, online, streaming or on the device](/course/open/sys-deployment-patterns?lang=en).",
            "event": "A prediction after every event, without a request, is streaming inference. What sets online inference off? See [Batch, online, streaming or on the device](/course/open/sys-deployment-patterns?lang=en).",
            "phone": "On the phone answers *where* the model runs, not *when*. When does online inference make its prediction? See [Batch, online, streaming or on the device](/course/open/sys-deployment-patterns?lang=en).",
        },
    )
    quick_online
    return (quick_online,)


@app.cell(hide_code=True)
def quick_check_package():
    quick_package = L.choice(
        "quick-package",
        question="A server loads a model once, when it starts. What does it need besides model.safetensors?",
        options={
            "nothing": "Nothing more: the weights file holds the whole network",
            "right": "config.json, with the class of the network and the scaling of the input",
            "data": "The training data, to look up the hours most like each request",
            "notebook": "The notebook the model was trained in, to run all of its cells again at every start",
        },
        correct="right",
        why={
            "right": "The weights alone do not say which class to build, or how an input is scaled: config.json does, with the class and its settings, and the mean and standard deviation of every feature. A server reads both once, when it starts, and keeps the model in memory; each request then only runs it.",
            "nothing": "A safetensors file holds tensors by name, each with its shape. Does it say which class to build, or how to scale an input? See [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).",
            "data": "The model learned from the data. To answer, does it still need the data itself? See [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).",
            "notebook": "A package exists so that nobody has to train the model again. What does it hold beside the weights? See [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en).",
        },
    )
    quick_package
    return (quick_package,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A trained model in a folder is of use to one kind of program: a Python program that loads it. But the programs that need its predictions are often others. At a bike-share company such as Capital Bikeshare in Washington, D.C., a planner for the trucks wants to know how many bikes will be rented at 17:00, a dashboard wants the numbers for the whole day, and the app wants one for its map. Other teams wrote them, perhaps in other languages, and they run on other computers. They cannot load a PyTorch model, and they should not have to.

    The usual answer: put the model behind an API. A small program, a server, keeps the model loaded and waits. Any program sends it a request over HTTP, the language of the web, with the input as JSON; the server checks the request, runs the model, and sends the answer back as JSON, with a number that says how it went. The other programs only need to know what to send, and what comes back.

    **An everyday example.** At a town hall, you hand in a filled-in form at the right counter. A clerk checks it first: with a field left empty, the form comes back with a note that says which one; at the wrong counter, you are sent to another; with a good form, you get your answer, without knowing how the town hall works inside. The form and the counters are the agreement between you and the town hall, and every visitor uses the same ones.
    """)
    return


@app.cell(hide_code=True)
def predict_midnight():
    guess_midnight = L.guess(
        "midnight",
        question="A model predicts how many bikes are rented in an hour, from the hour and 10 features of the day. A colleague's app writes midnight as hour 24 instead of 0, and sends it straight to the model, without a check. What happens, do you think?",
        options={
            "next": "The model answers for 0:00, the start of the next day",
            "zero": "The model answers 0 bikes: it never saw hour 24",
            "last": "The model answers for 23:00, the last hour it knows",
            "right": "The code of the model stops with an error",
        },
        reveal="**The code of the model stops with an error.** The hour goes into the model as 24 one-hot columns, one for each hour from 0 to 23, as in [One-hot vectors for classes](/course/open/ml-one-hot?lang=en). Hour 24 has no column, so PyTorch's `one_hot` stops with *Class values must be smaller than num_classes*. A server that passes such a request on to the model answers with status 500, *Internal Server Error*: it says that the server broke, while it was the request that was wrong. A missing field, or the word \"warm\" as a temperature, stops the code too. In this lesson, the server checks every request before the model sees it.",
    )
    guess_midnight
    return (guess_midnight,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Model serving.** **Model serving** is running a model as a service: a program keeps the model loaded and answers requests for predictions, one after another, for any program that asks. It is how a model does online inference, from [Batch, online, streaming or on the device](/course/open/sys-deployment-patterns?lang=en): a prediction at the moment of each request.

        **Web server.** That program is a **web server**: it waits for requests and answers each one. It listens on a *port*, a number that tells the programs on one computer apart, such as 8000. The address 127.0.0.1, also called localhost, is the computer itself: a server that listens there takes requests from programs on the same computer only.

        **HTTP request.** An **HTTP request** has a method, such as GET to fetch something or POST to send something; a path, such as /predict, that says what the request is for; headers, short lines such as Content-Type: application/json; and a body with the data: here, the input of the model as JSON. A path that a server answers, with its method, is an **endpoint**: POST /predict for a prediction, and GET /health to see that the server is up. The first line of a request holds the method, the path and the version of HTTP, such as `HTTP/1.1`; the server of this lesson answers in the older `HTTP/1.0` and closes the connection after each answer.

        **Status code.** Every answer starts with an **HTTP status code**: a number of three digits that says how it went, before the client reads the body. The first digit says whose turn it is: a 2 for success, a 4 for a request the client must change, a 5 for a server that broke on it, where the same request may work later.

        | Code | Name | Means |
        |---|---|---|
        | 200 | OK | it went well, and the body holds the answer |
        | 400 | Bad Request | the request is wrong: a field is missing, or a value cannot be right |
        | 404 | Not Found | the server has no such path |
        | 405 | Method Not Allowed | the path exists, but not with this method, such as a GET to /predict |
        | 500 | Internal Server Error | the server broke on the request, for instance because its code stopped with an error |

        **API.** The endpoints together, with what each takes and what it gives back, are the **API** of the service: the agreement between the server and every program that uses it. A client that keeps to it needs to know nothing about PyTorch, or about the model.

        **The handler.** On the server, one function gets every request and decides the answer: the handler. It looks at the path and the method, checks the input, runs the model, and turns its output into values JSON can hold. In this lesson, you write the handler, `handle`, and the course runs it in a real server on your own computer: `ThreadingHTTPServer` from Python's `http.server`, on 127.0.0.1. Behind it is a small neural network that predicts, for one hour, how many bikes Capital Bikeshare rents, from the hour and 10 features of the day and the weather: the model of *Project: a demand model, tracked and packed*, which the course trains here again.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **At scale.** **FastAPI** is a Python library that does the work around the handler. A decorator, `@app.post("/predict")` above a function, makes it an endpoint. FastAPI reads the JSON, checks it against the fields you declare, answers 422 when a field is missing or of the wrong kind, a 4xx code that is more specific than 400, and writes what the function gives back as JSON. Python's documentation warns that `http.server` is not meant for production: it is fine on your own computer, and a service for real users runs on a framework such as FastAPI. Between the programs of one company, gRPC sends requests as compact bytes instead of JSON text. And NVIDIA's Triton Inference Server serves many models at once on graphics cards, and puts requests that come in close together into one batch. The idea stays the same: a request comes in, gets checked, goes through the model, and an answer goes out.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **model serving** | | running a model as a service that answers requests for predictions | |
        | **web server** | | a program that waits for requests over HTTP, on a port, and answers each one | `ThreadingHTTPServer` |
        | **HTTP request** | | a message to a server: a method, a path, headers and a body | `POST /predict` |
        | **endpoint** | | a path a server answers, with its method | `GET /health` |
        | **HTTP status code** | | the number of three digits at the start of an answer that says how it went | `200`, `404` |
        | **API** | A P I | the endpoints of a service, with what each takes and gives back | |
        | **FastAPI** | fast A P I | a Python library to build an API, in which a decorator makes a function an endpoint | `@app.post("/predict")` |
        """),
        L.watch_out("Status 200 does not mean that the prediction is right: it means that the server could answer. A temperature of 26.2, in degrees, where the data has 0.64, gets status 200 too, with a number that is far off. What has to happen to an input before the model sees it is the subject of the next lesson."),
        L.watch_out("A path has to match letter for letter: /Predict, with a capital P, or /predict/, with a slash at the end, is another path, and gets 404 unless the server says otherwise."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Putting a model behind an API:

    1. Load the model once, when the server starts, and keep it in memory: not again at every request.
    2. Choose the endpoints, with a method and a path for each: POST /predict for a prediction, GET /health to see that the server is up.
    3. Check every request before the model sees it: a missing field, a value that is not a number, an hour the model does not know. Answer such a request with 400 and what is wrong.
    4. Run the model, and turn its output into values JSON can hold: a whole number of bikes, not a tensor.
    5. Send the status code with the answer as JSON, and keep the server on 127.0.0.1 while you build.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.start(C.handle)` starts the course's server on 127.0.0.1, on a free port, with the course's own `handle`. `C.send(server, method, path, body)` sends it a request, as any client would, with `body` as JSON, and gives back the status code and the JSON of the answer. `C.DAY[8]` is the body for 8:00 on Thursday 14 July 2011, a clear summer day: the hour and the 10 features of that hour, as the data has them. The cell starts the server with `with`, sends three requests, and the server stops again when the lines under `with` are done, even when one of them stops with an error. Run the cell.
    """)
    return


@app.cell
def example_server():
    with C.start(C.handle) as _server:
        print("listens on", _server.address, "port", _server.port)
        print(C.send(_server, "GET", "/health"))
        print(C.send(_server, "POST", "/predict", C.DAY[8]))
        print(C.send(_server, "POST", "/forecast", C.DAY[8]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The server listens on 127.0.0.1, on a port the computer gave it, so the number differs each time. GET /health gets 200 and `{"status": "ok"}`. POST /predict with the 11 numbers of 8:00 gets 200 and `{"bikes": 431}`: on that morning, 486 bikes were rented. /forecast is not an endpoint of this server, so it gets 404, with what is wrong. All of it went over HTTP on your own computer, and the server is stopped again. In Exercise 1, you write the line that turns the output of the model into that 431.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_answer():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. From a tensor to JSON

        `answer(request)` gets the body of a request as a dict, with `hr` and the 10 features, and gives back the dict the server sends as JSON. `expect_one(request)` runs the course's model on that hour and gives `pred`: a tensor with no axes, holding the bikes the model expects. But json cannot write a tensor, and no truck can bring 431.2 bikes. Replace the `...` so that `bikes` is the number in `pred` as a plain Python number, rounded to a whole number of bikes.

        The check calls `answer(C.DAY[8])`, where the model expects 431.2 bikes. Then 10:00, 17:00 and 12:00 of the same day, and 17:00 on 7 December 2011, in light rain.
        """),
        L.task("Make `bikes` a whole number of bikes, as a plain Python number."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("answer", lines=4)
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
            "Step 4 of the recipe: turn the output into a value JSON can hold. `pred` is a tensor, and `round` only rounds a plain Python number: first take the number out of `pred`, then round it, and put it in `bikes`.",
            "In [PyTorch tensors](/course/open/torch-tensors?lang=en), a tensor with no axes gave its number as a Python number with one of its methods, for a `print` or a list of losses. Which method was it?",
            "`    bikes = round(pred.___)`",
        ],
        why="`bikes = round(pred.item())`: `pred.item()` takes the number out of the tensor as a plain Python float, 431.2 at 8:00, and `round` makes it the whole number 431. json can write that, and a program in any other language can read it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_handle():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A request the model cannot take

        `handle(method, path, request)` gets the method, the path and the body of a request, and gives back two things: the status code and the dict the server sends as JSON. Its first lines answer `/health`, give 404 for a path the server does not have, and 405 for a method other than POST. `problem(request)` gives `""` for a good request, and otherwise a short text that says what is wrong: `trouble`. `answer(request)` is the course's answer of Exercise 1. Replace the `...` with two lines: when `trouble` is not empty, give back the status code for a bad request and `{"error": trouble}`, so that the model never sees the request.

        The check calls `handle("POST", "/predict", C.HOUR_24)`: the body of midnight, written as hour 24. Then a good request, one without temp, one with "warm" as temp, a GET to /predict, a POST to /forecast and a GET to /health.
        """),
        L.task("When `trouble` is not empty, answer with an error before the model sees the request."),
    ])
    return


@app.cell(hide_code=True)
def exercise_handle():
    box_handle = L.code_box("handle", lines=11)
    box_handle
    return (box_handle,)


@app.cell(hide_code=True)
def check_handle(box_handle):
    result_handle = L.check("handle", box_handle.value)
    result_handle
    return (result_handle,)


@app.cell(hide_code=True)
def hints_handle():
    L.hints(
        "handle",
        [
            "Step 3 of the recipe. `trouble` is `\"\"` for a good request. For any other text, `handle` gives back at once, with a status code and `{\"error\": trouble}`, and never gets to its last line.",
            "The first digit of a status code says whose turn it is: a 4 for a request the client must change, a 5 for a server that broke. Which code in the table of the idea says: bad request?",
            "`        return ___, {\"error\": trouble}`",
        ],
        why="`if trouble != \"\":` and then `return 400, {\"error\": trouble}`: a bad request gets 400, with what is wrong, and the model never sees it. Hour 24 gets *hr must be a whole number from 0 to 23, not 24*, and the client knows what to change.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_codes():
    mo.md(r"""
    ### Exercise 3. Which status code?

    Eight requests to the server of this lesson, whose endpoints are POST /predict and GET /health. Put each under the status code its answer should have.
    """)
    return


@app.cell(hide_code=True)
def sort_codes_cell():
    sort_codes = L.sort("codes")
    sort_codes
    return (sort_codes,)


@app.cell(hide_code=True)
def check_codes(sort_codes):
    result_codes = L.check("codes", sort_codes)
    result_codes
    return (result_codes,)


@app.cell(hide_code=True)
def hints_codes():
    L.hints(
        "codes",
        [
            "Step 3 of the recipe, and the table of the idea. Ask for each request: did it go well, is the body wrong, is the path wrong, or did the server break on a good request?",
            "Two requests go well, and two have a body the model cannot use. Which two ask for a path the server does not have?",
            "200: rush hour and \\_\\_\\_. 400: warm and \\_\\_\\_. 404: forecast and \\_\\_\\_. 500: a bug and \\_\\_\\_.",
        ],
        why="**200:** rush hour and /health, since nothing is wrong. **400:** \"warm\" and no humidity, since the client has to change the request. **404:** /forecast and /Predict, since the server has no such path, letter for letter. **500:** the bug and the full memory, since the request was good, and the server broke on it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_reach():
    mo.md(r"""
    ### Exercise 4. Who can reach it?

    Your server listens on 127.0.0.1, port 8000. On your Wi-Fi, your computer also has the address 192.168.1.20, and a colleague on the same Wi-Fi sends a request to 192.168.1.20, port 8000.
    """)
    return


@app.cell(hide_code=True)
def choice_reach_cell():
    choice_reach = L.choice(
        "reach",
        question="What happens?",
        options={
            "answer": "The colleague gets the bikes: both computers are on the same Wi-Fi",
            "right": "Nothing answers: 127.0.0.1 takes requests from your own computer only",
            "404": "The colleague gets status 404: the endpoint exists only on 127.0.0.1",
            "router": "The colleague gets the bikes, a little later than you, through the router",
        },
        correct="right",
        why={
            "right": "A server on 127.0.0.1 is out of reach for every other computer, even on the same Wi-Fi: no program at 192.168.1.20, port 8000, takes the request, so the colleague's program reports that the connection was refused, and no HTTP answer comes at all. That is what you want while you build. A server for others listens on the address of the network, and then needs more: a key for each client, encryption with HTTPS, and a limit on requests.",
            "answer": "The server listens on 127.0.0.1 only. Which computers can send a request to that address?",
            "404": "A 404 comes from a server that got the request. Does any server get this one?",
            "router": "The router brings the request to 192.168.1.20. Does the server listen on that address?",
        },
    )
    choice_reach
    return (choice_reach,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your server, on your own computer

    The demo starts a real server on 127.0.0.1, with your `handle` once Exercise 2 is right, and until then with the course's. Choose a method, type a path, and change the JSON of the body: the server gets your request over HTTP, and the picture shows what it got, what it sent back, and how long the client waited. It all stays on your computer, and the server stops again after each request.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_method = mo.ui.dropdown(options=["POST", "GET"], value="POST", label="Method")
    demo_path = mo.ui.text(value="/predict", label="Path")
    demo_body = mo.ui.text_area(value=C.DEMO_BODY, label="Body (JSON)", rows=7, full_width=True)
    mo.vstack([
        mo.hstack([demo_method, demo_path], justify="start"),
        demo_body,
        L.task("Ask for hour 17, leave out temp, write \"warm\" as temp, send a GET, and try hour 4 and a temp of 26.2. What does each get?"),
    ])
    return demo_body, demo_method, demo_path


@app.cell(hide_code=True)
def demo_send(box_handle, demo_body, demo_method, demo_path, result_handle):
    _fn = C.mine(box_handle.value, result_handle.state == "correct")
    mo.Html(C.exchange_picture(demo_method.value, demo_path.value, demo_body.value, _fn, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Hour 17 gets 497 bikes: the rest of the body still describes 8:00, and with the weather of 17:00 itself the model expects 537; 595 were rented. Without temp, or with "warm", the server answers 400 with what is wrong, and the model never sees the request; a GET to /predict gets 405. Hour 4 gets −3 bikes: the server passes on what the model says, even a number below 0. And a temp of 26.2, in degrees instead of the 0.64 the data uses, gets status 200 and 4,855 bikes instead of 431: the request was well formed, so nothing stopped it.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Batch, online, streaming or on the device](/course/open/sys-deployment-patterns?lang=en), online inference made a prediction at the moment of each request. This lesson built that moment: a server with endpoints, a check before the model, and an answer in JSON with a status code, as FastAPI and Triton do at scale. A real service loads a package such as the one of [Packing a model: weights, settings and a safe format](/course/open/sys-model-package?lang=en) once, when it starts. In the next lesson, *Before and after the model: the serving path*, the input gets the same preparation as in training, and the output becomes a decision: no answer below 0 bikes, and no 4,855 for a temperature in degrees.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_reach, result_answer, result_codes, result_handle):
    L.wrap_up(result_answer, result_handle, result_codes, choice_reach)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
