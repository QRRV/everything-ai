# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-project-rag. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-project-rag", "en")
    C = L.checks   # the course's runs about the handbook of Spoke & Chain, the stand-ins, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Project: your assistant, prompted well](/course/open/build-project-prompts?lang=en), and one about [Answers from the sources](/course/open/build-grounded-answers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="Your assistant should use another model on the server. Which of its files do you change?",
        options={
            "prompts": "`app/prompts.py`, where the words of your assistant are",
            "client": "`app/client.py`, which talks to the server",
            "answers": "`app/answers.py`, from where `ask_json` asks",
        },
        correct="client",
        why={
            "client": "Your `chat` writes the name of the model in every request; the other files only hand it messages.",
            "prompts": "`app/prompts.py` holds the words: `SYSTEM` and `TEMPLATES`. Which file builds the request that names the model? See [Project: your assistant, prompted well](/course/open/build-project-prompts?lang=en).",
            "answers": "`ask_json` asks through `chat`, and never names a model. See [Project: your assistant, prompted well](/course/open/build-project-prompts?lang=en).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def quick_check_grounded():
    quick_grounded = L.choice(
        "quick-grounded",
        question="Besides answering only from the numbered sources, what does the grounded prompt ask of the model?",
        options={
            "copy": "To copy the source word for word, so that nothing is made up",
            "all": "To use every source at least once, so that none is left out",
            "cite": "To name the number of its source, or to answer I don't know",
        },
        correct="cite",
        why={
            "cite": "The number lets your code find the source the answer comes from; I don't know is the answer when no source says it.",
            "copy": "It asks for one sentence, in the model's own words. See [Answers from the sources](/course/open/build-grounded-answers?lang=en).",
            "all": "Most questions are answered by one source; the others are only there because the search found them. See [Answers from the sources](/course/open/build-grounded-answers?lang=en).",
        },
    )
    quick_grounded
    return (quick_grounded,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your assistant knows what Qwen3 learned in training, and nothing about your own documents: the handbook of a shop, your notes, the rules of your club. Since [Project: your assistant, prompted well](/course/open/build-project-prompts?lang=en) you have built everything it takes to change that: sources in front of the question in [Look it up first: what RAG is](/course/open/build-rag-idea?lang=en), chunks in [Cutting documents into pieces](/course/open/build-chunking?lang=en), embeddings in [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en), a search in [Search by meaning](/course/open/build-retrieval-embeddings?lang=en), and a prompt that keeps the model to its sources and names them in [Answers from the sources](/course/open/build-grounded-answers?lang=en). Today it comes together in your project: your assistant reads a folder of documents, answers from them, and says which part of which document the answer comes from.

        **An everyday example.** A new colleague at a help desk does not know the rules by heart. For each question, they look up the right page in the binder, answer from it, and say which page it was, so that the customer can read it too. Your assistant gets the same: the library is the binder with its index, the search finds the pages, and the answer names its page. That is only a comparison; below you see what Qwen3 does with it.
        """),
        draw.project_steps(["Your assistant answers", "Your assistant, prompted well", "Your assistant reads your documents"], 2,
                           here_label="← today", title="Where your project stands: your assistant"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_without():
    _f = C.findings()
    guess_without = L.guess(
        "without",
        question="Spoke & Chain is a bike shop the course made up, with a handbook of {words} words. The course asked Qwen3-0.6B {k} questions the handbook answers, twice each, without the handbook: only a system prompt that says it answers questions about the shop. How many of the {runs} answers were right?".format(
            words=_f["words"], k=_f["questions"], runs=_f["runs"]),
        options={
            "none": "None of them",
            "some": "About a quarter: the prices of bike shops are much alike",
            "most": "Most of them: a model has read a lot about bike shops",
        },
        reveal="**None: {plain} of {runs}.** Qwen3 made up prices in dollars, such as a deposit of *around $25 to $50*, and for the two questions the handbook does not answer, a phone number, *555-1234*, and a street. Not once did it say that it did not know. No wonder: the shop and its handbook are the course's own, so they were in nothing Qwen3 learned from.".format(
            plain=_f["plain"], runs=_f["runs"]),
    )
    guess_without
    return (guess_without,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### Two new files, on top of six you made

        After today, your package `app` has two files more, each with one job:

        - `app/library.py` has `make_library(url, chunks, about)`. It embeds the chunks of your documents with your `embed`, and gives back your library: a dictionary with `"about"`, a few words on what the documents are about; `"chunks"`, the chunks as texts; and `"vectors"`, a NumPy array with one row per chunk, so that row `i` is the embedding of `chunks[i]`.
        - `app/rag.py` has `ask(url, library, question, k=3)`. It embeds the question, finds the `k` chunks with your `meaning_search`, asks with the grounded prompt of [Answers from the sources](/course/open/build-grounded-answers?lang=en) and the sources from your `with_sources`, through your `chat`, and keeps the chunks the answer names with your `cited`. It gives back the answer, the chunks it found as `"sources"`, and the ones the answer names as `"named"`.

        The other six come from earlier lessons, whose checks kept your code as a file. Where one is missing or broken, the course's version stands in, and the note at *Where your project stands* says which.
        """),
        mo.Html(C.files_picture(L.lang)),
        mo.md(r"""
        The grounded prompt is a template with one field, `{{about}}`: the course's prompt was about the course, and yours is about whatever your library says it holds. And `make_library` does not send all chunks in one request. The course's server takes at most 2,048 texts in one request, like OpenAI's, and answers more with status 400, the number a server sends back for a request it refuses; a folder with a few books has more chunks than that. So it sends `BATCH` chunks at a time, 64, as the course did when it embedded the 1,266 points of its summaries: {seconds} seconds in all, on the course's computer.
        """.format(seconds=C.number(C.batch_seconds(), L.lang, 1))),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: your assistant reads your documents

    1. **Cut:** each document into chunks with your `chunk` of [Cutting documents into pieces](/course/open/build-chunking?lang=en): 80 words that overlap by 20.
    2. **The library:** `make_library(url, chunks, about)` sends the chunks `BATCH` at a time to `embed`, adds the embeddings of each batch to one list, and gives back `"about"`, `"chunks"` and the embeddings as an array, `"vectors"`.
    3. **Search:** `ask(url, library, question, k=3)` embeds the question, and finds the `k` chunks whose embeddings have the highest dot product with it.
    4. **Ask:** two messages through `chat`: the system prompt `GROUNDED`, filled in with the library's `"about"`, and the question with the chunks as numbered sources, from `with_sources`.
    5. **Name:** `cited` finds the numbers in the answer; each number `n` from 1 to the number of sources is the chunk `passages[n - 1]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    _f = C.findings()
    mo.md(r"""
    ### Worked example

    The course's run with seed 0, on the handbook of Spoke & Chain: `ask(url, library, "{question}")`.

    1. **Cut:** the handbook, {words} words, becomes {chunks} chunks of 80 words, the last of {last}.
    2. **The library:** {chunks} chunks fit in one request, so `make_library` sends one, and gets {chunks} embeddings back.
    3. **Search:** the chunk with the highest dot product starts with *{start}*, and holds *cash up to 500 euros*. It becomes source 1; sources 2 and 3 are the two chunks just before it in the handbook.
    4. **Ask:** the system prompt says the model answers questions about the bike shop Spoke & Chain, using only the numbered sources; the user's message holds the three chunks, and then *Question: {question}*
    5. **Name:** Qwen3 answered *{answer}* `cited` finds `{cited}`, so `named` holds source 1: the chunk with the answer, which your app can show next to it. Without the handbook, Qwen3 answered *{plain}*
    """.format(question=_w["question"], words=_f["words"], chunks=_f["chunks"], last=len(C.CHUNKS[-1].split()),
               start=" ".join(_w["passages"][0].split()[:12]) + " …", answer=_w["answer"], cited=_w["cited"], plain=_w["plain"]))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        ### What the course measured

        The course asked all {k} questions about the handbook, and {none} it does not answer, twice each: once with only the system prompt about the shop, and once as your `ask` does, with the library of the handbook.
        """.format(k=_f["questions"], none=_f["none"])),
        mo.Html(C.runs_picture(L.lang)),
        mo.md(r"""
        With your `ask`, {yours} of {runs} answers were right, against {plain} without the handbook, and to the questions the handbook does not answer, it said *I don't know* in {yours_none} of {none_runs} runs. Every right answer named a chunk that holds the answer.

        Where it went wrong tells you more. Search by meaning found a chunk with the answer for {hits} of the {k} questions. It missed the chunk with the full service for both questions about it, also for *How much does a full service cost?*, and then the model did not say *I don't know*: it took another price from the chunks it got, such as *120 euros*, the service of electric bikes. Search by words, the BM25 of [Search by words: BM25](/course/open/build-bm25?lang=en), does put the chunk with the full service first for that question. Where the search did find the chunk, {right_found} of {runs_found} answers were right: the model mixed up numbers inside one chunk, such as *21:00* of the repair evening for the closing time on Saturday, and said *I don't know* {refused} times although the chunk held the answer. The {other_named} wrong answers there all named the chunk that holds the right answer, so a reader who opens the named source can see the mistake.
        """.format(yours=_f["yours"], runs=_f["runs"], plain=_f["plain"], yours_none=_f["yours_none"], none_runs=_f["none_runs"],
                   hits=_f["hits"], k=_f["questions"], right_found=_f["right_found"], runs_found=_f["runs_found"],
                   refused=_f["refused"], other_named=_f["other_named"])),
    ])
    return


@app.cell(hide_code=True)
def text_library():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your library

        `make_library(url, chunks, about)` should give back your library, with the embeddings of all chunks in their order. The box sends all chunks in one request. Send them `BATCH` at a time instead, one request per batch, and add the embeddings of each batch to one list. The check calls it against a stand-in of the server, which answers with made-up embeddings, with 150, 64 and 65 chunks. When it is right, the check saves it in your project, as `work/projects/app/library.py`.
        """),
        L.task("Send the chunks `BATCH` at a time, and add the embeddings of each batch to `vectors`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_library():
    box_library = L.code_box("make_library", lines=13)
    box_library
    return (box_library,)


@app.cell(hide_code=True)
def check_library(box_library):
    result_library = L.check("make_library", box_library.value)
    result_library
    return (result_library,)


@app.cell(hide_code=True)
def hints_library():
    L.hints(
        "make_library",
        [
            "Start with an empty list, and fill it in a loop: one request per batch of `BATCH` chunks.",
            "`for start in range(0, len(chunks), BATCH):` gives 0, 64, 128, …; the batch is `chunks[start:start + BATCH]`.",
            "In the loop: `vectors = vectors + embed(url, ...)`, with the batch as the texts. `+` joins two lists into one: `[a, b] + [c]` is `[a, b, c]`.",
        ],
        why="""```python
def make_library(url, chunks, about):
    vectors = []
    for start in range(0, len(chunks), BATCH):
        vectors = vectors + embed(url, chunks[start:start + BATCH])
    return {"about": about, "chunks": chunks, "vectors": np.array(vectors)}
```

The last slice may be shorter: `chunks[128:192]` of 150 chunks has 22.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your assistant answers from the library

        `ask(url, library, question, k=3)` already embeds the question and finds the passages. But it sends the model the question alone, and names no source. Make the two messages: the system prompt `GROUNDED` filled in with `library["about"]`, and as the user's, what `with_sources` makes of the question and the passages. Then keep in `named` the passages the answer cites: for each number `n` that `cited(answer)` finds, `passages[n - 1]`, but only when `n` is from 1 to `len(passages)`. The check calls it against a stand-in that answers with a fixed text, first with the library of the handbook and the question about cash. When it is right, the check saves it in your project, as `work/projects/app/rag.py`.
        """),
        L.task("Send the system prompt and the question with its sources through `chat`, and keep in `named` the passages the answer cites."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=24)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask",
        [
            "It is the request of [Answers from the sources](/course/open/build-grounded-answers?lang=en), with `{about}` filled in, and then a loop over the numbers the answer cites.",
            "Two messages in a list: first the system prompt, `GROUNDED` with `.format` and `library[\"about\"]` for its field, then the user's, which `with_sources(question, passages)` makes. Send the list with `chat`.",
            "After `answer = chat(url, messages)`: `for n in cited(answer):`, and `named.append(passages[n - 1])` when `n >= 1 and n <= len(passages)`.",
        ],
        why="""```python
    messages = [{"role": "system", "content": GROUNDED.format(about=library["about"])},
                {"role": "user", "content": with_sources(question, passages)}]
    answer = chat(url, messages)
    named = []
    for n in cited(answer):
        if n >= 1 and n <= len(passages):
            named.append(passages[n - 1])
```

An answer can cite a number that is no source, such as `[4]` with three sources: `passages[3]` would stop with an IndexError, and `passages[-1]`, for `[0]`, would be the last source: a negative index counts from the end.""",
    )
    return


@app.cell(hide_code=True)
def text_new_document():
    mo.md(r"""
    ### Exercise 3. A new document
    """)
    return


@app.cell(hide_code=True)
def choice_new_document_cell():
    choice_new_document = L.choice(
        "new-document",
        question="You put a new document in the folder of your assistant. What has to be embedded before `ask` can find its chunks?",
        options={
            "new": "Only the chunks of the new document",
            "all": "All chunks again, because every embedding depends on the others",
            "nothing": "Nothing: `ask` embeds the question, which is enough to find them",
        },
        correct="new",
        why={
            "new": "The embedding of a chunk depends on its own text only, so the old rows stay as they are: add the new chunks and their rows. The demo below simply makes the whole library again when a document changes, which is fine for a few documents.",
            "all": "Does the embedding of one chunk change when another chunk is added? Look at what `embed` gets: a list of texts, and one embedding per text back.",
            "nothing": "The search compares the question with the rows of the library. A chunk without a row is never compared, so how could it be found?",
        },
    )
    choice_new_document
    return (choice_new_document,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### What your project can do now

    The cell takes `make_library` and `ask` from your project, and reads the documents in `work/projects/app/documents/`: the files that end in `.md` or `.txt`. The first time, the course puts its handbook of Spoke & Chain there. Add your own files, or take the handbook out. The cell cuts each document with your `chunk`, makes the library with your `make_library`, and keeps it until the documents change. Then your `ask` asks Qwen3 your question. Everything runs on your own computer: your documents go to no one. Write in the box what your documents are about, in English, because it goes into the system prompt.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B and Granite Embedding (1.7 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The demo needs two models that are not both on your computer yet: the chat model Qwen3-0.6B, and Granite Embedding, the embedding model. The button downloads what is missing once from Hugging Face, at fixed versions: `Qwen/Qwen3-0.6B`, 1.5 GB, with the tokenizer of Qwen3-0.6B-Base, and `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, both under the Apache-2.0 license.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B and Granite Embedding are downloaded.*")
    return


@app.cell(hide_code=True)
def question_cell():
    about_box = mo.ui.text(value=C.ABOUT, label="What your documents are about, in English", full_width=True)
    question_box = mo.ui.text_area(value="Up to what amount can I pay in cash?", label="Your question", full_width=True, rows=2)
    ask_button = mo.ui.run_button(label="Ask your assistant")
    mo.vstack([about_box, question_box, ask_button])
    return about_box, ask_button, question_box


@app.cell(hide_code=True)
def demo_answer(about_box, ask_button, question_box, result_ask, result_library):
    _docs = C.documents()
    _names = ", ".join("`{name}` ({n} words)".format(name=_name, n=len(_text.split())) for _name, _text in _docs)
    _folder = mo.md("*In `work/projects/app/documents/`: {names}.*".format(names=_names) if _docs else
                    "*There are no `.md` or `.txt` files in `work/projects/app/documents/`. Put a document there, and press the button again.*")
    if not C.downloaded():
        _out = mo.md("*Download the two models first, with the button above.*")
    elif result_library.state != "correct" or result_ask.state != "correct":
        _out = mo.md("*Finish exercises 1 and 2 first: then your `make_library` and `ask` are in your project, and this cell uses them.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    elif not _docs:
        _out = mo.md("")
    else:
        _make, _why = C.learner("library", "make_library")
        _ask, _why_ask = C.learner("rag", "ask")
        _chunk, _why_chunk = C.learner("chunks", "chunk")
        if _make is None or _ask is None or _chunk is None:
            _out = mo.md("*Your project does not run as it is now: {why}*".format(why=_why or _why_ask or _why_chunk))
        else:
            _words = {"title": "What your assistant answered: {question}", "embedding": "Your make_library embeds {n} chunks of {d} documents …",
                      "asking": "Your ask searches, and Qwen3 writes …",
                      "library": "Library: {n} chunks of {d} documents, made in {seconds} s; answered in {asked} s.",
                      "answer": "The answer:", "named": "It names: {names}", "none": "It names no source.",
                      "sources": "The sources it found, the ones it names marked:", "yours": "All modules were yours.",
                      "course": "The course's version stood in for: {files}.", "failed": "Your assistant stopped with an error against the real server: {why}",
                      "memory": "The memory of your device is full. Close other programs, and press the button again."}
            mo.output.replace(mo.md("*Your assistant reads your documents …*"))
            _out = mo.Html(C.demo(_make, _ask, _chunk, C.documents(), about_box.value.strip() or C.ABOUT, question_box.value.strip(),
                                  lambda _t: mo.output.replace(mo.md("*" + _t + "*")), _words, L.lang))
    mo.vstack([_folder, _out])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The sources of [Look it up first: what RAG is](/course/open/build-rag-idea?lang=en), the chunks of [Cutting documents into pieces](/course/open/build-chunking?lang=en), the embeddings of [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en), the search of [Search by meaning](/course/open/build-retrieval-embeddings?lang=en) and the grounded prompt of [Answers from the sources](/course/open/build-grounded-answers?lang=en) now work together in your project, through the client of [Project: your assistant answers](/course/open/build-project-call?lang=en). The measurements showed where it falls short: an answer can only be as good as the chunks the search finds. The next lessons of this module make the search better: faster with an index, with filters, by words and meaning together, and with a second look at the chunks it found; a later one measures a RAG system as a whole.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_document, result_ask, result_library):
    L.wrap_up(result_library, result_ask, choice_new_document)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
