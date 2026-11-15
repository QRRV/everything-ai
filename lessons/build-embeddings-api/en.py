# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-embeddings-api. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-embeddings-api", "en")
    C = L.checks   # the course's recorded runs of Granite Embedding, the stand-in of the checks, and the demo with your best_match


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Calling a model over HTTP](/course/open/build-call-api?lang=en), and one about [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_usage():
    quick_usage = L.choice(
        "quick-usage",
        question="An answer of the course's server says `\"usage\": {\"prompt_tokens\": 19, \"completion_tokens\": 13, \"total_tokens\": 32}`. What are the 19?",
        options={
            "sent": "The tokens your request sent",
            "written": "The tokens the model wrote",
            "words": "The words of your question",
        },
        correct="sent",
        why={
            "sent": "`prompt_tokens` counts what went in, `completion_tokens` what came out. An answer with embeddings has only the first: nothing is written.",
            "written": "Those are the 13 under `completion_tokens`. What goes in a prompt? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "words": "A server counts tokens, not words, and a token is often a part of a word. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_usage
    return (quick_usage,)


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Two embeddings have a cosine similarity of 0.98. What does that say?",
        options={
            "length": "They are almost the same length",
            "far": "They lie far apart",
            "same": "They point almost the same way",
        },
        correct="same",
        why={
            "same": "Cosine similarity looks at the direction only: 1 is exactly the same way, 0 at right angles, −1 the opposite way. Almost the same way means the model sees the two things as alike.",
            "length": "The cosine similarity divides the lengths away: a long and a short arrow in the same direction also give 1. See [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
            "far": "Close to 1 means close in direction. See [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A customer of a web shop types "How do I change my password?". The help page that answers it says "Je wachtwoord wijzig je onder Instellingen, bij Account." Not one word is the same, not even the language, so a search for words finds nothing. A search by meaning finds it at once. For that, every text needs an embedding, and an embeddings API gives you those: many texts in one request. And sometimes a question is about a whole document, a contract or a manual; then the document has to go along with the request.

    **An everyday example.** In a library the books stand by subject, not by the color of their cover: a book about bicycles stands among other books about bicycles, whatever its title says, and in whatever language it is written. An embedding gives every text such a place on a shelf. That is only a comparison; below you see the numbers.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **embedding model** | a model that writes no text: it turns a text into one embedding, always the same number of numbers, so that texts with the same meaning get embeddings that point the same way |
    | **embeddings API** | the endpoint `/v1/embeddings`: you send `"input"`, a text or a list of texts, and get back under `"data"` one `"embedding"` per text, in the same order, each with its `"index"` |
    | **file upload** | sending a file to a provider once, apart from a request: you get an id back, such as `file-abc123`, and later requests name the file by that id |

    ### Many texts, one request

    Next to Qwen3, the course's server has an embedding model: Granite Embedding 107M multilingual, of IBM, a download of 223 MB. It gives 384 numbers per text, and knows twelve languages, English and Dutch among them. The course sent it four questions of customers in English and the same four in Dutch, all eight in one request, and worked out the dot product of every English question with every Dutch one:
    """)
    return


@app.cell(hide_code=True)
def picture_pairs():
    mo.Html(C.pairs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_pairs():
    mo.vstack([
        mo.md(r"""
        Granite gives every embedding length 1, so the dot product is already the cosine similarity of [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en): there are no lengths to divide by. A question and its translation score 0.87 to 0.96; a question and another question 0.51 to 0.64. Not 0: with Granite, even "The sun is a star." scores 0.473 against the question about the password. So a score alone says little; what counts is which text scores highest.
        """),
        L.watch_out("Embeddings of two models do not fit together: each model has coordinates of its own. Switch to another embedding model, and you embed all your texts again. And a model reads only so many tokens of a text, Granite 512: the rest of a longer text is left out, or another API refuses it."),
    ])
    return


@app.cell(hide_code=True)
def text_documents():
    mo.md(r"""
    ### Documents along with a request

    A question about a contract, a manual or a report needs the document itself. It can go along in two ways.

    **As text in the message.** Your program reads the file, and puts its text in the prompt between delimiters, as in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en). Every chat API takes that, the course's too. The document's tokens count in every request, and it has to fit in the context window.

    **As a file upload.** Your program sends the file to the provider once, to its endpoint for files, and gets an id back. Each later request names the file by that id; in OpenAI's chat API, the content of the message is then a list, with the piece `{"type": "file", "file": {"file_id": "file-abc123"}}` next to the question. For a PDF, many providers read the text and also look at each page as an image, so tables and charts count too. OpenAI, Anthropic and Google each have such a files API (examples in November 2026). The course's server has none: an upload is something for a provider's servers.

    Keep in mind that an uploaded file stays at the provider until you delete it, or until the provider removes it after a set time.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: find the text that fits a question best

    1. **Put all texts in one list,** the question first: `[question] + texts`.
    2. **Ask for their embeddings:** a POST to `url + "/embeddings"` with the body `{"model": ..., "input": that_list}`, the way you send a chat request.
    3. **Take the embeddings out:** each item under `"data"` holds the `"embedding"` of one text, in the order you sent them.
    4. **Score every text:** the dot product of the question's embedding, `vectors[0]`, with the text's; for embeddings of length 1 that is the cosine similarity.
    5. **Take the text with the highest score.**
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course's question about the password, with two help pages: one in Dutch about the password, one in English about delivery. The request its code sent, and what came back:
    """)
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_worked_steps():
    mo.md(r"""
    1. **The input:** one list with three texts, so one request gives all three embeddings.
    2. **The answer:** under `"data"`, three items, with `"index"` 0, 1 and 2, in the order of the texts. Each embedding has 384 numbers; the picture shows the first four.
    3. **The tokens:** `"prompt_tokens": 35`: 9, 16 and 10 for the three texts. Nothing is written, so there are no `completion_tokens`.
    4. **The scores:** the question with the Dutch page 0.686, with the English page about delivery 0.489. The Dutch page wins, without one word of the question in it.
    5. **One request:** the question with all six help pages took 0.025 seconds in one request on the course's computer, and 0.072 seconds in 7 requests of one text each, 2.9 times as long. At a hosted API, every request also travels over the internet.
    """)
    return


@app.cell(hide_code=True)
def text_search():
    mo.md(r"""
    The course asked three questions this way against the six help pages of a made-up web shop, three pages in English and three in Dutch. Under each question, every page with its score:
    """)
    return


@app.cell(hide_code=True)
def picture_search():
    mo.Html(C.search_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_embed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Embeddings through the API

        `embed(url, texts)` sends a request and reads the answer already, as your `ask` did in [Calling a model over HTTP](/course/open/build-call-api?lang=en). Fill in the three `...`: the texts as the input, the endpoint for embeddings, and the list of embeddings, one per text. The check calls it with a stand-in of the server, which answers at once with made-up embeddings: 384 numbers per text, drawn at random, so they mean nothing. The picture under the check shows what your function sent, and what came back.
        """),
        L.task("Finish `embed`: send `texts` as the input to the endpoint for embeddings, and give back the list of their embeddings."),
    ])
    return


@app.cell(hide_code=True)
def exercise_embed():
    box_embed = L.code_box("embed", lines=11)
    box_embed
    return (box_embed,)


@app.cell(hide_code=True)
def check_embed(box_embed):
    result_embed = L.check("embed", box_embed.value)
    result_embed
    return (result_embed,)


@app.cell(hide_code=True)
def hints_embed():
    L.hints(
        "embed",
        [
            "Steps 2 and 3 of the recipe: what goes in, where it goes, and what you take out of the answer.",
            "The texts go under `\"input\"` as the list itself; the endpoint is `/embeddings`, after `url`. In the answer, each item under `\"data\"` has its numbers under `\"embedding\"`.",
            "`\"input\": texts`, `url + \"/embeddings\"`, and `return [item[...] for item in answer[...]]`, with the right keys in place of the dots.",
        ],
        why="""```python
def embed(url, texts):
    body = {"model": "granite-embedding-107m-multilingual", "input": texts}
    request = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return [item["embedding"] for item in answer["data"]]
```

One request gives the embeddings of all texts; the list comprehension takes the numbers out of each item, in the order of the texts.""",
    )
    return


@app.cell(hide_code=True)
def text_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The best match

        Now find the text that fits a question best, as in the recipe. `best_match(url, question, texts)` has the loop already. Fill in the two `...`: the embeddings of the question and all texts in one request, with your `embed` from exercise 1, and the score of `texts[i]`. The check calls it twice with a stand-in, whose embeddings for these texts are made so that each text scores about what Granite gives it. The picture under the check shows the scores, and the text your function picked.
        """),
        L.task("Finish `best_match`: get the embeddings of the question and the texts in one request, and score each text by its dot product with the question."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best_match", lines=11)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best, box_embed):
    result_best = L.check("best_match", box_best.value, C.best_inputs(box_embed.value))
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best_match",
        [
            "Steps 1 and 4 of the recipe: first one list with the question in front of the texts, then for each text the dot product with the question.",
            "`[question] + texts` is that list, and your `embed` gives its embeddings. `vectors[0]` is then the question's; at which place is the embedding of `texts[i]`?",
            "`vectors = embed(url, [question] + texts)`, and `score = np.dot(q, vectors[...])`, with the place of `texts[i]` in place of the dots.",
        ],
        why="""```python
def best_match(url, question, texts):
    vectors = embed(url, [question] + texts)
    q = np.array(vectors[0])
    best, best_score = None, -2
    for i in range(len(texts)):
        score = np.dot(q, vectors[i + 1])
        if score > best_score:
            best, best_score = texts[i], score
    return best
```

The question comes first, so the embedding of `texts[i]` is at place `i + 1`. Granite's embeddings have length 1, so the dot product is the cosine similarity; dividing by the two lengths, as in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en), picks the same text.""",
    )
    return


@app.cell(hide_code=True)
def text_no_clear():
    mo.md(r"""
    ### Exercise 3. No clear winner

    Look again at the course's third question, in the picture with the six help pages.
    """)
    return


@app.cell(hide_code=True)
def choice_no_clear_cell():
    choice_no_clear = L.choice(
        "no-clear-match",
        question="`Kan ik iemand bellen?` (Can I call someone?) got the page about returns as its best match. What do its six scores tell you?",
        options={
            "dutch": "Granite does not understand Dutch questions",
            "close": "No page stands out: the scores lie close together",
            "returns": "You return an order by phone, so the page about returns is the answer",
        },
        correct="close",
        why={
            "close": "The six scores lie close together, 0.522 to 0.600, so the best page is no sure answer. A clear match stands out, as the password page does with 0.686 against 0.519 for the next page. Here the best page beats the next by 0.015. A search can then show several pages, or say it found nothing. The page about the customer service says when it can be reached, not that you can call.",
            "dutch": "Look at the second question, in Dutch too: which page did it find, and in which language is that page written?",
            "returns": "Read the page about returns again: does it say anything about calling?",
        },
    )
    choice_no_clear
    return (choice_no_clear,)


@app.cell(hide_code=True)
def text_upload():
    mo.md(r"""
    ### Exercise 4. One PDF, forty questions
    """)
    return


@app.cell(hide_code=True)
def choice_upload_cell():
    choice_upload = L.choice(
        "upload",
        question="You ask a model 40 questions about one PDF of 30 pages, in 40 requests. You upload the PDF once, and name it by its id in each request. What do you pay for?",
        options={
            "each": "The PDF's tokens in every one of the 40 requests",
            "once": "The PDF's tokens once, at the upload, and after that only the questions",
            "nothing": "Nothing for the PDF: an uploaded file is free to use",
        },
        correct="each",
        why={
            "each": "The upload saves sending the file every time, not the reading: the model reads the document again in every request, and every request pays for its tokens. A later lesson shows how to pay less when every request starts with the same long text.",
            "once": "Does the model remember the PDF from one request to the next, or does it read it again to answer question 37?",
            "nothing": "Some providers keep a file for free for a while. But what happens when the model reads it, to answer a question?",
        },
    )
    choice_upload
    return (choice_upload,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Search the help pages with your own best_match

    Once exercises 1 and 2 are right, your functions can search the six help pages for real: the course starts its server on your computer, with Granite behind it. Ask a question in English, in Dutch, or in another of Granite's twelve languages, and see which page your `best_match` picks.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The demo needs Granite Embedding, which is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, under the Apache-2.0 license.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Granite Embedding is downloaded.*")
    return


@app.cell(hide_code=True)
def question_box_cell():
    question_box = mo.ui.text(value="Can I still cancel my order?", label="Your question", full_width=True)
    search_button = mo.ui.run_button(label="Search with your best_match")
    mo.vstack([question_box, search_button])
    return question_box, search_button


@app.cell(hide_code=True)
def demo_answer(box_best, box_embed, question_box, search_button):
    _match = C.learner_best_match(box_best.value, box_embed.value)
    if not C.downloaded():
        _out = mo.md("*Download Granite Embedding first, with the button above.*")
    elif _match is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo searches with your own `embed` and `best_match`.*")
    elif not search_button.value:
        _out = mo.md("*Press the button to search.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `best_match` has sent the question and the six pages; the server loads Granite and works out their embeddings …*"))
        _words = {"title": "What your best_match found",
                  "mark": "← your best_match picked this one",
                  "done": "{seconds} seconds, with loading Granite the first time.",
                  "failed": "Your best_match stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_match, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The request is the one of [Calling a model over HTTP](/course/open/build-call-api?lang=en), at another endpoint; the dot product and the cosine similarity are those of [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en). The next lesson, [Ready-made AI: vision, speech, translation and classic NLP](/course/open/build-ai-services?lang=en), looks at ready-made services for one task. Later, the module on RAG, answering from your own documents, searches the course's own summaries this way, with the same model, after cutting longer texts into pieces.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_no_clear, choice_upload, result_best, result_embed):
    L.wrap_up(result_embed, result_best, choice_no_clear, choice_upload)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
