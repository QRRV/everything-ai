# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-rag-first. Route: course/route/lessen/bouwen.yaml. Checks, the recorded runs and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-rag-first", "en")
    C = L.checks   # the knowledge base, the course's recorded runs of Qwen3, the checks, and the demo with your rag


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [System prompts and response prefill](/course/open/build-system-prompt?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="Your app has a system prompt and sends the user's questions to a model over HTTP. Where does the system prompt go?",
        options={
            "every": "In every request, as the first message, with the role system",
            "once": "In the first request only; after that, the server remembers it",
            "end": "At the end of the user's message, so that the model reads it last",
        },
        correct="every",
        why={
            "every": "The server remembers nothing between requests, so the system prompt goes in with every request, as the first message with the role system. That is also why it costs input tokens every time.",
            "once": "The server remembers nothing between requests: the system prompt goes in with every request, as the first message with the role system. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
            "end": "The system prompt is a message of its own, with the role system, at the start of the conversation, in every request. See [System prompts and response prefill](/course/open/build-system-prompt?lang=en).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Look it up first: what RAG is](/course/open/build-rag-idea?lang=en), the course handed Qwen3 the right source. An assistant has to find it itself: among thousands of texts, for questions nobody thought of in advance. In this lesson you build the whole thing, in three short functions: one that finds the words of a text, one that searches, and one that asks the model with what the search found.

    **An everyday example.** To find something in a thick handbook, you do not read it from the start: you look up a word of your question in the index at the back, and read the pages it points to. If you look up a word the book does not use, say *price* where the book says *cost*, you find nothing, even though the answer is in the book. That is only a comparison; below you see the same with a search that counts words.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **knowledge base** | the collection of texts an assistant can search through for its answers |
    | **RAG pipeline** | the steps from question to answer, one after the other: search the knowledge base, put the best texts in the prompt, ask the model |
    | `a & b` | the items that two sets `a` and `b` share, as a new set |

    The knowledge base of this lesson is the course itself: every lesson ends with a summary of a few points that say what it measured, and together they are {points} points from {lessons} lessons. Each point is a few sentences in English.

    The search counts, for every point, how many words it shares with the question, and keeps the points with the most. Before it counts, it makes every word lower case and takes the punctuation off its ends, so that *Cache?* and *cache* count as the same word. A set holds each word once, and `&` gives what two sets share: `{{"a", "cache", "the"}} & {{"the", "cache", "shop"}}` gives `{{"cache", "the"}}`, two words.
    """.format(points=C.number(len(C.POINTS), L.lang, 0), lessons=C.lesson_count()))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a first RAG

    1. **Words:** make the set of the words of the question, in lower case and without punctuation at their ends.
    2. **Search:** count for every point how many of those words it has, and keep the three with the most.
    3. **Prompt:** put those three in front of the question, numbered, after the system prompt.
    4. **Ask:** send it to the model, and give back its answer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Worked example

    *{question}*

    1. **Words:** the question has {n} words: {asked}.
    2. **Search:** a point of the lesson on prompt caching shares {gold} of them: {shared}. No other point shares more than {second}; of all {points} points, {many} share exactly {common} words with the question, mostly *a* and *the*, and {zero} share none. The three with the most share {top0}, {top1} and {top2} words.
    3. **Prompt:** the system prompt of this lesson, *{system}*, and then the three points with their numbers in front of the question, as `with_sources` makes it.
    4. **Ask:** in both runs, Qwen3 answered: *{answer}* The point says *4.78 seconds*, so the answer is right. Without the sources, Qwen3 said {alone0} and {alone1}.
    """.format(question=_w["question"], n=len(_w["asked"]), asked=", ".join(["*" + w + "*" for w in _w["asked"]]),
               gold=_w["shares"][0], shared=", ".join(["*" + w + "*" for w in _w["shared"]]), second=_w["shares"][1],
               points=C.number(len(C.POINTS), L.lang, 0), many=C.number(_w["many"], L.lang, 0), common=_w["common"], zero=_w["zero"],
               top0=_w["shares"][0], top1=_w["shares"][1], top2=_w["shares"][2],
               system=C.SYSTEM, answer=C.answers("q11", "rag")[0]["text"].strip(), alone0="10 seconds", alone1="12 seconds"))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### What the course measured

    The course asked Qwen3 {all} questions about the course, in English, {seeds} times each: without sources, with only the system prompt; and with RAG, with the three points the word search found among all {points}. A point answers {questions} of them: {words} share words with that point, and {meaning}, marked in the picture, ask the same in other words. The picture shows those {questions}. Each square is one answer; point at a square to read it:
    """.format(all=len(C.QUESTIONS), questions=_f["questions"], seeds=C.SEEDS, points=C.number(len(C.POINTS), L.lang, 0), words=_f["words"], meaning=_f["meaning"]))
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        The search found the point that answers the question for {hits} of the {questions}: for {words_hits} of the {words} questions that share words with it, but for only {meaning_hits} of the {meaning} that ask in other words. Without sources, Qwen3 was right {alone} times out of {runs}, both times on something it knew from outside the course: that a SHA-256 checksum has 64 hex digits, and that halving the uncertainty of an average takes four times the data. With RAG, {rag} of the {runs} answers were right, all of them where the search had found the point: {rag_hit} of {hit_runs} there, against {rag_miss} of {miss_runs} where it had not.

        A found point is not yet a right answer. The point about the share of customers says *21.7%*, and adds that counting all orders would give *26.1%*; Qwen3 answered *26.1%* in both runs. Where the search missed, Qwen3 took a number from what it got: the point about the small CNN was not among the three, and one answer gave it *786 parameters*, from a point about 786 shapes.

        Then the {none} questions that no point answers, such as the city of the bike shop. Without sources, not one of their {none_runs} answers said that Qwen3 did not know: it gave *$25 per month* and *Paris*, or said only that the course recorded its runs *on a graphics card*. With RAG, {unknown} of the {none_runs} said that the sources do not say; the others still made something up, such as a graphics card *that uses HistGradientBoostingRegressor*.
        """.format(hits=_f["hits"], questions=_f["questions"], words_hits=_f["words_hits"], words=_f["words"], meaning_hits=_f["meaning_hits"],
                   meaning=_f["meaning"], alone=_f["alone"], runs=_f["runs"], rag=_f["rag"], rag_hit=_f["rag_hit"], hit_runs=_f["hit_runs"],
                   rag_miss=_f["rag_miss"], miss_runs=_f["miss_runs"], none=_f["none"], unknown=_f["none_unknown_rag"], none_runs=_f["none_runs"])),
        L.watch_out("A word search counts every word the same: *the* and *a* count as much as *cache*, so most points share a word or two with any question without being about it. And a question in other words shares hardly a word with its point. Wrong sources can even push out what the model knew: without sources, Qwen3 once answered that halving the uncertainty takes *quadruple* the data; with three points about something else, it talked about those in both runs. [Search by words: BM25](/course/open/build-bm25?lang=en) weighs rare words more; a later lesson searches by meaning instead of words."),
    ])
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The words of a text

        `words(text)` should give the set of the words of a text, the way the worked example counted them: in lower case, and without punctuation at their ends. The box splits the text at the spaces and makes a set, but keeps capitals and punctuation, so *Cache?* and *cache* stay two words. `text.lower()` gives the text in lower case. `strip` takes off more than spaces when you give it characters: `"(cache?)".strip(".,:;!?()")` takes every one of those characters off both ends, and gives `"cache"`. Punctuation inside a word stays, as in *4.78*.
        """),
        L.task("Make the text lower case, and take `.,:;!?()` off both ends of every word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("words", lines=3)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("words", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "words",
        [
            "Two things change before the set is made: the whole text gets lower case, and every word loses the punctuation at its ends.",
            "`text.lower().split()` gives the words in lower case. A list comprehension, `[... for word in ...]`, changes every word on the way.",
            "`[word.strip(\".,:;!?()\") for word in text.lower().split()]` gives the clean words; put `set(...)` around it.",
        ],
        why="""```python
def words(text):
    return set([word.strip(".,:;!?()") for word in text.lower().split()])
```

A set holds each word once, so a word that comes twice in a point counts once. `strip` and not `replace`, because punctuation inside a word belongs to it: *4.78* and *1,038* stay whole.""",
    )
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The points that share the most words

        `search(question, points, k)` should give the `k` points that share the most words with the question, the most first. The box makes a list of pairs, a number and a point, sorts them from large to small with `sorted(found, reverse=True)`, and keeps the points of the first `k`. Pairs sort by their first item, the number; only when two numbers are equal does the point decide. But the box gives every point the number 0. `words` is given, as it is after exercise 1.
        """),
        L.task("Give each point the number of words it shares with the question, instead of 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("search", lines=7)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Make the set of the question's words once, before the loop, and compare it with the words of each point in the loop.",
            "`asked = words(question)` before the loop. `asked & words(point)` is the set of the words they share, and `len(...)` counts them.",
            "In the loop: `found.append((len(asked & words(point)), point))`.",
        ],
        why="""```python
def search(question, points, k):
    asked = words(question)
    found = []
    for point in points:
        found.append((len(asked & words(point)), point))
    return [point for shared, point in sorted(found, reverse=True)[:k]]
```

The question's words are the same for every point, so you make them once. Points that share as many words come in reverse alphabetical order: a word search has nothing better to choose between them with.""",
    )
    return


@app.cell(hide_code=True)
def text_rag():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A whole RAG

        The box is the `ask` of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with the system prompt of this lesson, `SYSTEM`, before the question and room for 60 tokens. Change `rag(url, question)` so that it finds the three points that share the most words with the question among all points, `POINTS`, and sends the prompt `with_sources` makes of the question and those three as the user's message. `SYSTEM`, `POINTS`, `search` and the `with_sources` of the previous lesson are given. The check runs your function against a stand-in.
        """),
        L.task("Find three points with `search`, and send the question with those sources as the user's message."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rag():
    box_rag = L.code_box("rag", lines=15)
    box_rag
    return (box_rag,)


@app.cell(hide_code=True)
def check_rag(box_rag):
    result_rag = L.check("rag", box_rag.value)
    result_rag
    return (result_rag,)


@app.cell(hide_code=True)
def hints_rag():
    L.hints(
        "rag",
        [
            "Two steps come before the messages: find the passages, then make the prompt of them.",
            "`search(question, POINTS, 3)` gives the three points, and `with_sources(question, passages)` makes the prompt of them.",
            "`passages = search(question, POINTS, 3)` before `messages`, and in the user's message `\"content\": with_sources(question, passages)`.",
        ],
        why="""```python
import json
import urllib.request


def rag(url, question):
    passages = search(question, POINTS, 3)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": with_sources(question, passages)}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 60}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

That is the whole pipeline: search, prompt, model. Each step can be swapped for a better one without touching the others, and the next lessons do that with the search.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once the three exercises are right, you can ask your own questions about the course, with your own `rag`: the course starts its own server on your computer, with Qwen3-0.6B behind it. Ask in English, because the points are in English. Try a question that shares words with a point, and one that asks the same in other words, and look at what your search finds.
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
def settings_cell():
    question_text = mo.ui.text(value=C.question("q11")["text"]["en"], label="Question", full_width=True)
    ask_button = mo.ui.run_button(label="Ask with your rag")
    mo.vstack([question_text, ask_button])
    return ask_button, question_text


@app.cell(hide_code=True)
def demo_answer(ask_button, box_rag, box_search, box_words, question_text):
    _fn = C.learner_rag(box_words.value, box_search.value, box_rag.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _fn is None:
        _out = mo.md("*Finish the three exercises first: the demo runs your own `words`, `search` and `rag`.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    else:
        mo.output.replace(mo.md("*Your search looks through the points, the server loads Qwen3, and Qwen3 writes …*"))
        _words = {"title": "Your rag: {question}", "sources": "The three points your search found, with the lesson of each:",
                  "answer": "Qwen3's answer, after {seconds} seconds:", "empty": "Type a question first.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fn, question_text.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The knowledge base here was clean text already. Most documents are PDFs and web pages, and [From files to text](/course/open/build-documents-to-text?lang=en) gets the text out of them. Each point here was a few sentences; a handbook of a hundred pages has to be cut into pieces first, in [Cutting documents into pieces](/course/open/build-chunking?lang=en). And [Search by words: BM25](/course/open/build-bm25?lang=en) makes the search count rare words more than *the* and *a*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rag, result_search, result_words):
    L.wrap_up(result_words, result_search, result_rag)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
