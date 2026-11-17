# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-grounded-answers. Route: course/route/lessen/bouwen.yaml. Checks, the recorded runs and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-grounded-answers", "en")
    C = L.checks   # the knowledge base, the recorded answers of Qwen3, the checks, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Search by meaning](/course/open/build-retrieval-embeddings?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_once():
    quick_once = L.choice(
        "quick-once",
        question="A RAG system searches 10,000 chunks by meaning, and gets 100 questions. How many texts does it embed in all?",
        options={
            "again": "1,000,100: the question and all chunks, for every question",
            "once": "10,100: every chunk once, and each question",
            "questions": "100: only the questions, since the chunks are text",
        },
        correct="once",
        why={
            "once": "The embeddings of the chunks do not depend on the question, so they are made once: 10,000. Each question then needs one embedding of its own: 100 more.",
            "again": "That works, but why embed the chunks again when they did not change? See [Search by meaning](/course/open/build-retrieval-embeddings?lang=en).",
            "questions": "Search by meaning compares the embedding of the question with the embeddings of the chunks, so the chunks need them too. See [Search by meaning](/course/open/build-retrieval-embeddings?lang=en).",
        },
    )
    quick_once
    return (quick_once,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    In [Your first RAG](/course/open/build-rag-first?lang=en), Qwen3 got sources in its prompt, but nothing told it to use them. With the three points search by meaning finds and the system prompt of that lesson, {plain} of its {runs} answers were right. *{sam}* One source says: in 8 of 8 runs with the earlier turn, in 0 of 8 without it. Qwen3 answered: *{plain_sam}* And to a question no point answers, it made something up: *{city}* From such an answer, you cannot tell whether it came from a source, or from which.

    **An everyday example.** In an open-book exam, the teacher asks for the page next to each answer. Whoever has to write the page looks it up instead of guessing, the teacher can check the answer in a moment, and a question the book does not answer gets *not in the book*.
    """.format(plain=_f["plain"], runs=_f["runs"], sam=C.question("q05")["text"]["en"], plain_sam=C.answers("q05", "plain")[0],
               city=C.answers("q27", "plain")[1]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **grounded generation** | writing an answer only from the sources the model got, with the number of the source, and saying so when they do not answer the question |
    | `text.split("[")` | the pieces of `text` between the `[`s: `"a [1] b".split("[")` gives `["a ", "1] b"]` |
    | `inside.isdigit()` | `True` when `inside` holds only digits, at least one: `"12".isdigit()` is `True`, `"a".isdigit()` and `"".isdigit()` are `False` |
    | `a or b` | `True` when at least one of the two is `True` |

    In [Prompts that check themselves](/course/open/start-prompt-chains?lang=en), Qwen3 answered from one text you gave it, with a citation: grounding. In a RAG system, the texts are the sources the search found, numbered the way `with_sources` of [Look it up first: what RAG is](/course/open/build-rag-idea?lang=en) puts them in the prompt. A grounded system prompt asks three things: answer only from the numbered sources; end with the number of the source, in square brackets; and when the sources do not answer, say only *I don't know*. This is the one the course used:

    > {grounded}

    The number makes an answer checkable: you look up the source it names, and see whether it says that. A program can do part of it: find the numbers in brackets, and look whether every number in the answer is in the sources it names.
    """.format(grounded=C.GROUNDED))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: answering from the sources

    1. **Search:** the `k` chunks that fit the question best.
    2. **Number:** put them in the prompt as [1], [2], [3], with the question after them.
    3. **Instruct:** a system prompt that asks for an answer only from the sources, with the number of its source, and only *I don't know* when they do not answer.
    4. **Check:** find the numbers in brackets in the answer, and look whether every number in it is in the sources it names.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Worked example

    *{question}*

    1. **Search:** by meaning, the three best points come from *{t1}*, *{t2}* and *{t3}*.
    2. **Number:** source [{n}] is the point of *{tn}*: *{source}*
    3. **Instruct:** with the plain system prompt, Qwen3 answered *{plain}* With the grounded prompt: *{grounded}*
    4. **Check:** the answer names source {n}. Its words with a digit in them are {numbers}, and all {k} are in source {n}: the check says `True`.

    Source {n} has two numbers for the runs, 8 of 8 and 0 of 8, and the plain answer took the wrong one. Had it named source {n}, a check that only looks at the numbers would have said `True` for it too.
    """.format(question=_w["question"], t1=C.lesson_of(_w["passages"][0])["title"][L.lang], t2=C.lesson_of(_w["passages"][1])["title"][L.lang],
               t3=C.lesson_of(_w["passages"][2])["title"][L.lang], n=_w["cited"][0], tn=C.lesson_of(_w["source"])["title"][L.lang],
               source=_w["source"], plain=_w["plain"], grounded=_w["grounded"],
               numbers=", ".join("*" + x + "*" for x in _w["numbers"][:-1]) + " and *" + _w["numbers"][-1] + "*", k=len(_w["numbers"])))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course asked Qwen3 all {all} questions about the course, each with the three points search by meaning finds: twice with the plain system prompt, and twice with the grounded one. The sources are the same; only the system prompt differs.
    """.format(all=len(C.QUESTIONS)))
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
        With the grounded prompt, {grounded} of the {runs} answers to the {questions} questions a point answers were right; with the plain prompt, {plain}. Every grounded answer named a source; of the plain answers, {plain_cites}. On the {none} questions no point answers, the grounded prompt answered *I don't know* in {gnone} of the {nruns} runs. With the plain prompt, {pnone} of the {nruns} said they could not answer; the others answered anyway.
        """.format(grounded=_f["grounded"], runs=_f["runs"], questions=_f["questions"], plain=_f["plain"], plain_cites=_f["plain_cites"],
                   none=_f["none"], gnone=_f["grounded_none"], nruns=_f["none_runs"], pnone=_f["plain_none"])),
        L.watch_out("A number in brackets is not proof. *{q}* Source 1 says that a sentence is 6 tokens in English, and the same sentence in Dutch 15. Both grounded runs answered with [1]: *{a}*".format(
            q=C.question("q23")["text"]["en"], a=C.answers("q23", "grounded")[0].strip())),
    ])
    return


@app.cell(hide_code=True)
def text_cited():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The sources an answer names

        `cited(answer)` should give the numbers of the sources an answer names: the whole numbers in square brackets, in the order they come, each once. For *{example}* that is `[2]`. Brackets with anything else in them, such as `[a]` or `[1, 2]`, name no source: the grounded prompt asks for one number. The box cuts the answer at every `[`, and keeps of each piece what comes before the next `]`. When it is right, the check also keeps it as a file, `work/projects/app/citations.py`: [Project: your assistant reads your documents](/course/open/build-project-rag?lang=en) uses it.
        """.format(example=C.CITED_EXAMPLE)),
        L.task("Keep a piece only when it is a number, as a whole number, and only once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cited():
    box_cited = L.code_box("cited", lines=6)
    box_cited
    return (box_cited,)


@app.cell(hide_code=True)
def check_cited(box_cited):
    result_cited = L.check("cited", box_cited.value)
    result_cited
    return (result_cited,)


@app.cell(hide_code=True)
def hints_cited():
    L.hints(
        "cited",
        [
            "`inside.isdigit()` says whether a piece holds only digits; `int(inside)` makes a whole number of it.",
            "Add a number only when it is not in `found` yet; `not in` tells you that, but test the whole number, not the text.",
            "`if inside.isdigit() and ___:`, with the `append` of `int(inside)` under it.",
        ],
        why="""```python
def cited(answer):
    found = []
    for part in answer.split("[")[1:]:
        inside = part.split("]")[0]
        if inside.isdigit() and int(inside) not in found:
            found.append(int(inside))
    return found
```

`[1:]` skips the piece before the first `[`. And `and` stops at the first `False`: when `inside` is not a number, Python never gets to `int(inside)`, which would stop with an error.""",
    )
    return


@app.cell(hide_code=True)
def text_supported():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Are its numbers in the sources it names?

        `supported(answer, passages)` should give `True` only when the answer names at least one source, every source it names is there, and every number in the answer is in the sources it names. `passages` are the sources in the order of the prompt, so source 1 is `passages[0]`. `cited` and `numbers` are given: `numbers(text)` gives the words of a text with a digit in them, without the citations, such as `{{'0', '8', 'qwen3'}}` for the answer above. The check uses four sources of a bike shop, `SHOP`:

        {shop}

        The box puts the sources the answer names together in `text`, but then says `True` whatever is in them.
        """.format(shop="\n".join("{i}. *{t}*".format(i=i, t=t) for i, t in enumerate(C.SHOP, start=1)))),
        L.task("Give back `False` when the answer names a source that is not there, or holds a number that is not among the numbers of `text`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_supported():
    box_supported = L.code_box("supported", lines=12)
    box_supported
    return (box_supported,)


@app.cell(hide_code=True)
def check_supported(box_supported):
    result_supported = L.check("supported", box_supported.value)
    result_supported
    return (result_supported,)


@app.cell(hide_code=True)
def hints_supported():
    L.hints(
        "supported",
        [
            "Two checks: in the loop, before `passages[n - 1]`, whether source `n` is there; after the loop, every number of the answer against `numbers(text)`.",
            "Source `n` is there when it is from 1 to `len(passages)`, so `n < 1 or n > len(passages)` means it is not. After the loop, go through `numbers(answer)`.",
            "After the loop: `for number in numbers(answer):`, then `if ___:` with `return False` under it; `return True` stays at the end.",
        ],
        why="""```python
def supported(answer, passages):
    found = cited(answer)
    if len(found) == 0:
        return False
    text = ""
    for n in found:
        if n < 1 or n > len(passages):
            return False
        text = text + " " + passages[n - 1]
    for number in numbers(answer):
        if number not in numbers(text):
            return False
    return True
```

Compare numbers with numbers, not with the text: as a piece of text, *1* is in *15*. The picture runs your function on Qwen3's grounded answers.""",
    )
    return


@app.cell(hide_code=True)
def text_named_source():
    mo.md(r"""
    ### Exercise 3. A number from the source

    Every number in this answer is in the source it names.
    """)
    return


@app.cell(hide_code=True)
def choice_named_source_cell():
    _f = C.findings()
    choice_named_source = L.choice(
        "named-source",
        question="For *{q}*, Qwen3 answered with the grounded prompt: *{a}* Source 3 is: *{s}* So `supported` says `True`. What does that show?".format(
            q=C.question("q01")["text"]["en"], a=C.answers("q01", "grounded")[0].strip(), s=C.found(C.question("q01"))[2]),
        options={
            "right": "That the answer is right: the number is in the source it names",
            "search": "That the search found the right point, as source 3",
            "about": "Only that 34,090 is in source 3, which is about another CNN",
        },
        correct="about",
        why={
            "about": "Source 3 is a point of *{t}*: another CNN, with a bottleneck, of 34,090 parameters. The small CNN of the course has 9,098, and its point was not among the three. A check by numbers sees whether a number comes from the sources, not whether it answers the question: of the {other} grounded answers that were not right, it let all {other} through.".format(
                t=C.lesson_of(C.found(C.question("q01"))[2])["title"][L.lang], other=_f["other_supported"]),
            "right": "Read source 3 again: which CNN has 34,090 parameters there, and is it the small CNN of the course?",
            "search": "For this question, search by meaning missed the point that answers it: it is not among the three. Which CNN is source 3 about?",
        },
    )
    choice_named_source
    return (choice_named_source,)


@app.cell(hide_code=True)
def text_refused():
    mo.md(r"""
    ### Exercise 4. I don't know

    The grounded prompt asks for *I don't know* when the sources do not answer.
    """)
    return


@app.cell(hide_code=True)
def choice_refused_cell():
    _f = C.findings()
    choice_refused = L.choice(
        "refused",
        question="For *{q}*, source 1 says: *{s}* One of the two grounded runs answered: *{a}* What does that show?".format(
            q=C.question("q08")["text"]["en"], s=C.found(C.question("q08"))[0], a=C.answers("q08", "grounded")[1].strip()),
        options={
            "missed": "That the search missed the point with the answer, so Qwen3 had nothing",
            "percent": "That Qwen3 cannot read a share written as a percentage, such as 21.7%",
            "price": "That the strict prompt sometimes refuses where a source answers",
        },
        correct="price",
        why={
            "price": "Source 1 holds the answer, and came first. A prompt that makes *I don't know* a safe answer gets it now and then where it should not: in {refused} of the {runs} grounded answers to the questions a point answers. Against that, it said *I don't know* in {gnone} of the {nruns} runs where no point answers, where the plain prompt said in {pnone} that it could not answer. Whether one refusal too many is worth that depends on what a wrong answer costs.".format(
                refused=_f["refused"], runs=_f["runs"], gnone=_f["grounded_none"], nruns=_f["none_runs"], pnone=_f["plain_none"]),
            "missed": "Source 1 is the point that answers it: 21.7% is in it. So why would Qwen3 still say it does not know?",
            "percent": "Qwen3 read 95% right in both its answers to another question. What in the system prompt makes *I don't know* an answer it may give?",
        },
    )
    choice_refused
    return (choice_refused,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Ask Qwen3 with your own system prompt

    Once exercises 1 and 2 are right, you can ask Qwen3 one of the course's questions, with the three points search by meaning finds, after a system prompt of your own. The questions are in English, like the sources. Change the grounded prompt, or leave a rule out, and see what your `cited` and `supported` say about the answer.
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
    question_pick = mo.ui.dropdown(options={q["text"][L.lang]: q["id"] for q in C.QUESTIONS}, value=C.question("q05")["text"][L.lang],
                                   label="Question", full_width=True)
    system_text = mo.ui.text_area(value=C.GROUNDED, label="System prompt", full_width=True, rows=4)
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([question_pick, system_text, ask_button])
    return ask_button, question_pick, system_text


@app.cell(hide_code=True)
def demo_answer(ask_button, box_cited, box_supported, question_pick, system_text):
    _checks = C.learner_checks(box_cited.value, box_supported.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _checks is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo checks the answer with your own `cited` and `supported`.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    else:
        _words = {"title": "Qwen3 with your system prompt: {question}", "sources": "The three sources in the prompt, with the lesson of each:",
                  "answer": "Qwen3's answer:", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds.", "checked": "Your cited: {cited}. Your supported: {supported}.",
                  "failed": "Your functions stopped with an error on this answer: {why}",
                  "load": "Qwen3 could not be loaded. Download it again, with the button above.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_checks, question_pick.value, system_text.value, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Project: your assistant reads your documents](/course/open/build-project-rag?lang=en), your assistant does it all: it cuts documents into chunks, searches them, and answers from the sources with the number of its source. A check by numbers is a first sieve. Whether a source says what the answer claims, a person sees, or a second model asked to judge it, in a later lesson.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_named_source, choice_refused, result_cited, result_supported):
    L.wrap_up(result_cited, result_supported, choice_named_source, choice_refused)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
