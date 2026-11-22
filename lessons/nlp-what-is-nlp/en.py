# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-what-is-nlp. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-what-is-nlp", "en")
    C = L.checks   # nine made-up systems, and nine made-up requests to the assistant of a bike shop


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en) and [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_numbers():
    quick_numbers = L.choice(
        "quick-numbers",
        question="What does `[ord(ch) for ch in \"hi\"]` give?",
        options={
            "bytes": "b'hi', the two characters as bytes of UTF-8 text",
            "right": "[104, 105], one code point for each character",
            "count": "[1, 1], how often each character occurs",
        },
        correct="right",
        why={
            "right": "`ord` gives the code point of a character: 104 for h and 105 for i. For a computer, text is a sequence of numbers; this lesson asks what a computer can do with that sequence.",
            "bytes": "Bytes come from `\"hi\".encode()`. `ord` gives the code point of one character, a number. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "count": "`ord` counts nothing: it gives the code point of each character, 104 and 105. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
        },
    )
    quick_numbers
    return (quick_numbers,)


@app.cell(hide_code=True)
def quick_check_bigrams():
    quick_bigrams = L.choice(
        "quick-bigrams",
        question="A bigram model counts the pairs of characters in the text `banana`. Which character does it give after `n`?",
        options={
            "n": "n, since a bigram repeats the character it is given",
            "b": "b, the first character of the text",
            "right": "a, since every n in the text is followed by an a",
        },
        correct="right",
        why={
            "right": "Both pairs that start with n are na, so the model gives a. It writes text without knowing what any of it means: one side of what this lesson is about.",
            "n": "A bigram is two characters next to each other in the text. In banana, every n is followed by an a. See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
            "b": "The model looks at the character before, not at the start of the text: after n comes a, twice. See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
        },
    )
    quick_bigrams
    return (quick_bigrams,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You type three words into a search box and get the page you meant. A mail program moves a message about a prize you never entered to the spam folder. An app turns a Spanish menu into Dutch. Each time, a computer works with human language, which was written for people, not for computers.

    This lesson starts the domain on language and search. It says what that field is called and what it covers, sorts systems by what they do with language, and builds the first step of a small assistant: finding what a question is about.

    **An everyday example.** A post office does two jobs with the same letter: reading the address, and printing a label. One gets something out of text; the other makes text.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A computer can get something out of text, such as what a question is about, or it can write text. A chat assistant that answers questions about a shop: which of the two does it do?",
        options={
            "read": "Only getting something out of text: it sorts questions, and people answer them",
            "write": "Only writing text: it writes what is likely to follow",
            "right": "Both: it gets what you ask, then writes an answer",
        },
        reveal="**Both.** To answer, the assistant first has to get what you ask, and then write its answer in words. *Understanding and making* in *The idea* gives the two sides their names.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Natural language processing

    A natural language is a language people speak, write or sign, such as Dutch or English; Python is not one. In 2015, Julia Hirschberg and Christopher Manning wrote in *Science*: "Natural language processing employs computational techniques for the purpose of learning, understanding, and producing human language content." **Natural language processing**, NLP for short, is that field: computers that work with human language.

    The Association for Computational Linguistics (ACL) lists what such systems are: "speech recognition systems, text-to-speech synthesizers, automated voice response systems, web search engines, text editors, language instruction materials, to name just a few." Hirschberg and Manning pointed to products of 2015, "such as Apple's Siri and Skype Translator". The textbook of Jurafsky and Martin, in its draft of August 19, 2026, calls spam detection "another important commercial application".

    ### Computational linguistics

    The ACL's page, under the heading ‘What is Computational Linguistics?’, says: "Computational linguistics is the scientific study of language from a computational perspective." **Computational linguistics** uses computers to answer questions about language. Is that a different field from NLP? The sources do not agree. The same page speaks of "a field often referred to as either computational linguistics or natural language processing (NLP)", and Hirschberg and Manning write "Computational linguistics, also known as natural language processing (NLP)". Jurafsky and Martin note that the two are sometimes kept apart: computing the structure of language helps answer "questions about language itself, a research area called computational linguistics that is sometimes distinguished from natural language processing."

    When people do keep them apart, computational linguistics is the science that wants to know how language works, and NLP the engineering that wants systems that work. The ACL page names both motivations: in some cases scientific, and in other cases "more purely technological in that one wants to provide a working component of a speech or natural language system."

    ### Understanding and making

    The definition of Hirschberg and Manning has three verbs: learning, understanding and producing. The last two give NLP two sides.

    **Natural language understanding**, NLU for short, goes from language to what it means. Emily Bender and Alexander Koller (2020) take meaning to be the link between what is said and what someone wants to get across with it, and write: "we can now use understand to refer to the process of retrieving i given e". Here *e* is the expression, the words, and *i* the intent behind them. They call human-like NLU "a grand challenge of artificial intelligence". A spam filter is on this side: text goes in, and a label comes out.

    **Natural language generation**, NLG for short, goes the other way. Albert Gatt and Emiel Krahmer (2018) define it as "the task of generating text or speech from non-linguistic input": input that is not language, such as the numbers in a table. A weather site that writes a forecast from its measurements is on this side. They note that, as others have pointed out, "precisely defining nlg is rather difficult", and they count making text from text, such as a summary, as NLG too.

    Many systems do both. Translating a menu means getting what the Spanish says and writing it in Dutch. This lesson sorts systems with a simple rule the course made up: text in is understanding, text out is making, text in and text out is both, and when neither side is text, it is not NLP. Real systems are less tidy: a search engine reads your words, and what it gives back is a list of pages that others wrote.
    """)
    return


@app.cell(hide_code=True)
def sides_cell():
    mo.Html(C.sides_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **natural language processing (NLP)** | computers that work with human language: understanding it, producing it, or learning from it |
        | **computational linguistics** | the study of language itself with computers; often another name for NLP |
        | **natural language understanding (NLU)** | the side of NLP that goes from language to what it means |
        | **natural language generation (NLG)** | the side of NLP that goes from data or meaning to language |
        | `C.SYSTEMS` | nine made-up systems: dicts with `"text"`, `"input"` and `"output"` |
        | `C.REQUESTS`, `C.KEYWORDS` | nine made-up requests to a bike shop, and a dict from each kind of request to words that point to it |
        | `elif` | else if: Python checks this condition only when the `if` before it was false |
        | `a in s` | `True` when the string `s` contains the string `a`: `"pen" in "open"` is `True` |
        """),
        L.watch_out("NLU is the name of a task, not a claim that a system understands as people do. Bender and Koller argue, in what they call a position paper, that \"a system trained only on form has a priori no way to learn meaning\"."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        1. Look at what goes in: human language, written or spoken, or data such as numbers and labels.
        2. Look at what comes out, in the same way.
        3. Ask first whether text goes in and text comes out. Then the system does both sides.
        4. Else, text that only goes in is understanding (NLU), and text that only comes out is making (NLG).
        5. If no language goes in or comes out, it is not NLP, however clever the system is.
        """),
        mo.Html(C.grid_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes the third of the course's systems and prints its text, what goes in and what comes out. `C.SYSTEMS[2]` is a dict; `system["input"]` looks up the value under the key `"input"`.
    """)
    return


@app.cell
def example_system():
    system = C.SYSTEMS[2]   # the third system
    print(system["text"])
    print("in:", system["input"])
    print("out:", system["output"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Text goes in and text comes out: by the recipe, both sides. The app has to get what the Spanish says, and then write it in Dutch.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_side():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. side: which side of NLP

        `C.SYSTEMS` holds nine made-up systems, each a dict with the keys `"text"`, `"input"` and `"output"`; under the last two is `"text"` or `"data"`. Write `side(system)`: the side of NLP of the dict `system`, as one of the strings `"both"`, `"NLU"`, `"NLG"` or `"none"`. Ask the questions of the recipe in their order, with `if` and `and`, then `elif`. Replace the `...`. The check calls `side(SYSTEMS[0])`, then the other systems and one of its own.
        """),
        L.task("Compare what goes in and what comes out with \"text\", and give back the side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_side():
    box_side = L.code_box("side", lines=8)
    box_side
    return (box_side,)


@app.cell(hide_code=True)
def check_side(box_side):
    result_side = L.check("side", box_side.value)
    result_side
    return (result_side,)


@app.cell(hide_code=True)
def hints_side():
    L.hints(
        "side",
        [
            "*Recipe* gives the four answers, and which question comes first.",
            "Which question must come first, so that a system with text on both sides does not stop at a question about one side? And which answer is left for the last line?",
            "`    if system[\"input\"] == \"text\" and ___:`",
        ],
        why="`if system[\"input\"] == \"text\" and system[\"output\"] == \"text\":` with `return \"both\"` under it, then `elif system[\"input\"] == \"text\":` with `return \"NLU\"`, `elif system[\"output\"] == \"text\":` with `return \"NLG\"`, and last `return \"none\"`. `side(SYSTEMS[0])` gives `\"NLU\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_side():
    mo.md(r"""
    Two systems only understand, three only make text, three do both, and one is not NLP at all. The spam folder and the sorted emails end in a label; the forecast, the match report and the delay notice start from numbers. The spreadsheet is clever with numbers, but no language goes in or comes out.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_intent():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. intent: what a request is about

        A made-up bike shop wants a program that answers questions. Its first step is understanding: what is a request about? `C.REQUESTS` holds nine requests, and `C.KEYWORDS` is a dict from each kind of request (`"hours"`, `"repair"`, `"price"`) to a list of words that point to it. Write `intent(request, keywords)`: the first key of the dict `keywords` with a word that occurs in the string `request`, and `"unknown"` when no word does. The words are lowercase, so make the request lowercase first, with `.lower()`. Go over the keys and their lists with `for name, words in keywords.items():`, and over each list with a second `for` loop. Replace the `...`. The check calls `intent(REQUESTS[0], KEYWORDS)`, then the other requests and dicts of its own.
        """),
        L.task("Go over the kinds and their words, and give back the first kind with a word in the request."),
    ])
    return


@app.cell(hide_code=True)
def exercise_intent():
    box_intent = L.code_box("intent", lines=7)
    box_intent
    return (box_intent,)


@app.cell(hide_code=True)
def check_intent(box_intent):
    result_intent = L.check("intent", box_intent.value)
    result_intent
    return (result_intent,)


@app.cell(hide_code=True)
def hints_intent():
    L.hints(
        "intent",
        [
            "Read the request the way the program does: one kind at a time, and within a kind one word at a time.",
            "What do you ask of each word, and what do you give back as soon as the answer is yes?",
            "`            if ___:`",
        ],
        why="`text = request.lower()`, `for name, words in keywords.items():`, `for word in words:`, then `if word in text:` with `return name` under it, and after both loops `return \"unknown\"`. `intent(REQUESTS[0], KEYWORDS)` gives `\"hours\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_intent():
    mo.md(r"""
    Seven of the nine are right, and two are wrong: find them in the picture under the check.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: both sides at work

    Pick a request, and see the two sides one after the other: the course's `intent` finds what the request is about, and a fixed reply for each kind of request makes the answer: the simplest kind of making.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_request = mo.ui.dropdown(options=C.request_options(L.lang), value=list(C.request_options(L.lang))[0], label="A request")
    demo_request
    return (demo_request,)


@app.cell(hide_code=True)
def demo_look(demo_request):
    mo.Html(C.demo_picture(demo_request.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_names():
    mo.md(r"""
    ### Exercise 3. Two projects that count words

    A research group counts the words that children of two to four use, to find out at what age they start to use the past tense. A company counts the words its users type, to build a spelling app that suggests the right word.
    """)
    return


@app.cell(hide_code=True)
def choice_names_cell():
    choice_names = L.choice(
        "names",
        question="Where people keep computational linguistics and NLP apart, which of the two projects is computational linguistics?",
        options={
            "right": "The research on children, since it asks how language is learned",
            "app": "The spelling app, since it puts computing on language to work for real users",
            "neither": "Neither, since a program that counts words does no linguistics",
        },
        correct="right",
        why={
            "right": "The group wants to know how language works: when children learn the past tense. Jurafsky and Martin call that research on \"questions about language itself\". The spelling app is NLP: a system that has to work.",
            "app": "A working app is the engineering side: NLP. Computational linguistics, where people keep it apart, asks questions about language itself. See *Computational linguistics* in *The idea*.",
            "neither": "Counting is computing, and the question about children is a question about language: that makes it computational linguistics. See *Computational linguistics* in *The idea*.",
        },
    )
    choice_names
    return (choice_names,)


@app.cell(hide_code=True)
def text_exercise_open():
    mo.md(r"""
    ### Exercise 4. A lock that won't open

    For the request "The lock on my bike won't open.", the course's `intent` gives `"hours"`.
    """)
    return


@app.cell(hide_code=True)
def choice_open_cell():
    choice_open = L.choice(
        "open",
        question="Why does it get this request wrong?",
        options={
            "list": "The repair words lack lock, and adding it fixes every request like this one for good",
            "right": "Open means something else here, and a keyword cannot tell the two apart",
            "lower": "Making the request lowercase changed what it means",
        },
        correct="right",
        why={
            "right": "The word open is in the request, but it is about a lock, not about the shop. A program that only looks at letters cannot see that: keywords find letters, not meaning. A strange noise when braking goes wrong the other way: it is a repair, but none of the repair words is in it.",
            "list": "Adding lock would not help: hours comes first in the dict, so open still wins. And the next request about a lock might be about buying one. See `C.KEYWORDS`.",
            "lower": "Lowercase letters do not change what a request means; without them, *Open on Sunday?* would be missed. The trouble is the word open, which means something else here.",
        },
    )
    choice_open
    return (choice_open,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en) showed how a computer stores text, and [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) made text by counting, without understanding any of it. This lesson named the field: NLP, with two sides, understanding (NLU) from text to meaning and making (NLG) from meaning to text. Computational linguistics studies language itself with computers, and is often another name for the same field. A program with keywords found what a request was about, and got two of nine wrong. The next lesson, *Why language is hard for computers*, shows where such mistakes come from.

    **An everyday example.** A good interpreter does both sides: first get what is said, then say it again in another language.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_names, choice_open, result_intent, result_side):
    L.wrap_up(result_side, result_intent, choice_names, choice_open)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
