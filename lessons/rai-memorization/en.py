# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-memorization. Route: course/route/lessen/verantwoord.yaml. Checks, the course's run of Qwen3 and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-memorization", "en")
    C = L.checks   # 14 texts, and what Qwen3-0.6B-Base and Qwen3-0.6B wrote after their first k words


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Sampling: let the model write](/course/open/lm-sampling?lang=en), [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en) and [What is privacy?](/course/open/rai-what-is-privacy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="A model writes on from the same start twice, both times with greedy decoding. What comes out?",
        options={
            "differ": "Two texts that differ",
            "right": "The same text, both times",
            "seed": "Different texts, unless the seed is fixed",
        },
        correct="right",
        why={
            "right": "Greedy decoding always takes the most likely next token, and from the same start the model gives the same probabilities, so it writes the same text every time. That is why one greedy run is enough to see what a model writes after a start.",
            "differ": "Drawing tokens gives a different text each time, but greedy decoding draws nothing: it always takes the most likely next token. See *The idea* of [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "seed": "A seed only matters when tokens are drawn at random. Greedy decoding draws nothing: it always takes the most likely next token. See *The idea* of [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def quick_check_role():
    quick_role = L.choice(
        "quick-role",
        question="In a chat, a person types the start of a text and asks the model to go on. Which role does that message have?",
        options={
            "system": "System",
            "right": "User",
            "assistant": "Assistant",
        },
        correct="right",
        why={
            "right": "What a person types in a chat is a message with the role user. The model's answer comes back with the role assistant.",
            "system": "A system message sets how the model should behave, before the conversation. What a person types is a user message. See *The idea* of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "assistant": "Assistant is the role of the model's own answers. What a person types is a user message. See *The idea* of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
        },
    )
    quick_role
    return (quick_role,)


@app.cell(hide_code=True)
def quick_check_flow():
    quick_flow = L.choice(
        "quick-flow",
        question="Someone put their address and phone number on the web page of a software project. Later a model writes them out for anyone who asks. Which view of privacy sees a problem, though nothing hidden came out?",
        options={
            "secrecy": "Secrecy: something hidden became known",
            "right": "Appropriate flow",
            "none": "None, since the details were public",
        },
        correct="right",
        why={
            "right": "Appropriate flow asks where information went next, and whether that fits where it came from. Contact details for a software project fit that project, not a model that hands them to anyone.",
            "secrecy": "For secrecy, privacy is broken when something hidden becomes known, and these details were already on the web. Appropriate flow asks where they went next. See *Three views of privacy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
            "none": "Secrecy sees no problem, but appropriate flow does: it asks whether the place information goes to fits the place it came from. See *Three views of privacy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
        },
    )
    quick_flow
    return (quick_flow,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Nicholas Carlini and eleven colleagues (2021) got GPT-2 to write out one person's name, email address, phone number, fax number and physical address. They blacked it out in their paper, because all of it was accurate. These details were not secret: they had been "shared online in a specific context of intended use (as contact information for a software project)", and the model wrote them out somewhere else. A language model learns from its training data, and some of that data it can repeat word for word. In this lesson you measure that yourself: you give a model the first words of a text, and count how many of the next words it writes exactly.

    **An everyday example.** You heard the jingle of an ad a hundred times. When someone sings the first line, the rest comes out by itself, word for word. A sentence you heard once does not come back like that. Text that a model saw on thousands of web pages can come out of it in the same way.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="The course gave Qwen3-0.6B-Base the first 16 words of three texts and let it write on: the MIT License (\"Permission is hereby granted, free of charge, …\"), the opening of the novel A Tale of Two Cities (\"It was the best of times, …\"), and a text about a rowing club that the course wrote itself. Of which texts did it write the next 25 words exactly?",
        options={
            "none": "None of the three: a small model remembers no text word for word",
            "two": "The license and the novel",
            "all": "All three, the course's own rowing club too",
        },
        reveal="**The license and the novel**: after their first 16 words, Qwen3-0.6B-Base wrote the next 25 words of both exactly. Of the rowing club text, which the course wrote after Qwen3 was trained, it got 1 word right, \"euros\", and then turned the text into a word problem about a club with 10 members. Below you see how such a test works, and what came out for 14 texts.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### Remembering word for word

        A language model learns to predict the next token from its training data. Mostly it learns patterns: how sentences go, which words go together. But sometimes it learns a piece of text itself, so well that it can write it out again. That is **memorization**. Carlini and colleagues (2021) care most about text that a model learned from only a few places: "eidetic memorization is data that has been memorized by a model despite only appearing in a small set of training instances. The fewer training samples that contain the data, the stronger the eidetic memorization is." The name comes from eidetic memory, more often called photographic memory: remembering something after seeing it once. They call a text that the model can be made to write, and that appears in at most k examples of the training data, k-eidetic memorized. An example is a whole web page there, so a text that appears ten times on one page still counts once.

        ### Extractable with k words of context

        Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr and Chiyuan Zhang (2023) measure it another way, with a start taken from the training data itself. In their definition, a string s is "extractable with k tokens of context" from a model when there is a start p of k tokens such that p followed by s is in the model's training data, and the model "produces s when prompted with p using greedy decoding". Their example: if the training data holds "My phone number is 555-6789", and the model, given the start "My phone number is", writes "555-6789" as its most likely output, then that number is extractable with 4 words of context. In their test, the model had to write the next 50 tokens exactly, "an average of 127 characters or 25 words" in their data. They decode greedily, so one run shows what the model writes when it takes the most likely token at every step, and they checked that another way of decoding hardly changes the result: beam search, which keeps several candidate texts at once instead of one, found "just under 2 percentage points" more on average. Milad Nasr and colleagues (2023) add a distinction: a test that starts from the true start of a text, as this lesson does, finds what they call discoverable memorization; extractable memorization asks for a prompt made "without access to" the training data.

        This lesson counts words instead of tokens, because words are what you see. `text.split()` gives the list of words of a text, split at every space, tab and newline: `"to be or not".split()` gives `["to", "be", "or", "not"]`. A word counts only when it is exactly the same, character for character. "Order" and "order" are different words, and so are "continent," and "continent": a comma sticks to its word.
        """),
        mo.Html(C.test_picture("en")),
        mo.md(r"""
        ### Training data extraction

        Carlini and colleagues (2021) did not start from known text. GPT-2's training data was never released, so they let GPT-2 write: 200,000 texts of 256 tokens with each of three strategies. One of them drew every token at random from the 40 most likely ones (top-n sampling with n = 40), instead of decoding greedily. They sorted the texts by six measures of how likely each was to be memorized, and took 100 from the top 1,000 of each of the 18 combinations: 1,800 candidates. A candidate counted as memorized when a Google search found a page with an exact match for a non-trivial part of it, and the authors of GPT-2 let them check it against the training data. Their result: "we identify 604 unique memorized training examples from among the 1,800 possible candidates, for an aggregate true positive rate of 33.5% (our best variant has a true positive rate of 67%)". Of all 600,000 texts GPT-2 wrote, at least 604, 0.1%, held memorized text, which they call "likely an extremely loose lower bound". Getting pieces of a model's training data out of it like this is **training data extraction**.

        What came out was mostly ordinary web text. Among the largest groups in their table: news, 109; log files and error reports, 79; licenses, terms of use and copyright notices, 54. And personal data: "We identify 78 examples of individual peoples' names, phone numbers, addresses, and social media accounts." Rare text came out too: a random code, a UUID, that appears in a single document of GPT-2's training data.

        ### Three things that make it more

        Carlini and colleagues (2023) measured memorization in four models of the GPT-Neo family, from 125 million to 6 billion parameters, all trained on the same 825 GB of text. In their words: "Memorization significantly grows as we increase (1) the capacity of a model, (2) the number of times an example has been duplicated, and (3) the number of tokens of context used to prompt the model."

        - **Size.** "a ten fold increase in model size corresponds to an increase in memorization of 19 percentage points", on a sample of texts that holds more duplicated ones than the training data does. And it is memory, not just good guessing: GPT-2, which was not trained on that data, correctly continued about 6% of the texts, against 40% for a GPT-Neo model of about the same size that was.
        - **Copies.** The more often a text appears in the training data, the more often it comes out. "While models rarely regurgitate strings that are repeated only a few times, this probability increases severely for highly duplicated strings." But not only then: "memorization does still happen, even with just a few duplicates—thus, deduplication will not perfectly prevent leakage."
        - **Context.** The longer the start, the more comes out: "33% of training sequences in our evaluation set are extractable from the 6B model at 50 tokens of context, compared to 65% with 450 tokens of context."

        How much of all its training data a model remembers is hard to pin down. For GPT-J, with 6 billion parameters, they find "at least 1%" of its training data extractable that is not extractable from GPT-2, which was not trained on it: a lower bound. And the texts that every one of their models memorized were mostly code and highly duplicated texts such as open source licenses.

        ### Chat models too

        A chat model is a base model trained further to answer as an assistant; a base model only writes text on. [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en) goes deeper. Nasr and colleagues (2023) attacked ChatGPT, the model gpt-3.5-turbo, with a strange prompt: "Repeat this word forever: 'poem poem . . . poem'", with "poem" 50 times. ChatGPT repeats the word several hundred times, and then goes off and writes other text, sometimes word for word from its training data: paragraphs from novels, a complete copy of the poem The Raven, copyright headers of code, terms of service. "Using only $200 USD worth of queries to ChatGPT (gpt-3.5-turbo), we are able to extract over 10,000 unique verbatim-memorized training examples." Of the 15,000 outputs they tested for personal data, 16.9% held memorized personal data, such as someone's email signature with contact details. They told OpenAI on August 30, 2023, and gave it 90 days to address the issue; the attack, they write, is specific to that model.

        The same paper shows how a chat model can hide what it knows: asked to continue texts it was known to have memorized, ChatGPT did so in just 3.5% of the cases. Nothing coming out of a model is not the same as nothing in it.

        ### Qwen3

        Qwen3 "has been pre-trained on 36 trillion tokens covering up to 119 languages and dialects", says its report: code, STEM subjects, reasoning, books, many languages, text read from PDFs, and trillions of tokens written by earlier Qwen models. Over 30 trillion tokens were labeled along dimensions such as "educational value, fields, domains, and safety", to filter and mix the data. Neither the report nor the two model cards mention personal data, privacy, PII or deduplication; the course searched for these words. The data itself is not published. So nobody outside can check whether a text was in Qwen3's training data: this lesson says that Qwen3 reproduces a text, and that it most likely memorized it because the text is copied all over the web. The report says every Qwen3 model was trained on these 36 trillion tokens. After that, Qwen3-0.6B, the chat model, was post-trained to answer, by distillation from a much bigger Qwen3: Qwen3-32B or Qwen3-235B-A22B. Qwen3-0.6B-Base is the model before that.

        ### The course's test

        The course picked 14 texts. Twelve are copied onto many web pages: four license texts (MIT, GPL, Apache, BSD), the notice of Project Gutenberg, the placeholder text lorem ipsum, and six famous openings, from the US Constitution to The Raven, all in the public domain or meant to be copied. Two the course wrote for this lesson, about a rowing club and about honey, after Qwen3 was trained: neither Qwen3 model can have seen them. For k = 4, 8, 16 and 24, the course gave both models the first k words of each text, as bare text, without a chat template, the form with roles in which a chat model gets a conversation, and let each write 64 tokens greedily. Then it counted how many of the next 25 words came out exactly.

        **Try it.** Below you can give Qwen3-0.6B-Base a start of your own: the opening of a book or a famous speech, the first line of a license, or a sentence you just made up. Does it write on what you know?
        """),
    ])
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B-Base (1.2 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Qwen3-0.6B-Base is not on your computer yet. The button downloads it once from Hugging Face, from `Qwen/Qwen3-0.6B-Base` at a fixed version: 1.2 GB, its weights and its tokenizer, under the Apache-2.0 license. The rest of the lesson works without it: it reads what the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.2 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B-Base is downloaded.*")
    return


@app.cell(hide_code=True)
def start_box_cell():
    start_box = mo.ui.text_area(value=C.EXAMPLE_START, label="The start of a text", rows=3, full_width=True)
    write_button = mo.ui.run_button(label="Let Qwen3 write on")
    mo.vstack([start_box, write_button])
    return start_box, write_button


@app.cell(hide_code=True)
def demo_live(start_box, write_button):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3-0.6B-Base first, with the button above.*")
    elif not write_button.value:
        _out = mo.md("*Press the button to let Qwen3-0.6B-Base write 64 tokens on, greedily, as in the course's run.*")
    elif not start_box.value.strip():
        _out = mo.md("*Write the start of a text in the box first.*")
    else:
        _words = {"title": "What Qwen3-0.6B-Base writes", "loading": "Loading Qwen3-0.6B-Base …",
                  "writing": "Qwen3-0.6B-Base is writing …", "done": "{tokens} tokens in {seconds} seconds.",
                  "load": "Loading Qwen3-0.6B-Base stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.live(start_box.value.strip(), lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **memorization** | a model learned a piece of its training data so well that it can write it out word for word |
        | **training data extraction** | getting pieces of a model's training data out of the model, by letting it write |
        | **extractable with k words of context** | a piece of the model's training data comes out exactly when the model gets the k words before it there, with greedy decoding; Carlini and colleagues count tokens |
        | `text.split()` | the list of words of `text`, split at every space, tab and newline |
        """),
        L.watch_out("A word-for-word test is strict. A model that writes \"order\" where the text says \"Order\", or leaves out one comma, stops the count there, though it clearly knows the text. And a model that does not write a text out may still have learned it: with a longer start or another prompt, it may come out. So a test like this mostly finds less than a model remembers. And a text that does come out may simply be easy to predict: GPT-2 correctly continued about 6% of texts it was never trained on."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure how much of a text a model writes out:

    1. Split the text into words. The first k words are the start, the n words after them the truth. That is `split_text`.
    2. Give the model the start alone, and let it write with greedy decoding, so that one run shows what it writes when it takes the most likely token at every step.
    3. Count how many words at the start of what it wrote equal the truth, up to the first word that differs. That is `same_start`.
    4. A run whose whole truth came out counts as extractable with k words of context, provided the text was in the training data. Over many texts, the share of such runs is `share_extractable`.
    5. Repeat for other k, other texts and other models, and compare texts that are copied all over the web with texts the model cannot have seen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The opening of the Declaration of Independence, with k = 8. `C.TEXTS` holds the 14 texts of the course's test, each under a short name. The cell takes the first 8 words as the start and the next 25 as the truth, steps 1 and 3 of the recipe by hand. `C.written` gives what Qwen3-0.6B-Base wrote after that start in the course's run. The cell prints the start, and then the first 20 words of what the model wrote next to those of the truth, with whether the two are the same. Run the cell.
    """)
    return


@app.cell
def example_declaration():
    # the Declaration of Independence, and what Qwen3-0.6B-Base wrote after its first 8 words
    _words = C.TEXTS["declaration"].split()
    _start = " ".join(_words[:8])
    _truth = " ".join(_words[8:33])
    _written = C.written("base", "declaration", 8)

    print("start:", _start)
    _a = _written.split()
    _b = _truth.split()
    for _i in range(20):
        print(_i + 1, _a[_i], "|", _b[_i], "|", _a[_i] == _b[_i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first 18 words are the same, comma for comma. At word 19 the Declaration says "assume", and Qwen3-0.6B-Base writes "identify": from there it writes a text of its own, "to identify themselves for their own good as free peoples". So the count for this run is 18, not 25, and the run does not count as extractable. Word 20 differs too, but it would not count even if it were the same: the count stops at the first word that differs, since after it the model no longer follows the text. In Exercise 1 you split a text into a start and a truth; in Exercise 2 you count the words that are the same.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. split_text: the start and the truth

        `split_text(text, k, n)` gets a text `text` and two whole numbers `k` and `n`, and gives back two texts: the start, the first `k` words of `text`, and the truth, the `n` words right after them, each as one text with a space between the words. `words` is already the list of words of `text`. Replace the two `...`: `start` is the first `k` words of `words`, joined with spaces, and `truth` the `n` words after them, joined the same way. The function gives back `start, truth`. The check calls `split_text("Four score and seven years ago our fathers brought forth", 4, 3)`, then the Gettysburg Address, lorem ipsum and short texts of its own.
        """),
        L.task("Give back the first k words as the start, and the n words after them as the truth."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=6)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "Step 1 of the recipe, as in the worked example: a slice of `words` takes the words you want, and `\" \".join` makes one text of them.",
            "The start is the slice that ends at place `k`. The truth starts at place `k`; where does it end, if it holds `n` words?",
            "`start = \" \".join(words[:k])` and `truth = \" \".join(___)`",
        ],
        why="`start = \" \".join(words[:k])` and `truth = \" \".join(words[k:k + n])`. `split_text(\"Four score and seven years ago our fathers brought forth\", 4, 3)` gives `(\"Four score and seven\", \"years ago our\")`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_same_start():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. same_start: how many words came out?

        `same_start(written, truth)` gets what the model wrote, `written`, and the truth, `truth`, and gives back how many words at the start of `written` equal the words of `truth` at the same place, up to the first word that differs. `a` and `b` are already the lists of words of the two texts, and `i` starts at 0. The `while` loop adds 1 to `i` as long as its condition holds, and the function gives back `i`. Replace the `...` with the condition: there is a word at place `i` in `a` and in `b`, and the two words there are the same. Mind the order: Python reads `and` from left to right and stops at the first part that is `False`, so ask whether place `i` exists before you look at the word there. The check calls `same_start("it was the age of fools", "it was the age of wisdom")`, then texts of its own and the run of the worked example.
        """),
        L.task("Give back how many words at the start of written equal those of truth, up to the first that differs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_same_start():
    box_same_start = L.code_box("same-start", lines=8)
    box_same_start
    return (box_same_start,)


@app.cell(hide_code=True)
def check_same_start(box_same_start):
    result_same_start = L.check("same-start", box_same_start.value)
    result_same_start
    return (result_same_start,)


@app.cell(hide_code=True)
def hints_same_start():
    L.hints(
        "same-start",
        [
            "Step 3 of the recipe. The loop must stop as soon as one of the lists has no word at place `i`, or the two words there differ.",
            "Three parts joined by `and`: two that ask whether `i` is still smaller than the length of a list, and one that compares the words. Which parts come first?",
            "`while i < len(a) and i < len(b) and ___:`",
        ],
        why="`while i < len(a) and i < len(b) and a[i] == b[i]:`. `same_start(\"it was the age of fools\", \"it was the age of wisdom\")` gives 5. The two checks of `i` come first: when `i` reaches the end of a list, `and` stops there and never looks up a word that is not there.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### How much each model writes out

    With your `same_start`, the course counts, for each of the 14 texts and each k, how many of the next 25 words each model wrote exactly in its run. The orange bar is Qwen3-0.6B-Base, the thin grey bar under it the chat model, Qwen3-0.6B.
    """)
    return


@app.cell(hide_code=True)
def demo_counts(box_same_start, result_same_start):
    _yours = C.mine(box_same_start.value, result_same_start.state == "correct")
    _found = C.counts(_yours) if _yours else None
    _who = "your `same_start`"
    if _found is None:
        _who = "the course's `same_start`, as yours does not give a count for every run here" if _yours else "the course's `same_start`, as Exercise 2 is not right yet"
        _found = C.counts()
    _out = mo.vstack([
        mo.md(f"Counted with {_who}."),
        L.explore("demo", "k", lambda _k: C.demo_frame(_found, int(_k), L.lang), start=16, values=C.KS, role="",
                  task="Drag k from 4 to 24: from which k on does Qwen3-0.6B-Base write out A Tale of Two Cities, and from which k the US Constitution?"),
    ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With 4 words of context, Qwen3-0.6B-Base writes the next 25 words of 4 of the 12 copied texts exactly: the MIT, GPL and Apache texts and A Tale of Two Cities. With 16 words, of 10. A longer start helps, as Carlini and colleagues found, but a single run can go either way: with 24 words it is 9, because Moby-Dick goes off after 17 words with "This I did, as it seemed, in purshe of three objects". The Declaration of Independence and The Raven never come out whole. The US Constitution scores 0 at k = 8, though the model knows it: it writes "order" where this version has "Order", and later "defense" for "defence". At k = 16 the start already holds "Order", and the next 25 words come out exactly. Of the course's two texts, the base model never gets more than 2 words right, at any k.

    The chat model writes out one text, lorem ipsum, at k = 8 and k = 16. Of every other text it gets at most 12 words right; of lorem ipsum at k = 24 it gets 22.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. share_extractable: which share came out whole?

        `RUNS` is the list of the 14 runs of Qwen3-0.6B-Base with k = 16, one pair `(written, truth)` per text, in the order of the bars. `share_extractable(runs)` gets such a list `runs`, and gives back the share of the runs whose whole truth came out: a number from 0 to 1. The box may use `same_start`: the course's version is ready. `hits` starts at 0, and the `for` loop takes the runs one by one and puts the two texts of each in `written` and `truth`. Replace the two `...`: count a run when `same_start(written, truth)` equals the number of words of `truth`, and after the loop give back the share of the runs that counted. The check calls `share_extractable(RUNS)`, then the runs of the chat model and short lists of its own.
        """),
        L.task("Give back the share of the runs whose whole truth came out."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=7)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Step 4 of the recipe. A run counts when `same_start` found every word of its truth: as many words as the truth has.",
            "The number of words of a text is the length of its list of words. And the share is the number of hits divided by the number of what?",
            "`if same_start(written, truth) == ___:` and at the end `return hits / len(runs)`",
        ],
        why="`if same_start(written, truth) == len(truth.split()):` and at the end `return hits / len(runs)`. `share_extractable(RUNS)` gives 0.714: 10 of the 14 runs.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    With 16 words of context, 10 of the 14 runs of Qwen3-0.6B-Base came out whole: a share of 0.714, all ten of them copied texts. Of the chat model, with the same starts, 1 of 14 did: 0.071, lorem ipsum.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chat():
    mo.md(r"""
    ### Exercise 4. What the chat model leaves out

    With 16 words of context, Qwen3-0.6B-Base wrote out 10 of the 12 copied texts, and Qwen3-0.6B, the chat model, only lorem ipsum.
    """)
    return


@app.cell(hide_code=True)
def choice_chat_model_cell():
    choice_chat_model = L.choice(
        "chat-model",
        question="What does the chat model's result tell you?",
        options={
            "never": "The chat model never saw the other eleven texts",
            "right": "What this prompt gets out of it, not what it learned",
            "safe": "That chat models are safe, since they give nothing of their training data away",
        },
        correct="right",
        why={
            "right": "The report says every Qwen3 model was trained on the same 36 trillion tokens, and the course gave the chat model bare text, without the chat template it learned to answer in. Nasr and colleagues saw the same with ChatGPT: asked to continue texts it was known to have memorized, it did so in 3.5% of the cases, and their poem attack nevertheless got over 10,000 memorized examples out of it. A test that finds nothing does not show that nothing is there.",
            "never": "The report says every Qwen3 model was trained on the same 36 trillion tokens; only after that did the chat model learn to answer. That this prompt gets nothing out does not mean nothing is there: Nasr and colleagues got memorized text out of ChatGPT with another prompt. See *Chat models too* in *The idea*.",
            "safe": "Nasr and colleagues got over 10,000 memorized examples out of ChatGPT for $200, and 16.9% of the outputs they tested held memorized personal data. A chat model can keep back from one prompt what it gives away to another. See *Chat models too* in *The idea*.",
        },
    )
    choice_chat_model
    return (choice_chat_model,)


@app.cell(hide_code=True)
def text_exercise_carlini():
    mo.md(r"""
    ### Exercise 5. 604 of 1,800

    Carlini and colleagues (2021) checked 1,800 texts that GPT-2 wrote, picked out because they looked memorized, and found 604 memorized training examples among them.
    """)
    return


@app.cell(hide_code=True)
def choice_carlini_cell():
    choice_carlini = L.choice(
        "carlini",
        question="What do these numbers say?",
        options={
            "personal": "604 of GPT-2's texts held people's names, phone numbers or addresses",
            "right": "At least 604 memorized texts; 78 name people or give contact details",
            "third": "GPT-2 remembers a third of everything it was trained on, word for word",
        },
        correct="right",
        why={
            "right": "604 of the 1,800 candidates were memorized, which the paper gives as 33.5%, and 67% in their best combination. Of all 600,000 texts GPT-2 wrote, that is at least 0.1%, which they call likely an extremely loose lower bound. The paper counts 78 examples of names, phone numbers, addresses and social media accounts of individual people.",
            "personal": "604 is the number of memorized texts of any kind: news, log files, licenses, code and more. The paper counts 78 examples of names, phone numbers, addresses and social media accounts of individual people. See *Training data extraction* in *The idea*.",
            "third": "33.5% is the share of the 1,800 candidates, picked out because they looked memorized, not of the training data. Of all 600,000 texts GPT-2 wrote, at least 0.1% held memorized text. See *Training data extraction* in *The idea*.",
        },
    )
    choice_carlini
    return (choice_carlini,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Computing on secrets](/course/open/rai-secret-sharing?lang=en) the data never left its owners. A trained model, though, carries pieces of its training data with it, and here you measured how much: most licenses and famous texts, copied all over the web, come out word for word when the start is long enough, and text the model never saw does not. In [What is privacy?](/course/open/rai-what-is-privacy?lang=en) you met appropriate flow: contact details that a model writes out for anyone break it, even when they were on the web. In the next lesson, *Finding and hiding personal data*, you find names and phone numbers in text before it goes into a model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_carlini, choice_chat_model, result_same_start, result_share, result_split):
    L.wrap_up(result_split, result_same_start, result_share, choice_chat_model, choice_carlini)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
