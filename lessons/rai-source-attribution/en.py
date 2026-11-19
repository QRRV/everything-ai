# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-source-attribution. Route: course/route/lessen/verantwoord.yaml. Checks and the course's run of Qwen3: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-source-attribution", "en")
    C = L.checks   # the course's sources and questions, and Qwen3's answers with every source and with one left out


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Can a model explain itself?](/course/open/rai-llm-self-explanations?lang=en) and [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_faithful():
    quick_faithful = L.choice(
        "quick-faithful",
        question="A model explains its answer, and every fact in the explanation is right. Is the explanation faithful?",
        options={
            "yes": "Yes: right facts show what led to the answer",
            "right": "Not always: it may not be what led to it",
            "size": "Only when the model is large enough to know",
        },
        correct="right",
        why={
            "yes": "Right facts make an explanation plausible. Faithful means it shows what really led to the answer, and the planet answer in [Can a model explain itself?](/course/open/rai-llm-self-explanations?lang=en) had right facts that led to (A) while the answer was (B).",
            "right": "Plausible is how convincing an explanation is; faithful is whether it shows what led to the answer. This lesson asks the same about a citation: a source can say what the answer says without being where the answer came from.",
            "size": "Size does not make an explanation faithful; Lanham and colleagues even found less faithful reasoning in larger models on most of their tasks. See [Can a model explain itself?](/course/open/rai-llm-self-explanations?lang=en).",
        },
    )
    quick_faithful
    return (quick_faithful,)


@app.cell(hide_code=True)
def quick_check_change():
    quick_change = L.choice(
        "quick-change",
        question="How did the course test whether Qwen3's explanation says what changed its answer?",
        options={
            "ask": "It asked Qwen3 whether its explanation was honest",
            "right": "It added a hint and saw whether the explanation named it",
            "read": "It read the explanations and judged how good they sounded",
        },
        correct="right",
        why={
            "ask": "Its answer to that would be one more self-explanation, text it writes like any other. The test changed something that should not matter, a hint, and looked at what the explanation said. See [Can a model explain itself?](/course/open/rai-llm-self-explanations?lang=en).",
            "right": "Change something, and see whether the answer and the explanation change with it. This lesson does the same with sources: leave one out, and see whether the answer changes.",
            "read": "Reading cannot tell plausible from faithful. The test changed something that should not matter, a hint, and looked at what the explanation said. See [Can a model explain itself?](/course/open/rai-llm-self-explanations?lang=en).",
        },
    )
    quick_change
    return (quick_change,)


@app.cell(hide_code=True)
def quick_check_cite():
    quick_cite = L.choice(
        "quick-cite",
        question="A model answers from a text you give it. Why ask it for a citation, such as [2]?",
        options={
            "right": "So you can check the answer against that place",
            "true": "So the answer is right: a model cannot cite what is not there",
            "fast": "So the model answers faster",
        },
        correct="right",
        why={
            "right": "A citation names the place in the given text that a part of the answer comes from, so you can look it up. Whether that place says what the answer says, and whether the answer really came from it, is what this lesson tests.",
            "true": "A model can write any number in brackets, also a wrong one. A citation lets you check the answer; it does not make it right. See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
            "fast": "A citation is a few more tokens to write, not fewer. It lets you check the answer against the text. See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
        },
    )
    quick_cite
    return (quick_cite,)




@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Answers with sources look checkable. A chatbot that writes [2] after a sentence seems to say: this is where I got it. But a citation can point to a source that does not say what the answer says. And even a citation that holds up need not be where the answer came from: the model may have answered from what it remembered, and then cited a source that fits. In this lesson you test both, on Qwen3-0.6B: does the source it cites say what the answer says, and does the answer change when you leave that source out?

    **An everyday example.** A student hands in an essay with a list of sources. A teacher can look up whether a source says what the essay claims. Whether the student got the idea from that source, or wrote the essay first and found a source that fits afterwards, the list cannot show.
    """)
    return


@app.cell(hide_code=True)
def predict_wrote():
    guess_wrote = L.guess(
        "wrote",
        question="The course asked Qwen3 12 questions, each with five numbered sources, and asked it to say so when the sources do not answer. Then it asked each question again without the source Qwen3 had cited. What did Qwen3 write most often?",
        options={
            "says": "That the sources do not answer the question",
            "memory": "The right answer anyway, from what it knows",
            "other": "Another answer, with a citation in it",
        },
        reveal="Another answer, with a citation in it: in 8 of the 12, such as \"The ticket for the ferry costs [5] dollars.\" In 3, it said that the sources do not answer, and only the library's opening time stayed right, because another source says it too. In the exercises below, you find out which sources each answer needed.",
    )
    guess_wrote
    return (guess_wrote,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Source attribution.** In [Prompts that check themselves](/course/open/start-prompt-chains?lang=en), a model answered from a text you gave it, with a citation, so that you could check the answer. Showing which sources an answer is based on is called **source attribution**. A citation raises two questions that are easy to mix up: does the source say what the answer says, and did the answer come from that source?

    **Does the source say it?** Nelson F. Liu, Tianyi Zhang and Percy Liang (2023) call an answer verifiable when "each generated statement about the external world should be fully supported by a set of in-line citations, and each provided citation should support its associated statement". They measured both in four generative search engines of early 2023, Bing Chat, NeevaAI, perplexity.ai and YouChat, on 1,450 queries: citation recall, the share of the statements worth checking that their citations fully support, and **citation precision**, the share of citations that support their statement. Over the four engines, "a mere 51.5% of generated statements are fully supported with citations (recall), and only 74.5% of citations fully support their associated statements (precision)". Per engine, recall ran from 68.7% for perplexity.ai down to 11.1% for YouChat. And the engines whose answers people found more useful had less precise citations (a correlation of −0.96): Liu and colleagues suspect that this is because answers that copy their sources closely have precise citations, but seem less useful. Hannah Rashkin and colleagues (2023) give support a fixed test: a reader must be able to say "According to P, s", with P the source and s the statement; if only part of the statement is in the source, it does not count. Tianyu Gao and colleagues (2023) let a model judge this, on their benchmark ALCE; it agreed with people substantially for citation recall and moderately for citation precision. And supported is not the same as true: Rashkin and colleagues leave out whether a statement is true, and Liu and colleagues write that "verifiability is not factuality".

    **Did the answer come from it?** A citation can support its statement without the model having used it. Jonas Wallat, Maria Heuss, Maarten de Rijke and Avishek Anand (2024) call the first question citation correctness, and keep faithfulness for the second: a citation is faithful when the cited document supports the statement and also had a causal effect on it. A model can answer from its own memory and then look for a document that matches, in a shallow way, such as by matching words; they write that such "unfaithful attributions might still be factually correct and, therefore, difficult to spot". They tested this on Command-R+, a model of Cohere, by adding a short statement that the model had written with a citation, such as "Emperor penguin", to other documents. When the model gave the same statement again, it cited such an altered document in 12% of the cases when the document was a random one, and in 57% when the document was relevant but had not been cited before. Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev and Aleksander Madry (2024) put the test of the second question in one sentence: "if a part of the context actually led to a particular generated response, then removing it should substantially affect this response". With words from a paper they cite, they call a source that caused the answer contributive, and one that supports it corroborative.

    **Leave one out.** The simplest test of the second question: leave one source out, ask again, and see whether the answer changes. Do it for every source, and that is **leave-one-out**. Cohen-Wang and colleagues compare their method, ContextCite, with it: by how much less likely the answer becomes without the most important source, leave-one-out is the best there is, but it costs one more run of the model for every source. ContextCite instead runs the model with random sets of sources left out, in one of their examples 32 times for 98 sources, and fits a small linear model on the results. They also found that the model's attention is no safe shortcut: with one model it came close to ContextCite, with another it did poorly.

    **What leaving out cannot show.** If two sources say the same thing, leaving one out changes little: the other one still says it. Cohen-Wang and colleagues name that case for their own linear model, and it holds for leave-one-out too. If the model already knew the answer, leaving the source out may change nothing either, even though the source supports the answer: a low score "even for a source that supports a statement", in their words. And removing a sentence can change what a word such as "it" in another sentence refers to.

    **The course's test.** The course wrote four short texts about a made-up island, Kestrel Island, each in five numbered sources, and 12 questions about them, each answered by one or two of the sources. Two sources of the library's text give the same opening time in other words, and the quiz night's text holds two facts that a model has seen often, the capital of France and the boiling point of water. Qwen3 got each question with its five sources, and the instruction to answer in one short sentence with the number of the source in square brackets, or to say so when the sources do not answer. Then the course asked each question once more for every source, with that source left out and the other numbers kept: with the first round, 72 prompts. It asked without thinking and greedily, always taking the most likely next token, so that each prompt has one answer. Qwen3's model card warns against greedy decoding in its thinking mode, which this run does not use. Whether an answer is right, the course decides by whether it holds the words of the right answer, such as 7:40 or 6 euros.

    **Try it.** Below you can ask Qwen3 one of the course's questions with its five sources. Delete a source, ask again, and see whether the answer and its citation change.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the answers the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def ask_box_cell():
    ask_box = mo.ui.text_area(value=C.ASK_EXAMPLE, label="Your sources and question", rows=13, full_width=True)
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([ask_box, ask_button])
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3, without thinking and greedily, as in the course's run. Delete a line with a source to leave it out.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Write sources and a question in the box first.*")
    else:
        _words = {"title": "What Qwen3 writes", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds.",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(chat.demo(C.ask_messages(ask_box.value), lambda _html: mo.output.replace(mo.Html(_html)), _words,
                                 L.lang, max_tokens=C.MAX_TOKENS, temperature=0,
                                 note=lambda _text: C.cited_note(ask_box.value, _text, L.lang)))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **source attribution** | showing which sources an answer is based on, for example with citations, so that readers can check it |
        | **citation precision** | the share of the citations in answers that support the statement they belong to |
        | **leave-one-out** | leaving out one source at a time, asking again, and seeing whether the answer changes |
        """),
        L.watch_out("A citation that supports the answer does not show that the answer came from it: the model may have known the answer, or found a source that fits afterwards. And a source whose removal changes nothing may still have been used, when another source says the same thing."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Testing a model's citations against its sources:

    1. Give the model numbered sources and a question, and ask for a short answer with the number of the source in square brackets, or to say so when the sources do not answer.
    2. Read which numbers the answer cites. That is `cited`.
    3. For every citation, look whether that source says what the answer says. The share of citations that do is the citation precision.
    4. Ask the question again once for every source, each time with that source left out and the other numbers kept.
    5. Note the sources without which the answer no longer holds the right answer. That is `needed`.
    6. Count the share of the citations whose source the answer needed: the citations that leaving out confirms. That is `confirmed_share`.
    7. Write down what neither can show: a cited source the answer did not need may still be the one it used, when another source says the same; and a source that supports the answer need not be where it came from.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up answer that cites two sources, and the answers to the same question with one of the five sources left out. The cell goes over the five, and prints for each source whether the answer cites it, and whether the answer without it still says 9:00.
    """)
    return


@app.cell
def example_left_out():
    # a made-up answer with all five sources, and the answers with one source left out
    _answer = "The museum opens at 9:00 [3][5]."
    _left_out = [(1, "The museum opens at 9:00 [3][5]."),
                 (2, "The museum opens at 9:00 [3][5]."),
                 (3, "The sources do not say when the museum opens."),
                 (4, "The museum opens at 9:00 [3][5]."),
                 (5, "The museum opens at 9:00 [3].")]

    for _number, _text in _left_out:
        _cited = f"[{_number}]" in _answer
        _still = "9:00" in _text
        print("source", _number, "| cited:", _cited, "| without it still 9:00:", _still)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The answer cites [3] and [5]. Without source 3, it no longer says 9:00: the answer needed it, so leaving it out confirms that citation. Without source 5, it still says 9:00: leaving it out does not confirm [5]. Perhaps the answer did not use it, or another source says the same. Of the 2 citations, leaving out confirms 1: a share of 0.5. `f"[{_number}]" in _answer` is True when the answer cites that source.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cited():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. cited: which sources does an answer cite?

        `cited(text, n)` gets an answer `text` and the number of sources `n`, and gives back, smallest first, the numbers from 1 to `n` that the text cites in square brackets, such as `[2]`. `found` starts as an empty list, and the `for` loop goes over every number `i` from 1 to `n`. Replace the `...` with two lines: when the text holds the number `i` between square brackets, add `i` to `found`. The function gives back `found`. The check calls `cited(ANSWER, 5)`, with `ANSWER` Qwen3's answer to the first question, "The ferry to Kestrel Island leaves at 7:40 every morning [1].", then other texts.
        """),
        L.task("Give back the numbers from 1 to `n` that the text cites in square brackets."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cited():
    box_cited = L.code_box("cited", lines=7)
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
            "Step 2 of the recipe, as `_cited` in the worked example. Look for the number with its square brackets, so that the 4 in 7:40 does not count.",
            "An `if` that asks whether the text holds `i` between square brackets, and under it `found.append(i)`.",
            "`if ___ in text:`",
        ],
        why="`if f\"[{i}]\" in text:` and under it `found.append(i)`. `\"[\" + str(i) + \"]\"` works too. On Qwen3's 12 answers with all five sources, `cited` finds one citation in each.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_needed():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. needed: which sources could the answer not do without?

        `needed(without, value)` gets a list `without` of Qwen3's answers to one question with one source left out, each a tuple of two: the number of the source that was left out and the answer. It also gets `value`, the words of the right answer, such as `"7:40"`. `found` starts as an empty list, and the `for` loop takes each tuple apart into `number` and `text`. Replace the `...` with two lines: when `value not in text`, which is True when the answer no longer holds the words, add `number` to `found`. The function gives back `found`. The check calls `needed(WITHOUT, "7:40")`, with `WITHOUT` Qwen3's five answers to the first question, each with one source left out, then `needed(OPENING, "10:00")`, its answers about the library's opening time, and short lists.
        """),
        L.task("Give back the numbers of the sources without which the answer no longer holds `value`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_needed():
    box_needed = L.code_box("needed", lines=7)
    box_needed
    return (box_needed,)


@app.cell(hide_code=True)
def check_needed(box_needed):
    result_needed = L.check("needed", box_needed.value)
    result_needed
    return (result_needed,)


@app.cell(hide_code=True)
def hints_needed():
    L.hints(
        "needed",
        [
            "Step 5 of the recipe, as `_still` in the worked example, but the other way round: you want the sources without which the answer no longer holds `value`.",
            "An `if` with `not in`, which asks whether the words of the right answer are missing from the answer, and under it add the number of the source to `found`.",
            "`if value not in ___:`",
        ],
        why="`if value not in text:` and under it `found.append(number)`. `needed(WITHOUT, \"7:40\")` is `[1]`: only without source 1 does the answer lose 7:40. For the library's opening time, `needed(OPENING, \"10:00\")` is `[]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_confirmed():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. confirmed_share: how many citations does leaving out confirm?

        `confirmed_share(rows)` gets a list `rows` with a dict per question: `"cited"`, the numbers that the answer with all five sources cites, as your `cited` finds them, and `"needed"`, the numbers without which the answer no longer holds the right answer, as your `needed` finds them. Each dict also has `"question"`, the number of the question counted from 0, and `"says"`, the sources that say the right answer; this exercise needs neither. `citations` and `confirmed` start at 0, and the `for` loop goes over every `row`. Replace the `...` with four lines: a second `for` loop over every `number` in `row["cited"]`, in which you add 1 to `citations`, and also 1 to `confirmed` when `number in row["needed"]`. The function gives back `confirmed / citations`. The check calls `confirmed_share(ROWS)`, with `ROWS` the 12 questions of the course's run, then three short lists.
        """),
        L.task("Give back, of all citations, the share whose source the answer needed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_confirmed():
    box_confirmed = L.code_box("confirmed", lines=10)
    box_confirmed
    return (box_confirmed,)


@app.cell(hide_code=True)
def check_confirmed(box_confirmed):
    result_confirmed = L.check("confirmed", box_confirmed.value)
    result_confirmed
    return (result_confirmed,)


@app.cell(hide_code=True)
def hints_confirmed():
    L.hints(
        "confirmed",
        [
            "Step 6 of the recipe. A citation is one number in `row[\"cited\"]`, so you need a second loop inside the first.",
            "`for number in row[\"cited\"]:`, inside it `citations = citations + 1`, and an `if` that asks whether the number is also among the sources the answer needed.",
            "`if number in ___:`",
        ],
        why="Inside the loop over the rows: `for number in row[\"cited\"]:`, inside it `citations = citations + 1` and `if number in row[\"needed\"]:` with under it `confirmed = confirmed + 1`. `confirmed_share(ROWS)` is 0.917: leaving out confirms 11 of the 12 citations.",
    )
    return


@app.cell(hide_code=True)
def text_measured():
    mo.vstack([
        mo.md(r"""
        ### What Qwen3's answers show

        With all five sources, Qwen3 cited one source in each of the 12 answers, and every cited source holds the words of the right answer: by the course's reading, a citation precision of 1.0. Leaving out the cited source took the right answer away in 11 of the 12; leaving out any other source took it away in none. So leave-one-out confirms 11 of the 12 citations, a share of 0.917. The one it does not confirm is the library's opening time: without [2], Qwen3 gave the same time and cited [4], and without [4], it answered as with all five. Even the capital of France needed its source: without [2], Qwen3 wrote "The capital of France is [1]."
        """),
        mo.Html(C.measured_picture("en")),
        mo.md(r"""
        Without the source it cited, Qwen3 said in only 3 of the 12 questions that the sources do not answer, and even then it put numbers in brackets. In 8, it wrote another answer with a citation in it: a wrong one, such as "The trip to the island takes 7 hours and 30 minutes. [1]", a broken one, such as "The ticket for the ferry costs [5] dollars.", or one about something else, such as "The bakery is closed on Mondays. [5]" to the question when the bakery closes. A reader who only looks whether there is a citation would not notice. The test is small: one small model, asked greedily, 12 easy questions with short sources the course wrote, and right decided by matching words. It shows which sources the answers could not do without; it does not show how Qwen3 used them, and where two sources say the same thing, as at the library, it cannot tell which one was used.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_duplicate():
    mo.md(r"""
    ### Exercise 4. Two sources that say the same

    Two sources of the library's text say when the library opens on Saturdays: [2] "The library opens at 10:00 on Saturdays." and [4] "On Saturdays, the doors of the library open at 10:00." With all five sources, Qwen3 answered "The library opens at 10:00 on Saturdays. [2]". Without [2], it answered "The library opens on Saturdays at 10:00. [4]", and without [4], the same as with all five.
    """)
    return


@app.cell(hide_code=True)
def choice_duplicate_cell():
    choice_duplicate = L.choice(
        "duplicate",
        question="What does leave-one-out tell you about where this answer came from?",
        options={
            "neither": "That it came from neither of the two sources",
            "right": "Little: one of the two was enough",
            "both": "That it needed both sources together",
            "memory": "That Qwen3 already knew the opening time",
        },
        correct="right",
        why={
            "neither": "Leaving out one source changed nothing, because the other one still says 10:00. That does not show that neither was used; it shows that each one alone was enough. See *What leaving out cannot show* in *The idea*.",
            "right": "Leave out either one, and the other still gives 10:00, so no single removal takes the right answer away, and `needed` gives an empty list. The answer may come from [2], as it cites, or from [4]; leaving out one at a time cannot tell. Leaving out both at once would show whether it needs at least one of them.",
            "both": "Then leaving out one would have taken the right answer away. It did not: each one alone was enough. See *What leaving out cannot show* in *The idea*.",
            "memory": "The island and its library are made up for the course, so Qwen3 cannot have known the time. Leaving out one source changed nothing because the other one still says 10:00.",
        },
    )
    choice_duplicate
    return (choice_duplicate,)


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Exercise 5. A headline
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="A news site writes, about the study of Liu and colleagues: \"Half of what AI search engines say is wrong.\" What does the study support?",
        options={
            "exact": "The headline, as it stands",
            "right": "Less: about half lacked full support by citations",
            "more": "More: a quarter of the citations were made up",
            "useful": "Nothing: it measured only how useful answers seemed",
        },
        correct="right",
        why={
            "exact": "Liu and colleagues measured whether statements were supported by their citations, not whether they were true: \"verifiability is not factuality\". 51.5% of the statements were fully supported; the others can still be true. And it was four engines, in early 2023. See *Does the source say it?* in *The idea*.",
            "right": "51.5% of the statements worth checking were fully supported by their citations, and 74.5% of the citations supported their statement, in four engines in early 2023. A statement without full support is not therefore wrong: verifiability is not factuality.",
            "more": "74.5% of the citations fully supported their statement. The others did not fully support it, which does not mean that they were made up. See *Does the source say it?* in *The idea*.",
            "useful": "They had people judge citation recall and citation precision. How useful people found the answers was a separate question, and it went with less precise citations. See *Does the source say it?* in *The idea*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Can a model explain itself?](/course/open/rai-llm-self-explanations?lang=en), you tested whether an explanation says what changed the answer; a citation is another explanation that a model writes, and leaving a source out tests it the same way: change something, and see whether the answer changes with it. [Prompts that check themselves](/course/open/start-prompt-chains?lang=en) asked for citations so that you can check an answer; this lesson showed that a citation that checks out need not be where the answer came from, and that an answer without its source can still carry a citation. This is the last lesson of the module on explaining models; the next module starts with *What is privacy?*
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_duplicate, choice_headline, result_cited, result_confirmed, result_needed):
    L.wrap_up(result_cited, result_needed, result_confirmed, choice_duplicate, choice_headline)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
