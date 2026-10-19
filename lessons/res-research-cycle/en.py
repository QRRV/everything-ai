# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-research-cycle. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-research-cycle", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_review():
    quick_review = L.choice(
        "quick-review",
        question="Who does the peer review of a paper?",
        options={
            "right": "A few other researchers in the same field",
            "journalists": "The journalists who write about the paper",
            "authors": "The authors themselves, before they put it online",
        },
        correct="right",
        why={
            "journalists": "Journalists write about a paper, often after it appeared; they do not judge it for a conference or journal. See [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).",
            "authors": "Checking your own work is not peer review: the peers are other researchers. See [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).",
            "right": "A conference or a journal sends the paper to a few experts, the peers, who advise whether to publish it and what must be fixed.",
        },
    )
    quick_review
    return (quick_review,)


@app.cell(hide_code=True)
def quick_check_paper():
    quick_paper = L.choice(
        "quick-paper",
        question="What does a paper report?",
        options={
            "results": "Only the results, so that others can use them",
            "news": "A short summary of a result, for the news",
            "right": "A piece of research, from its question to its results",
        },
        correct="right",
        why={
            "results": "Results alone cannot be checked: a paper also says what the question was and what the researchers did. See [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).",
            "news": "That is a news story or a press release, someone's summary of a paper. See [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).",
            "right": "And how it relates to earlier work, so that others can read it, check it and build on it.",
        },
    )
    quick_paper
    return (quick_paper,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A paper reports a piece of research. But how does the research itself go? Say you think that one sentence makes a chatbot give better answers. How do you find out whether that is true, in a way that others can check and trust? Knowing how researchers do that helps you read a paper: what did they want to know, what did they expect, and what did they do? And later in this course, you run experiments yourself, up to repeating the result of a paper.

    **An everyday example.** A doctor who thinks you have the flu, and wants to be sure, does not just say so. They order a test whose result could show that it is something else.
    """)
    return


@app.cell(hide_code=True)
def predict_plan():
    guess_plan = L.guess(
        "plan",
        question="Researchers want to know whether a chatbot solves maths word problems better when \"Let's think step by step.\" is added after each problem. Which plan tells them most?",
        options={
            "until": "Keep trying problems until they find some where the sentence helps, and show those",
            "right": "200 problems, each with and without the sentence; count the right answers",
            "ask": "Ask the chatbot whether the sentence helps it to think better",
        },
        reveal="**200 problems, each with and without the sentence.** Then the sentence is the only difference between the two counts, so a clear difference in right answers comes from it. Trying until it helps finds some problems where it helps, even if on average it does nothing. And what the chatbot says about itself measures nothing: it can say yes either way.",
    )
    guess_plan
    return (guess_plan,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Research goes round in a cycle. It starts with a **research question**: what do you want to find out? "Does a chatbot solve maths problems better when it is told to think step by step?"

        A question is not yet something you can test. So you turn it into a **hypothesis**: a precise answer you expect, stated so that a test could prove it wrong. "With the sentence added, the chatbot gets more of these 200 problems right." It says what you expect, on what, and what you will count. Before the test, you also say which result would prove it wrong: as many right answers with the sentence as without, or fewer.

        Then the **experiment**: a test of the hypothesis. You change one thing, keep everything else the same, and measure. Here you change only the sentence; the problems, the chatbot and the way you mark the answers stay the same. If you changed two things at once, you could not tell which one made the difference. A small difference can also be chance: a chatbot does not always give the same answer. Later in the course, you learn how big a difference has to be.

        Last, you write it up as a paper, with what you did and found, so that others can check it and repeat it. A result raises new questions, and the cycle starts again.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **research question** | what you want to find out |
        | **hypothesis** | a precise answer you expect to the question, stated so that a test could prove it wrong |
        | **experiment** | a test of a hypothesis: you change one thing, keep the rest the same, and measure what happens |
        """),
        L.watch_out("A result that agrees with the hypothesis does not prove it for good: it survived this test, on these problems, with this chatbot. A result against it from a fair test shows that it is wrong, at least for this test. That is why a good experiment is one that could have gone against you."),
        mo.md(r"""
        The picture: the research cycle.
        """),
        mo.Html(C.cycle({"title": "The research cycle",
                         "boxes": [("Question", "what do you want to find out?"),
                                   ("Hypothesis", "a precise answer you expect, which a test could prove wrong"),
                                   ("Experiment", "change one thing, keep the rest the same, and measure"),
                                   ("Paper", "write down what you did and found, so that others can check it")],
                         "again": "new questions",
                         "alt": "Four boxes in a square, with arrows going round. 1, Question: what do you want to find out? 2, Hypothesis: a precise answer you expect, which a test could prove wrong. 3, Experiment: change one thing, keep the rest the same, and measure. 4, Paper: write down what you did and found, so that others can check it. An arrow goes from the paper back to the question, marked new questions."})),
        mo.md(r"""
        The arrow back matters: a result raises new questions, and other researchers pick up where a paper ends.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To turn a question into an experiment:

    1. Write down the question: what do you want to find out?
    2. Make it a hypothesis: a precise statement of what you expect, about something you can count or measure.
    3. Say, before the experiment, which result would prove the hypothesis wrong.
    4. Design the experiment: change one thing, keep everything else the same, and measure.
    5. Write down what you did and found, also when it went against you, so that others can check it and repeat it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A real study from 2022, by Takeshi Kojima and four co-authors, with a generative AI for text, of the kind behind today's chatbots. The paper does not use these headings; here the study is written as the steps of the recipe.

    1. **Question:** does such a program solve maths word problems better when it is told to think step by step?
    2. **Hypothesis:** with "Let's think step by step." added after each problem, the program gets more of a fixed set of 600 maths word problems right.
    3. **Wrong if:** it gets as many right with the sentence as without, or fewer.
    4. **Experiment:** the same 600 problems and the same program, once with each problem alone and once with the sentence added. Every answer is marked right or wrong, and the right answers are counted both times.
    5. **Paper:** they put it on arXiv in May 2022, and it passed peer review at NeurIPS that year. The main program they tested got about 18% of the problems right without the sentence, and about 79% with it.

    The hypothesis survived this test. And the result raised new questions, which other researchers took up: why does the sentence help, and is there an even better one?
    """)
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which step?

    Six sentences, in mixed order, from one study that compares a chatbot in Dutch and in English. Put each with its step of the cycle.
    """)
    return


@app.cell(hide_code=True)
def exercise_steps():
    sort_steps = L.sort("cycle-steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("cycle-steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "cycle-steps",
        [
            "For each sentence, ask: is it something they want to know, a precise expectation, something they do, or something they share afterwards?",
            "Look back at the picture: the arrow from the paper goes back to the question. Which sentence comes after the result?",
            "Two sentences ask something they want to know, one of them only after the result. Exactly one is a precise expectation that a count could prove wrong. Two describe what they do to test it, and one what they do afterwards.",
        ],
        why="**Question:** worse at Dutch than at English; as big a gap for other languages. **Hypothesis:** fewer of the 300 test questions right in Dutch. **Experiment:** a translator translates the questions; the same chatbot answers them in both languages. **Paper:** write down what they did and found, and put it online.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_testable():
    mo.md(r"""
    ### Exercise 2. Can it be wrong?
    """)
    return


@app.cell(hide_code=True)
def choice_testable_cell():
    choice_testable = L.choice(
        "testable",
        question="Which of these is a hypothesis that an experiment could prove wrong?",
        options={
            "understands": "The chatbot truly understands the meaning of every question it answers",
            "future": "One day, chatbots will be better than people at everything they do",
            "right": "Told to answer in one word, the chatbot gets fewer of these 200 questions wrong",
            "maybe": "In Dutch, the chatbot may make a mistake now and then, or it may not, depending on the question",
        },
        correct="right",
        why={
            "understands": "Which count would show \"truly understands\"? As it stands, no result could prove it wrong, so no experiment can test it.",
            "future": "\"One day\" never has to arrive: no result today can prove it wrong.",
            "maybe": "Every result fits \"may or may not\": it cannot be wrong, so it says nothing.",
            "right": "It says what is expected, on which questions, and what to count. If the chatbot gets as many wrong in one word, or more, the hypothesis is wrong.",
        },
    )
    choice_testable
    return (choice_testable,)


@app.cell(hide_code=True)
def text_exercise_design():
    mo.md(r"""
    ### Exercise 3. Design an experiment
    """)
    return


@app.cell(hide_code=True)
def explain_design():
    explain_check = L.explain(
        "design",
        question="Someone's hypothesis: \"A chatbot gets more questions right when they are written without spelling mistakes.\" Design an experiment with the recipe. Write down what you change, what you keep the same, what you count, and which result would prove the hypothesis wrong.",
        model_answer="**Change:** only the spelling: every question once with spelling mistakes and once without. **Keep the same:** the questions themselves, the chatbot, and the way the answers are marked. **Count:** how many questions it gets right in each version. **Wrong if:** the version without spelling mistakes gets as many right as the version with them, or fewer.",
        points=[
            "you change only the spelling, on the same questions",
            "you keep the chatbot and the marking the same",
            "you count the right answers in both versions",
            "you say which result proves the hypothesis wrong",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en) showed how research is shared; now you know what comes before the paper: a question, a hypothesis that could be wrong, and an experiment that tests it. In the research part of the course, you design experiments like this yourself, and in the end you repeat the result of a paper. With this lesson, the basics part of the course is done. Later lessons build on it, such as *Change one thing at a time* and *Is the difference real?*. The next lesson, *Derivative rules*, starts the intermediate part.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_testable, explain_check, result_steps):
    L.wrap_up(result_steps, choice_testable, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
