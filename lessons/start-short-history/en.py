# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-short-history. Route: course/route/lessen/stam-1-basis.yaml. Checks and the timeline's events: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-short-history", "en")
    C = L.checks   # the events, winters and the sketched wave for the timeline


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about the four circles.
    """)
    return


@app.cell(hide_code=True)
def quick_check_typing():
    quick_typing = L.choice(
        "quick-typing",
        question="Around 2010, a phone keyboard suggested your next word by counting, in millions of messages, which word usually came next. There was no neural network in it. What is the smallest circle it fits in?",
        options={"ai": "AI", "ml": "Machine learning", "dl": "Deep learning"},
        correct="ml",
        why={
            "ai": "It fits in AI, but there is a smaller circle: it found its rule in examples. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "ml": "It found its rule in examples, so it is machine learning, and without a neural network it is not deep learning.",
            "dl": "Deep learning needs a neural network with many layers, and this keyboard had none. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
        },
    )
    quick_typing
    return (quick_typing,)


@app.cell(hide_code=True)
def quick_check_rules():
    quick_rules = L.choice(
        "quick-rules",
        question="A chess program scores every position only with rules that chess players wrote down. Which circle is it in?",
        options={"ml": "Machine learning", "ai": "AI, but not machine learning", "genai": "Generative AI"},
        correct="ai",
        why={
            "ai": "People wrote its rules, so it did not learn from examples: it is AI, but not machine learning.",
            "ml": "Machine learning finds its rules in examples. This program got its rules from people. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "genai": "Generative AI lies inside machine learning, so it always learns from examples. This program got its rules from people. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
        },
    )
    quick_rules
    return (quick_rules,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    AI feels new, but the name is about seventy years old. Since then AI has gone up and down several times: big promises, a lot of money, then disappointment, and years later a new start. If you know that history, you read today's news about AI better: what is really new, and what was promised before.

    **An everyday example.** Electric cars. Around 1900 there were already quite a few of them, but gasoline cars won, and for most of a century electric cars almost disappeared. They came back when batteries had become good and cheap enough, and governments wanted cleaner air. The idea was old; it had to wait. You will see the same thing in the history of AI.
    """)
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-learning",
        question="When do you think the first neural networks that learn from examples were built?",
        options={
            "1950s": "In the 1950s",
            "1980s": "In the 1980s",
            "2012": "Around 2012",
        },
        reveal="**In the 1950s.** The best known is the perceptron from 1958, and there were a few even earlier ones. Neural networks are an old idea. In this lesson you see why it took until about 2012 before they really took off.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    The history of AI goes in waves. In a good period there is a lot of hope and money, and people promise a lot. When the promises do not come true, money and interest dry up for years.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **AI winter** | a period in which the big promises of AI did not come true, so money and interest dried up; the best-known came in the 1970s and in the late 1980s |

    The word *winter* is a comparison: AI did not stop, but there was much less money and attention for it. Not every historian agrees that the 1970s were a real winter: in the United States, the total money for AI research kept growing (Haigh, 2023). In Britain, the cut was real.
    """)
    return


@app.cell(hide_code=True)
def picture_timeline():
    mo.vstack([
        draw.timeline(
            [_e[0] for _e in C.EVENTS["en"]], C.WINTERS, C.WAVE, start=1950, end=2026,
            winter_label="AI winter", wave_label="hope and money",
            title="Seventy years of AI, in waves"),
        mo.md(r"""
        Each winter follows a peak, and the latest wave is the highest so far (2026). The line is a sketch of hope and money, not a measurement; the gray bands are the two AI winters, and their exact years differ from book to book. The numbers on the timeline match the list below.
        """),
        mo.md("\n".join(f"{_i}. **{_e[0]} · {_e[1]}.** {_e[2]}" for _i, _e in enumerate(C.EVENTS["en"], 1))),
    ])
    return


@app.cell(hide_code=True)
def text_two_ways():
    mo.md(r"""
    ### Two ways, taking turns

    In [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) the first question was: did people write the rules, or did the computer find them in examples? The history of AI is largely about those two ways taking turns.

    1. **Rules written by people (1956 to the 1990s).** Programs that reason with rules, and in the 1980s *expert systems*: programs that give advice in one narrow field with hundreds of rules, written down together with human experts. They worked in a small world, but the real world needed more rules than anyone could write.
    2. **Learning from examples.** It started early. In 1958 the *perceptron* learned from its mistakes: in essence a single artificial *neuron*, a small calculation with numbers that it adjusts a little after each mistake. (Rosenblatt's machine had several of them; one neuron is a simplification.) In 1986 *backpropagation* became widely known: a way to let neural networks with several layers learn from their mistakes. But for a long time there was too little data, and computers were too slow.
    3. **Deep learning (from about 2012).** Three things came together: much more data, such as photos and text on the internet; fast GPUs, chips first made for the graphics of games, that do the arithmetic of a neural network many times faster; and better ways to let networks with many layers learn. In 2012 AlexNet, a deep neural network, recognized things in photos far better than anything before.
    4. **Large language models (from about 2018).** Very large neural networks that learned from huge amounts of text how text goes on. With ChatGPT, at the end of 2022, they reached the general public; in October 2025, OpenAI reported about 800 million people using ChatGPT every week.

    Twice, more was promised than the programs could deliver, and a winter followed: in the 1970s, when both reasoning programs and the first neural networks fell short, and at the end of the 1980s, when expert systems disappointed. And as with the electric car, neural networks were an old idea that had to wait for data and fast computers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two systems from that history. For each, ask the first question from [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en): did people write its rules, or did the computer find them in examples?

    1. **XCON (1980)** helped the computer maker DEC put together the right parts for each computer that was ordered, with a few hundred rules at first and thousands later, all written down by its makers. **People wrote the rules.**
    2. **A neural network from 1989** learned to read the digits of handwritten US ZIP codes (they are only digits) from thousands of handwritten digits, each marked with the digit it shows. Nobody wrote down what a 7 looks like; it found that in the examples. **It learned from examples.**
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ways():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Rules or learning?

        Six systems from the history of AI. Ask the same question as in the worked example.
        """),
        L.task("Put each system in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ways():
    sort_ways = L.sort("rules-or-learning")
    sort_ways
    return (sort_ways,)


@app.cell(hide_code=True)
def check_ways(sort_ways):
    result_ways = L.check("rules-or-learning", sort_ways)
    result_ways
    return (result_ways,)


@app.cell(hide_code=True)
def hints_ways():
    L.hints(
        "rules-or-learning",
        [
            "For each system, ask: who decided how it answers, a person or the examples?",
            "A system that adjusts its own numbers, or gets good by seeing many examples, learned. A system whose patterns, rules or steps someone wrote down did not.",
            "Three systems got their rules from people, and three learned. Which three are neural networks, or a neuron?",
        ],
        why="Rules written by people: **ELIZA**, **Expert system** and **SHRDLU**; their makers wrote down the patterns, the rules and the steps. Learned from examples: **Perceptron**, **AlexNet** and **ChatGPT**; they adjusted their own numbers by seeing examples.",
    )
    return


@app.cell(hide_code=True)
def text_winter():
    mo.md(r"""
    ### Exercise 2. What is an AI winter?
    """)
    return


@app.cell(hide_code=True)
def choice_winter_cell():
    choice_winter = L.choice(
        "winter",
        question="Which sentence describes an AI winter?",
        options={
            "broken": "A time when AI programs stopped working.",
            "impossible": "A time when scientists proved that AI is impossible.",
            "promises": "A time when AI had promised more than it could do, and money and attention fell away.",
        },
        correct="promises",
        why={
            "promises": "The programs kept working, and research went on; there was just much less money and attention.",
            "broken": "The programs kept working as before. Look at the table with the new word again: what was it that dried up?",
            "impossible": "Nobody proved that, and research went on. Look at the table with the new word again.",
        },
    )
    choice_winter
    return (choice_winter,)


@app.cell(hide_code=True)
def text_why_2012():
    mo.md(r"""
    ### Exercise 3. Why only in 2012?
    """)
    return


@app.cell(hide_code=True)
def choice_why_2012_cell():
    choice_why_2012 = L.choice(
        "why-2012",
        question="The perceptron dates from 1958. Why did deep learning only take off around 2012?",
        options={
            "invented": "Nobody had thought of neural networks before 2012.",
            "together": "Much more data, fast GPUs and better ways to let networks learn came together.",
            "rules": "In 2012, neural networks were finally given rules about what is in a photo.",
        },
        correct="together",
        why={
            "together": "As with the electric car, the idea was old; it had to wait for enough data and computers that were fast enough.",
            "invented": "The timeline shows the perceptron in 1958 and backpropagation in 1986. Look again at step 3 of *Two ways, taking turns*.",
            "rules": "A neural network gets no rules from people: it finds them in examples. Look again at step 3 of *Two ways, taking turns*.",
        },
    )
    choice_why_2012
    return (choice_why_2012,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) you learned to tell the two ways apart; here you saw them take turns over seventy years. Later in the course you build that history yourself: a single neuron, then a network with layers that learns through backpropagation you write yourself, and at the end of the basics a small language model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why_2012, choice_winter, result_ways):
    L.wrap_up(result_ways, choice_winter, choice_why_2012)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
