# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-annotation. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-annotation", "en")
    C = L.checks   # 20 messages, the labels of the annotators A, B and C the course made up, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Where labels come from](/course/open/data-labels?lang=en) and one about [Probability: counting how often](/course/open/prob-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_labels():
    quick_labels = L.choice(
        "quick-labels",
        question="Which of these gives an annotation label?",
        options={
            "click": "Whether a reader clicked a headline",
            "right": "A person who reads each message and marks it spam or not",
            "sensor": "A sensor that records when a machine fails",
        },
        correct="right",
        why={
            "click": "A click comes for free from what people do: a natural label. See *The idea* in [Where labels come from](/course/open/data-labels?lang=en).",
            "sensor": "A sensor gives a label from a measurement. See *The idea* in [Where labels come from](/course/open/data-labels?lang=en).",
            "right": "Right: a person judges each example. Today is about making those judgments agree.",
        },
    )
    quick_labels
    return (quick_labels,)


@app.cell(hide_code=True)
def quick_check_prob():
    quick_prob = L.choice(
        "quick-prob",
        question="Annotator A labels 13 of 20 messages spam. You pick one of the 20 at random. What is the probability that A labeled it spam?",
        options={
            "count": "13",
            "ok": "0.35",
            "right": "0.65",
        },
        correct="right",
        why={
            "count": "13 is how many, not how likely: a probability is a number from 0 to 1. See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "ok": "0.35 is 7 / 20, the probability that A labeled it ok. See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "right": "Right: 13 / 20 = 0.65, how often A says spam. You need that number today.",
        },
    )
    quick_prob
    return (quick_prob,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A team labels messages for a spam filter. A model can learn no more than its labels say: if two careful people would give a message different labels, its label says more about who happened to label it than about the message. Before you trust labels from people, you want to know how well they agree, and how to make them agree more.

    **An everyday example.** Two teachers grade the same essays. Without a grading rubric, one takes points off for spelling and the other does not, and their grades differ. With a rubric and a few graded example essays, they agree far more often. And agreeing that an empty sheet gets the lowest grade proves little: anyone would.
    """)
    return


@app.cell(hide_code=True)
def predict_chance_cell():
    guess_chance = L.guess(
        "chance-guess",
        question="Annotators A and B agree on 14 of 20 messages, 70%. A calls 13 messages spam, B 9. If both picked their labels at random, A still 13 times spam and B 9 times, how often would they agree?",
        options={
            "low": "Hardly ever, about 10%",
            "right": "About half of the time",
            "same": "About 70%, as often as now",
        },
        reveal="**About half of the time: 48.5%.** Both say spam by chance in $0.65 \\cdot 0.45 = 0.2925$ of the messages, and both say ok in $0.35 \\cdot 0.55 = 0.1925$: together 0.485. A large part of the 70% would happen by chance too.",
    )
    guess_chance
    return (guess_chance,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Guidelines.** An **annotation guideline** tells annotators what each label means, how to decide the hard cases, with examples of each label, and what to do when they are unsure. Here are two for the same task.

        > **Guidelines 1.** Spam is a message you do not want.
        >
        > **Guidelines 2.** Spam is a message from a sender who does not know you that offers a prize, money or something for free, or asks you to pay, click, call or text. Everything else is ok, also an ad that only tells you something, a message from a school or a shop you deal with, and a friend telling you they won something. Examples: "join our club and get your first month free" is spam; "a new cafe opened on your street" is ok; "we won free concert tickets text me now" from a friend is ok.

        The course took 20 messages: 12 clear ones from its message set, 6 spam and 6 ok, and 8 hard ones, one from the set and 7 written for this lesson. Then it made up two annotators, A and B, who label all 20 under each guideline. On the 12 clear messages they always agree. On the hard ones:

        | Message | Guidelines 1: A | B | Guidelines 2: A | B |
        |---|---|---|---|---|
        | we won free concert tickets text me now | spam | ok | ok | ok |
        | the shoe shop has a sale this weekend all boots half price | spam | ok | ok | ok |
        | join our club today and get your first month free | spam | spam | spam | spam |
        | your school trip money is due friday pay at the office | ok | spam | ok | ok |
        | rate your last order and get ten percent off the next one | spam | ok | spam | ok |
        | vote for our band in the contest text yes to win a signed album | spam | spam | spam | spam |
        | a new cafe opened on your street come and try it | spam | ok | ok | ok |
        | forward this to ten friends for good luck | spam | ok | ok | ok |

        **Agreement.** Give the same examples to two annotators, each on their own, and count how often they give the same label: the simplest measure of their **inter-annotator agreement**, written $p_o$. Where they differ, the guideline is often unclear, which says more about the guideline than about who is right.

        **Agreeing by chance.** Two people who never read a single message still agree now and then. Say A calls 65% of the messages spam and B 45%, and each picks without looking at the other. Of the 65% that A calls spam, B calls 45% spam too: both say spam in $0.65 \cdot 0.45 = 0.2925$ of the messages. In counts: of A's 13 spam messages, B calls 45% spam, $13 \cdot 0.45 = 5.85$, and $5.85 / 20 = 0.2925$. Both say ok in $0.35 \cdot 0.55 = 0.1925$. Together they agree by chance in $p_e = 0.485$ of the messages.

        **Kappa.** **Cohen's kappa**, $\kappa$, named after Jacob Cohen (1960), says what part of the room above chance two annotators use:

        $$\kappa = \frac{p_o - p_e}{1 - p_e}$$

        Kappa is 1 when they always agree, 0 when they agree no more often than chance, and below 0 when they agree less often. There is no fixed line for good enough: compare guidelines on the same messages, and read the messages on which people differ.

        **Gold questions.** A **gold question** is an example whose label is certain, mixed in among the others without saying which. An annotator who gets the gold questions wrong is working carelessly, or understood the guideline differently; a gold question that many get wrong points at the guideline.

        **Crowdsourcing and the people who do it.** Large datasets are often labeled through **crowdsourcing**: a big job cut into many small tasks, such as labeling one message, done by many people online, often paid per task. Per hour that can come to less than a minimum wage, and some tasks show violent or hateful texts and images. Pay for the real time a task takes, the right to skip what someone does not want to see, and support belong to the process, just like consent did for data in [Where does this data come from?](/course/open/data-provenance?lang=en).

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **annotation guideline** | what each label means, how to decide the hard cases, with examples, and what to do when unsure | |
        | **inter-annotator agreement**, $p_o$ | how often annotators give the same examples the same label | `same / len(a)` |
        | $p_e$ | how often they would agree by chance, given how often each gives each label | `chance` |
        | **Cohen's kappa**, $\kappa$ | what part of the room above chance two annotators use: 1 always the same, 0 no more than chance | `(agree - chance) / (1 - chance)` |
        | **gold question** | an example whose label is certain, mixed in to check annotators | |
        | **crowdsourcing** | a big job cut into small tasks that many people do online, often paid per task | |
        """),
        L.watch_out("High agreement is not good agreement by itself. When nearly all messages are ok, someone who always says ok agrees with a careful annotator on nearly all of them. Kappa takes chance out."),
        mo.md(r"""
        The slider goes through three comparisons: A and B under guidelines 1, A and B under guidelines 2, and A with a third annotator, C, who says ok to everything. Left, the labels of both, a column per message, filled for spam and open for ok, in a gray band where they differ; the clear messages left of the dashed line, the hard ones right of it. Right, how often they agree, how often they would by chance, and kappa. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_case_cell():
    def _frame(case):
        said = C.case_said(case, L.lang)
        alt = ("Left: two rows of 20 dots, the labels of two annotators, filled for spam and open for ok, with a gray band "
               "where they differ. Right: three bars, how often they agree, how often they would by chance, and kappa. " + said)
        return (C.case_picture(case, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_case = L.explore("case", "comparison", _frame, start=1, values=[1, 2, 3], role="",
                             task="Go through the three comparisons. On which messages do A and B differ under guidelines 1, and what is C's kappa?")
    explore_case
    return


@app.cell(hide_code=True)
def text_after_case():
    mo.md(r"""
    Under guidelines 1, A and B differ on 6 of the 8 hard messages: A calls spam everything that sells or pushes something, even a friend's message and a chain letter; B mostly what asks for money, even the school's. They agree on 70%, but 48.5% would happen by chance, and kappa is 0.42. Guidelines 2 settles 5 of the 6 with a rule and examples; only "ten percent off" stays a doubt: is a discount from a shop you ordered from an offer of money, or a message from a shop you deal with? A place to sharpen the rule next time. Agreement rises to 95% and kappa to 0.90. C agrees with A on 55% of the messages, on every message A calls ok, yet C's kappa is 0: saying ok to everything adds nothing beyond chance. If only 1 message in 10 were spam, C would agree on 90%, and kappa would still be 0.

    ### Recipe

    To get labels from people that you can trust:

    1. Write guidelines: what each label means, rules for the hard cases, examples of each label, and what to do when unsure.
    2. Let two or more annotators label the same sample, each on their own.
    3. Compute their agreement and kappa, and read the examples on which they differ.
    4. Sharpen the guidelines on those examples, and measure again on new examples.
    5. Mix in gold questions, and check every annotator on them.
    6. Pay for the real time the work takes, let people skip what they do not want to see, and write in the datasheet who labeled, under which guidelines, with what kappa.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The labels of A and B under guidelines 1, in the order of the picture: 6 clear spam, 6 clear ok, then the 8 hard messages. The cell counts how often each says spam, turns those counts into shares, and prints how often both would say spam and both ok by chance. Run the cell.
    """)
    return


@app.cell
def example_chance():
    a = ["spam", "spam", "spam", "spam", "spam", "spam", "ok", "ok", "ok", "ok", "ok", "ok",
         "spam", "spam", "spam", "ok", "spam", "spam", "spam", "spam"]
    b = ["spam", "spam", "spam", "spam", "spam", "spam", "ok", "ok", "ok", "ok", "ok", "ok",
         "ok", "ok", "spam", "spam", "ok", "spam", "ok", "ok"]

    a_count = 0
    b_count = 0
    for i in range(len(a)):
        if a[i] == "spam":
            a_count = a_count + 1
        if b[i] == "spam":
            b_count = b_count + 1
    a_spam = a_count / len(a)                              # how often A says spam
    b_spam = b_count / len(b)
    print(a_count, b_count, a_spam, b_spam)
    print(round(a_spam * b_spam, 4), round((1 - a_spam) * (1 - b_spam), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A calls 13 of the 20 messages spam and B 9: shares of 0.65 and 0.45. By chance both say spam in 0.2925 of the messages and both say ok in 0.1925, together $p_e = 0.485$. With the agreement of 0.7, kappa is $(0.7 - 0.485) / (1 - 0.485) = 0.42$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_agreement():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How often two annotators agree

        `agreement(a, b)` gets the labels of two annotators for the same messages, each `"spam"` or `"ok"`, and should give the share of messages on which they give the same label. Replace the `...`. The check calls `agreement(["spam", "ok", "ok", "spam", "ok"], ["spam", "ok", "spam", "spam", "ok"])`, which should give 0.8, and four more pairs of lists, among them A and B under guidelines 1.
        """),
        L.task("Let `agreement` count the messages on which the two agree."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agreement():
    box_agreement = L.code_box("agreement", lines=7)
    box_agreement
    return (box_agreement,)


@app.cell(hide_code=True)
def check_agreement(box_agreement):
    result_agreement = L.check("agreement", box_agreement.value)
    result_agreement
    return (result_agreement,)


@app.cell(hide_code=True)
def hints_agreement():
    L.hints(
        "agreement",
        [
            "For every message, one question: did the two give it the same label?",
            "Look at one message at a time: `a[i]` is A's label of message `i`, and `b[i]` B's label of the same message. An `if` can let `same` grow only for some messages.",
            "`if a[i] == ___:` with `same = same + 1` one step in.",
        ],
        why="Both spam counts as agreeing, and so does both ok. Under guidelines 1, A and B give the same label to 14 of the 20 messages: 14 / 20 = 0.7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kappa():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Cohen's kappa

        `kappa(agree, a_spam, b_spam)` gets the agreement of two annotators and how often each of them says spam, and should give Cohen's kappa. Replace the two `...`: first how often they would agree by chance, then kappa. The check calls `kappa(0.7, 0.65, 0.45)`, the numbers of guidelines 1, and five more.
        """),
        L.task("Let `kappa` take chance out of the agreement."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kappa():
    box_kappa = L.code_box("kappa", lines=5)
    box_kappa
    return (box_kappa,)


@app.cell(hide_code=True)
def check_kappa(box_kappa):
    result_kappa = L.check("kappa", box_kappa.value)
    result_kappa
    return (result_kappa,)


@app.cell(hide_code=True)
def hints_kappa():
    L.hints(
        "kappa",
        [
            "First chance: in which two ways can two annotators agree by chance? Then kappa: what part of the room above chance do they use?",
            "By chance both say spam, `a_spam` times `b_spam`, or both say ok, with for each the share that is not spam. The room above chance runs from `chance` up to 1.",
            "`chance = a_spam * b_spam + (1 - ___) * (1 - ___)` and `return (agree - ___) / (1 - ___)`.",
        ],
        why="With 0.7, 0.65 and 0.45, chance is $0.65 \\cdot 0.45 + 0.35 \\cdot 0.55 = 0.485$, and kappa $(0.7 - 0.485) / (1 - 0.485) = 0.42$: A and B use 42% of the room above chance.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gold():
    mo.md(r"""
    ### Exercise 3. What a gold question tells you
    """)
    return


@app.cell(hide_code=True)
def choice_gold_cell():
    choice_gold = L.choice(
        "gold",
        question="Annotator D labels 200 messages in ten minutes. Among them are 12 gold questions, 6 spam and 6 ok, and D gets 7 of them right. What do you make of it?",
        options={
            "fine": "D is fine: more than half is right",
            "right": "D hardly beats a coin on clear messages: check D's labels and ask why",
            "drop": "Leave the gold questions out next time: they cost D time",
        },
        correct="right",
        why={
            "fine": "Flipping a coin gets 6 of 12 right on average, and these are clear messages. 7 of 12 says that D's labels hardly follow the messages.",
            "drop": "Gold questions are how you notice careless work at all. They also show where guidelines are unclear: a gold question that many people get wrong points at the guideline, or at the gold label itself.",
            "right": "Right. 200 messages in ten minutes is 3 seconds a message. Look at how the work is set up, too: pay per task rewards speed, not care.",
        },
    )
    choice_gold
    return (choice_gold,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Annotation labels from [Where labels come from](/course/open/data-labels?lang=en) are as good as the guidelines behind them, and agreeing by chance is built from how often each says spam, from [Probability: counting how often](/course/open/prob-basics?lang=en): both spam multiplied, plus both ok multiplied. The next lesson, *Fix the data, not the model*, finds the wrong labels that slip through anyway.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gold, result_agreement, result_kappa):
    L.wrap_up(result_agreement, result_kappa, choice_gold)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
