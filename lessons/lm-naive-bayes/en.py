# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-naive-bayes. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-naive-bayes", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Bayes' rule](/course/open/prob-bayes?lang=en) and [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bayes():
    quick_bayes = L.choice(
        "quick-bayes",
        question="1 in 10 people has an illness. A test is positive for 90% of the people who are ill, and for 10% of the healthy people. Someone tests positive. How likely is it that they are ill?",
        options={
            "hit": "0.9",
            "right": "0.5",
            "joint": "0.09",
        },
        correct="right",
        why={
            "hit": "That is P(positive | ill), the other way round. See [Bayes' rule](/course/open/prob-bayes?lang=en).",
            "joint": "That is P(ill and positive), not yet divided by P(positive). See [Bayes' rule](/course/open/prob-bayes?lang=en).",
            "right": "0.9 · 0.1 = 0.09 are ill and positive, and 0.1 · 0.9 = 0.09 healthy and positive: half of the positive tests.",
        },
    )
    quick_bayes
    return (quick_bayes,)


@app.cell(hide_code=True)
def quick_check_smooth():
    quick_smooth = L.choice(
        "quick-smooth",
        question="A row of a count table adds up to 40, in a table of 10 characters. One pair of that row never came. With Laplace smoothing, k = 1, what probability does it get?",
        options={
            "zero": "0",
            "row": "0.025: 1 out of 40",
            "right": "0.02: 1 out of 40 + 10",
            "even": "0.1: 1 out of 10",
        },
        correct="right",
        why={
            "zero": "That is without smoothing. See [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en).",
            "row": "The row sum grows too: each of the 10 counts gets 1 more. See [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en).",
            "even": "That would be a row without any counts. See [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en).",
            "right": "(0 + 1) / (40 + 10) = 1 / 50 = 0.02.",
        },
    )
    quick_smooth
    return (quick_smooth,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Bayes' rule](/course/open/prob-bayes?lang=en), one word, "free", raised the probability of spam from 0.2 to 0.75. A real message has many words, and Bayes' rule then needs P(message | spam): how often exactly this message is among the spam. Almost every message is new, so its count is 0, as for the pairs you never saw. Counting single words does work.

    **An everyday example.** You find an unsigned note on the fridge: "back late, pizza in the oven". Your brother always writes "pizza", your mother often writes "late". Each word is a hint, and you weigh the hints one by one, without asking whether they belong together.
    """)
    return


@app.cell(hide_code=True)
def predict_right():
    guess_right = L.guess(
        "how-many",
        question=f"A filter counts the words of {len(C.messages()['train'])} short messages the course wrote, {C.model()['messages']['spam']} of them spam, and takes every word as a hint of its own. How many of {len(C.messages()['test'])} new messages do you think it sorts right?",
        options={
            "coin": "About 10",
            "most": "About 15",
            "right": "19 or 20",
        },
        reveal=f"**{C.right()} of the {len(C.messages()['test'])}.** Spam has words of its own, such as \"free\", \"claim\" and \"cash\", so counting words is enough for most messages. Which one it gets wrong, you find in the play block.",
    )
    guess_right
    return (guess_right,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Sorting whole texts into classes, such as spam and ok, is called **text classification**. For a message with the words $w_1, \dots, w_n$, Bayes' rule gives for each class $c$:
        """),
        L.formula(r"P(c \mid w_1, \dots, w_n) = \frac{P(w_1, \dots, w_n \mid c) \cdot P(c)}{P(w_1, \dots, w_n)}"),
        mo.md(r"""
        The bottom is the same for both classes, so to choose between them you only need the top. **Naive Bayes** makes it countable with one assumption: within a class, the words are independent of each other. That is **conditional independence**, and with it the probability of the whole message is the product of the probabilities of its words. That product is the score of the class, and the class with the highest score wins:
        """),
        L.formula(r"\text{score}(c) = P(c) \cdot P(w_1 \mid c) \cdot P(w_2 \mid c) \cdots P(w_n \mid c)"),
        mo.md(r"""
        The assumption is not true, hence the name: in the spam of the course, half of the messages have "free", but two thirds of those with "now", so "now" does say something about "free". The filter counts part of the same hint twice and is too sure, yet it often picks the right class.

        The filter reads a message as a **bag of words**: which words it has and how often, not their order. Each probability comes from counts in the training messages: P(c) is the share of the messages of class c, and P(w | c) gets the Laplace smoothing of [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en), so that a word a class never had does not make its whole product 0. With $V$ the size of the vocabulary, the number of different words in the training messages:
        """),
        L.formula(r"[[param:P(w \mid c)]] = \frac{(\text{count of } w \text{ in the messages of } c) + 1}{(\text{number of words in the messages of } c) + V}"),
        mo.md(r"""
        A word in no training message at all is left out: neither class has seen it. And as in [How good is a language model?](/course/open/lm-nll?lang=en), a product of many small probabilities becomes 0 in the computer, so you add their logs:
        """),
        L.formula(r"[[pred:\log \text{score}(c)]] = [[param:\log P(c)]] + \sum_{i=1}^{n} [[param:\log P(w_i \mid c)]]"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **text classification** | | sorting whole texts into classes, such as spam and ok | |
        | **naive Bayes** | | a classifier that gives a text the class with the highest score | `log_score(...)` |
        | **conditional independence** | | within a class, one word says nothing about another: P(free and now \| spam) = P(free \| spam) · P(now \| spam) | |
        | **bag of words** | | a text as the words in it and how often each comes, without their order | `text.split()` |

        The play block runs the filter on each of the 20 test messages and follows one number, log score(spam) − log score(ok). Each <span class="c-param">orange bar</span> is the push of one word, log P(w | spam) − log P(w | ok): a fixed number of the trained filter, not an output as the probabilities of the bigram model were, so orange. The first bar is the prior, and the <span class="c-pred">blue bar</span> the sum: right of the dashed line at 0, spam has the higher score. Slide from message to message.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_messages_cell():
    _labels = {"title": "How each word moves the score", "ok_side": "← ok", "spam_side": "spam →", "prior": "prior",
               "left_out": "left out", "sum": "sum", "spam": "spam", "ok": "ok"}

    def _frame(_i):
        _v = C.frame(_i, L.lang)
        _alt = "Message {i}, {text}: the score starts at the prior, {prior}, and each word it knows moves it: {steps}. The sum is {sum}, so the filter says {said}."
        _said = "Message {i}: *{text}*. It is {label}, and the filter says **{said}**."
        return C.waterfall(_i, _labels | {"alt": _alt.format(**_v)}, L.lang), _said.format(**_v)

    _explore = L.explore("message", "message", _frame, start=1, low=1, high=len(C.messages()["test"]), step=1, role="",
                         task="Find the message the filter gets wrong. Which word pushes it hardest toward spam?")
    _explore
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Message 12, "we won free concert tickets text me now", is ok, but the filter says spam: "free" pushes it hardest, by +2.3, then "now" and "text". All three are words of spam, and the filter adds them up as three separate hints.

    ### Recipe

    To sort a message with naive Bayes:

    1. Count, in the training messages: for each class, its number of messages, how often each word appears in them, and their number of words; and the vocabulary.
    2. Keep the words of the new message that are in the vocabulary.
    3. For each class: start at log P(c), and add log((count + 1) / (words of the class + V)) for every word.
    4. The class with the highest score wins.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Five training messages, two of them spam, and the new message "free cash now please". `_counts["spam"]` is a dictionary inside a dictionary: the counts of the words of spam. `set(_all_words)` keeps every word once, and `_w in _vocab` asks whether a word is in that set, as with a dictionary. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_naive_bayes():
    import numpy as np

    _train = [("spam", "win cash now"), ("spam", "free cash prize now"),
              ("ok", "lunch now"), ("ok", "call me"), ("ok", "free for lunch")]
    # 1. per class: its messages, how often each word appears in them, and their number of words
    _messages = {"spam": 0, "ok": 0}
    _counts = {"spam": {}, "ok": {}}
    _n_words = {"spam": 0, "ok": 0}
    _all_words = []
    for _label, _text in _train:
        _messages[_label] = _messages[_label] + 1
        for _w in _text.split():
            if _w in _counts[_label]:
                _counts[_label][_w] = _counts[_label][_w] + 1
            else:
                _counts[_label][_w] = 1
            _n_words[_label] = _n_words[_label] + 1
            _all_words.append(_w)
    _vocab = set(_all_words)   # the vocabulary: every different word
    print(_messages, _n_words, len(_vocab))
    print(_counts["spam"])
    # 2. the words of the new message that are in the vocabulary
    _words = []
    for _w in "free cash now please".split():
        if _w in _vocab:
            _words.append(_w)
    print(_words)
    # 3. for each class: log P(c), plus log P(w | c) for every word, with Laplace smoothing
    _scores = {}
    for _label in ["spam", "ok"]:
        _score = np.log(_messages[_label] / len(_train))
        for _w in _words:
            if _w in _counts[_label]:
                _n = _counts[_label][_w]
            else:
                _n = 0
            _score = _score + np.log((_n + 1) / (_n_words[_label] + len(_vocab)))
        _scores[_label] = _score
        print(_label, _score)
    # 4. the class with the highest score
    if _scores["spam"] > _scores["ok"]:
        print("spam")
    else:
        print("ok")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Both classes have 7 words, and the vocabulary has 9: hence the 16 at the bottom. "please" is not in it, so 3 words are left. For spam, with 2 of the 5 messages: log 0.4 + log(2/16) + log(3/16) + log(3/16), about −6.34. "cash" never came in ok and gets 1/16 there instead of 0; ok ends at about −7.44, so the message is spam.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A word that spam never had
    """)
    return


@app.cell(hide_code=True)
def choice_unseen_cell():
    choice_unseen = L.choice(
        "unseen-word",
        question=f"The filter of the course counted {C.model()['words']['spam']} words in its {C.model()['messages']['spam']} spam messages, and its vocabulary has {len(C.model()['vocab'])} words. \"lunch\" is in it, but never in spam. What is P(lunch | spam)?",
        options={
            "zero": "0: it never came in spam",
            "row": f"1 / {C.model()['words']['spam']}, about 0.007",
            "right": f"1 / {C.model()['words']['spam'] + len(C.model()['vocab'])}, about 0.0032",
            "vocab": f"1 / {len(C.model()['vocab'])}, about 0.0059",
        },
        correct="right",
        why={
            "zero": "That is without smoothing: \"lunch\" would make the score of spam 0. See the formula for P(w | c) in *The idea*.",
            "row": "The bottom grows too: the size of the vocabulary is added to the words of spam. See *The idea*.",
            "vocab": "The words of spam count too. See the formula for P(w | c) in *The idea*.",
            "right": f"(0 + 1) / ({C.model()['words']['spam']} + {len(C.model()['vocab'])}) = 1 / {C.model()['words']['spam'] + len(C.model()['vocab'])}: small, but not 0.",
        },
    )
    choice_unseen
    return (choice_unseen,)


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The log score of a class

        `log_score(words, prior, counts, n_words, n_vocab)` gives the log score of one class: `words` are the words of the message that are in the vocabulary, `prior` is P(c), `counts` the dictionary of how often each word appears in that class, `n_words` the number of words of the class and `n_vocab` the size of the vocabulary. Replace the two `...` with step 3 of the recipe. The check calls `log_score(["free", "cash", "now"], 0.6, counts, 7, 9)`, with the counts of ok from the worked example, which should give about `-7.442`; it also tries other messages, one of them 300 words long.
        """),
        L.task("Make `log_score` start at the log of the prior, and add the log of P(w | c) for every word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("log-score", lines=12)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("log-score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "log-score",
        [
            "Step 3 of the recipe, as in the worked example.",
            "The score starts at the log of the prior. Every word adds the log of its probability: 1 added to its count, and the size of the vocabulary to the number of words of the class.",
            "`score = np.log(___)`, and in the loop `score = score + np.log((n + 1) / (___))`.",
        ],
        why="For ok in the worked example: log 0.6 is about −0.511, and \"free\", \"cash\" and \"now\" get 2/16, 1/16 and 2/16. Their logs added to it give about −7.442.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_naive():
    mo.md(r"""
    ### Exercise 3. Two words that come together
    """)
    return


@app.cell(hide_code=True)
def choice_naive_cell():
    choice_naive = L.choice(
        "naive",
        question="In the spam of the course, \"free\" and \"now\" often come in the same message. A new message has both. What does naive Bayes do with them?",
        options={
            "right": "It adds the push of each toward spam, as if the second word told it something new",
            "once": "It counts them as one hint, because they come together",
            "pair": "It reads them as one pair, as a bigram model would",
        },
        correct="right",
        why={
            "once": "Naive Bayes never looks at which words come together. See *The idea*.",
            "pair": "That is what a bigram model does; naive Bayes reads a bag of words, without their order. See *The idea*.",
            "right": "That is the naive assumption: the two hints overlap, so the filter is too sure.",
        },
    )
    choice_naive
    return (choice_naive,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Bayes' rule](/course/open/prob-bayes?lang=en) gave the probability of spam after one word; naive Bayes does it for every word at once. Its probabilities come from counts with the Laplace smoothing of [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en), and their logs are added up as in [How good is a language model?](/course/open/lm-nll?lang=en). Each class is in fact a small language model of single words, and the filter asks which of the two, weighed by its prior, makes the message more likely. In the next lesson of the main route, *Project: the same bigram model, learned with gradient descent*, your bigram model learns its table instead of counting it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_naive, choice_unseen, result_score):
    L.wrap_up(choice_unseen, result_score, choice_naive)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
