# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-nll. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-nll", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en) and [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_certain():
    quick_certain = L.choice(
        "quick-certain",
        question="A model gave the outcome that really happened the probability 1. What is the surprise, minus the log of that probability?",
        options={
            "right": "0: the model was sure, and right",
            "one": "1: the probability itself",
            "inf": "Infinitely large: nothing is ever certain",
        },
        correct="right",
        why={
            "one": "The surprise is minus the log of the probability, and log 1 is 0. See [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "inf": "Infinitely large is the surprise of probability 0, an outcome the model called impossible. See [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "right": "log 1 is 0, so minus log 1 is 0 too: what the model was sure of costs nothing when it happens.",
        },
    )
    quick_certain
    return (quick_certain,)


@app.cell(hide_code=True)
def quick_check_cell():
    quick_cell = L.choice(
        "quick-cell",
        question="What is `P[i, j]` in your table of bigram probabilities?",
        options={
            "count": "How often the pair `chars[i] + chars[j]` came in the text",
            "right": "How likely it is that `chars[j]` comes after `chars[i]`",
            "reverse": "How likely `chars[i]` is to come before `chars[j]`",
        },
        correct="right",
        why={
            "count": "Those are the counts of the count matrix `N`; `P` holds every row of it divided by its sum. See [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).",
            "reverse": "Row `i` is the character before, and its numbers add up to 1 over what comes after it, not over what comes before `chars[j]`. See [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).",
            "right": "Row `i` is the distribution of what comes after `chars[i]`, and column `j` picks `chars[j]` from it.",
        },
    )
    quick_cell
    return (quick_cell,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your bigram model gives a probability to every character that can come next. But how good is it, as a number? To compare it with the better models of later lessons, you need one measure that says how well a model predicts real text. Every language model is measured this way, up to the largest LLMs.

    **An everyday example.** A weather forecaster who gave a 90% chance of rain on the days it rained, and 10% on the dry days, did well; one who said 50% every day did not. You judge a forecaster by the probabilities they gave to what really happened.
    """)
    return


@app.cell(hide_code=True)
def predict_plays():
    guess_plays = L.guess(
        "plays-nll",
        question=f"A model that knows nothing gives each of the 65 characters the probability 1/65. Its average surprise per character is log 65, about {L.number(C.uniform_nll(), 2)}. What do you think your bigram model of the plays gets, on the plays themselves?",
        options={
            "barely": "About 4: one character before tells little",
            "right": "About 2.5: less surprised, though still far from sure",
            "tiny": "About 0.5: it is rarely surprised at all",
        },
        reveal=f"**About 2.5: {L.number(C.plays_nll(), 2)}**, averaged over all {C.grouped(C.pair_count(), L.lang)} pairs of the plays. As a probability: e to the power minus {L.number(C.plays_nll(), 2)} is about {L.number(2.718281828459045 ** -C.plays_nll(), 3)}, so on a typical character the model gave what really came next a chance of about 1 in {round(2.718281828459045 ** C.plays_nll())}, against 1 in 65 for a model that knows nothing. Better, but far from sure: one character is not much to go on.",
    )
    guess_plays
    return (guess_plays,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Give the model a real text, and look at the probability it gave to each next character that really came. The **likelihood** of the text is all those probabilities multiplied: for "the", $P(\text{h} \mid \text{t}) \cdot P(\text{e} \mid \text{h})$. The higher it is, the better the model predicted the text.

        For a long text, that product is tiny: a million probabilities of about 0.1 multiplied give a number with almost a million zeros after the decimal point, which a computer stores as 0. So you take the log of each probability and add the logs up, since the log of a product is the sum of the logs. With a minus sign in front, so that the number is positive, that is the **negative log-likelihood**, NLL for short: the total surprise of the text. Divided by the number of predictions, it is the average surprise per character:
        """),
        L.formula(r"[[loss:\text{average NLL}]] = -\frac{1}{n} \sum_{t=1}^{n} \log [[pred:P(c_t \mid c_{t-1})]]"),
        mo.md(r"""
        Here $c_0, c_1, \dots, c_n$ are the characters of the text, so a text of $n + 1$ characters has $n$ pairs. Lower is better. It is the same mean surprise of what really happened as in the log loss of [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en), now with 65 possible outcomes at every step instead of 2.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **likelihood** | | the probability a model gives to a whole text: the probabilities of all the next characters that really came, multiplied | |
        | **negative log-likelihood** (NLL) | | minus the log of the likelihood: the sum of the surprises; divided by the number of pairs, the average NLL | `total - np.log(p)` |
        | $P(c_t \mid c_{t-1})$ | P of c t given c t minus 1 | the probability the model gives to character $c_t$ after the character before it | `P[row_of["t"], row_of["h"]]` |
        """),
        L.watch_out("A pair the model never saw has probability 0, and its surprise, minus log 0, is infinite. One such pair makes the NLL of the whole text infinite. For the model of the plays, \"Zebra\" has one: a \"Z\" was never followed by an \"e\". The next lesson fixes that."),
        mo.md(r"""
        The picture: the surprise of every pair of "To be, or not to be", for the model of the plays, with their average as a dashed line.
        """),
        mo.Html(C.surprise_bars("To be, or not to be", {"title": "Surprise per pair", "mean": f"average {L.number(C.text_nll('To be, or not to be'), 2)}",
                                                         "alt": "18 bars, one per pair of the line To be, or not to be. The highest are the n after a space, about 3.5, the comma after be and the o after a space, both about 3.1 to 3.2. The lowest is the space after the comma, about 0.3. A dashed line marks the average, about 2.2."})),
        mo.md(r"""
        The highest bars are characters the model could not see coming: the "n" of " not" after a space, and the comma after "be". The lowest is the space after the comma: after a comma, a space usually comes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the average NLL of a text under a table of bigram probabilities `P`:

    1. Note the row of every character: `row_of[chars[k]] = k`, for every `k`.
    2. For every pair of the text, `text[i]` and `text[i + 1]`, look up the probability of the second after the first: `p = P[row_of[text[i]], row_of[text[i + 1]]]`.
    3. Add its surprise to the total: `total = total - np.log(p)`.
    4. Divide the total by the number of pairs, `len(text) - 1`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The table of "abab ba" from [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en), and the text "ab ba". The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_nll():
    import numpy as np

    _P = np.array([[0, 0, 1], [0, 0, 1], [1/3, 2/3, 0]])   # the table of "abab ba"
    _chars = [" ", "a", "b"]
    _text = "ab ba"
    # 1. the row of every character
    _row_of = {}
    for _k in range(len(_chars)):
        _row_of[_chars[_k]] = _k
    _total = 0
    for _i in range(len(_text) - 1):
        # 2. the probability of the next character after this one
        _p = _P[_row_of[_text[_i]], _row_of[_text[_i + 1]]]
        print(_text[_i] + _text[_i + 1], _p)
        # 3. its surprise, added to the total
        _total = _total - np.log(_p)
    # 4. divided by the number of pairs
    print(_total / (len(_text) - 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The pairs "ab" and " b" had probability 1, so they cost nothing. "b " had 1/3 and cost about 1.10, "ba" had 2/3 and cost about 0.41. Together about 1.50, over 4 pairs: an average NLL of about 0.376.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_by_hand():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. By hand
    """)
    return


@app.cell(hide_code=True)
def choice_by_hand_cell():
    choice_by_hand = L.choice(
        "by-hand",
        question="A model gives the two pairs of the text \"abc\" the probabilities 0.5 and 0.25. What is the average negative log-likelihood? (log 0.5 is about −0.69, and log 0.25 about −1.39.)",
        options={
            "mean-p": "About 0.375",
            "right": "About 1.04",
            "sum": "About 2.08",
            "minus": "About −1.04",
        },
        correct="right",
        why={
            "mean-p": "That is the mean of the probabilities themselves. The NLL averages their surprises, minus their logs.",
            "sum": "That is the total surprise of the two pairs. The average divides it by the number of pairs, 2.",
            "minus": "That is the mean of the logs, without the minus sign. The surprise is minus the log, so the NLL is positive.",
            "right": "The surprises are 0.69 and 1.39; together 2.08, and over 2 pairs that is 1.04.",
        },
    )
    choice_by_hand
    return (choice_by_hand,)


@app.cell(hide_code=True)
def text_exercise_nll():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The NLL of a text

        `nll(P, chars, text)` gives the average NLL of `text` under the table `P`. The dictionary `row_of`, the loop over the pairs and the last line are there. Replace the `...` with steps 2 and 3 of the recipe. The check calls `nll(P, ["a", "b", "c"], "abca")` with `P = np.array([[0.1, 0.6, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]])`, which should give about `0.415`, and tries other tables and texts, also one with a pair of probability 0.
        """),
        L.task("Make `nll` look up the probability of each pair and add its surprise to `total`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nll():
    box_nll = L.code_box("nll", lines=12)
    box_nll
    return (box_nll,)


@app.cell(hide_code=True)
def check_nll(box_nll):
    result_nll = L.check("nll", box_nll.value)
    result_nll
    return (result_nll,)


@app.cell(hide_code=True)
def hints_nll():
    L.hints(
        "nll",
        [
            "Steps 2 and 3 of the recipe, as in the worked example.",
            "The row is the character at `i`, the column the one at `i + 1`. Then the surprise of that probability goes into `total`.",
            "`p = P[row_of[text[___]], row_of[text[___]]]` and `total = total - np.log(___)`.",
        ],
        why="In \"abca\", \"ab\" has 0.6, \"bc\" 0.6 and \"ca\" 0.8. Their surprises are about 0.51, 0.51 and 0.22; together 1.24, over 3 pairs about 0.415.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_most():
    mo.md(r"""
    ### Exercise 3. The most surprising text
    """)
    return


@app.cell(hide_code=True)
def choice_most_cell():
    choice_most = L.choice(
        "most-surprising",
        question="Which of these texts surprises the model of the plays most, per character?",
        options={
            "king": "\"the king\"",
            "hamlet": "\"To be, or not to be\"",
            "right": "\"jazz\"",
            "caps": "\"THE KING\"",
        },
        correct="right",
        why={
            "king": f"\"the king\" is ordinary text for the plays: its average NLL is {L.number(C.text_nll('the king'), 2)}, below the {L.number(C.plays_nll(), 2)} of the plays as a whole.",
            "hamlet": f"\"To be, or not to be\" is not in these plays, but its pairs are ordinary English: its average NLL is {L.number(C.text_nll('To be, or not to be'), 2)}.",
            "caps": f"The plays have names in capitals, so capitals after capitals do occur: the average NLL of \"THE KING\" is {L.number(C.text_nll('THE KING'), 2)}.",
            "right": f"After \"j\" an \"a\" is rare in the plays, after \"a\" a \"z\" rarer still, and \"zz\" too: the average NLL of \"jazz\" is {L.number(C.text_nll('jazz'), 2)}, higher than even the {L.number(C.uniform_nll(), 2)} of a model that knows nothing.",
        },
    )
    choice_most
    return (choice_most,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The surprise of [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en), averaged over every pair of a text, measures a language model, with the probabilities of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en). Your model of the plays gets about 2.45 on the plays themselves, against 4.17 for a model that knows nothing. But a text with a pair the plays never had gets an infinite NLL. In the next lesson, *Pairs you never saw: smoothing*, you fix that.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_by_hand, choice_most, result_nll):
    L.wrap_up(choice_by_hand, result_nll, choice_most)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
