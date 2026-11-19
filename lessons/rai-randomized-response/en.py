# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-randomized-response. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-randomized-response", "en")
    C = L.checks   # a made-up survey: TRUTH, the coins FIRST and SECOND, ANSWERS, and eight people FEW_*


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Linking tables](/course/open/rai-linkage-attacks?lang=en), [Probability: counting how often](/course/open/prob-basics?lang=en) and [Random numbers and seeds](/course/open/np-random?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_name():
    quick_name = L.choice(
        "quick-name",
        question="A released visit matches exactly one voter, on a voter list that holds about two thirds of the town. What does that match give?",
        options={
            "right": "A name that may be wrong",
            "sure": "A name that is surely right",
            "none": "No name: one match is too few",
        },
        correct="right",
        why={
            "right": "Someone with the same values who is not on the list does not show, so the single match can name the wrong person: in the made-up town, 48 of the 622 names were wrong.",
            "sure": "The list does not hold everyone, so someone with the same values may be missing from it. See *A name from an incomplete list* in *The idea* of [Linking tables](/course/open/rai-linkage-attacks?lang=en).",
            "none": "A released row that matches exactly one voter gets that voter's name; the question is whether the name is right. See *Two tables, one key* in *The idea* of [Linking tables](/course/open/rai-linkage-attacks?lang=en).",
        },
    )
    quick_name
    return (quick_name,)


@app.cell(hide_code=True)
def quick_check_prob():
    quick_prob = L.choice(
        "quick-prob",
        question="Of 200 answers, 50 are yes. What is the probability of a yes, counted from these answers?",
        options={
            "count": "50, the number of yes answers",
            "right": "0.25, or 50 of 200",
            "coin": "0.5: it is yes or no",
        },
        correct="right",
        why={
            "right": "A probability counted from tries is how often it happened, divided by the number of tries: 50 / 200 = 0.25.",
            "count": "50 is how often it happened; a probability divides that by the number of tries, and lies between 0 and 1. See *The idea* of [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "coin": "Two possible outcomes need not be equally likely; counted from these answers, a yes came 50 times in 200. See *The idea* of [Probability: counting how often](/course/open/prob-basics?lang=en).",
        },
    )
    quick_prob
    return (quick_prob,)


@app.cell(hide_code=True)
def quick_check_rng():
    quick_rng = L.choice(
        "quick-rng",
        question="`rng = np.random.default_rng(7)`. Which numbers can `rng.integers(0, 2, size=5)` give?",
        options={
            "three": "0, 1 and 2",
            "right": "Only 0 and 1",
            "floats": "Any number from 0 to 2",
        },
        correct="right",
        why={
            "right": "`rng.integers(low, high)` gives whole numbers up to, but not including, `high`: here 0 and 1, like a coin.",
            "three": "`high` is not included: `rng.integers(0, 2)` gives 0 or 1. See *New words and symbols* in [Random numbers and seeds](/course/open/np-random?lang=en).",
            "floats": "`rng.integers` gives whole numbers; `rng.random` gives floats. See *New words and symbols* in [Random numbers and seeds](/course/open/np-random?lang=en).",
        },
    )
    quick_rng
    return (quick_rng,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Ask people a question they would rather not answer, and some of them will not tell the truth. In 1965, Stanley Warner called the error this causes "evasive answer bias", and proposed a way around it: every person adds chance to their own answer, so that no single answer gives them away, and whoever adds the answers up corrects for that chance. Local differential privacy, a way to measure such protection, has its roots in Warner's idea, and Google and Apple have used versions of it to collect data from their users. In this lesson you run a made-up survey of 10,000 people with coins, estimate from the answers how many said yes in truth, and measure how much privacy the coins give.

    **An everyday example.** A teacher asks who copied the homework, and hardly a hand goes up. Next time, everyone first flips a coin under the desk: on heads, the hand goes up whatever the truth; on tails, it goes up only for a yes. A raised hand now says little about anyone, yet the teacher can still work out roughly how many copied.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="In an example, Warner wants the share of a group, about one half, with a standard deviation of .05. Asked straight, with everyone telling the truth, about 100 interviews do. How many does it take with the spinner Warner proposed, set to p = .75, which you meet below?",
        options={
            "same": "Just about 100",
            "twice": "Around 200",
            "right": "About 400",
        },
        reveal="**About 400**: Warner (1965) computes that the spinner needs \"about 400\" interviews where the direct question needs \"only about 100\". The chance that protects each person also blurs the group, so it takes more answers. In Exercise 2 you see a price like it with two coins.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A spinner instead of an answer

    Warner (1965) wanted to remove "evasive answer bias": the error in a survey that comes from people who do not answer a sensitive question truthfully. The device was a spinner: "each interviewer is furnished with an identical spinner with a face marked so that the spinner points to the letter A with probability p and to the letter B with probability (1- p). Then, in each interview, the interviewee is asked to spin the spinner unobserved by the interviewer and report only whether or not the spinner points to the letter representing the group to which the interviewee belongs." Group A can be the people who did the sensitive thing. The interviewer hears only yes or no, and never sees the spinner. Warner's title already names the method: **randomized response**.

    One answer then says little, yet the answers of many still say something. With a share π of the people in group A, a yes comes with chance πp + (1 − π)(1 − p), and Warner solves that for π, from the share of yes. The spinner must lean one way: at p = 1/2, "the interviewee would be furnishing no information at all", and at p = 1 it is the direct question. In between, the privacy costs answers: with π = .5 and p = .75, Warner needs "about 400" interviews for a standard deviation of .05, against "only about 100" for the direct question, if everyone told the truth.

    ### Two coins

    Cynthia Dwork and Aaron Roth (2014) describe a later variant of Warner's idea, with coins: "Study participants are told to report whether or not they have property P as follows: 1. Flip a coin. 2. If tails, then respond truthfully. 3. If heads, then flip a second coin and respond 'Yes' if heads and 'No' if tails." This too is randomized response: each person adds chance to their own answer before giving it.

    Someone whose truth is yes answers yes with chance 1/2 + 1/4 = 3/4: half of the time from the truth, a quarter of the time from the second coin. Someone whose truth is no answers yes with chance 1/4. So "even a 'Yes' answer is not incriminating, since this answer occurs with probability at least 1/4 whether or not the respondent actually has property P." With a true share p of yes, the share of yes among the answers is on average (1/4)(1 − p) + (3/4)p = 1/4 + p/2, and turned around: "we can estimate p as twice the fraction answering 'Yes' minus 1/2". Two fair coins give a yes with the same chances as Warner's spinner with p = 3/4 (there p is the chance that the spinner points to A, not the share), and so the same estimate.

    ### How much privacy: ε

    For one person, a yes is 3/4 likely with a true yes and 1/4 likely with a true no: three times as likely. That ratio is the measure. An answer method is ε-differentially private, in the local sense, when for any two truths a person could have, every answer is at most e<sup>ε</sup> times as likely with the one as with the other. Dwork and Roth: "The version of randomized response described above is (ln 3, 0)-differentially private." Here ln 3 is the log of 3, about 1.10, and the 0 says that the bound holds without exceptions. A smaller ε means answers that say less about a person; at ε = 0, an answer does not depend on the truth at all. And "privacy is obtained by process, there are no 'good' or 'bad' responses."

    This is **local differential privacy**: each person adds the chance before anyone collects the answer. "Because nobody other than its owner ever touches any piece of private data, the local setting is far more secure", Dwork and Roth write, and "The local model of data privacy has its roots in randomized response, as first proposed by Warner in 1965".

    ### The price

    Coins hide a person, and they also blur the group: the same precision takes more answers, and in Exercise 2 you see how many more. Úlfar Erlingsson, Vasyl Pihur and Aleksandra Korolova (2014), who designed RAPPOR, a form of randomized response that Google's Chrome has used, give an example at ε = ln 3: "if one would like to detect items with frequency 1%, then one million samples are required". Asking again uses privacy up: "this privacy guarantee degrades if the survey is repeated", so RAPPOR makes one noisy answer per person once, keeps it, and adds fresh noise to it at every report. RAPPOR collected its data from "some of the Chrome users who have opted in to send usage statistics to Google".

    Apple uses local differential privacy too, with methods more involved than two coins but the same measure. Its *Differential Privacy Technical Overview* says: "For emoji, Apple uses a privacy budget with epsilon of 4, and submits one donation per day. For QuickType, Apple uses a privacy budget with epsilon of 8, and submits two donations per day." At ε = 8, an answer may be up to e<sup>8</sup>, about 2,981 times, as likely with one truth as with another; at ln 3, 3 times.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **randomized response** | each person adds chance to their own answer, as with the two coins, so that no single answer gives them away while the share in the group can still be estimated |
        | **local differential privacy** | privacy where each person adds the chance before the answer leaves them, so that not even the collector sees the true answers; ε says how much one answer may say |
        | **ε** (epsilon) | how much one answer may say: every answer is at most e<sup>ε</sup> times as likely with one truth as with another; a smaller ε is more private |
        | **ln 3** | the log of 3 with base e, `np.log(3)`, about 1.10: the ε of two fair coins |
        | **2 × share − 1/2** | the estimate of the true share of yes, from the share of yes among answers with two fair coins |
        """),
        L.watch_out("The two coins are a later variant of Warner's idea: Warner used a spinner. ε = ln 3 holds for one answer: the same question asked again with fresh coins says more each time. And the estimate need not lie between 0 and 1: Warner notes that it can fall outside, and 2 × share − 1/2 is below 0 whenever fewer than a quarter of the answers are yes."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To ask a question that people would rather not answer, with two coins:

    1. Each person flips the first coin. On tails, they answer truthfully; on heads, they flip the second coin and answer yes on heads and no on tails. That is `answer`.
    2. Add the answers up: the share of yes, times 2, minus 1/2, estimates the true share. That is `estimate`.
    3. Measure the privacy: how many times as likely a yes is with a true yes as with a true no. The log of that ratio is ε. That is `epsilon`.
    4. Count the price: the estimate spreads more than with the direct question, so you need more answers, and every new answer of the same person uses privacy up.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell flips the two coins for 10,000 people, with 1 for heads and 0 for tails, and counts what the coins decide on their own.
    """)
    return


@app.cell
def example_coins():
    import numpy as np

    _rng = np.random.default_rng(1965)
    _first = _rng.integers(0, 2, size=10000)    # 1 = heads: the second coin decides
    _second = _rng.integers(0, 2, size=10000)   # 1 = heads: answer yes
    print("heads on the first coin:", _first.mean())
    # both coins heads: a yes, whatever the truth
    print("forced yes:", (_first * _second).mean())
    print("forced no:", (_first * (1 - _second)).mean())
    print("tells the truth:", (1 - _first).mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    About half of the people tell the truth, about a quarter say yes and about a quarter say no, whatever the truth: here 50.15%, 24.15% and 25.7%. Multiplying arrays of 0s and 1s picks people out: `_first * _second` is 1 only where both coins are heads, and `1 - _first` is 1 only where the first coin is tails. So someone whose truth is yes answers yes about 1/2 + 1/4 = 3/4 of the time, and someone whose truth is no about 1/4 of the time. In Exercise 1 you put the coins and the truth together into answers.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_answer():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. answer: what each person says

        `TRUTH` holds the true answers of the 10,000 people of a made-up survey on a question they would rather not answer, such as whether they ever cheated on a test: 1 for yes, 0 for no; 2,000 of them are 1. `FIRST` and `SECOND` hold their two coins: 1 for heads, 0 for tails. `FEW_TRUTH`, `FEW_FIRST` and `FEW_SECOND` are eight people, one for each combination of truth and coins.

        `answer(truth, first, second)` gets three NumPy arrays of 0s and 1s of the same length: the true answers `truth`, the first coins `first` and the second coins `second`. It gives back what each person answers, as an array of 0s and 1s: the truth where the first coin is tails (0), and the second coin where the first is heads (1). Replace the `...`. The check calls `answer(FEW_TRUTH, FEW_FIRST, FEW_SECOND)`, then `TRUTH`, `FIRST` and `SECOND`, and other coins.
        """),
        L.task("Give back the truth where first is 0, and second where first is 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("answer", lines=3)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "answer",
        [
            "Step 1 of the recipe. In the worked example, `1 - _first` is 1 exactly for the people who tell the truth.",
            "Multiplying by an array of 0s and 1s keeps the values where it is 1 and makes the rest 0, so `(1 - first) * truth` is the truth for the people who tell it, and 0 for the others. Add the part for the people whose first coin is heads.",
            "`return (1 - first) * truth + ___`",
        ],
        why="`return (1 - first) * truth + first * second` gives the truth where `first` is 0 and the second coin where `first` is 1. In the survey, 1,499 of the 2,000 people whose truth is yes answer yes, and 1,988 of the 8,000 whose truth is no: about 3 in 4 and 1 in 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_estimate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. estimate: the true share from the answers

        `ANSWERS` holds the answers of the 10,000 people of the survey, made with the course's `answer` from `TRUTH`, `FIRST` and `SECOND`.

        `estimate(answers)` gets a NumPy array `answers` of 0s and 1s: the answers of a survey with two fair coins. It gives back the estimate of the true share of yes, as Dwork and Roth make it. Replace the `...`. The check calls `estimate(ANSWERS)`, then other answers.
        """),
        L.task("Give back the estimate of the true share of yes, from answers with two fair coins."),
    ])
    return


@app.cell(hide_code=True)
def exercise_estimate():
    box_estimate = L.code_box("estimate", lines=3)
    box_estimate
    return (box_estimate,)


@app.cell(hide_code=True)
def check_estimate(box_estimate):
    result_estimate = L.check("estimate", box_estimate.value)
    result_estimate
    return (result_estimate,)


@app.cell(hide_code=True)
def hints_estimate():
    L.hints(
        "estimate",
        [
            "Step 2 of the recipe, and the last quote of Dwork and Roth in *Two coins*.",
            "The share of yes among 0s and 1s is their mean. On average that share is 1/4 + p/2, with p the true share: solve that for p.",
            "`return 2 * ___ - 0.5`",
        ],
        why="`return 2 * answers.mean() - 0.5` turns the share of yes back into the true share. `estimate(ANSWERS)` gives 0.1974: 34.87% of the answers are yes, and the true share is 0.2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_epsilon():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. epsilon: how much privacy

        `epsilon(t)` gets a number `t` from 0 to below 1: the chance that the first coin says "tell the truth". With a fair coin, `t` is 0.5. When the first coin does not say so, the second coin, still fair, decides between yes and no: yes half of those times. It gives back ε: the log of how many times as likely a yes is for someone whose truth is yes, as for someone whose truth is no. `yes_if_yes` is the chance of a yes for someone whose truth is yes, and `yes_if_no` the same for someone whose truth is no. Replace the three `...`. The check calls `epsilon(0.5)`, then other values of `t`.
        """),
        L.task("Give back the log of how many times as likely a yes is with a true yes as with a true no."),
    ])
    return


@app.cell(hide_code=True)
def exercise_epsilon():
    box_epsilon = L.code_box("epsilon", lines=8)
    box_epsilon
    return (box_epsilon,)


@app.cell(hide_code=True)
def check_epsilon(box_epsilon):
    result_epsilon = L.check("epsilon", box_epsilon.value)
    result_epsilon
    return (result_epsilon,)


@app.cell(hide_code=True)
def hints_epsilon():
    L.hints(
        "epsilon",
        [
            "Step 3 of the recipe. With fair coins, `t` is 0.5: a yes comes 3/4 of the time with a true yes and 1/4 with a true no, and ε is ln 3.",
            "Someone whose truth is yes says yes when the first coin says tell the truth, with chance `t`, or when it does not and the second coin says yes, with chance `(1 - t) / 2`. Someone whose truth is no says yes only in the second case. `np.log` gives the log, as in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "`yes_if_yes = t + (1 - t) / 2`, `yes_if_no = (1 - t) / 2` and `return np.log(___)`",
        ],
        why="`yes_if_yes = t + (1 - t) / 2` and `yes_if_no = (1 - t) / 2` are the two chances of a yes, and `return np.log(yes_if_yes / yes_if_no)` is ε. `epsilon(0.5)` gives 1.10, ln 3; `epsilon(0.9)` gives 2.94: a yes is then 19 times as likely with a true yes.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_a_yes():
    mo.md(r"""
    ### Exercise 4. One yes

    In the survey with two fair coins, someone answered yes.
    """)
    return


@app.cell(hide_code=True)
def choice_a_yes_cell():
    choice_a_yes = L.choice(
        "a-yes",
        question="What does that one answer show about them?",
        options={
            "sure": "That they have it: a yes is never forced",
            "right": "Little: a yes is 3/4 likely if true, 1/4 if not",
            "nothing": "Nothing at all: the coins decide every answer alone",
        },
        correct="right",
        why={
            "right": "A yes comes with chance 3/4 for someone whose truth is yes, and 1/4 for someone whose truth is no: three times as likely, ε = ln 3. That shifts the odds, but never to certainty, which is why Dwork and Roth call a yes not incriminating.",
            "sure": "Both coins heads force a yes, whatever the truth: that happens a quarter of the time. See *Two coins* in *The idea*.",
            "nothing": "Half of the time the first coin says tell the truth, so a yes is three times as likely with a true yes as with a true no. See *How much privacy: ε* in *The idea*.",
        },
    )
    choice_a_yes
    return (choice_a_yes,)


@app.cell(hide_code=True)
def text_exercise_every_day():
    mo.md(r"""
    ### Exercise 5. Every day again

    A company asks the same people the same question every day, each time with two fresh coins.
    """)
    return


@app.cell(hide_code=True)
def choice_every_day_cell():
    choice_every_day = L.choice(
        "every-day",
        question="What happens to their privacy?",
        options={
            "same": "Nothing: each answer still has ε = ln 3",
            "right": "It wears away as the answers add up",
            "grows": "It grows, because every day adds new noise",
        },
        correct="right",
        why={
            "right": "Each answer is yes 3/4 of the time with a true yes and 1/4 with a true no, so over many days one person's share of yes comes close to 3/4 or 1/4, and that gives their truth away. Erlingsson, Pihur and Korolova write that the guarantee degrades if the survey is repeated; for that reason RAPPOR keeps one noisy answer per person, and adds fresh noise to it at every report.",
            "same": "Each answer has ε = ln 3, but they add up: over many days, the share of yes of one person comes close to 3/4 or 1/4, and that gives the truth away. See *The price* in *The idea*.",
            "grows": "New coins every day give new answers, and their share over many days comes close to 3/4 or 1/4: the truth shows through. See *The price* in *The idea*.",
        },
    )
    choice_every_day
    return (choice_every_day,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Linking tables](/course/open/rai-linkage-attacks?lang=en), the data were collected first, and names came back by joining them with a voter list. Here each person protects their own answer before giving it, and ε says how much: no table to link can give back what the coins hid, yet the share in the group can still be estimated, as in [Probability: counting how often](/course/open/prob-basics?lang=en), at the price of more answers. In the next lesson, *Training without collecting the data*, the data stay on each device, and only what a model learns from them travels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_a_yes, choice_every_day, result_answer, result_epsilon, result_estimate):
    L.wrap_up(result_answer, result_estimate, result_epsilon, choice_a_yes, choice_every_day)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
