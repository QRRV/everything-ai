# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-goodhart. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-goodhart", "en")
    C = L.checks   # the course's contests: options with a true value, a score with noise, and a winner


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en) and [Random numbers and seeds](/course/open/np-random?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaming():
    quick_gaming = L.choice(
        "quick-gaming",
        question="On the course's race track, an agent that got 3 points for each target went back and forth between two targets instead of finishing. Why?",
        options={
            "right": "The points paid more for it than finishing did",
            "bug": "A bug in the code of its Q-learning",
            "few": "Too few episodes, so it never found the finish line",
        },
        correct="right",
        why={
            "right": "Going back and forth paid 2 points a step, and the agent did what the points paid for: specification gaming. This lesson asks why a number that is pushed on stops measuring what it was meant to.",
            "bug": "The learning worked: value iteration, which needs no learning at all, gave the same way. The points paid for going back and forth. See [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en).",
            "few": "In none of its 500 episodes did it reach the finish, and value iteration, which needs no episodes, gives the same way: the points pay for going back and forth. See [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en).",
        },
    )
    quick_gaming
    return (quick_gaming,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="You make `rng = np.random.default_rng(3)` and ask `rng.random(size=2)`. Later you make a new `np.random.default_rng(3)` and ask the same. What do you get?",
        options={
            "new": "Two new numbers, as random numbers should be",
            "right": "The same two numbers as the first time",
            "first": "The same first number, and then a different second one",
        },
        correct="right",
        why={
            "right": "A generator with the same seed gives the same numbers, in the same order. The course's contests in this lesson are drawn with seed 0, so anyone who runs them gets the same ones.",
            "new": "The numbers only look random: they follow from the seed. The same seed gives the same numbers. See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "first": "The whole row of numbers follows from the seed, not only the first. See [Random numbers and seeds](/course/open/np-random?lang=en).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of AI is picking by a number: the model with the lowest loss, the prompt with the best score on a test, the answer a reward model rates highest. The number stands in for what you want, and while nobody pushes on it, it often measures that well. Pick by it, and pick hard, and it starts to drift away from what it measured.

    Part of that drift needs no cheating at all, only noise and a choice. This lesson shows why, and lets you measure it in contests the course drew.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Ten options are all equally good: each has a true value of 0. You score each one with a measurement that is right on average, but has noise with a standard deviation of 1, and you pick the option with the highest score. On average, how far does the winner's score lie above its true value?",
        options={
            "zero": "About 0, since the score is right on average",
            "half": "About 0.5",
            "right": "About 1.5",
        },
        reveal="**About 1.5.** James Smith and Robert Winkler (2006) worked it out: \"with 10 it reaches 154% of the standard deviation of the value estimates.\" The winner is the option whose noise was luckiest. *Picking the best by a noisy score* in *The idea* says more.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Goodhart's law

    In 1975, Charles Goodhart wrote, in *Problems of Monetary Management: The U.K. Experience*, that "any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes." Goodhart wrote it about monetary control: a relationship between numbers that held while nobody steered by it tends to break once someone does.

    The sentence most people know is not Goodhart's. Marilyn Strathern wrote it in 1997, in *'Improving ratings': audit in the British University system*: "When a measure becomes a target, it ceases to be a good measure." Strathern credits the name "Goodhart's law" to Hoskin, and says of audit that it "does more than monitor—it has a life of its own that jeopardizes the life it audits."

    Donald Campbell had said much the same, in a paper of 1976, of which a version was presented in 1974: "The more any quantitative social indicator is used for social decision-making, the more subject it will be to corruption pressures and the more apt it will be to distort and corrupt the social processes it is intended to monitor." In Texarkana, in the United States, contractors "were to be paid on the basis of the achievement test score gains of individual pupils. It turned out that the contractors were teaching the answers to specific test items that were to be used on the final play-off testing." Soviet factories judged on weight, Campbell read, made only their heaviest item, such as the largest nails; judged on the number of items, only the easiest, such as the smallest nails. Campbell called this evidence "predominantly anecdotal".

    **Goodhart's law**, in short: push hard on a measure, and it stops measuring what you wanted. When a system raises a measure in a way that does not raise what the measure stood for, that is **proxy gaming**: the contractors gamed the test, the factories the weight, and in the lesson before, the course's agent gamed the points.

    ### Picking the best by a noisy score

    Gaming needs someone who games. One part of Goodhart's law needs no one at all. In 2018, David Manheim and Scott Garrabrant worked out four kinds of Goodhart's law, which Garrabrant had already described in 2017. The first is regressional Goodhart: "When selecting for a proxy measure, you select not only for the true goal, but also for the difference between the proxy and the goal." Think of a score as the true value plus noise. The option with the highest score probably has a high true value, but it probably also had lucky noise. Of this kind they write: "While this is the simplest Goodhart effect, it is also the most fundamental: it cannot be avoided."

    In 2006, James Smith and Robert Winkler had worked this out for decision analysis, under the name the optimizer's curse: "if we take these value estimates at face value and select accordingly, we should expect the value of the chosen alternative to be less than its estimate, even if the value estimates are unbiased." With options that are all equally good and noise with a standard deviation of 1, the winner's score lies on average 0.85 above its true value with three options, 1.03 with four, and 1.54 with ten: "This expected disappointment increases with the number of alternatives considered."

    That does not make the winner a bad pick. When the options really differ, the winner is still expected to be better than an average option; Manheim and Garrabrant note that an option with a high score is still expected to have a higher true value than others. Only its score promises too much.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Four kinds of Goodhart

    Manheim and Garrabrant's four kinds, in their words: "1) Regressional, where selection for an imperfect ... proxy necessarily also selects for noise, 2) Extremal, where selection for the metric pushes the state distribution into a region where old relationships no longer hold, 3) Causal, where an action on the part of the regulator causes the collapse, and 4) Adversarial, where an agent with different goals than the regulator causes the collapse." The regulator is whoever sets the measure. Extremal means: pushed far enough, you end up in situations unlike those in which the measure was seen to follow the goal. Under adversarial they place Campbell's law and the "Cobra Effect", "named after a supposed situation in colonial India where British authorities offered a reward for dead cobras."

    ### Being skeptical

    Smith and Winkler's way out is not to take the scores at face value. Suppose you know beforehand roughly where the true values lie, around a mean with a standard deviation that we call the spread, and how large the noise is. Then mix each score with that mean:

    estimate = α · score + (1 − α) · mean, with α = 1 / (1 + noise² / spread²).

    With little noise, α is close to 1 and the score counts almost fully; with a lot of noise, the estimate stays close to the mean. Of such estimates they write: "the decision maker who interprets value estimates as a Bayesian skeptic should not be expected to be disappointed on average." A Bayesian skeptic mixes a new score with what they expected beforehand, as here. In a study at Kodak from 1999 with seven alternatives, they took the mean to be 0 and the noise variance to be 20% of the spread variance, so α = 1 / 1.2 = 0.8333, and the value of the recommended alternative went from 80.0 to 66.7 million dollars.

    ### Reward models pushed too far

    In 2022, Leo Gao, John Schulman and Jacob Hilton of OpenAI measured Goodhart's law in language models. A reward model gives an answer a score for how good it is. A policy can be trained to get high scores; or you can draw many answers and keep the one with the highest score. To know the true quality as well, they let one reward model stand in for people: "a fixed ‘gold-standard’ reward model plays the role of humans, providing labels used to train a proxy reward model." The gold reward model had 6B parameters, the proxy reward models from 3M to 3B, and the policy 1.2B; a comparison used a 6B policy as well. What they found: "when we optimize for a learned proxy of the gold reward, the gold reward initially increases and later decreases." Drawing n answers and keeping the one the proxy scores highest is the contest of this lesson, with answers as the options. For the best of n answers, they chose the form of their formula on up to 1,000 answers and then tested it, as a prediction, on up to 60,000.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **Goodhart's law** | push hard on a measure, and it stops measuring what you wanted |
        | **proxy gaming** | raising a proxy in a way that does not raise what it stands for |
        | regressional Goodhart | picking by a score with noise also picks lucky noise |
        | optimizer's curse | the option you pick is expected to be worth less than its estimate, even when the estimates are right on average |
        | α (alpha) | the weight a skeptical estimate gives the score: 1 / (1 + noise² / spread²) |
        | reward model | a model that gives an answer a score for how good it is |
        | `C.TRUES`, `C.SCORES` | 2,000 contests of 10 options that really differ: the true values and the scores, a row per contest |
        | `C.EQUAL_TRUES`, `C.EQUAL_SCORES` | 2,000 contests of 10 options that are all equally good: every true value is 0 |
        """),
        L.watch_out("The winner of a contest with noise is not a bad pick: when the options really differ, it is usually better than an average option, though often not the truly best. It is worse than its score."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Find out how noisy your score is: score the same options twice, or on fresh questions, and see how much the scores move.
    2. Expect the winner to fall short of its score, the more so when you compared more options or the score has more noise.
    3. Measure the winner again, on fresh data that played no part in picking it.
    4. If you know roughly where the true values lie, shrink each score toward that mean before you compare, with α = 1 / (1 + noise² / spread²).
    5. When a number improves fast under pressure, look at what it stands for: did the goal improve, or only the number?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes contest 6 of the course from `C.TRUES` and `C.SCORES`: 10 options that really differ. It finds the winner with `np.argmax`, the option with the highest score, and compares the winner's score with its true value.
    """)
    return


@app.cell
def example_contest():
    true, score = C.TRUES[6], C.SCORES[6]   # contest 6: the true values and the scores of 10 options
    best = np.argmax(score)                  # the winner: the option with the highest score
    print("winner: option", best)
    print("its score:", round(score[best], 2), " its true value:", round(true[best], 2))
    print("gap:", round(score[best] - true[best], 2))
    print("truly best: option", np.argmax(true))
    return


@app.cell(hide_code=True)
def contest_cell():
    mo.Html(C.contest_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The winner, option 7, scores 3.0 but is worth 1.32: its noise gave it 1.68 extra. Two options are truly better, and option 2 is the truly best. One contest can be luck; in Exercise 1 you average the gap over all 2,000.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_average_gap():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. average_gap: how far the winner's score lies above its true value

        Write `average_gap(trues, scores)`: the average, over all contests, of how far the winner's score lies above its true value. `trues` and `scores` are NumPy arrays with a row per contest and a column per option. Keep the gaps in a list `gaps`. Go over the contests with a `for` loop over `range(len(scores))` that takes each row number as `i`: find the winner `best` with `np.argmax` of the row of `scores`, and add the winner's score minus its true value to `gaps`. After the loop, return the average with `np.mean`. Replace the `...`. The check calls `average_gap(TRUES, SCORES)`, the 2,000 contests of options that really differ, and then contests of its own.
        """),
        L.task("For every contest, take the winner by its score, and average its score minus its true value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average_gap():
    box_average_gap = L.code_box("average_gap", lines=9)
    box_average_gap
    return (box_average_gap,)


@app.cell(hide_code=True)
def check_average_gap(box_average_gap):
    result_average_gap = L.check("average_gap", box_average_gap.value)
    result_average_gap
    return (result_average_gap,)


@app.cell(hide_code=True)
def hints_average_gap():
    L.hints(
        "average_gap",
        [
            "The *Worked example* finds the winner of one contest and its gap. Do the same for every row, and take the average.",
            "`gaps` starts as an empty list. In the loop, `best` is the index of the highest number in `scores[i]`; the gap is the winner's number in `scores[i]` minus its number in `trues[i]`, both at index `best`. After the loop, `return np.mean(gaps)`.",
            "`        gaps.append(___)`",
        ],
        why="`gaps = []`, then `for i in range(len(scores)):` with `best = np.argmax(scores[i])` and `gaps.append(scores[i][best] - trues[i][best])` in it; after the loop `return np.mean(gaps)`. `average_gap(TRUES, SCORES)` gives about 1.11.",
    )
    return


@app.cell(hide_code=True)
def text_after_average_gap():
    mo.md(r"""
    With options that really differ, the winner's score lies 1.11 above its true value on average; with options that are all equally good, 1.56, close to the 1.54 of Smith and Winkler. Over all options, not just the winners, the gap is as good as 0: every score is right on average. It is the picking that makes the winner's score too high.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: more options, a larger gap

    For nine numbers of options from 1 to 1,000, the course drew 2,000 contests each: once with options that are all equally good, and once with options that really differ, with true values that have a standard deviation of 1. The slider sets the number of options; the bars show the winner's score, its true value and the gap, on average.
    """)
    return


@app.cell(hide_code=True)
def demo_options():
    options = L.explore("options", "options in a contest", lambda _v: C.demo_frame(_v, L.lang), start=10,
                        values=C.NS, role="",
                        task="Find the smallest number on the slider at which the gap, with all options equally good, is more than 2.")
    options
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The gap grows with every option you add, but ever more slowly: with all options equally good, 1.56 for 10 options, 2.50 for 100 and 3.24 for 1,000. When the options really differ, the winner's true value grows as well, from 1.08 with 10 options to 2.30 with 1,000: picking harder still picks better, only the score promises more than that. This is what noise alone does. Gao et al. make the same point: "if this is the only kind of Goodhart present, the gold reward must always increase monotonically with the proxy reward". Their gold reward did fall after a while, so either the noise was not of this simple kind or other kinds were at work as well; they expect extremal Goodhart to be mainly responsible.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_skeptical():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. skeptical: a score you do not take at face value

        Write `skeptical(scores, mean, spread, noise)`: the skeptical estimates of Smith and Winkler for a NumPy array `scores`, when the true values lie around `mean` with standard deviation `spread`, and the noise has standard deviation `noise`. First work out the weight `alpha` with the formula in *Being skeptical* in *The idea*. Then return `alpha` times `scores` plus `1 - alpha` times `mean`. Replace the `...`. The check calls `skeptical(SCORES[6], 0.0, 1.0, 1.0)`, the scores of contest 6 in the course's setting, and then settings of its own.
        """),
        L.task("Work out alpha from the noise and the spread, and mix the scores with the mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_skeptical():
    box_skeptical = L.code_box("skeptical", lines=3)
    box_skeptical
    return (box_skeptical,)


@app.cell(hide_code=True)
def check_skeptical(box_skeptical):
    result_skeptical = L.check("skeptical", box_skeptical.value)
    result_skeptical
    return (result_skeptical,)


@app.cell(hide_code=True)
def hints_skeptical():
    L.hints(
        "skeptical",
        [
            "*Being skeptical* in *The idea* gives the formula for α and for the estimate.",
            "In Python, a square is `**2`, so noise² is `noise**2`. `alpha` is 1 divided by 1 plus the noise variance over the spread variance. Then `return alpha * scores + (1 - alpha) * mean`.",
            "`    alpha = 1 / (1 + ___)`",
        ],
        why="`alpha = 1 / (1 + noise**2 / spread**2)` and `return alpha * scores + (1 - alpha) * mean`. With `noise` and `spread` both 1, `alpha` is 0.5: every score moves halfway to the mean.",
    )
    return


@app.cell(hide_code=True)
def text_after_skeptical():
    mo.md(r"""
    With as much noise as spread, α = 0.5: the skeptical estimate halves the distance of every score to the mean. In the course's contests the winners score 2.19 on average, their skeptical estimates come to 1.10, and their true values to 1.08: the skeptic is, on average, not disappointed. The price is what you rarely know exactly: where the true values lie, and how large the noise is. And when some options are harder to estimate than others, shrinking can change which one comes out best: Smith and Winkler show, with the Kodak study, that an option with a high but uncertain estimate can be passed over by one with a lower, more accurate estimate.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prompts():
    mo.md(r"""
    ### Exercise 3. The best of 50 prompts

    A team tries 50 prompts on the same 20 questions, scored by a judge, and keeps the prompt with the highest score: 85%. The prompts really differ in quality. The team then runs the winning prompt on 200 fresh questions.
    """)
    return


@app.cell(hide_code=True)
def choice_prompts_cell():
    choice_prompts = L.choice(
        "prompts",
        question="What do you expect on the fresh questions?",
        options={
            "same": "About 85%, since the score was right on average",
            "right": "Below 85%, but above an average prompt",
            "worse": "Below an average prompt, since the winner was only lucky",
        },
        correct="right",
        why={
            "right": "The winner was picked on 20 questions, so its score includes lucky noise: on fresh questions that luck is gone. But the prompts really differ, so picking the highest score also picked a good prompt: it stays above average, as the winner in the course's contests did.",
            "same": "Every single score is right on average, but the winner's is not: it was picked because it was high, and part of that was noise. See *Picking the best by a noisy score* in *The idea*.",
            "worse": "When the options really differ, the winner is still expected to be better than an average option; only its score promises too much. In the course's contests, the winner of 10 was worth 1.08 against 0 for an average option. See *Picking the best by a noisy score* in *The idea*.",
        },
    )
    choice_prompts
    return (choice_prompts,)


@app.cell(hide_code=True)
def text_exercise_fraud():
    mo.md(r"""
    ### Exercise 4. A fraud model in use

    A bank tests a model that flags fraudulent payments on last year's cases: it catches 95% of the fraud. Once the model is in use, fraudsters try out which payments it lets through, and change how they work. The share of fraud it catches falls.
    """)
    return


@app.cell(hide_code=True)
def choice_fraud_cell():
    choice_fraud = L.choice(
        "fraud",
        question="Which of the four kinds of Goodhart fits best?",
        options={
            "regressional": "Regressional: the model was picked for lucky noise on last year's cases",
            "right": "Adversarial: people with goals of their own react to the model",
            "extremal": "Extremal: the model was pushed to extreme scores",
        },
        correct="right",
        why={
            "right": "The share fell because others, with goals of their own, changed what they do once the model was used against them. Manheim and Garrabrant call that adversarial: \"an agent with different goals than the regulator causes the collapse\".",
            "regressional": "Lucky noise would make the 95% too high from the start, and it would not keep falling as the fraudsters learn. Here the fall comes from what the fraudsters do. See *Four kinds of Goodhart* in *The idea*.",
            "extremal": "Nobody pushed the model's scores to extremes; the fraudsters changed how they work. Goodhart caused by others with goals of their own is adversarial. See *Four kinds of Goodhart* in *The idea*.",
        },
    )
    choice_fraud
    return (choice_fraud,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en) showed agents that game their reward; this lesson gave the pattern its name, Goodhart's law, and showed a part of it that needs no gaming at all: picking the best by a score with noise also picks lucky noise, the more so the more options you compare. The skeptical estimate of Smith and Winkler takes that luck back out on average, and Gao et al. saw the true reward of a language model rise and then fall as it was pushed against a proxy. The next lesson, *Outer and inner alignment*, splits the problem in two: does the objective say what we want, and does the trained model pursue that objective?

    **An everyday example.** The restaurant with the best reviews out of fifty in town is often a little less good than its reviews when you go: part of its lead was luck.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fraud, choice_prompts, result_average_gap, result_skeptical):
    L.wrap_up(result_average_gap, result_skeptical, choice_prompts, choice_fraud)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
