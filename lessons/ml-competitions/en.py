# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-competitions. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-competitions", "en")
    C = L.checks   # the made-up competition of coin flips, its public and private scores, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) and one about [The default for tables: gradient-boosted trees](/course/open/ml-gbdt-default?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_luck():
    quick_luck = L.choice(
        "quick-luck",
        question="Ten models are scored on the same validation set, and you keep the best. What is its validation score?",
        options={
            "right": "A little too optimistic: the winner is partly the luckiest on those examples",
            "fair": "An honest estimate of how it does on new examples",
            "low": "Too pessimistic: the validation set is small",
        },
        correct="right",
        why={
            "fair": "One model on a validation set gets an honest estimate. The best of ten does not: among several models, the winner is partly the one that was luckiest on those examples. See the *Watch out* box in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "low": "A small validation set makes the score noisy, both ways. Keeping the best of ten picks the noise that went up. See the *Watch out* box in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "right": "That is why you measure on the test set at the end. Today you see what happens when the test set is looked at again and again.",
        },
    )
    quick_luck
    return (quick_luck,)


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="`HistGradientBoostingRegressor(early_stopping=True)` gets `X_val` and `y_val` in `fit`. What does it choose with them?",
        options={
            "right": "How many trees it keeps",
            "bins": "How many bins every feature gets",
            "features": "Which features it uses",
        },
        correct="right",
        why={
            "bins": "The bins are made from the training rows, before the first tree, with `max_bins`. See *Early stopping* in *The idea* in [The default for tables](/course/open/ml-gbdt-default?lang=en).",
            "features": "It uses every feature it gets. See *Early stopping* in *The idea* in [The default for tables](/course/open/ml-gbdt-default?lang=en).",
            "right": "It stops once 10 trees in a row did not lower the error on the validation set. On 7,837 hours of 2016 that choice is safe; today you see when choosing on the same examples is not.",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In a competition on a site such as Kaggle, teams get a training set with labels and a test set without. They send in a prediction for every test example, a submission, and a leaderboard shows their score on part of the test set: the **public leaderboard**. The rest of the test set stays hidden until the end, and its score, the **private leaderboard**, decides who wins. Teams may send in several submissions a day. Some climb the public leaderboard by trying again and again and keeping what scores best, and drop when the private leaderboard is shown. What happens to them happens to anyone who chooses many times on the same examples: in a competition, at work, or in a paper.

    **An everyday example.** A student practices for an exam with the same 20 questions of last year, again and again, until they know every answer. Their score on those 20 questions says little about the exam, which has new questions.
    """)
    return


@app.cell(hide_code=True)
def predict_coins_cell():
    guess_coins = L.guess(
        "coins-guess",
        question="200 teams do not even look at the data: each sends in one submission with a coin flip for every test example, 1 or 0. The public leaderboard scores 300 of the 1,000 test examples. What is the best public score?",
        options={
            "coin": "About 50%: a coin is a coin",
            "right": "About 57%",
            "high": "About 75%",
        },
        reveal="**57.0%.** The best of 200 coin flips gets 57.0% of the 300 public examples right. On the other 700 examples, the private part, the same submission gets 48.6%. The public leaderboard rewarded luck.",
    )
    guess_coins
    return (guess_coins,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The leaderboard.** The test set is split at random into a public part and a private part. Every submission gets a score on each: here the share right, on 300 public and 700 private examples. During the competition, the **leaderboard** shows only the public scores.

        **Every submission is a look.** A public score is a score on a validation set, one that you look at every time you send something in. One look gives an honest estimate. The best of many looks does not: as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), the winner is partly the luckiest. A coin flip per example gets 50% right on average, but on 300 examples its score varies, with a standard deviation of about 2.9 percentage points. The more submissions, the higher the best of them climbs by luck alone, while the private score of that same submission stays at its real level. Choosing on the public part until it flatters you is **leaderboard overfitting**.

        **Luck that adds up.** Submissions can also be combined. Keep the coin flips that happened to score above 50% on the public part, and let them vote on every example. On the public part each of them was right a little more often than wrong, so their vote is right much more often there. On the private part none of them knows anything, and neither does the vote. You build that vote in exercise 2.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **public leaderboard** | the scores on the part of the test set that is shown during a competition | `right[:, public].mean(axis=1)` |
        | **private leaderboard** | the scores on the rest of the test set, shown at the end; it decides who wins | `right[:, private].mean(axis=1)` |
        | **leaderboard overfitting** | choosing on the public part until its score flatters you, so the private score is lower | |
        | `np.argmax(a)` | the position of the largest value of `a` | `np.argmax([0.2, 0.7, 0.5])` gives `1` |
        """),
        L.watch_out("The public score of your best submission is not your score. It is the best of many looks at the same few examples. Judge your work by your own validation, such as cross-validation on the training set, and use the public leaderboard as one more noisy look."),
        mo.md(r"""
        The slider sets how many of the coin-flip submissions are sent in. Every dot is one submission: its public score on the upper row, its private score on the lower row. The dark line joins the public and the private score of the one that is best on the public part. The dashed line is 50%. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_submissions_cell():
    def _frame(k):
        said = C.strip_said(k, L.lang)
        alt = ("Two rows of dots between 40% and 65% right: above, the public scores of the first submissions; below, "
               "their private scores; a line joins the two scores of the best on the public part. " + said)
        return (C.strip_picture(k, {"title": "Coin flips on the leaderboard", "x": "share right", "public": "public part",
                                    "private": "private part", "alt": alt}), said)

    explore_submissions = L.explore("submissions", "submissions", _frame, start=1, values=C.SLIDER, role="hyper",
                                    task="Go from 1 submission to 1,000. What does the best public score do, and the private score of that same submission?")
    explore_submissions
    return


@app.cell(hide_code=True)
def text_after_submissions():
    mo.md(r"""
    With 10 submissions, the best public score is 53.7%; with 100, 56.7%; with 200, 57.0%; with 1,000, 61.3%. The private score of the winner stays near 50% every time: 48.9%, 51.7%, 48.6% and 54.4%. Even 54.4% is luck: on 700 examples a coin varies by about 1.9 percentage points. Not one submission knows anything; only the luck on 300 examples grows with the number of tries.

    **Real models too.** With real models it happens as well, more softly, because their scores differ for real. The course took 24 settings of `HistGradientBoostingRegressor` on the traffic of [The default for tables](/course/open/ml-gbdt-default?lang=en), and split the 15,246 test hours at random into a public part and a private part, 200 times for each size. With 100 public hours, the setting that is best on the public part was also best on the private part in 10% of the splits; with 300 hours in 32%, with 1,000 in 55%, with 4,500 in 75%. And with 100 public hours, the private MAE of the winner was on average 6.5 vehicles an hour higher than its public MAE; with 1,000, 1.4 higher. The smaller the part you choose on, and the more you choose, the more the winner is luck.

    ### Recipe

    To keep a leaderboard honest:

    1. Set your own validation aside, or use cross-validation on the training set, and choose by that.
    2. Send in rarely, and read every public score as one more noisy look.
    3. Distrust a jump on the public leaderboard that your own validation does not show.
    4. Choose your final submissions by your own validation, not by the public score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes the competition of the slider: 1,000 hidden labels, 0 or 1; a random order of the 1,000 examples, whose first 300 are the public part and the other 700 the private part; and 1,000 submissions with a coin flip for every example. `submissions == y` compares every row with the labels, by broadcasting, and the mean of a row of True and False is its share right. The cell prints the mean public and private score of all submissions, then, of the first 200, which one is best on the public part, its public score and its private score. Run the cell.
    """)
    return


@app.cell
def example_competition():
    import numpy as np

    _rng = np.random.default_rng(0)
    _y = _rng.integers(0, 2, 1000)                      # the hidden labels of the 1,000 test examples
    _order = _rng.permutation(1000)
    _public = _order[:300]                              # the 300 examples of the public leaderboard
    _private = _order[300:]                             # the 700 of the private leaderboard
    _submissions = _rng.integers(0, 2, (1000, 1000))    # 1,000 submissions: a coin flip for every example

    _right = _submissions == _y                         # True where a submission is right
    _public_scores = _right[:, _public].mean(axis=1)
    _private_scores = _right[:, _private].mean(axis=1)
    print(round(_public_scores.mean(), 3), round(_private_scores.mean(), 3))

    # the first 200 submissions: the best on the public part, and the same one on the private part
    _first = _public_scores[:200]
    _best = np.argmax(_first)
    print(_best, _first[_best], round(_private_scores[_best], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    On average the submissions get 49.9% right on the public part and 50.0% on the private part, as coins should. Of the first 200, submission 115 is best on the public part, with 57.0%, and gets 48.6% on the private part. On a leaderboard of these 200, it would have been first during the competition, and 159th at the end, tied with two others.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_of():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The winner of the public leaderboard

        `best_of(public_scores, private_scores)` gets the public and the private score of every submission, in the same order, and should give the private score of the submission with the highest public score: what the leader of the public leaderboard gets at the end. Replace the two `...`. The check calls `best_of(public_scores, private_scores)` with the scores of the first 200 submissions of the worked example, and tries three short arrays.
        """),
        L.task("Let `best_of` find the leader of the public leaderboard and give its private score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_of():
    box_best_of = L.code_box("best-of", lines=6)
    box_best_of
    return (box_best_of,)


@app.cell(hide_code=True)
def check_best_of(box_best_of):
    result_best_of = L.check("best-of", box_best_of.value)
    result_best_of
    return (result_best_of,)


@app.cell(hide_code=True)
def hints_best_of():
    L.hints(
        "best-of",
        [
            "Two steps: find where the highest public score is, then look up the private score at that same place.",
            "You need the position of the highest value, not the value itself; NumPy has a function for that, as in the worked example. The private score of that submission is at the same position in `private_scores`.",
            "`best = np.___(public_scores)` and `return private_scores[___]`.",
        ],
        why="The leader of the public leaderboard of 200 coin flips scores 57.0% there and 48.6% on the private part. The best private score of the 200 is 56.4%, but nobody could see which one that was.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The vote of the lucky ones

        `vote(submissions, public_scores)` gets the submissions, a row per submission and a column per example, and their public scores. It should keep the submissions with a public score above 0.5, and give for every example whether more than half of them say 1. Replace the two `...`. The check calls `vote(submissions, public_scores)` with the first 200 submissions of the worked example, and tries three made-up sets, each with one public score of exactly 0.5.
        """),
        L.task("Let `vote` combine the submissions that were lucky on the public part."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("vote", lines=5)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "vote",
        [
            "Two steps: keep the rows of the lucky submissions, then for every column, every example, the share of them that say 1.",
            "A mask between the brackets keeps the rows where it is True, as in [Selecting parts of an array](/course/open/np-indexing?lang=en). The mean of 0s and 1s is the share of 1s, and `axis` says whether it runs down the columns or along the rows.",
            "`lucky = submissions[public_scores ___ 0.5]` and `votes = lucky.mean(axis=___)`.",
        ],
        why="Of the first 200 submissions, 100 score above 50% on the public part. Their vote gets 67.0% right there, and 50.9% on the private part. With all 1,000 submissions the vote gets 84.0% on the public part and 49.9% on the private part: the higher the public score, the more luck it holds.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_final():
    mo.md(r"""
    ### Exercise 3. The final submissions
    """)
    return


@app.cell(hide_code=True)
def choice_final_cell():
    choice_final = L.choice(
        "final-pick",
        question="A competition ends tomorrow, and you may choose two of your submissions for the private leaderboard. Which two do you choose?",
        options={
            "right": "The two with the best score in your own cross-validation on the training set",
            "public": "The two with the best public score",
            "vote": "A vote of your submissions that did best on the public part",
        },
        correct="right",
        why={
            "public": "The public score of your best submission is the best of many looks at the same few examples: the leader of 200 coin flips had 57.0% there and 48.6% on the private part.",
            "vote": "That is the vote of exercise 2: 67.0% on the public part and 50.9% on the private part. It adds up the luck on the public part, nothing else.",
            "right": "Your cross-validation uses all of your training set, more examples than the public part, and you did not climb it by sending in. After many choices it can flatter a little too, which is why the private leaderboard, the test set, has the last word.",
        },
    )
    choice_final
    return (choice_final,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The public leaderboard is a validation set that everyone looks at again and again, so the warning of [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) grows with every submission. The vote of exercise 2 is the majority vote of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en), of models that know nothing. Early stopping in [The default for tables](/course/open/ml-gbdt-default?lang=en) also chooses on a validation set, but it picks one number, how many trees, on 7,837 hours. In the next lesson, *When one class is rare*, a model that is 97% right can be useless.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_final, result_best_of, result_vote):
    L.wrap_up(result_best_of, result_vote, choice_final)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
