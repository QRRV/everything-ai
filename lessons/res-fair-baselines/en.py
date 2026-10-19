# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-fair-baselines. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-fair-baselines", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Always beat a simple baseline](/course/open/ml-baselines?lang=en) and [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="A model gets an accuracy of 91% on the test set. What do you compare that with first?",
        options={
            "train": "The accuracy on the training set",
            "right": "A simple baseline on the same test set",
            "perfect": "100%, the best possible score",
        },
        correct="right",
        why={
            "train": "The training set says how well the model fits what it has seen, not whether 91% is good. See [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "perfect": "100% is the ceiling, not a comparison: it does not tell you whether 91% is hard or easy to get on this test set. See [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "right": "If always guessing the most common class also gets 90%, then 91% says little.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_noise():
    quick_noise = L.choice(
        "quick-noise",
        question="With only another seed, the small GPT ends a few hundredths higher or lower. One run at a learning rate of 0.003 ends 0.3 below one run at 0.001. What does that tell you?",
        options={
            "nothing": "Nothing: one run is never a result",
            "seed": "Only something about that one seed",
            "right": "The learning rate matters here",
        },
        correct="right",
        why={
            "nothing": "One run cannot show a difference of a few thousandths, but 0.3 is ten times the noise between seeds. See [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
            "seed": "With another seed, each run moves a few hundredths, not 0.3. See [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
            "right": "A difference ten times the noise between seeds does not come from the seed. For a small difference you need several seeds.",
        },
    )
    quick_noise
    return (quick_noise,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A new method is only as convincing as the baseline it beats. In research, the baseline is the method you want to improve on, and it should be its best version, not a weak one. If you tune your own method with care and run the baseline with whatever settings it came with, your method wins partly because of the tuning, not because it is better.

    That happens in published work. In 2018, Gábor Melis, Chris Dyer and Phil Blunsom tuned an older, simpler kind of language model and newer ones in the same way, with the same large automatic search over their settings. The older model then came out ahead of the newer ones ([Melis et al., 2018](https://arxiv.org/abs/1707.05589)).

    **An everyday example.** Two runners race: one in running shoes and warmed up, the other in boots, straight out of bed. The winner says little about who runs faster.
    """)
    return


@app.cell(hide_code=True)
def predict_gap():
    guess_gap = L.guess(
        "tuned-gap",
        question="In Change one thing at a time, the GPT with 4 layers ended 0.07 below the one with 2, both at a learning rate of 0.001. Now you try nine learning rates for the 4-layer GPT and take its best, and leave the 2-layer GPT at 0.001. How far apart do they end?",
        options={
            "same": "About 0.07 again",
            "right": "Much further apart",
            "closer": "Closer together",
        },
        reveal="**Much further apart: 0.62.** The 4-layer GPT at its best rate, 0.03, ends at 1.777 on average over seeds 0 to 2; the 2-layer GPT at 0.001 at 2.400. Almost all of that is tuning: give the 2-layer GPT the same nine rates, and its best, also at 0.03, ends at 1.779. Then the gap is 0.002, smaller than the spread between seeds.",
    )
    guess_gap
    return (guess_gap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **fair comparison** gives your method and the baseline the same chances:

        - the same data, split the same way into training and validation set;
        - the same compute, the amount of calculation that training takes: in practice, as many epochs, or as much time on the same computer;
        - the same seeds;
        - the same **tuning effort**: as many settings tried, chosen the same way, on the validation set.

        Tuning effort is the easiest to miss. The best of nine tries beats a single try, even for two methods that are just as good: the best try finds a better setting, and also gets some luck with the seeds. So the method you tuned more looks better than it is. Report how you tuned each method, the baseline included.

        The picture: the course's runs of the small GPT with 2 layers (dashed) and with 4 layers, at nine learning rates. Each line goes through the mean of seeds 0 to 2, the small circles are the single runs, and a ring marks the best rate of each model. At 0.001, the 4-layer GPT is clearly lower. At their best rates, the two end at the same place. And the 4-layer GPT took about twice as long per run: as many epochs gave it twice the time. With the same time, the 2-layer GPT could have trained longer. Say which of the two you kept the same.
        """),
        mo.Html(C.tuning_picture({"title": "Nine learning rates each", "x": "learning rate", "y": "validation loss", 2: "2 layers", 4: "4 layers", "lang": L.lang,
                                  "alt": "Validation loss against the learning rate, from 0.001 to 0.1, for the GPT with 2 and with 4 layers. Both fall from about 2.4 and 2.33 at 0.001 to about 1.78 at 0.03, and rise again at 0.05 and 0.1. At 0.03 the two lie on top of each other."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **fair comparison** | a comparison in which the baseline gets the same data, compute, seeds and tuning effort as your own method |
        | **tuning effort** | how many settings you try for a method before you take its best |
        """),
        L.watch_out("Tune on the validation set, never on the test set. A setting that you pick because it scores best on the test set makes that test score too good."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compare your method with a baseline fairly:

    1. Fix what they share: the data and its split, the number of epochs, the seeds.
    2. Give each the same tuning effort: the same learning rates, or as many settings, chosen on the validation set.
    3. Take each method at its own best setting, by the mean over the seeds.
    4. Compare those two, and report how you tuned each.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course's runs at three of the nine learning rates, seeds 0 to 2 each. The comments say which step each part is. It is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_fair():
    import numpy as np

    # 1. the same data, epochs and seeds for both
    _two = {0.001: [2.3902, 2.3924, 2.4174],
            0.003: [2.0827, 2.0987, 2.085],
            0.03: [1.7859, 1.7773, 1.7738]}
    _four = {0.001: [2.3335, 2.3245, 2.3374],
             0.003: [2.0117, 1.9931, 2.0099],
             0.03: [1.7745, 1.7776, 1.7776]}
    # 2. the same learning rates for both; 3. the mean over the seeds of each
    for _lr in _two:
        print("rate", _lr, " 2 layers:", round(np.mean(_two[_lr]), 3), " 4 layers:", round(np.mean(_four[_lr]), 3))
    # 3-4. each at its own best rate, and the difference
    _gap = np.mean(_four[0.03]) - np.mean(_two[0.001])  # complete: the 2-layer GPT at its own best rate, not at 0.001
    print("4 layers minus 2 layers:", round(_gap, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With the 2-layer GPT at 0.001, the difference is −0.623: the 4-layer GPT looks far better. At its own best rate, 0.03, the difference is −0.002, smaller than the spread between seeds of one rate. Almost all of the 0.62 was tuning.

    That holds for this small setup: 6 epochs on the first quarter of Tiny Shakespeare, 3 seeds. With longer training, the 4-layer GPT might pull ahead. A fair comparison tells you which method is better under the conditions you tested, not in general.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The best setting of a method

        Complete `best_setting(results)`: step 3 of the recipe as a function. `results` is a dictionary with the learning rates as keys and the validation losses of each rate's seeds as values, such as `{0.001: [2.3902, 2.3924, 2.4174], 0.003: [...]}`. The function gives back the learning rate with the lowest mean over its seeds, and that mean, as `[best_lr, best_mean]`. Replace the two `...`. The check calls `best_setting(two_layers)`, with all nine rates of the course's 2-layer GPT, and tries other results.
        """),
        L.task("Make `best_setting(results)` give back the learning rate with the lowest mean validation loss over its seeds, and that mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-setting", lines=10)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-setting", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-setting",
        [
            "Step 3 of the recipe: for each learning rate, the mean of its losses, as in the worked example. A loss is better when it is lower, so you keep a rate when its mean is lower than the best so far.",
            "`mean = np.mean(results[___])` and `if best_mean is None or mean ___ best_mean:`.",
        ],
        why="```python\nimport numpy as np\n\n\ndef best_setting(results):\n    best_lr = None\n    best_mean = None\n    for lr in results:\n        mean = np.mean(results[lr])\n        if best_mean is None or mean < best_mean:\n            best_lr = lr\n            best_mean = mean\n    return [best_lr, best_mean]\n```\n\nThe mean over the seeds keeps one lucky seed from choosing the rate. The first rate is the best so far, because there is nothing to compare with yet; each next rate takes its place when its mean is lower.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Three ways to compare

    Your `best_setting` picks the best rate of each model from the course's nine; until Exercise 1 is right, the course's `best_setting` does it. The table puts three comparisons next to each other.
    """)
    return


@app.cell(hide_code=True)
def demo_compare(box_best, result_best):
    _best, _yours = C.best_setting_of(box_best.value if result_best.state == "correct" else None)
    _lr2, _m2 = _best({_k: list(_v) for _k, _v in C.GRID[2].items()})
    _lr4, _m4 = _best({_k: list(_v) for _k, _v in C.GRID[4].items()})
    _d2, _d4 = C.mean_of(2, C.DEFAULT_LR), C.mean_of(4, C.DEFAULT_LR)
    _n, _s, _r = (lambda x: C.number(x, 3, L.lang)), (lambda x: C.signed(x, 3, L.lang)), (lambda x: C.rate(x, L.lang))
    _who = "Your `best_setting` chose" if _yours else "The course's `best_setting` chose"
    mo.md(f"""
    | comparison | 4 layers | 2 layers | difference |
    |---|---|---|---|
    | both at {_r(C.DEFAULT_LR)} | {_n(_d4)} | {_n(_d2)} | {_s(_d4 - _d2)} |
    | only 4 layers tuned | {_n(_m4)} at {_r(_lr4)} | {_n(_d2)} at {_r(C.DEFAULT_LR)} | {_s(_m4 - _d2)} |
    | both tuned the same way | {_n(_m4)} at {_r(_lr4)} | {_n(_m2)} at {_r(_lr2)} | {_s(_m4 - _m2)} |

    {_who} the rates. The first row treats both models alike, but at a rate that suits neither. Only the last row is fair: both models got the same nine tries, and each is taken at its own best rate.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Exercise 2. Which comparison is fair?
    """)
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair",
        question="You give your GPT 4 layers instead of 2 and want to show that it helps. Which comparison gives the 2-layer GPT a fair chance?",
        options={
            "yours": "The 4-layer GPT tuned over nine rates, the 2-layer GPT at the 0.001 it always had",
            "right": "Both tuned over the same nine rates, with the same seeds and epochs",
            "longer": "Both at 0.001, the 4-layer GPT with twice the epochs because it is bigger",
        },
        correct="right",
        why={
            "yours": "The 4-layer GPT gets nine tries and the 2-layer GPT one: most of the gap is then tuning. See the picture in *The idea*.",
            "longer": "Twice the epochs is more compute: the bigger GPT then wins partly because it trained longer. And 0.001 suits neither of them.",
            "right": "Same tuning effort, same seeds, same epochs: what is left of the gap comes from the layers. Here that is almost nothing.",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_review():
    mo.md(r"""
    ### Exercise 3. Read a claim
    """)
    return


@app.cell(hide_code=True)
def explain_review():
    explain_review_cell = L.explain(
        "review-claim",
        question="A paper says: “Our model reaches a validation loss of 1.80, far below the baseline's 2.40. We tried 30 learning rates for our model and trained it for 20 epochs. The baseline's number comes from its original paper, which trained it for 6 epochs.” What makes this comparison unfair, and what would you ask the authors to do?",
        model_answer="**Unfair:** the two did not get the same chances. Their model got 30 learning rates, the baseline whatever its own paper tried; their model trained 20 epochs, the baseline 6, so it got more compute; and the baseline's number comes from another paper, with other code and perhaps other data or another split, so the two numbers were not even measured the same way. Each number may also be one run. And their 1.80 is the best of 30 tries on the validation set itself, so it is a little too good. **Ask:** run the baseline again in their own setup, on the same data and split, with the same 20 epochs and the same seeds, tuned over the same 30 learning rates on the validation set. Then report both as a mean over several seeds, with the spread, measure the chosen settings on a test set, and say how each was tuned.",
        points=[
            "you name the unequal tuning effort: 30 learning rates against those of the original paper",
            "you name the unequal compute, 20 against 6 epochs, and that the baseline's number comes from another setup",
            "you ask for the baseline to be run again in the same setup, with the same tuning effort and seeds, and for both to be reported the same way",
        ],
    )
    explain_review_cell
    return (explain_review_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A baseline came first in [Always beat a simple baseline](/course/open/ml-baselines?lang=en); here it is the method you want to beat, and it gets the same care as your own. The runs extend those of [Change one thing at a time](/course/open/res-hypothesis-control?lang=en), and comparing means over seeds comes from [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en). A gap of 0.002 between two tuned models: is that a difference or noise? The next lesson, *Is the difference real?*, tests it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, explain_review_cell, result_best):
    L.wrap_up(result_best, choice_fair, explain_review_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
