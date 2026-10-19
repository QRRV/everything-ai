# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-critical-reading. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-critical-reading", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading a paper in three passes](/course/open/res-three-pass?lang=en) and [Fair comparisons](/course/open/res-fair-baselines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pass():
    quick_pass = L.choice(
        "quick-second",
        question="What should you be able to do after the second pass of a paper?",
        options={
            "rebuild": "Build the method again from scratch",
            "right": "Tell someone its main claim and the evidence for it",
            "kind": "Say what kind of paper it is and which earlier work it builds on",
        },
        correct="right",
        why={
            "rebuild": "That is what the third pass is for. See [Reading a paper in three passes](/course/open/res-three-pass?lang=en).",
            "kind": "That is what the first pass gives you. See [Reading a paper in three passes](/course/open/res-three-pass?lang=en).",
            "right": "The claim and its evidence: this lesson checks whether the one really carries the other.",
        },
    )
    quick_pass
    return (quick_pass,)


@app.cell(hide_code=True)
def quick_check_fair():
    quick_fair = L.choice(
        "quick-fair",
        question="A new method was tuned for a week. The baseline was run once, with the settings of an old paper. The new method wins by a little. What does that show?",
        options={
            "wins": "That the new method is better: it won the comparison",
            "baseline": "That the old settings were best for the baseline",
            "right": "Not much: the baseline was not tuned as hard",
        },
        correct="right",
        why={
            "wins": "A week of tuning can be worth more than a small gap. See [Fair comparisons](/course/open/res-fair-baselines?lang=en).",
            "baseline": "Nobody searched for the baseline's best settings, so we do not know. See [Fair comparisons](/course/open/res-fair-baselines?lang=en).",
            "right": "This lesson puts that check next to three others.",
        },
    )
    quick_fair
    return (quick_fair,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A paper's abstract says what the authors believe they showed. Its tables and figures say what they did show. The two are not always the same, and the gap is rarely a lie: one run that looked good, a baseline that got less care, a metric that happened to favour the new method. Before you build on a paper, cite it, or set out to reproduce it, you check the claims against the evidence.
    """)
    return


@app.cell(hide_code=True)
def predict_best():
    guess_best = L.guess(
        "best-of-four",
        question="A new method is, in truth, exactly as good as the baseline. The authors try 4 variants of it, run each once, and report the best one against one run of the baseline. How likely is it that the best variant \"beats\" the baseline?",
        options={
            "half": "About 50%",
            "quarter": "About 20%",
            "right": "About 80%",
        },
        reveal="**About 80%.** All 5 runs are equally good, so each is equally likely to be the best of the 5. The baseline only wins when it happens to be the best: 1 time in 5. So a \"win\" comes 4 times in 5, though nothing is better.",
    )
    guess_best
    return (guess_best,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: each claim next to its evidence.** Write down every claim of the abstract and the conclusions, and next to it the table, figure or number that is meant to carry it. A claim with nothing next to it, or with less than it says, is an **overclaim**.

        **Step 2: the baselines.** Is there a baseline at all? Is it strong, the best method known, not an old weak one? Did it get the same tuning effort, data and compute as the new method?

        **Step 3: the noise.** How many seeds per method? Is the gap bigger than the run-to-run variance? With one run each, a gap smaller than the spread between seeds says nothing.

        **Step 4: the selection.** Is everything reported, or only where the new method wins: which metrics, which datasets, which model sizes, which variants? The best of $n$ equally good variants beats one run of the baseline $n / (n + 1)$ of the time.
        """),
        mo.Html(C.chance_picture({"title": "A win with nothing to win", "x": "variants tried, each run once", "y": "chance of a win", "lang": L.lang,
                                  "alt": "A curve of the chance that the best of n equally good variants beats one run of the baseline, for n from 1 to 20. It rises from 0.50 at 1 variant to 0.80 at 4, 0.90 at 9 and 0.95 at 19."})),
        mo.md(r"""
        The picture: the chance of a "win" when nothing is better, against the number of variants tried. Words that often go further than the evidence: *consistently* (one run?), *significantly* (one run each, no interval?), *state of the art* (one baseline?), *in general* or *for all* (one dataset?).

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **overclaim** | a claim that says more than the experiments show |
        | **strong baseline** | the best method known for the task, tuned as carefully as the new one |
        | **cherry-picking** | reporting only the runs, metrics or datasets where a method looks good |
        """),
        L.watch_out("An overclaim does not make the result false, and a paper without one is not therefore right. Critical reading finds what the evidence carries: often a smaller, true claim, which is the one to build on or to reproduce."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a paper's claims against its evidence:

    1. List the claims of the abstract and the conclusions, and next to each the result that should carry it.
    2. Check the baselines: present, strong, and tuned with the same effort.
    3. Check the noise: seeds per method, and a gap bigger than the spread between them.
    4. Check the selection: what was tried and not reported. Then write the claim the evidence does carry.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    What is new is the routine, applied to a whole paper. An invented abstract: *"FastDrop, a new kind of dropout, raises the accuracy of a small network on the handwritten digits by 1.2 points, and generalizes better than any dropout method."* Its table: dropout 91.0%, FastDrop 92.2%, each the mean of 5 seeds, with a spread of about 0.4 between the lowest and the highest seed; both tuned over the same 6 dropout rates.

    1. **Claims and evidence:** "+1.2 points on the digits" has a row in the table. "Better than any dropout method" has nothing next to it: only plain dropout was compared.
    2. **Baselines:** plain dropout, tuned like FastDrop: fair, but not the strongest dropout method there is.
    3. **Noise:** 5 seeds each, and a gap of 1.2 against a spread of 0.4: the runs of the two do not even overlap, so this is clearly more than noise.
    4. **Selection:** one dataset, one network, one metric. "Generalizes better" in general is more than one dataset shows.

    The claim the evidence carries: *on the handwritten digits, FastDrop beats plain dropout tuned the same way by about 1.2 points.* Smaller, and true.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overclaims():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Find the overclaims

        Another invented paper. Its setup and its table:
        """),
        mo.Html(C.paper_card({"title": "WarmNorm: a warmer normalization for small language models", "invented": "A paper invented for this lesson.",
                              "setup_name": "Setup.", "setup": "A GPT of 4 layers, trained on Tiny Shakespeare. One run per method. Learning rate 0.001, chosen for WarmNorm from five values on the validation set.",
                              "rows": [("validation loss", "1.62", "1.58"), ("time per training step", "100 ms", "103 ms"),
                                       ("numbers to learn, per place it is used", "2d", "2d + 2")]})),
        mo.md(r"""
        Its abstract makes six claims. Put each in its group.
        """),
    ])
    return


@app.cell(hide_code=True)
def exercise_overclaims():
    sort_overclaims = L.sort("overclaims")
    sort_overclaims
    return (sort_overclaims,)


@app.cell(hide_code=True)
def check_overclaims(sort_overclaims):
    result_overclaims = L.check("overclaims", sort_overclaims)
    result_overclaims
    return (result_overclaims,)


@app.cell(hide_code=True)
def hints_overclaims():
    L.hints(
        "overclaims",
        [
            "Use the recipe: for each sentence, find the row of the table or the line of the setup that should carry it. Is there one, and does it say as much as the sentence, also given how each method was tuned and how many runs there are?",
        ],
        why="**Shown:** the losses of one run each, 1.58 and 1.62, the 3% longer step, the 2 extra numbers to learn. **Claims more:** *a lower validation loss than LayerNorm* (the learning rate was chosen for WarmNorm only, and there is one run each), *a new state of the art* (only LayerNorm was compared), *at every learning rate* (the comparison was run at one learning rate).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fake():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Fake wins, simulated

        Check the picture with a simulation. Complete `fake_win_rate(n_variants, n_trials, seed)`: in each of `n_trials` experiments, every variant and the baseline get one score from the normal distribution with mean 0 and standard deviation 1, `rng.normal(...)`, so all are equally good, and higher is better. Give back the share of experiments in which the best variant scores higher than the baseline. The check calls `fake_win_rate(4, 20000, 0)`, which should give about 0.8, and tries 1, 9 and 19 variants.
        """),
        L.task("Complete `fake_win_rate`: draw the scores with `rng`, and give back how often the best variant beats the baseline."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fake():
    box_fake = L.code_box("fake-wins", lines=10)
    box_fake
    return (box_fake,)


@app.cell(hide_code=True)
def check_fake(box_fake):
    result_fake = L.check("fake-wins", box_fake.value)
    result_fake
    return (result_fake,)


@app.cell(hide_code=True)
def hints_fake():
    L.hints(
        "fake-wins",
        [
            "Draw an array with one row per experiment and one column per variant, and an array with one baseline score per experiment. The best variant of each experiment is the largest number in its row; comparing gives an array of True and False, and its mean is the share of True.",
        ],
        why="`variants = rng.normal(size=(n_trials, n_variants))`, `baseline = rng.normal(size=n_trials)`, and `return (variants.max(axis=1) > baseline).mean()`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ask():
    mo.md(r"""
    ### Exercise 3. What to ask for
    """)
    return


@app.cell(hide_code=True)
def choice_ask_cell():
    choice_ask = L.choice(
        "first-ask",
        question="A paper reports a validation loss 0.02 below the baseline's, with one run of each. In your own runs of a similar model, the loss moves about 0.05 from seed to seed. What do you ask the authors for first?",
        options={
            "size": "Results for a bigger model, to see whether the gain grows with size",
            "right": "Several seeds for both methods, with their spread",
            "metric": "A second metric next to the loss",
        },
        correct="right",
        why={
            "size": "A bigger model asks whether the result carries over. First ask whether there is a result at all: the gap is smaller than the noise.",
            "metric": "Another metric from the same single runs has the same problem: one run each cannot separate a gap of 0.02 from noise of 0.05.",
            "right": "With a spread of 0.05 between seeds, one run each cannot show a gap of 0.02. Several seeds, and the spread, show whether there is a gap at all.",
        },
    )
    choice_ask
    return (choice_ask,)


@app.cell(hide_code=True)
def text_exercise_carried():
    mo.md(r"""
    ### Exercise 4. The claim the evidence carries
    """)
    return


@app.cell(hide_code=True)
def explain_carried():
    explain_carried_cell = L.explain(
        "carried-claim",
        question="Go back to the setup and the table of WarmNorm in Exercise 1. Write, in one or two sentences, the claim its evidence does carry, as at the end of the worked example, and the two things you would ask the authors for first.",
        model_answer="**Carried:** with a GPT of 4 layers on Tiny Shakespeare, one run of WarmNorm, at a learning rate chosen for WarmNorm, ended at a validation loss of 1.58, and one run of LayerNorm at the same rate at 1.62; a training step takes 3% longer, and WarmNorm learns 2 numbers more in each place. Whether WarmNorm is better at all, this does not show. **Ask for:** LayerNorm tuned over the same five learning rates, and several seeds for both methods, with their spread.",
        points=[
            "your claim stays within the setup: one small GPT, Tiny Shakespeare, one run each",
            "you say that the learning rate was chosen for WarmNorm only, so part of the gap may be tuning",
            "you ask for LayerNorm tuned the same way, and for several seeds for both methods, with their spread",
        ],
    )
    explain_carried_cell
    return (explain_carried_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The baselines of [Fair comparisons](/course/open/res-fair-baselines?lang=en) and the claims and evidence of the second pass in [Reading a paper in three passes](/course/open/res-three-pass?lang=en), now as one routine for any paper. The next lesson, *Choosing a paper you can reproduce*, uses it to pick a claim worth testing yourself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ask, explain_carried_cell, result_fake, result_overclaims):
    L.wrap_up(result_overclaims, result_fake, choice_ask, explain_carried_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
