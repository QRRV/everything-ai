# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-choose-paper. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-choose-paper", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Critical reading: claims versus evidence](/course/open/res-critical-reading?lang=en) and [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_smaller():
    quick_smaller = L.choice(
        "quick-smaller",
        question="Critical reading finds overclaims in a paper. What is usually left?",
        options={
            "nothing": "Nothing: one overclaim makes the whole paper false",
            "right": "A smaller claim: the one the evidence does carry",
            "same": "The same claim: overclaims are only words",
        },
        correct="right",
        why={
            "nothing": "An overclaim says more than the evidence; the evidence itself can still be sound. See [Critical reading: claims versus evidence](/course/open/res-critical-reading?lang=en).",
            "same": "An overclaim says more than the experiments show; the claim to build on is the one they do show. See [Critical reading: claims versus evidence](/course/open/res-critical-reading?lang=en).",
            "right": "And a small, sound claim is just what this lesson looks for: one you can test yourself.",
        },
    )
    quick_smaller
    return (quick_smaller,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-6nd",
        question="A model trains for 10 epochs on a dataset of 100 million tokens. What is $D$ in $6ND$?",
        options={
            "epoch": "100 million",
            "right": "1 billion",
            "six": "6 billion",
        },
        correct="right",
        why={
            "epoch": "That is one epoch. $D$ counts every token the model trains on, epochs included. See [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).",
            "six": "That is $6D$: the 6 is the FLOPs per parameter per token, not part of $D$. See [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en).",
            "right": "$10 \\times 10^8 = 10^9$: every token counts, epochs included. This lesson counts one level up: runs times seeds.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Reproducing a paper is one of the best ways to understand it, and to find out whether it holds. But many attempts end before the first result: the experiment needs months of GPUs, the data cannot be had, or the claim is too vague to test. Since 2018, the ML Reproducibility Challenge has asked people to reproduce papers from the big conferences, and every such reproduction starts with a choice: which paper, and which claim. In the next lessons you reproduce a paper yourself, first guided, then on your own.
    """)
    return


@app.cell(hide_code=True)
def predict_code():
    guess_code = L.guess(
        "shared-code",
        question="In 2018, Odd Erik Gundersen and Sigbjørn Kjensmo looked at 400 papers from two big AI conferences. How many of them shared the code of their method?",
        options={
            "most": "About 90%",
            "half": "About half",
            "right": "About 6%",
        },
        reveal="**About 6%.** About a third shared the data they tested on, and about half gave pseudocode, the steps of the method written out in plain words and formulas. Since then, conferences ask authors to fill in a checklist on code and data, and far more papers share code. But not all, and code that runs on someone else's machine is rarer still.",
    )
    guess_code
    return (guess_code,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: does it fit your compute?** Estimate one run with 6ND, and the whole reproduction as runs × seeds. Then compare with what you have: a laptop's CPU reaches about $10^{11}$ FLOP/s when it trains a small model, so a day of it is about $10^{16}$ FLOPs; one GPU for a week, at 40% of an A100's peak, about $10^{20}$; a small model reaches far less of that peak. Look for the smallest setting of the paper where the claim still shows.
        """),
        mo.Html(C.budget_picture({"title": "One run of four candidates", "x": "FLOPs of one run",
                                  "names": ["0.4M parameters, 0.25B tokens", "10M parameters, 1B tokens", "124M parameters, 10B tokens", "7B parameters, 2T tokens"],
                                  "laptop": "laptop, a day", "gpu": "GPU, a week",
                                  "alt": "Four bars on a log scale of FLOPs. 0.4 million parameters on 0.25 billion tokens: about 6 times 10 to the 14, well within a day of a laptop. 10 million on 1 billion: 6 times 10 to the 16, a week of a laptop, less than an hour of a GPU. 124 million on 10 billion: about 7 times 10 to the 18, within a week of one GPU. 7 billion on 2 trillion: about 8 times 10 to the 22, a thousand times more than a week of one GPU."})),
        mo.md(r"""
        **Step 2: what is there?** Code, data, checkpoints, and the settings in the appendix. Not every missing piece costs the same: data you cannot get stops you, code that is missing you can write, settings that are missing you have to guess or ask the authors for.

        **Step 3: a small, testable claim.** One sentence, one figure or table it rests on, and a result that would show it false. "A step towards understanding" names no such result; "the method beats the baseline by 2 percentage points of accuracy, in each of 3 seeds" does.

        **Step 4: the cost before you start.** Runs × seeds × FLOPs per run, divided by what your machine does, times a margin of 2 or 3 for the runs that go wrong. If it does not fit, choose a smaller setting, or another claim.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **reproduction** | doing someone else's experiment again, from the paper, to see whether its result holds |
        | **falsifiable** | of a claim: there is a result that would show it is false |
        | **ML Reproducibility Challenge** | a yearly event, since 2018, in which people reproduce papers from big ML conferences and publish what they found |
        """),
        L.watch_out("A reproduction that does not find the paper's result is a result too, if you did what the paper said. Write down every choice you had to make yourself: that is often where the difference lies."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To choose a paper you can reproduce:

    1. Compute: estimate one run with 6ND, and find the paper's smallest setting where the claim shows.
    2. Availability: check for code, data, checkpoints and the settings; data you cannot get rules a paper out.
    3. Claim: write one sentence, the figure it rests on, and the result that would show it false.
    4. Cost: runs × seeds × FLOPs per run, divided by your machine's FLOP/s, times a margin of 2 or 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    What is new is the cost of a whole reproduction, not of one run: the second candidate of the picture, with its baseline, 3 seeds each.
    """)
    return


@app.cell
def example_cost():
    _n_params, _n_tokens = 10e6, 1e9  # one run: a small GPT on 1 billion tokens
    _runs, _seeds = 2, 3  # the method and its baseline, 3 seeds each
    _flops = 6 * _n_params * _n_tokens * _runs * _seeds
    for _machine, _speed in [("laptop", 1e11), ("GPU", 312e12 * 0.1)]:  # an A100 at 10% of its peak: the model is small
        _hours = _flops / _speed / 3600
        print(_machine, round(_hours, 1), "hours, with a margin of 2:", round(2 * _hours, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Six runs take $3.6 \times 10^{17}$ FLOPs: 1,000 hours on a laptop, six weeks without stopping, and about 3 hours on one GPU. The same reproduction is out of reach on one machine and easy on the other; with a smaller setting, such as 100 million tokens per run, it fits a laptop in about four days, eight with a margin of 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A good first reproduction?

    Six claims, from real and invented papers. Put each in its group, for a reproduction on your own machine.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_first = L.sort("first-repro")
    sort_first
    return (sort_first,)


@app.cell(hide_code=True)
def check_sort(sort_first):
    result_first = L.check("first-repro", sort_first)
    result_first
    return (result_first,)


@app.cell(hide_code=True)
def hints_sort():
    L.hints(
        "first-repro",
        [
            "Steps 1 to 3 of the recipe, for each claim: does it fit your machine, can you get the data, and which result would show it false?",
        ],
        why="**Good:** the small transformer on modular addition, dropout in a small GPT, batch normalization in a small network on the digits: small models, public or generated data, a number to measure. **Not now:** 70 billion parameters on 15 trillion tokens (far beyond one GPU), records of one hospital (data you cannot get), \"truly understand\" (no result could show it false).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_falsifiable():
    mo.md(r"""
    ### Exercise 2. A claim you can test
    """)
    return


@app.cell(hide_code=True)
def choice_falsifiable_cell():
    choice_falsifiable = L.choice(
        "falsifiable",
        question="Three claims someone could write about batch normalization. Which one could your reproduction show to be false?",
        options={
            "right": "With batch normalization, an MLP of 10 hidden layers reads at least 80% of the test digits right after one epoch at a learning rate of 1.0",
            "light": "Our results shed new light on why batch normalization makes deep networks easier to train",
            "deep": "Batch normalization reveals a deep link between the scale of the layers and how well they learn",
        },
        correct="right",
        why={
            "light": "No result could show that no light was shed: the sentence names nothing to measure.",
            "deep": "\"A deep link\" names no result: what would you measure, and which number would show it false?",
            "right": "One number, one point in training, one setting: if your runs stay below 80% after that epoch, the claim fails.",
        },
    )
    choice_falsifiable
    return (choice_falsifiable,)


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.md(r"""
    ### Exercise 3. Your plan
    """)
    return


@app.cell(hide_code=True)
def explain_plan():
    explain_plan_cell = L.explain(
        "my-plan",
        question="Choose the dropout or the batch normalization claim of Exercise 1, or a paper you pick yourself; the first claim, the late jump called grokking, is for the next lesson. Write your plan: the claim in one sentence, the figure it rests on, the result that would show it false, which code and data exist, and the cost of the reproduction in FLOPs and in hours on your machine.",
        model_answer="**Claim:** in the small GPT of about a million parameters that the course trains on a CPU, dropout of 0.1 gives a lower best validation loss on Tiny Shakespeare than no dropout. **Figure:** the validation loss after every epoch, with and without dropout. **False if:** with 3 seeds for each setting, the best validation loss with dropout is not lower than without, or the gap is smaller than the spread between seeds. **Code and data:** Tiny Shakespeare is in the course, my GPT and my Trainer are in my project, and dropout is one more layer, `nn.Dropout(0.1)`. **Cost:** about $10^6$ parameters, and 10 epochs over a million characters is $10^7$ tokens per run: $6 \\times 10^6 \\times 10^7 = 6 \\times 10^{13}$ FLOPs, about 10 minutes on a laptop at $10^{11}$ FLOP/s; with and without dropout, 3 seeds each and a margin of 2, about 2 hours.",
        points=[
            "you write the claim as one sentence, with the figure or table it rests on",
            "you name a result that would show the claim false",
            "you say which code and data exist, or how you get them",
            "you estimate the cost with 6ND, times runs and seeds, in hours on your machine",
        ],
    )
    explain_plan_cell
    return (explain_plan_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The smaller, sound claims of [Critical reading: claims versus evidence](/course/open/res-critical-reading?lang=en), costed with [Estimating training compute: the 6ND rule](/course/open/sys-compute-6nd?lang=en). The next lesson, *Guided reproduction 1: read the paper and pick the claim*, does this for the grokking paper, together with you.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_falsifiable, explain_plan_cell, result_first):
    L.wrap_up(result_first, choice_falsifiable, explain_plan_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
