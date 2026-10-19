# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-claim. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-claim", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en) and [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cost():
    quick_cost = L.choice(
        "quick-cost",
        question="You know the FLOPs of one run. What do you multiply them by, for the cost of a whole reproduction?",
        options={
            "params": "The number of parameters, once more",
            "right": "The runs and the seeds, and a margin for runs that go wrong",
            "runs": "Only the number of runs",
        },
        correct="right",
        why={
            "params": "6ND counts the parameters already: one run's FLOPs include them. See [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en).",
            "runs": "Every run needs several seeds, and some runs go wrong. See [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en).",
            "right": "For the grokking paper, the estimate of the previous lesson came to about 12 hours on a laptop: three seeds, with a margin of 2.",
        },
    )
    quick_cost
    return (quick_cost,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="A model gets 100% of its training examples right and 2% of its validation examples. What is going on?",
        options={
            "right": "It overfits: it has learned the training examples by heart",
            "under": "It underfits: it is too small to learn the task",
            "bug": "The data is broken: the validation set must have the wrong labels",
        },
        correct="right",
        why={
            "under": "A model that is too small gets the training examples wrong too. See [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "bug": "That can happen, but a perfect training score with a poor validation score is the usual sign of overfitting. See [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "right": "The paper of this lesson starts there: what if you keep training anyway?",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        From here on, you reproduce a paper, the whole way: *Grokking* by Alethea Power and others (2022). A small transformer learns a table of sums, such as all $a + b$ modulo 97. It soon gets every training sum right and the other sums wrong: it overfits. What happens when you keep training anyway is the paper's claim, and the claim you will test. Everyone does this paper together, in seven steps, and each step adds a part to your own reproduction kit, which you take to a paper of your own choice afterwards. This first step: read the paper, choose the one claim you test, and decide before any run when your reproduction has failed.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           0, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_later():
    guess_later = L.guess(
        "after-overfit",
        question="A small transformer gets all its training sums right, and almost none of the others. You keep training it for much longer, with weight decay. What do you think happens to its validation accuracy?",
        options={
            "stays": "It stays low: the model has learned the training sums by heart",
            "falls": "It falls further: the longer you train, the more it overfits",
            "right": "It rises to almost 100%: late, and then fast",
        },
        reveal="**It rises to almost 100%: late, and then fast.** That is what the paper calls grokking: generalization long after overfitting. It is the claim you will test, and the paper reports it for some settings and not for others, so a reproduction has to say which setting it tests.",
    )
    guess_later
    return (guess_later,)


@app.cell(hide_code=True)
def text_reading():
    mo.md(r"""
    ## Reading guide

    Download the paper from arXiv: *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177. Read it with the first two passes of [Reading a paper in three passes](/course/open/res-three-pass?lang=en), about an hour in all, with these questions; the third pass, every setting of the model, belongs to the next step, when you build it:

    1. **The abstract and Figure 1:** what is grokking, and on which task and fraction of data does Figure 1 show it?
    2. **How the data is made:** which equations, which numbers, and how they are split into training and validation.
    3. **The other figures:** what changes when the fraction of training data changes, and what does weight decay do? Note them for later steps; this step tests one claim.

    The paper's notation next to the course's and the code's:

    | Paper | Course | Code |
    |---|---|---|
    | $a \circ b = c$ | an equation of the table, with $\circ$ an operation such as $+$ modulo $p$ | `a`, `b`, `c` |
    | $p = 97$ | the numbers 0 to 96 | `p = 97` |
    | training data fraction | the share of all equations used for training | `train_fraction` |
    | optimization steps | training steps | `steps` |
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: one claim.** A paper makes several claims. For a reproduction you choose one, the core one, that one figure shows and your budget allows: for grokking, that validation accuracy reaches near 100% long after training accuracy did. The other claims, about the fraction of data and about weight decay, come back in the ablation.

        **Step 2: measurable.** Words such as "long after" and "near 100%" become numbers: an accuracy counts as reached at 99%, and "long after" means at a step at least 10 times as late. Then one run gives two step numbers, $t$ for training and $v$ for validation, and the claim holds for that run when $v \geq 10\,t$.

        **Step 3: every outcome named.** A run can go four ways: training never reaches 99% (*did not fit*), validation never does (*not reached*), validation does but not 10 times later (*no delay*), or it does (*grokked*). Each says something else about the reproduction.

        **Step 4: decided before any run.** Write the thresholds, the budget of steps and the number of seeds down first, in code, and keep them. Once you have seen curves, it is tempting to call any rise "late enough".

        The picture: a sketch of the claim, not the paper's data. Training accuracy, solid, reaches 99% at step $t$; validation accuracy, dashed, at step $v$, more than 10 times later.
        """),
        mo.Html(C.sketch_picture({"title": "The claim, sketched", "x": "training step", "y": "accuracy", "train": "training", "val": "validation",
                                  "alt": "A sketch on a log axis of training steps from 10 to a million. The training accuracy, a solid line, rises from about 0 to 100% between step 100 and 1,000 and reaches 99% at step t, about 1,000. The validation accuracy, a dashed line, stays near 0 and rises late, reaching 99% at step v, about 170,000. A thin line marks 99%."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **grokking** | generalization long after overfitting: validation accuracy rises late, after training accuracy has long been near 100% |
        | **reproduction plan** | the claim you test, how you measure it, and when it has failed, written down before any run |
        | $t$, $v$ | the first step at which the training accuracy, and the validation accuracy, reaches 99% |
        """),
        L.watch_out("A failed reproduction is a result, but only with a plan made beforehand. Without one, every outcome can be read as \"almost\", and nobody, you included, can tell what you found."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To write a reproduction plan:

    1. Choose the one claim you test, with the figure it rests on.
    2. Turn its words into numbers: thresholds, a budget of steps, a number of seeds.
    3. Name every outcome a run can have, and what each says about the claim.
    4. Put it in code before any run, and keep it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    New is reading a log for the first step at which something happens: the training accuracy of a made-up run, logged at four steps.
    """)
    return


@app.cell
def example_first():
    _steps = [50, 500, 5000, 50000]
    _train = [0.4, 0.98, 0.99, 1.0]
    for _i in range(len(_steps)):
        if _train[_i] >= 0.99:
            print("training reaches 99% at step", _steps[_i])
            break
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The loop goes through the log in order and stops at the first step where the accuracy reaches 0.99: step 5,000, where it is exactly 0.99. `>=` makes that count, and `break` stops at the first one, so a later step never replaces it. A log is only as fine as its steps: the run may have reached 99% anywhere between step 501 and 5,000.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The claim to test
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "which-claim",
        question="Three claims you could take from the grokking paper. Which one does this reproduction test first?",
        options={
            "fraction": "The smaller the fraction of training data, the more steps it takes to generalize",
            "right": "On one task and fraction of data, validation accuracy reaches near 100% long after training accuracy did",
            "understand": "Neural networks can come to understand arithmetic",
        },
        correct="right",
        why={
            "fraction": "That needs many runs at many fractions: it is a second claim, for the ablation, once the first one holds.",
            "understand": "\"Understand\" names no result you could measure; the paper's figures measure accuracy over steps.",
            "right": "One figure, one setting, two step numbers per run: small enough to test, and the core of the paper.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your plan, in code

        Complete `first_step(steps, accuracy, threshold)`, the first step at which the accuracy reaches the threshold, or `None`, and `verdict(steps, train_acc, val_acc)`, which gives back `"did not fit"`, `"not reached"`, `"no delay"` or `"grokked"` for one run, as in Step 3, with `THRESHOLD` and `DELAY`. The check calls `verdict([10, 100, 1000, 10000, 100000], [0.3, 0.95, 1.0, 1.0, 1.0], [0.0, 0.01, 0.02, 0.4, 1.0])`, which should give `"grokked"`, and tries every other kind of run. When it is right, it is saved in your project as `repro/plan.py`, the first part of your reproduction kit.
        """),
        L.task("Complete `first_step` and `verdict`: the four outcomes of Step 3, with the thresholds at the top of the plan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=20)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "`first_step` is the worked example as a function, with `return` instead of `print` and `break`, and `None` after the loop. `verdict` finds `t` and `v` with it, then checks training, validation and the delay, in that order.",
        ],
        why="`first_step`: `for i in range(len(steps)):` with `if accuracy[i] >= threshold: return steps[i]`, and `return None` after the loop. `verdict`: `t = first_step(steps, train_acc, THRESHOLD)` and `v` the same for `val_acc`; `\"did not fit\"` when `t is None`, `\"not reached\"` when `v is None`, `\"no delay\"` when `v < DELAY * t`, else `\"grokked\"`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your `verdict`, from your project, on three made-up runs that all reach 99% on their training data at step 300. In run A the validation accuracy reaches 99% at step 30,000, in run B at step 1,000, soon after training, and in run C it hardly rises.
    """)
    return


@app.cell(hide_code=True)
def demo_verdict(result_verdict):
    result_verdict  # run again when exercise 2 saves your plan
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercise 2 first: then your plan is in your project, and this cell uses it.*")
    elif _result == "error":
        _out = mo.md("*Your `repro/plan.py` stops with an error. Did you change it after the check? Delete it, and check exercise 2 again.*")
    else:
        _out = mo.md("\n".join(f"- **Run {_name}:** `{_v}`" for _name, _v in zip("ABC", _result)))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_before():
    mo.md(r"""
    ### Exercise 3. Why before
    """)
    return


@app.cell(hide_code=True)
def explain_before():
    explain_before_cell = L.explain(
        "why-before",
        question="Why do you write down when the reproduction has failed before the first run, and not after you have seen the curves?",
        model_answer="**Curves invite stories:** once you see a curve, almost any rise can be called \"late enough\" and any plateau \"near 100%\". Choosing the thresholds afterwards fits them to what you got, and then the reproduction can no longer fail. **A result either way:** with the plan fixed beforehand, a run that is not grokked is a finding too, and someone else can check your verdict with the same rule. **The same rule for every run:** all seeds, and later the ablation, are judged the same way, so differences between runs come from the runs, not from how you read them.",
        points=[
            "you say that after seeing the curves, you are tempted to fit the thresholds to the results, so the claim can no longer fail",
            "you say that a fixed plan makes a failed reproduction a result that others can check",
            "you say that every run and seed is judged by the same rule",
        ],
    )
    explain_before_cell
    return (explain_before_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The choice of [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en), made for one paper, and the overfitting of [Overfitting and underfitting](/course/open/ml-overfitting?lang=en), with a twist. The next lesson, *Guided reproduction 2: build the model and the data*, makes the table of sums and adapts your GPT to the paper, setting by setting; later steps run it and judge every run with your `verdict`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, explain_before_cell, result_verdict):
    L.wrap_up(choice_claim, result_verdict, explain_before_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
