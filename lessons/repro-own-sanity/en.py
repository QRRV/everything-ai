# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-sanity. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-sanity", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 3: build it](/course/open/repro-own-build?lang=en) and [A reusable training loop](/course/open/nn-trainer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_same():
    quick_same = L.choice(
        "quick-same",
        question="Two cells each call `make_model(\"claim\", 0)` from your experiment of Your reproduction 3 and measure the first loss on `one_batch()`. Both get 2.335. Why the same number?",
        options={
            "right": "`make_model` starts with `torch.manual_seed(0)`, so what is random after it comes out the same",
            "only-k": "Before any training, the first loss depends only on K, and K is 10 in both cells",
            "kept": "Python keeps the model of the first call in memory, and the second call gets that same model back",
        },
        correct="right",
        why={
            "only-k": "It is near ln K, 2.30, but not on it: the random starting weights, and dropout, make it 2.335. Seed 1 gives 2.322. See [Your reproduction 3](/course/open/repro-own-build?lang=en).",
            "kept": "Each call builds a new model. It is the same model because its random starting weights come from the same seed. See [Your reproduction 3](/course/open/repro-own-build?lang=en).",
            "right": "That is what the seed is for: the starting weights, and the dropout after them, come from the same random numbers each time.",
        },
    )
    quick_same
    return (quick_same,)


@app.cell(hide_code=True)
def quick_check_reshape():
    quick_reshape = L.choice(
        "quick-reshape",
        question="In A reusable training loop, the loss line is `F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))`. The logits of a model of text have the shape (8, 16, 50): 8 texts of 16 places, and 50 tokens. What shape goes into the loss?",
        options={
            "rows": "(8, 800): one row for each text, with all its places side by side",
            "right": "(128, 50): one row of 50 for each of the 8 × 16 places",
            "same": "(8, 16, 50): `reshape(-1, ...)` keeps the shape as it is",
        },
        correct="right",
        why={
            "rows": "That would be `reshape(len(logits), -1)`. Here the -1 comes first: as many rows as it takes, each of `logits.shape[-1]` = 50 numbers. See [A reusable training loop](/course/open/nn-trainer?lang=en).",
            "same": "`reshape` keeps the numbers, not the shape: the -1 means as many rows as it takes, so 8 × 16 = 128 rows of 50. See [A reusable training loop](/course/open/nn-trainer?lang=en).",
            "right": "Each place of each text gets its own row of 50 logits, and `yb.reshape(-1)` gives its 128 targets in the same order.",
        },
    )
    quick_reshape
    return (quick_reshape,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        The main result runs your experiment for hours, over every seed of your plan. A bug in the model or the training would cost you all of them, and you would see it only at the end, as a result that is a little off, or not at all. The sanity checks of your kit take a minute and look at the three things such a bug spoils first: what the model gives back, what it knows before it has learned anything, and whether it can learn at all. In Guided reproduction 3 they were written for the grokking model; here they work for any model, and you point them at your own experiment.

        **An everyday example.** Before a long drive, you walk around the car: tyres, oil, lights. Five minutes cannot tell you whether the trip goes well, but they catch what would end it on the motorway.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           3, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_trained():
    guess_trained = L.guess(
        "trained",
        question="A language model that was trained before, with a vocabulary of 151,936 tokens, has a first loss of 2.9 on your batch: far below ln 151,936, which is about 11.9. What does the sanity check say about its first loss?",
        options={
            "fails": "It fails: a first loss of 2.9 is far from ln K, so something is wrong",
            "leak": "It warns that the answer leaks into what the model sees",
            "right": "Nothing: it compares with ln K only for a model from scratch",
        },
        reveal="**Nothing: it compares with ln K only for a model from scratch.** ln K is the loss of a model that knows nothing yet. A model that was trained before knows more than chance, and that is what its training was for. For it, the shape of the logits and learning one batch still count.",
    )
    guess_trained
    return (guess_trained,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Three checks, any model.** The checks of Guided reproduction 3 work for any model whose logits for a batch `x` have the shape of its labels `y` and then K numbers. A classifier gives one row of K for each example; a model of text gives one row of K for each place, with a label for each place. Put one under the other with `reshape(-1, K)`, both get their loss the same way.

    **The shape.** The logits must have the shape of `y` and then K. When they do not, the other two checks make no sense, so the check stops there.

    **The first loss.** Before any training, a model from scratch gives every answer about the same chance, so its loss is about ln K. Much higher means starting weights that are too large; much lower means that the answer leaks into what the model sees. A model that was trained before, loaded from a checkpoint, starts lower, as it should: for it, this check says nothing.

    **One batch by heart.** 200 steps of AdamW at a learning rate of `1e-3` on a few dozen examples must bring the loss below 0.05. A model that cannot learn so little by heart will not learn the data either: a part may be frozen, cut off from the loss, or give an output that does not depend on its weights. Weight decay goes, as in Guided reproduction 3: it is part of the training, and the check trains in a way of its own. Dropout stays on: in `"claim"` it is part of the model the plan tests, and a big enough network learns one batch even with it. One batch cannot be learned by heart when the same input comes in it with different labels, as for a model of text that sees only the token before: choose a batch in which each input tells its label apart.

    **Your experiment's parts.** The check takes them from your experiment of Your reproduction 3: `make_model(setting, seed)`, `one_batch()`, `K` and `FROM_SCRATCH`, in both settings.

    ### New words and symbols

    | Symbol | What it means |
    |---|---|
    | ln K | the loss of a model that gives each of K answers the same chance, 1/K |

    The picture: the first loss of the course's experiment for the dropout paper, before any training, on its batch of 32 pictures, in setting `"claim"` with its dropout on, as the check measures it, with every starting weight 1 to 3 times as large as PyTorch makes it. The band is ln 10 plus or minus 0.5, where the check passes. Move the slider.
    """)
    return


@app.cell(hide_code=True)
def explore_first_cell():
    _labels = {"title": "First loss, by the size of the starting weights", "x": "starting weights, times as large", "y": "first loss",
               "band": "ln 10 ± 0.5",
               "alt": "The first loss of the course's model for the dropout paper against how many times as large its starting weights are, from 1 to 3. At 1 it is 2.34, inside a band from 1.80 to 2.80 around ln 10; at 1.75 it is 2.59; at 2 it is 2.81, just above the band; at 3 it is 5.27."}

    def _frame(s):
        _f = C.first_loss(s)
        _say = "so the check passes" if C.first_ok(s) else "so the check fails"
        return C.first_picture(s, _labels), f"Starting weights {C.number(s, 2, L.lang)} times as large: first loss {C.number(_f, 2, L.lang)} against ln 10 = {C.number(math.log(10), 2, L.lang)}, {_say}."

    explore_first = L.explore("scale", "scale", _frame, start=1, low=1, high=3, step=0.25, role="",
                              task="From which size on does the check fail? And why does the loss stay near ln 10 at first, and then shoot up?")
    explore_first
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Up to 1.75 times as large, the first loss stays within half a point of ln 10; at 2 times it is 2.81, just over, and the check fails; at 3 times it is 5.27. The loss stays near ln 10 at first and then shoots up, because the logits grow much faster than the weights: each of the three layers multiplies by the scale again, so 3 times the weights gives more than 20 times the logits. While they are small, every kind still gets about 1/10; once they are large, the model is sure, and mostly wrong. Dropout adds to it: it is on, as in training, and spreads the logits at the start further. Without it, as in `"other"`, 3 times is still 2.95. Starting weights that are too large come from an initialization of your own, or a scale copied wrong from a paper. A first loss that is much too low is the other warning: a model that knows the answer before it has learned anything sees it somewhere.
        """),
        L.watch_out("These checks look at the model and one batch, with a training of their own: not at the training in your `run`, nor at what you measure. A bug in that training, a test set that leaks into the training data, a wrong metric, or dropout left on while the test runs pass all three. Guard those in `run` and in your plan."),
        mo.md(r"""
        ### Recipe

        To put a model through the sanity checks:

        1. Take one small batch, a few dozen examples, and the model before any training.
        2. Check that its logits have the shape of the labels and then K.
        3. For a model from scratch, check that the first loss is about ln K.
        4. Learn the batch by heart: 200 steps of AdamW at `1e-3`, and the loss must go below 0.05.
        5. Do it in both settings, before any long run.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 by hand: logits that give every answer the same score, for 4 pictures and 10 kinds of clothing, and for 2 texts of 3 places. `reshape(-1, 10)` is the one of [A reusable training loop](/course/open/nn-trainer?lang=en): rows of 10, as many as it takes. Run the cell.
    """)
    return


@app.cell
def example_ln():
    # 4 pictures: the same score for each of 10 kinds
    _logits = torch.zeros(4, 10)
    _labels = torch.tensor([3, 0, 7, 9])
    _loss = F.cross_entropy(_logits, _labels).item()
    _ln = round(math.log(10), 4)
    print("pictures:", round(_loss, 4), "ln 10:", _ln)
    # 2 texts of 3 places: a row of 10 for each place
    _text = torch.zeros(2, 3, 10)
    _next = torch.tensor([[1, 4, 2], [0, 0, 9]])
    _rows = _text.reshape(-1, 10)   # 6 rows of 10
    _labels = _next.reshape(-1)     # and 6 labels
    _loss = F.cross_entropy(_rows, _labels).item()
    print("text:", tuple(_rows.shape), round(_loss, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Equal scores give each kind 1/10, so the loss is -ln(1/10) = ln 10 = 2.3026, whatever the labels are. `reshape(-1, 10)` makes the 2 times 3 places into 6 rows of 10, and `reshape(-1)` the labels into 6; then the loss of a model of text is computed just like that of a classifier, and it is ln 10 again.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sanity():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The sanity checks, for any model

        Complete `sanity(model, x, y, k, steps, from_scratch)`, after the comments. It gives back a report: `"shape"`, `True` when the logits of `model(x)` have the shape of `y` and then `k`; `"first"`, for a model from scratch whether the first loss is within 0.5 of ln k, and `None` for one that was trained before; `"learns"`, whether `steps` steps of AdamW at `1e-3` bring the loss on the batch below 0.05; and the first and the last loss; the first from the same logits as the shape check, with the model's dropout on. When the shape is wrong, it gives back `False` and four times `None`. The check tries a model that works, a wrong `k`, starting weights that are too large, a frozen model, a trained one, a model of text and one with dropout.
        """),
        L.task("Complete `sanity`: the shape first, then the first loss, then the batch by heart."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sanity():
    box_sanity = L.code_box("sanity", lines=20)
    box_sanity
    return (box_sanity,)


@app.cell(hide_code=True)
def check_sanity(box_sanity):
    result_sanity = L.check("sanity", box_sanity.value)
    result_sanity
    return (result_sanity,)


@app.cell(hide_code=True)
def hints_sanity():
    L.hints(
        "sanity",
        [
            "The shape: compare `tuple(logits.shape)` with `tuple(y.shape) + (k,)`, and return the report of `False` and four times `None` when they differ.",
            "The first loss over every row: `F.cross_entropy(logits.reshape(-1, k), y.reshape(-1)).item()`, before the loop.",
            "The loop of `overfit_one_batch` of Guided reproduction 3, with the same `reshape` in its loss, and the optimizer of the comment: AdamW at `1e-3` with `weight_decay=0.0`, without the paper's betas.",
        ],
        why="The shape comes first because the losses need it; the first loss before any step; and the loop at `1e-3` tests whether the model can learn at all.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run():
    mo.md(r"""
    ### Exercise 2. Your experiment

    Press the button: the check reads your experiment from `work/projects/own/experiment.py`, and puts its parts through the course's sanity checks in both settings, seed 0, with 200 steps for the batch. When one fails, its message names the setting and what to look at. Fix it in exercise 2 of [Your reproduction 3](/course/open/repro-own-build?lang=en), where your experiment is, and press again.
    """)
    return


@app.cell(hide_code=True)
def button_run():
    run_sanity = mo.ui.run_button(label="Put my experiment through the sanity checks")
    run_sanity
    return (run_sanity,)


@app.cell(hide_code=True)
def check_run(run_sanity):
    result_run = L.check("sanity-run", C.experiment_text() if run_sanity.value else "")
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "sanity-run",
        [
            "Each message names the check that fails and the setting. Look in `make_model` for the shape and the first loss, and, for learning the batch, at which of its parameters can learn.",
            "Try your parts with a few lines of your own first: `model = make_model(\"claim\", 0)`, `x, y = one_batch()`, then `model(x).shape` and `y.shape`.",
        ],
        why="This is the course's experiment for the dropout paper: its parts pass all three checks in both settings.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your own sanity check, `own/sanity.py`, on your own experiment, both settings: the report it gives. It runs when you press the button above.
    """)
    return


@app.cell(hide_code=True)
def demo_sanity(run_sanity, result_sanity, result_run):
    result_sanity, result_run  # run again when exercise 1 saves your check
    _out = mo.md("*Press the button above: then this cell runs your sanity check on your experiment.*")
    if run_sanity.value and result_run.state == "slow":
        _out = mo.md("*The check above ran out of time, so the demo does not run your sanity check as well: it would take as long. First make the check finish in time.*")
    elif run_sanity.value:
        _r = C.demo()
        if _r is None:
            _out = mo.md("*Finish exercise 1 here and exercise 2 of Your reproduction 3 first: then both files are in your kit, and this cell runs them.*")
        elif _r == "slow":
            _out = mo.md("*Your sanity check takes more than a minute for one setting, so the demo stops. Is your batch a few dozen examples?*")
        elif _r == "error":
            _out = mo.md("*Your `own/sanity.py` or `own/experiment.py` stops with an error. Did you change one of them after its check? Then check it again.*")
        else:
            _out = mo.Html(C.demo_table(_r, {"True": "yes", "False": "no", "None": "–"}, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. What the checks catch

    Your experiment passes the sanity checks. Say what each of them would have caught, and what none of them can.
    """)
    return


@app.cell(hide_code=True)
def explain_why_sanity():
    explain_why_sanity_cell = L.explain(
        "why-sanity",
        question="Which bug would each of the three sanity checks have caught in your experiment? And which bug can none of them catch?",
        model_answer="**The shape** catches a model with another number of outputs than K, such as a last layer of 5 for 10 kinds of clothing, or labels of another shape. **The first loss** catches starting weights that are too large, such as a scale I copied wrong, and an answer that leaks into the input, which makes it far too low. **One batch by heart** catches a model that cannot learn: a part I froze, a part computed under `torch.no_grad()`, cut off from the loss, an output that does not depend on the weights. **None of them** catches a bug in the training of my `run`, since the check trains with an AdamW of its own, nor what I do after training: dropout left on while the test runs, a metric that counts the wrong thing, or test pictures that are also in the training data. Those I guard in `run`, and by comparing with the paper.",
        points=[
            "you name a bug the shape check catches",
            "you name a bug the first loss catches, too high or too low",
            "you name a bug that keeps a model from learning one batch",
            "you name a bug none of the three catches, in the training of `run`, the test or the metric",
        ],
    )
    explain_why_sanity_cell
    return (explain_why_sanity_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The sanity checks of [Guided reproduction 3](/course/open/repro-guided-sanity?lang=en), now for any model; the parts of your experiment from [Your reproduction 3](/course/open/repro-own-build?lang=en); and the starting weights of [Initialization](/course/open/nn-initialization?lang=en). In the next lesson, *Your reproduction 5: the main result*, you run your experiment at the budget of your plan, over its seeds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_sanity, result_run, explain_why_sanity_cell):
    L.wrap_up(result_sanity, result_run, explain_why_sanity_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
