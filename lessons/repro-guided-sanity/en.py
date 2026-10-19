# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-sanity. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-sanity", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Guided reproduction 2: build the model and the data](/course/open/repro-guided-build?lang=en) and [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chance():
    quick_chance = L.choice(
        "quick-chance",
        question="Before any training, the loss of your ModernGPT on the answers was close to ln 99 = 4.60. Why that number?",
        options={
            "trained": "A model of this size reaches about 4.60 on modular arithmetic, before and after training",
            "batch": "The batch has 512 equations, and 4.60 is about ln 99, the number of answers in it",
            "right": "It gives each of the 99 tokens about the same probability, 1/99, and −ln(1/99) = ln 99",
        },
        correct="right",
        why={
            "trained": "Training brings the loss far below it: a model that has learned the answers gives them a probability close to 1, and −ln 1 = 0. See [Guided reproduction 2](/course/open/repro-guided-build?lang=en).",
            "batch": "The loss does not depend on the size of the batch: it is a mean over the equations. The 99 is the number of tokens the model chooses among. See [Guided reproduction 2](/course/open/repro-guided-build?lang=en).",
            "right": "A model that has learned nothing guesses evenly among its 99 tokens; the cross-entropy of the right one is then −ln(1/99).",
        },
    )
    quick_chance
    return (quick_chance,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Your network learns one mini-batch of 64 images by heart: after 300 steps, its loss on them is 0.03. What does that tell you?",
        options={
            "solved": "It will do well on all the data too: it has learned the task",
            "right": "The network and its training step can learn, and no more than that",
            "wrong": "Something is wrong: a network should not learn 64 images by heart",
        },
        correct="right",
        why={
            "solved": "Passing the test does not mean it will do well on all the data: a network that learned 64 images by heart can read new images badly. See [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
            "wrong": "Learning one batch by heart is what the test asks: a network that works can do it, and this one shows that it can. See [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
            "right": "The overfit test is a test, not a way to train: it rules out a broken network, start or training step, and nothing more.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(rf"""
        ## Why this matters

        Your reproduction trains for a long time. In the course's run at the fraction of your plan, validation reached 99% only at step {C.number(C.RUN_STEPS, 0, L.lang)}: at {C.number(C.SECONDS_PER_STEP, 2, L.lang)} seconds a step on the course's CPU, about {round(C.RUN_STEPS * C.SECONDS_PER_STEP / 60)} minutes, and the paper's budget of {C.number(C.BUDGET, 0, L.lang)} steps is more than {int(C.BUDGET * C.SECONDS_PER_STEP / 3600)} hours. A bug you find after that run cost you the run, and a bug you never find costs you the reproduction: a curve that does not jump says nothing about the paper if your model could not have learned anyway. So before any long run, a few quick tests whose outcome you know in advance. Together they take about a minute.

        **An everyday example.** Before a long drive, you check the fuel and the tires. It takes a minute, against being stuck halfway, and you know what you should see before you look.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           2, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_place():
    guess_place = L.guess(
        "place",
        question="Your ModernGPT has a bug: it reads its guess for $c$ at the place of $a$, `model(x)[:, 0]`, instead of at the place of $=$. Which of three quick tests do you think catches it?",
        options={
            "shape": "The shape of the logits: it is not what the settings give",
            "first": "The loss before training: it is far from ln 99",
            "right": "Learning one batch by heart: the loss does not go to almost 0",
        },
        reveal="**Learning one batch by heart.** The logits have the same shape either way, and before any training every place guesses about evenly, so the first loss is about ln 99 at either place. What differs is what each place has seen; *The idea* shows what that does.",
    )
    guess_place
    return (guess_place,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **sanity check** is a quick test that should obviously pass before you trust a long run. Each of the three below has a number that the setup tells you in advance, so you know what you should see before you look.

    1. **The shapes.** One batch of 512 equations of 4 tokens goes in: 512 × 4. Out come the logits: 512 × 4 × 99, a row of $p + 2$ scores at each place. Another number here means a model built with other settings than you think.
    2. **The loss before training.** A model that has learned nothing guesses about evenly among its 99 tokens, so its loss on the answers is about ln 99 = 4.60. Much higher: it starts out sure and wrong, such as with weights that start too large. Much lower: it knows something it cannot know yet, such as an answer that leaks into its input.
    3. **Learning one batch by heart.** Train on that one batch alone, again and again, without weight decay: the test asks only whether the model can learn, and weight decay pulls the other way. A model that works learns 512 equations in a few hundred steps, and its loss goes to almost 0. If it does not, the model or its training step is broken.

    Below are two runs of the course's ModernGPT with the paper's settings, on the first 512 training equations of your data, with the paper's AdamW without weight decay. One of them has the bug of the guess: it reads its guess at the place of $a$. Move the slider, or drag across the chart, to set the number of steps: the chart shows the loss of both on the batch for every step, the run with the bug dashed, and the grey dashed line is ln 99.
    """)
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    def _frame(step):
        right, wrong = C.RUNS["right"][step], C.RUNS["wrong"][step]
        return (C.picture(step, {"title": "One batch, learned by heart", "x": "step", "y": "loss on the batch", "even": "guessing evenly: ln 99",
                                 "right": "reads at =", "wrong": "reads at a: the bug",
                                 "alt": f"The loss on one batch of 512 equations over 150 steps, for two ModernGPTs. The one that reads at the place of = falls from 4.76 to 0.0041; the one that reads at the place of a falls from 4.79 too, faster at first, but stays near 1.78. A dashed line marks ln 99 = 4.60. After {step} steps: {C.number(right, 4, L.lang)} and {C.number(wrong, 4, L.lang)}."}),
                f"""After {step} steps, the loss of the model that reads at = is <span class="c-loss ea-num">{C.number(right, 4, L.lang)}</span>, and that of the model that reads at a <span class="c-loss ea-num">{C.number(wrong, 4, L.lang)}</span>.""")

    explore_steps = L.explore("steps", "step", _frame, start=20, low=0, high=150, step=5, role="", task="From which step is the loss of the model that reads at = below 0.01? Where does the other one stop?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(rf"""
    From step 90 the model that reads at $=$ is below 0.01, at 0.0098: it has learned the 512 equations by heart. The one that reads at $a$ falls even faster at first, but stops at about 1.78. Its shapes were right, and its first loss, 4.79, was close to ln 99, as was that of the other: only learning one batch by heart caught it.

    **Where a check stops often says why.** At the place of $a$, the model has seen only $a$. In the batch, most values of $a$ come with several different $b$, so with several different answers, and the model cannot tell them apart. If an $a$ comes $k$ times, the best it can do is give each of those $k$ answers $1/k$, a loss of $\ln k$ for each of those equations. Averaged over the batch:
    """)
    return


@app.cell(hide_code=True)
def text_idea_formula():
    mo.vstack([
        L.formula(r"[[loss:\mathcal{L}_{\min}]] = \frac{1}{n} \sum_{a} k_a \ln k_a"),
        mo.md(rf"""
        with $k_a$ the number of equations with that $a$ among the $n$ of the batch. For the first 512 training equations it is {C.number(C.floor(), 2, L.lang)}: the model with the bug stops just above the lowest loss it can reach. So a loss that stops at a number you can work out from what the model sees points at what it cannot see.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **sanity check** | a quick test that should obviously pass before you trust a long run, with an outcome you know in advance | `sanity(model, x, y, vocab, steps)` |
        | loss before training | the loss of a model that has learned nothing: about ln of the number of tokens it chooses among | `math.log(vocab)` |
        | $k_a$ | the number of equations in the batch with this $a$ | `counts[a]` |
        | $\mathcal{{L}}_{{\min}}$ | the lowest loss a model can reach with what it sees | `best_loss(rows)` |
        """),
        L.watch_out("A check that passes is no proof. If the answers in your data were wrong, say (a · b) mod p instead of (a + b) mod p, all three checks would pass: a model can learn any batch by heart, also one with wrong answers. Sanity checks catch a broken model and a broken training step; whether the experiment is the paper's, the settings tell you, each with its place in the paper."),
        mo.md(r"""
        ### Recipe

        To check a model before a long run:

        1. Take one batch of the training data, as big as those of the real run.
        2. Compare the shapes of the batch and of the logits with what the settings give.
        3. Compare the loss before training with ln of the number of tokens.
        4. Train on that batch alone, without weight decay, for a few hundred steps: the loss should go to almost 0. If it stops higher, work out what the model can see, and the lowest loss that allows.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The numbers to expect for a small batch of 4 equations of addition modulo 5, before you run anything. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_numbers():
    import math

    _p = 5
    _rows = [(1, 2, 3), (1, 4, 0), (2, 2, 4), (1, 0, 1)]   # 1. one batch: four equations (a, b, c)
    # 2. the shapes: 4 equations of 4 tokens in, and p + 2 scores at each place out
    print("x:", (len(_rows), 4), "logits:", (len(_rows), 4, _p + 2))
    # 3. the loss before training: guessing evenly among the p + 2 tokens
    print("first loss: about", round(math.log(_p + 2), 2))
    # 4. a model that sees only a: a = 1 comes 3 times, a = 2 once
    print("lowest loss when it sees only a:", round((3 * math.log(3) + 1 * math.log(1)) / 4, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `x` is 4 × 4 and the logits 4 × 4 × 7, and the first loss should be about ln 7 = 1.95. Learning the batch by heart should take the loss to almost 0; a model that sees only $a$ stops at 0.824 at best. The three equations with $a = 1$ have three different answers, 3, 0 and 1, so each costs ln 3 at best, and the one with $a = 2$ costs ln 1 = 0: $(3 \ln 3 + 1 \ln 1) / 4$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Where it stops

        Complete `best_loss(rows)`: the lowest loss a model can reach on the equations `rows`, each `(a, b, c)`, when it sees only `a`. Count how often each `a` comes, as in the worked example, and add up $k_a \ln k_a$ over every `a`, divided by the number of equations. In this data every pair of `a` and `b` comes once, so the equations of one `a` have different answers. The check calls `best_loss([(3, 0, 3), (3, 1, 4), (0, 2, 2), (0, 0, 0), (3, 4, 2)])`, and other batches.
        """),
        L.task("Complete `best_loss`: count the equations of each `a`, then the mean of $k_a \\ln k_a$."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-loss", lines=11)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-loss", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-loss",
        [
            "Counting per `a` is counting with a dictionary: when `a` is a key already, its count goes up by 1; when not, it starts at 1.",
            "In the worked example, `a = 1` came 3 times, and its equations cost `3 * math.log(3)` together. Do the same for every `a` in `counts`, with its own count.",
        ],
        why="An `a` that comes once costs ln 1 = 0: the model can learn its one answer. Only an `a` with more answers than one keeps the loss above 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_overfit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Learn one batch by heart

        Complete `overfit_one_batch(model, x, y, steps, lr)` for a GPT: `steps` steps of AdamW, every step on the same `x` and `y`, with the loss on the answers only, at the last place, that of $=$. It keeps the loss of every step, measured before its update, in `losses`. The optimizer is given: the paper's betas, without weight decay. The check trains small models on made-up batches and compares the losses and the model afterwards. When it is right, it is saved in your project as `repro/overfit.py`.
        """),
        L.task("Complete `overfit_one_batch`: the loss at the last place, kept, and a step of AdamW, `steps` times."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overfit():
    box_overfit = L.code_box("overfit", lines=13)
    box_overfit
    return (box_overfit,)


@app.cell(hide_code=True)
def check_overfit(box_overfit):
    result_overfit = L.check("overfit", box_overfit.value)
    result_overfit
    return (result_overfit,)


@app.cell(hide_code=True)
def hints_overfit():
    L.hints(
        "overfit",
        [
            "`model(x)` gives one row of logits for every place; the answer `c` is guessed at the last place. How do you take only those rows?",
            "`losses` keeps numbers, not tensors: which method of a tensor gives the number in it? Then come the three lines of every training step, in their order.",
        ],
        why="With the loss at the last place only, the test trains exactly what the real run trains; a loss over every place would ask the model to guess `c` where it has not seen `b` yet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bug():
    mo.md(r"""
    ### Exercise 3. Which bug?
    """)
    return


@app.cell(hide_code=True)
def choice_bug_cell():
    choice_bug = L.choice(
        "which-bug",
        question=f"You run the three checks on a model of a fellow learner, on the first 512 training equations. The shapes are right, the first loss is 4.81, and after {C.STEPS} steps of learning the batch by heart, the loss is 1.79 and hardly falls any more. `best_loss` of the batch gives {C.number(C.floor(), 2, L.lang)}. Where is the bug most likely?",
        options={
            "right": "The model guesses at a place that has not seen b",
            "lr": "The learning rate is too small: it needs more steps",
            "answers": "The answers in the data are wrong",
        },
        correct="right",
        why={
            "lr": f"With a learning rate that is too small, the loss falls slowly, but it goes on falling. This one stops, just above {C.number(C.floor(), 2, L.lang)}: the lowest loss of a model that sees only `a`.",
            "answers": "A model can learn any batch by heart, also one with wrong answers: the loss would still go to almost 0. See the watch-out in *The idea*.",
            "right": f"It stops just above {C.number(C.floor(), 2, L.lang)}, the lowest loss a model can reach that sees only `a`: it reads its guess at a place before `b`, such as that of `a`.",
        },
    )
    choice_bug
    return (choice_bug,)


@app.cell(hide_code=True)
def text_exercise_sanity():
    mo.vstack([
        mo.md(rf"""
        ### Exercise 4. The sanity check of your kit

        Complete `sanity(model, x, y, vocab, steps)`, the three checks in one function, for any model that takes a batch of tokens and gives a row of logits at each place. It gives back a report: `shape`, whether the logits are `len(x)` by the number of places by `vocab`; `first`, whether the loss before training is within {C.number(C.FIRST_TOLERANCE, 1, L.lang)} of ln `vocab`; and `learns`, whether after learning the batch by heart with your `overfit_one_batch` at the paper's learning rate, `1e-3`, the loss of the last step is below {C.number(C.LEARNED, 2, L.lang)}. The two losses come with it. The check runs it on small models, and on ones with a bug. When it is right, it is saved in your project as `repro/sanity.py`.
        """),
        L.task("Complete `sanity`: the shape, the first loss and the losses of learning the batch by heart."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sanity():
    box_sanity = L.code_box("sanity", lines=18)
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
            "`tuple(logits.shape)` turns the shape into a tuple, which you can compare with another tuple with `==`. Which three numbers should it have, from `x` and `vocab`?",
            "The first loss comes from the `logits` you already have, at the same place and in the same way as the loss in your `overfit_one_batch`. Then your `overfit_one_batch` trains, at the paper's learning rate.",
        ],
        why="Measure the first loss before learning the batch: afterwards the model has learned, and its loss says nothing about how it started.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### What your project can do now

    The button runs your `sanity`, from your project, twice, on the first batch of your training data: on your ModernGPT with your settings, and on the same model with the bug of the guess planted, so that it reads its guess at the place of $a$. Each learns the batch by heart for {C.STEPS} steps; together that takes about half a minute on a CPU. Where a part of your kit is missing or broken, the course's version stands in.
    """)
    return


@app.cell(hide_code=True)
def sanity_button_cell():
    sanity_button = mo.ui.run_button(label="Run your sanity check")
    sanity_button
    return (sanity_button,)


@app.cell(hide_code=True)
def demo_sanity(sanity_button, result_overfit, result_sanity):
    result_overfit, result_sanity  # run again when an exercise saves a part of the kit
    _ready = result_overfit.state == "correct" and result_sanity.state == "correct"
    if not sanity_button.value:
        _out = mo.md("*Press the button to run your sanity check.*" if _ready else "*Finish exercises 2 and 4 first, then press the button: it runs your `sanity` with your `overfit_one_batch`.*")
    else:
        with mo.status.spinner(title="Your sanity check is learning the batch by heart, twice: about half a minute"):
            _r = C.demo()
        if _r is None:
            _out = mo.md("*Finish exercises 2 and 4 first: then your `overfit_one_batch` and your `sanity` are in your project, and this cell uses them.*")
        elif isinstance(_r, str):
            _out = mo.md({"slow": "*Your sanity check takes more than three minutes here for one model. Does your `overfit_one_batch` or your `sanity` do more work than it needs, such as more steps than `steps`?*",
                          "sanity": "*Your `repro/sanity.py` or `repro/overfit.py` stops with an error. Did you change it after its check? Delete it, and check its exercise again.*",
                          "model": "*Your ModernGPT or your data stops with an error. Did you change `llm/modern_gpt.py`, `repro/data.py` or `repro/settings.py` after its check? Delete the file you changed, and check its exercise again.*"}[_r])
        else:
            _y, _b = _r["reports"]["yours"], _r["reports"]["planted"]
            _yes = lambda _v: "passes" if _v else "**fails**"
            _stand = (" The course's version stood in for: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
            _verdict = ("Your model passes all three checks, and your check catches the planted bug" if all(_y[_k] for _k in ("shape", "first", "learns")) and not _b["learns"]
                        else "Not every check came out as expected: look at the row that does not")
            _out = mo.md(f"""
| check | your ModernGPT | with the bug | what to expect |
|---|---|---|---|
| shape of the logits | {_yes(_y['shape'])} | {_yes(_b['shape'])} | {_r['batch']} × 4 × {_r['vocab']} |
| loss before training | {C.number(_y['first_loss'], 2, L.lang)}: {_yes(_y['first'])} | {C.number(_b['first_loss'], 2, L.lang)}: {_yes(_b['first'])} | about ln {_r['vocab']} = {C.number(_r['even'], 2, L.lang)} |
| loss after {_r['steps']} steps on the batch | {C.number(_y['last_loss'], 4, L.lang)}: {_yes(_y['learns'])} | {C.number(_b['last_loss'], 4, L.lang)}: {_yes(_b['learns'])} | below {C.number(C.LEARNED, 2, L.lang)}; a model that sees only a, {C.number(_r['floor'], 2, L.lang)} at best |

{_verdict}.{_stand}
""")
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Your kit now has a check you can point at any model before a long run: your own, a changed one, or a fellow learner's. Run it again after every change to the model or the training step. In the next lesson the long run comes, and it starts only once these three pass.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The overfit test of [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en), now for a GPT and with a loss on the answer alone; the loss before training of [Guided reproduction 2](/course/open/repro-guided-build?lang=en), now a check with a number to meet; the cross-entropy of a model that guesses evenly; and, as in [Gradient checking](/course/open/nn-gradient-check?lang=en), a test whose right outcome you know before you run it. Your kit now has `repro/overfit.py` and `repro/sanity.py`. The next lesson, *Guided reproduction 4: reproduce the key figure*, trains long enough for validation to jump, and redraws the figure from your own logs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_overfit, result_best, choice_bug, result_sanity):
    L.wrap_up(result_overfit, result_best, choice_bug, result_sanity)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
