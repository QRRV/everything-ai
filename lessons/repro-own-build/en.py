# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-build. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-build", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 2: claim and plan](/course/open/repro-own-claim?lang=en) and [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_verdict():
    quick_verdict = L.choice(
        "quick-verdict",
        question="A plan says 3 seeds, lower by at least 0.5. The gains of the seeds are 0.9, 0.2 and 0.6. What does `verdict` give back?",
        options={
            "one": "`\"not supported\"`: one seed is below 0.5",
            "few": "`\"too few runs\"`: the seed below 0.5 does not count",
            "right": "`\"supported\"`: the median of the three gains is 0.6",
        },
        correct="right",
        why={
            "one": "The median decides, not each seed: one seed that goes wrong cannot move it much. See [Your reproduction 2](/course/open/repro-own-claim?lang=en).",
            "few": "All three seeds of the plan have run, and every one counts. See [Your reproduction 2](/course/open/repro-own-claim?lang=en).",
            "right": "The middle one of 0.2, 0.6 and 0.9 is 0.6, and that is at least 0.5.",
        },
    )
    quick_verdict
    return (quick_verdict,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="In `run_seed` of Guided reproduction 4, the seed fixes three things. Which of these is not one of them?",
        options={
            "right": "How many steps it runs",
            "weights": "The starting weights",
            "batches": "The order of the batches",
        },
        correct="right",
        why={
            "weights": "`torch.manual_seed(seed)` sets them, just before the model is made. See [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).",
            "batches": "They come from a generator of their own, `torch.Generator().manual_seed(seed)`. See [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).",
            "right": "The steps are an argument of their own; the seed fixes the split, the starting weights and the batches.",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your plan says what to run; now you build it. The lessons after this one run your experiment many times: over seeds, at the budget of your plan, in the ablation. They can only do that when every run keeps a few promises: it gives back one number, the same one when you ask again, and its seed and its setting really change it. A run that breaks one still gives numbers, but numbers your plan cannot use, and nothing warns you. This lesson builds the experiment, and a contract that checks those promises in seconds.

        **An everyday example.** A recipe that says "a pinch of salt" and "bake until done" gives another cake in every kitchen, and what one change does, you cannot taste. One that says 5 grams and 25 minutes at 180 degrees gives the same cake twice, and when you change one thing, you taste what that one thing does.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           2, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_short():
    guess_short = L.guess(
        "short",
        question=f"The course's experiment for the dropout paper, run for {C.STEPS} steps with seed 0: with dropout the test error is {C.number(C.SHORT_RUN['claim'], 1, L.lang)}%, without {C.number(C.SHORT_RUN['other'], 1, L.lang)}%. What does that say about the paper's claim?",
        options={
            "fails": "It does not reproduce: with dropout the test error is clearly higher",
            "right": f"Nothing yet: {C.STEPS} steps show that the code runs, not the claim",
            "seeds": f"Not enough yet: with five seeds of {C.STEPS} steps it would show",
        },
        reveal=f"**Nothing yet: {C.STEPS} steps show that the code runs, not the claim.** After {C.STEPS} steps neither network has learned much, and dropout, which switches off neurons at random, learns slower at first. More seeds of a run that short measure the same thing more precisely. The picture below shows from which budget on dropout pays off.",
    )
    guess_short
    return (guess_short,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **One function, two settings.** Every experiment of your kit is one function, `run(setting, seed, steps)`. `"claim"` is the setting of your plan's claim, `"other"` the one it is compared with; both share all code but the one thing your plan changes, and the function gives back the metric of your plan, one number. These two names stay the same whatever your paper, so the lessons after this one can run any experiment.

    **The seed sets everything random.** `torch.manual_seed(seed)` at the start of the run sets the starting weights and the dropout masks, and the order of the batches gets a generator of its own, `torch.Generator().manual_seed(seed)`. Then seed 0 of `"claim"` and seed 0 of `"other"` see the same batches, even when only one of them draws dropout masks: the worked example shows why that takes a generator of its own. The same call then gives exactly the same number: that is determinism.

    **A contract.** The promises of a run: it gives back a number, not NaN; the same call gives the same number; and the steps, the seed and the setting each change it. A run that breaks one still gives numbers: with a seed that does nothing, your three seeds are one run three times. An unknown setting must stop it: look the setting up in a dict, and a typo stops with a KeyError instead of running something else.

    **Parts for the sanity checks.** Next to `run`, your experiment gives the sanity checks of the next lesson what they look at, made by the same code: `make_model(setting, seed)`, the model before any training, which `run` uses too; `one_batch()`, one small batch of the training data; `K`, how many numbers the model gives for each example; and `FROM_SCRATCH`, whether it starts from random weights or from a trained model.

    **A short run checks the code.** The contract runs 20 steps: enough to see in seconds that the code keeps its promises. About the claim, it says nothing.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **contract** | what a function promises whoever calls it: what it takes, what it gives back, and what it does with what it does not know |
    | **determinism** | getting exactly the same outcome when you run the same code on the same data again |

    The picture: the course's experiment for the dropout paper, the solution of exercise 2, over seeds 0, 1 and 2. For every budget of steps, each a run of its own, a dot for the gain of each seed: how many points lower the test error is with dropout. The rings are the medians, and the dashed line is the *at least* of the course's plan, 0.5. Move the slider, or drag along the picture: a thin line marks its budget, and the sentence under the picture gives the runs of that budget.
    """)
    return


@app.cell(hide_code=True)
def explore_gains_cell():
    _labels = {"title": "What dropout gains, by budget", "x": "steps of the run", "y": "gain, points", "line": "at least 0.5",
               "alt": "The gain of dropout per seed against the steps of a run, on a log axis from 100 to 12,800 steps. Up to 400 steps every gain is below 0, down to -3.8; at 800 and 1600 steps the medians are -1.0 and 0.3; from 3200 steps on they are 0.5, 0.4 and 0.7, around a dashed line at 0.5."}

    def _frame(s):
        _s = int(s)
        _k = C.BUDGETS.index(_s)
        _r = C.RUNS[_s]
        _say = "the plan's rule says supported" if C.supported(_s) else "the plan's rule says not supported"
        _with = ", ".join(C.number(_x, 1, L.lang) for _x in _r["claim"])
        _without = ", ".join(C.number(_x, 1, L.lang) for _x in _r["other"])
        return (C.gains_picture(_k, _labels),
                f"After {C.whole(_s, L.lang)} steps: with dropout {_with}% test error, without {_without}%. Median gain {C.number(C.median_gain(_s), 1, L.lang).replace('-', '−')}: {_say}.")

    explore_gains = L.explore("budget", "steps", _frame, start=C.BUDGETS[0], values=list(C.BUDGETS), role="",
                              task="From which budget on does dropout lead? And how far above the dashed line does it get?")
    explore_gains
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Up to 800 steps, dropout is behind, in the median by up to 3.7 points: switching off neurons at random slows learning down at first. At 1600 steps it leads by 0.3, and from 3200 steps on by about half a point, right around the *at least* of the plan: a median of 0.5 at 3200 steps, 0.4 at 6400 and 0.7 at 12,800. A run of 20 steps says nothing about that, and neither does the contract. What it does say is that the numbers your next lessons measure come from a run that keeps its promises.
        """),
        L.watch_out("Choose the steps of your main result before you look at gains like these: from the paper's setup and the hours of your plan. The course's own plan for the dropout paper names no number of steps, and the picture shows what that leaves open: by its rule, 3200 and 12,800 steps support the claim, 1600 and 6400 do not. Picking the budget where the gain looks best is choosing your rule after the results."),
        mo.md(r"""
        ### Recipe

        To build the experiment of your plan:

        1. Write down in a dict what `"claim"` and `"other"` are: the one thing your plan changes.
        2. Build the part both settings share, from the lessons your paper builds on: the data, the model in `make_model`, one batch in `one_batch`, the training in `run`.
        3. Seed everything random: `torch.manual_seed(seed)` in `make_model`, and a generator of its own for the order of the batches.
        4. Give back the metric of your plan, one number.
        5. Hold the run to the contract with 20 steps, before any long run.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 of the recipe, and why the batches need a generator of their own. Seed 0 in both settings: `"claim"` draws a dropout mask first, `"other"` does not. Run the cell.
    """)
    return


@app.cell
def example_generators():
    # "claim": a generator for its batches, and a mask
    torch.manual_seed(0)
    _g = torch.Generator().manual_seed(0)
    torch.rand(512)
    _c = torch.randint(0, 6000, (3,), generator=_g).tolist()
    # "other": the same generator, and no mask
    torch.manual_seed(0)
    _g = torch.Generator().manual_seed(0)
    _o = torch.randint(0, 6000, (3,), generator=_g).tolist()
    print("own generator:", _c, _o)

    # the same, with the one global generator
    torch.manual_seed(0)
    torch.rand(512)
    _c = torch.randint(0, 6000, (3,)).tolist()
    torch.manual_seed(0)
    _o = torch.randint(0, 6000, (3,)).tolist()
    print("one generator:", _c, _o)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With a generator of its own, the first batch of seed 0 is the same three pictures in both settings, whatever else the run draws. With the one generator that `torch.manual_seed` sets, the dropout mask of `"claim"` uses up random numbers, and everything drawn after it shifts: the two runs of seed 0 see other batches. Then the gain of a seed compares runs that differ in more than dropout, and pairing by seed loses what it was for.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_contract():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The contract

        Complete `contract(run, steps)`: it gives back the first of the five answers in its comments that holds for `run`, in that order, or `None` when the run keeps its promises. Call the run the way the comments say, starting with `a = run("claim", 0, steps)`. New is `math.isnan(a)`: `True` when `a` is NaN, the float that is not a number, which a training that breaks down gives back. A run that gives back no number at all, such as `None`, already stops `math.isnan` with an error; the check of exercise 2 tells you so. The check tries made-up runs, each named after what it does.
        """),
        L.task("Complete `contract`: call the run, and give back the first answer that holds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contract():
    box_contract = L.code_box("contract", lines=18)
    box_contract
    return (box_contract,)


@app.cell(hide_code=True)
def check_contract(box_contract):
    result_contract = L.check("contract", box_contract.value)
    result_contract
    return (result_contract,)


@app.cell(hide_code=True)
def hints_contract():
    L.hints(
        "contract",
        [
            "First `a = run(\"claim\", 0, steps)` and the check for NaN, then one `if` for each answer, in the order of the comments, each with a call of its own.",
            "\"steps do nothing\" compares `run(\"claim\", 0, 2 * steps)` with `a`, and \"seed does nothing\" compares `run(\"claim\", 1, steps)` with it.",
        ],
        why="Each `if` is one promise of the contract, in the order of the comments, so that the first answer is the first thing to fix.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_experiment():
    mo.md(r"""
    ### Exercise 2. Your experiment

    Build the experiment of your plan in the box: `run(setting, seed, steps)` and its parts, for the paper you chose. The help below says what `"claim"` and `"other"` are for your paper, and what you can reuse. The check holds your run to the contract with 20 steps, and names the first promise it breaks; it also checks that an unknown setting stops it. Once it keeps them all, it is saved as `work/projects/own/experiment.py`. Read data with a path from the course folder, such as `np.load("data/fashion-mnist-small.npz")`. On an NVIDIA card, move the model and the batches to it inside `run`: the lessons after this one call `run` the same way on any computer. For the dropout paper this exercise takes about half an hour; for another paper, which you build yourself, plan an evening.
    """)
    return


@app.cell(hide_code=True)
def help_for_your_paper():
    mo.vstack([
        mo.accordion({"Help for your paper": mo.md(C.outline_text(C.saved_choice(), C.saved_plan(), L.lang))}),
        L.task("Build `run` for your paper, and check it: the check names the first promise it breaks."),
    ])
    return


@app.cell(hide_code=True)
def exercise_experiment():
    box_experiment = L.code_box("experiment", lines=30)
    box_experiment
    return (box_experiment,)


@app.cell(hide_code=True)
def check_experiment(box_experiment):
    result_experiment = L.check("experiment", box_experiment.value)
    result_experiment
    return (result_experiment,)


@app.cell(hide_code=True)
def hints_experiment():
    L.hints(
        "experiment",
        [
            "Start with the part both settings share, and run it with one setting: does it give back one number?",
            "Then the one thing that differs, in the dict `SETTINGS`, and the seed: `torch.manual_seed(seed)` in `make_model`, `torch.Generator().manual_seed(seed)` for the batches in `run`, which makes its model with `make_model`.",
            "For the dropout paper: an MLP with two hidden layers of 512 on Fashion-MNIST, dropout of 0.2 on the input and 0.5 on the hidden layers, SGD with momentum, and a learning rate that decays to 0 over the steps.",
        ],
        why="This is the course's experiment for the dropout paper, whose runs the picture shows. For another paper the shape stays the same: a dict of the settings, the seed first, the part both share, and the metric as one number.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your experiment from your kit, seeds 0, 1 and 2 in both settings, for 200 steps, and what the rule of your plan says about them. 200 steps are not the budget of your plan: the next lessons run that.
    """)
    return


@app.cell(hide_code=True)
def demo_button():
    run_demo = mo.ui.run_button(label="Run seeds 0, 1 and 2 for 200 steps")
    run_demo
    return (run_demo,)


@app.cell(hide_code=True)
def demo_experiment(run_demo, result_experiment):
    result_experiment  # run again when exercise 2 saves your experiment
    _out = mo.md("*Press the button: it runs your `own/experiment.py` six times.*")
    if run_demo.value:
        _runs = C.demo_runs()
        if _runs is None:
            _out = mo.md("*Finish exercise 2 first: then your experiment is in your kit, and this cell runs it.*")
        elif _runs == "slow":
            _out = mo.md("*One run of 200 steps of your experiment takes more than two minutes, so the demo stops. The main result will run your plan's budget without this limit.*")
        elif _runs == "error":
            _out = mo.md("*Your `own/experiment.py` stops with an error at 200 steps, more than the 20 and 40 of the check. Does every step get a batch, also after one pass through the data? Or did you change the file after the check? Then delete it, and check exercise 2 again.*")
        else:
            _said = C.demo_verdict(_runs, C.saved_plan())
            _rule = f"Your rule says `{_said}`: after 200 steps, that is about the code, not the claim." if _said else "With a plan and a rule in your kit, your rule would judge these too."
            _out = mo.vstack([mo.Html(C.demo_table(_runs, {"seed": "seed"}, L.lang)), mo.md(f"*{_rule}*")])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. What a short run says

    Your experiment keeps the contract with 20 steps. Say what that tells you, and what not yet.
    """)
    return


@app.cell(hide_code=True)
def explain_why_short():
    explain_why_short_cell = L.explain(
        "why-short",
        question="Your experiment keeps the contract with 20 steps. What does that tell you about it, and what not yet?",
        model_answer="**What it tells me:** my run gives back one number, the same one for the same call, and its steps, its seed and its setting each change it: three seeds are three runs, and the two settings really differ. An unknown setting stops it, so a typo cannot run the wrong experiment. **What not yet:** whether the claim holds. After 20 steps the course's runs have a test error of 67% with dropout and 53% without: dropout learns slower at first, and only from about 1600 steps on does it lead, by half a point from 3200 on. Whether mine does is for the main result, at the budget of my plan, which I choose before I look.",
        points=[
            "you name what the contract shows about your run",
            "you say why that matters for your plan, such as seeds that really differ",
            "you say that a run this short says nothing about the claim",
            "you say where the claim is tested: in the main result, at the budget of your plan",
        ],
    )
    explain_why_short_cell
    return (explain_why_short_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The plan of [Your reproduction 2](/course/open/repro-own-claim?lang=en), now as code; the seed of [Guided reproduction 4](/course/open/repro-guided-figure?lang=en), which fixed the split, the starting weights and the batches; and [Dropout](/course/open/nn-dropout?lang=en), which slows learning down at first. In the next lesson, *Your reproduction 4: sanity checks*, you point the sanity checks of your kit at the parts of your experiment, before any long run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_contract, result_experiment, explain_why_short_cell):
    L.wrap_up(result_contract, result_experiment, explain_why_short_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
