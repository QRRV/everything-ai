# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-ablation. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-ablation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 5: the main result](/course/open/repro-own-figure?lang=en) and [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="Your experiment log has one line: `{\"setting\": \"claim\", \"seed\": 0, \"steps\": 400, \"result\": 16.4}`. Your `run_plan` is called with seeds `[0]` and budgets `[400, 800]`. Which runs does it run?",
        options={
            "two": "Two: both settings at 800 steps, as seed 0 is in the log already",
            "four": "Four: a line from before does not count for a new call",
            "right": "Three: `\"other\"` at 400 steps, and both settings at 800",
        },
        correct="right",
        why={
            "two": "The setting, the seed and the steps together name a run: only `\"claim\"` of seed 0 at 400 steps is done. See [Your reproduction 5](/course/open/repro-own-figure?lang=en).",
            "four": "`run_plan` reads the log first, and skips every run whose line is in it, whichever call wrote it. See [Your reproduction 5](/course/open/repro-own-figure?lang=en).",
            "right": "Only a run whose setting, seed and steps stand together on a line of the log is skipped.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def quick_check_grid():
    quick_grid = L.choice(
        "quick-grid",
        question="Guided reproduction 5 ran an ablation grid of weight decay and the fraction of the table. A grid of weight decay 1, 0.1 or 0, times the fractions 30%, 50%, 70% and 90%: how many runs is that, and why?",
        options={
            "right": "12: every value of the one, with every value of the other",
            "one": "6: the control run, and each other value on its own once",
            "three": "4: one for each fraction, the control run among them",
        },
        correct="right",
        why={
            "one": "That would change one thing at a time from the control run. A grid runs every combination, so it also shows whether weight decay matters more at one fraction than at another. See [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en).",
            "three": "Four runs vary only the fraction: they cannot show what weight decay does. See [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en).",
            "right": "3 × 4 = 12: each combination once, the control run among them.",
        },
    )
    quick_grid
    return (quick_grid,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your main result says that the method works in your runs. It does not say which part of the method does the work. Many methods are several changes at once: the dropout paper switches off neurons at the input, with a chance of 0.2, and in the hidden layers, with a chance of 0.5. An ablation takes one part out, keeps all the rest, and runs again at the budget and over the seeds of your main result: what is left of the gain is what the method does without that part. Reviewers ask for it, and so will the readers of your report.

        **An everyday example.** Your bike is faster after a new chain and new tyres. Ride it once more on the old tyres: what is left of the gain is what the chain did on its own.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           5, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_input():
    guess_input = L.guess(
        "no-input",
        question="In the course's main result for the dropout paper, dropout lowers the test error at 12,800 steps by 0.7 points in the median. Take out the dropout at the input, and keep the 0.5 in the hidden layers. What do you think is left of the gain?",
        options={
            "most": "Most of it: the hidden layers, with their larger chance of 0.5, do the work",
            "half": "About half of it: each of the two parts brings its own share",
            "right": "None of it: the network does even a little worse than without dropout",
        },
        reveal="**None of it.** Without the dropout at the input, the gains over no dropout are −0.9, 0.4 and −0.5 for seeds 0, 1 and 2: a median of −0.5. Dropout in the hidden layers alone does not help this small network at 12,800 steps; with the dropout at the input as well, the two give 0.7. *The idea* shows the other ablation too.",
    )
    guess_input
    return (guess_input,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **One part out.** The ablation is the setting of the claim with one part of its method taken out, and everything else the same: a third setting of your experiment, `"ablation"`. The runs of the claim are its control runs; the other setting is what the claim is compared with. Take out two parts at once, and the ablation could not say which of them did the work. When the method is one change, such as a rate, an ablation can take part of it: half the rate.

    **What it keeps.** Seed by seed, the gain of the ablation over the other setting, against the gain of the claim: the share it keeps is the median of its gains, divided by the median of the claim's. A share of 1 keeps all of the gain, 0 none of it, and below 0 the ablation does worse than the other setting.

    **The rule, first.** Your result depends on the part when the ablation keeps less than half of the gain. You fix the part and the rule before the runs, as with the rule of your plan, and the ablation runs at the budget and over the seeds of your main result, so that its runs compare with the claim's seed by seed.

    **Your experiment stays as it is.** Your `own/ablation.py` imports your experiment and adds `"ablation"` to its `SETTINGS` while it runs: the file of your experiment does not change, and your main result stays what it was.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | ablation setting | the setting of the claim with one part of its method taken out |
    | share kept | the median gain of the ablation over the other setting, divided by the median gain of the claim |

    The picture: the course's ablations of its experiment for the dropout paper at 12,800 steps, with the claim and the other setting: the four ways to have the dropout at the input (0.2) and in the hidden layers (0.5), or not. Each dot is the test error of one seed, the ring the median of three; the solid line is the median of the claim, the dashed one that of the other setting. Move the slider to read each row.
    """)
    return


@app.cell(hide_code=True)
def explore_rows_cell():
    _names = ["both: the claim", "hidden layers only", "input only", "neither: the other"]
    _labels = {"title": "Test error at 12,800 steps, by dropout", "x": "test error (%)",
               "alt": "The test error of three seeds at 12,800 steps for four settings. Both dropouts, the claim: 12.5, 12.9 and 12.4. Hidden layers only: 13.8, 13.2 and 14.1. Input only: 13.5, 13.3 and 12.7. Neither, the other setting: 12.9, 13.6 and 13.6."}
    _rows = ["claim", "no-input", "no-hidden", "other"]

    def _frame(k):
        _k = int(k) - 1   # rows 1 to 4, from the top
        _row = _rows[_k]
        _g = C.gains(C.course_runs("claim"), C.course_runs("other"))
        if _row == "claim":
            _say = f"Both, the claim: gains of {', '.join(C.signed(g, L.lang) for g in _g)} over no dropout, a median of {C.signed(C.median(_g), L.lang)}: the gain the ablations are measured against."
        elif _row == "other":
            _say = "Neither, the other setting: the runs every gain is counted from."
        else:
            _a = C.gains(C.course_runs(_row), C.course_runs("other"))
            _part = "the dropout at the input" if _row == "no-input" else "the dropout in the hidden layers"
            _share = round(100 * C.course_kept(_row))
            _say = (f"Without {_part}: gains of {', '.join(C.signed(g, L.lang) for g in _a)}, a median of {C.signed(C.median(_a), L.lang)}. "
                    f"It keeps {C.signed(_share, L.lang)}% of the gain: less than half, so by the rule the result depends on {_part}.")
        return C.ablation_picture(_k, _names, _labels), _say

    explore_rows = L.explore("row", "row", _frame, start=2, low=1, high=4, step=1, role="",
                             task="Which of the two parts does the gain depend on? And do the shares of the two ablations add up to the whole gain?")
    explore_rows
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        With either part alone, less than half of the gain is left: without the dropout at the input none of it, −71%, as the network does a little worse than without any dropout; without the dropout in the hidden layers, 43%. By the rule, the result depends on both: what dropout does here, the two parts do together. And the shares do not add up: −71% and 43% are far from the whole gain. When parts work together, an ablation says what a part is needed for, not how much of the gain is its own. With three seeds, the first answer is clear, a median of −0.5 against 0.7; the second is close to the line, 43% against 50%.
        """),
        L.watch_out("An ablation of three seeds sees a large change, not a small one: a share near one half can tip with one seed more. Fix the part and the rule before the runs, run one ablation, and say in your report how close it was. Trying parts until one gives the answer you hoped for is choosing your result."),
        mo.md(r"""
        ### Recipe

        To run the ablation of your reproduction:

        1. Name the parts of the method of your claim, and choose the one to take out, before any run.
        2. Add the setting `"ablation"` to your experiment: that of the claim, with that one part out.
        3. Run it with the runner of your kit, over the seeds of your plan, at your budget.
        4. Work out the share it keeps: its gains over the other setting, seed by seed, against the claim's.
        5. Read it by the rule: below one half, your result depends on that part.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The share the other ablation keeps, without the dropout in the hidden layers, by hand: the gain of each seed over no dropout, for the claim and for the ablation, and the median of each. New is `zip`: it goes through several lists at once, one item of each per round, so that `c` and `o` are always of the same seed. Run the cell.
    """)
    return


@app.cell
def example_share():
    _claim = [12.5, 12.9, 12.4]      # both dropouts: seeds 0, 1, 2
    _other = [12.9, 13.6, 13.6]      # no dropout
    _ablation = [13.5, 13.3, 12.7]   # only the dropout at the input
    _g_claim = [round(o - c, 1) for c, o in zip(_claim, _other)]
    _g_ablation = [round(o - a, 1) for a, o in zip(_ablation, _other)]
    print("claim:", _g_claim, np.median(_g_claim))
    print("ablation:", _g_ablation, np.median(_g_ablation))
    print("kept:", round(np.median(_g_ablation) / np.median(_g_claim), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The test error should go down, so a gain is the other setting minus the setting itself. The claim gains 0.7 in the median, the ablation 0.3, so it keeps 0.3 / 0.7 = 0.43 of the gain: less than half. Seed 0 of the ablation is even behind no dropout, by 0.6, and seed 2 ahead by 0.9: the median takes the middle gain, 0.3, so neither end decides it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ablation():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your ablation

        Complete `own/ablation.py` in the box: the value of the setting `"ablation"`, the claim's with one part of its method taken out, and in `PART` which part that is. The box imports your experiment from your kit, and adds the setting to its `SETTINGS` while it runs. The check compares the new setting with `"claim"` and `"other"`, and runs `run("ablation", 0, 20)` twice. For the dropout paper, `"claim"` is `(0.2, 0.5)`.
        """),
        L.task("Complete the ablation: the claim's setting without one part, and the name of that part."),
    ])
    return


@app.cell(hide_code=True)
def course_note():
    L.course_versions()
    return


@app.cell(hide_code=True)
def exercise_ablation():
    box_ablation = L.code_box("ablation", lines=9)
    box_ablation
    return (box_ablation,)


@app.cell(hide_code=True)
def check_ablation(box_ablation):
    result_ablation = L.check("ablation", box_ablation.value)
    result_ablation
    return (result_ablation,)


@app.cell(hide_code=True)
def hints_ablation():
    L.hints(
        "ablation",
        [
            "Look at the value of `\"claim\"` in the `SETTINGS` of your experiment: which of its parts is the method, and which could go without the others?",
            "Copy the value of `\"claim\"`, and set one of its parts to what `\"other\"` has there: for the dropout paper, one of its two numbers to 0.0.",
        ],
        why="One part out, the rest as the claim has it, so that a change in the result can come from that part only.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kept():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What the ablation keeps

        Complete `kept(direction, claim, ablation, other)`. Each list holds the metric of one setting, seed 0 first, at your budget; `direction` is `"lower"` or `"higher"`, the way the claim says the metric goes. It gives back the share of the claim's gain that the ablation keeps, as in the worked example, or `None` when the claim has no gain to keep: a median gain of 0 or less. The check tries the course's ablation, the other one, an accuracy that should go up, a claim without a gain, and four seeds.
        """),
        L.task("Complete `kept`: the gains of each seed the way `direction` says, then `None` or the share."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kept():
    box_kept = L.code_box("kept", lines=18)
    box_kept
    return (box_kept,)


@app.cell(hide_code=True)
def check_kept(box_kept):
    result_kept = L.check("kept", box_kept.value)
    result_kept
    return (result_kept,)


@app.cell(hide_code=True)
def hints_kept():
    L.hints(
        "kept",
        [
            "Go through the three lists together with `zip`, one seed at a time, as the worked example does; for `\"higher\"`, a gain is the setting minus the other.",
            "After the loop: `None` when `np.median(claim_gains) <= 0`; else the median of the ablation's gains divided by the median of the claim's.",
        ],
        why="A gain of a seed compares two runs that saw the same starting weights and batches; the medians keep one seed that runs away from deciding it; and without a gain there is nothing to keep.",
    )
    return


@app.cell(hide_code=True)
def text_run():
    mo.md(rf"""
    ### What your project can do now

    The button runs your ablation with your kit: your `run_plan` with the `run` of your `own/ablation.py`, every seed of your plan, at your budget, into a log of its own, `work/projects/own/ablation.jsonl`. Then it reads the runs of the claim and the other setting at that budget from your main result, and gives all three to your `kept`. For the course's experiment for the dropout paper, the three runs take about {C.ABLATION_MINUTES} minutes on one thread of the course's computer. As with your main result, a stop costs the run that was going. Change your ablation after its runs, and delete `work/projects/own/ablation.jsonl` first: the button runs only what has no line yet.
    """)
    return


@app.cell(hide_code=True)
def button_ablation():
    run_ablation = mo.ui.run_button(label="Run my ablation")
    run_ablation
    return (run_ablation,)


@app.cell(hide_code=True)
def demo_ablation(run_ablation, result_ablation, result_kept):
    result_ablation, result_kept  # run again when an exercise saves a part of the kit
    ablation_lines = None
    if not run_ablation.value:
        _out = mo.md("*Press the button to run your ablation with your kit.*")
    else:
        def _progress(_line):
            _n = len(C.my_ablation_log() or [])
            mo.output.replace(mo.md(f"**{_n} {'run' if _n == 1 else 'runs'} of the ablation in its log** · the newest: seed {_line['seed']}, {C.whole(_line['steps'], L.lang)} steps: {C.short(_line['result'], L.lang)}"))

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Your kit needs four parts first, and one of them is not there yet: your ablation, exercise 1 here; your runner, exercise 1 of [Your reproduction 5](/course/open/repro-own-figure?lang=en); your experiment, exercise 2 of [Your reproduction 3](/course/open/repro-own-build?lang=en); and your plan, of [Your reproduction 2](/course/open/repro-own-claim?lang=en).*")
        elif _r == "budget":
            _out = mo.md("*Fix your budget in exercise 3 of [Your reproduction 5](/course/open/repro-own-figure?lang=en) first: the ablation runs at it.*")
        elif _r == "log":
            _out = mo.md("*A line of your ablation log, `work/projects/own/ablation.jsonl`, is not a run of the ablation as `run_plan` writes it: `\"setting\"` `\"ablation\"`, a whole `\"seed\"` and `\"steps\"`, and a number as `\"result\"`. Did you change your `own/runner.py` after its check, or the log by hand?*")
        elif _r == "error":
            _out = mo.md("*Your `own/runner.py`, `own/ablation.py` or `own/experiment.py` stops with an error. Did you change one of them after its check? Then check it again. The runs that were done are in the log.*")
        else:
            ablation_lines = _r
            _out = mo.md(f"*Done: {len(_r)} runs of the ablation are in its log, `work/projects/own/ablation.jsonl`.*")
    _out
    return (ablation_lines,)


@app.cell(hide_code=True)
def demo_reading(ablation_lines, result_kept):
    ablation_lines, result_kept  # read the logs again after a run
    _plan, _budget = C.P.saved_plan(), C.F.saved_budget()
    _r = C.my_reading(_plan, _budget["steps"]) if _plan and _budget else None
    if _r is None:
        _out = mo.md("*Your ablation shows here once its runs, and those of your main result at your budget, are in your logs.*")
    else:
        _a, _b = C.P._settings(_plan)
        _rows = "\n".join(f"| {_i} | {C.short(_c, L.lang)} | {C.short(_x, L.lang)} | {C.short(_o, L.lang)} |"
                          for _i, (_c, _x, _o) in enumerate(zip(_r["claim"], _r["ablation"], _r["other"])))
        if _r["kept"] == "error":
            _say = "Your `own/kept.py` stops with an error, or is not in your kit yet: check exercise 2."
        elif _r["kept"] is None:
            _say = "Your `kept` gives `None`: the claim has no gain at your budget, so there is nothing for the ablation to keep."
        else:
            _share = round(100 * _r["kept"])
            _say = f"Your ablation keeps **{C.signed(_share, L.lang)}%** of the gain: " + ("less than half, so by the rule your result depends on the part it takes out." if C.depends(_r["kept"]) else "at least half, so by the rule your result does not depend on the part it takes out.")
        _out = mo.md(f"""
| seed | {_a} | ablation | {_b} |
|---|---|---|---|
{_rows}

{_say}
""")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. What your ablation says

    Your ablation is in its log. Say what your result depends on, by your rule, and what would make you surer.
    """)
    return


@app.cell(hide_code=True)
def explain_why_ablation():
    explain_why_ablation_cell = L.explain(
        "why-ablation",
        question="Which part of the method does your result depend on, by your rule? And what would make you surer of it?",
        model_answer="**By my rule:** without the dropout at the input, my ablation keeps −71% of the gain, far below one half: my result depends on the dropout at the input. Without it, the network is even a little worse than without any dropout, so the dropout in the hidden layers does not help on its own here. **To be surer:** more seeds, since my seeds differ by up to 0.9 points and three cannot see a small change; and the other ablation, without the dropout in the hidden layers, which the course measured at 43%, close to the line. I would write both in my report, with the rule I fixed before the runs.",
        points=[
            "you give the share your ablation keeps, and what the rule says",
            "you say what that means for the part you took out",
            "you name something that would make you surer, such as more seeds or another ablation",
        ],
    )
    explain_why_ablation_cell
    return (explain_why_ablation_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The ablation of [Ablation: what happens if you take a part out?](/course/open/res-ablation?lang=en) and the grid of [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en), now for the method of your own paper; and the runner and the budget of [Your reproduction 5](/course/open/repro-own-figure?lang=en). In the next lesson, *Your reproduction 7: compare with the paper*, you test whether the difference between your numbers and the paper's is real.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ablation, result_kept, explain_why_ablation_cell):
    L.wrap_up(result_ablation, result_kept, explain_why_ablation_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
