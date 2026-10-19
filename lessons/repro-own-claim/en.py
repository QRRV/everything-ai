# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-claim. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-claim", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 1: choose a paper](/course/open/repro-own-choose?lang=en) and [Guided reproduction 1](/course/open/repro-guided-claim?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_margin():
    quick_margin = L.choice(
        "quick-margin",
        question="A paper needs an NVIDIA card with 8 GB, and the course estimates 2 hours. Your computer has an NVIDIA card with 12 GB, and you have one evening of 4 hours. With a margin of 3, does it fit?",
        options={
            "right": "No: 2 hours times 3 is 6, more than the 4 you have",
            "card": "Yes: 12 GB is more than the 8 it needs",
            "hours": "Yes: 2 hours fit in 4",
        },
        correct="right",
        why={
            "card": "The card is enough, but the time has to fit as well, with the margin. See [Your reproduction 1](/course/open/repro-own-choose?lang=en).",
            "hours": "Without a margin they do; with a margin of 3, the estimate counts as 6 hours. See [Your reproduction 1](/course/open/repro-own-choose?lang=en).",
            "right": "The card is enough; the evening is not. A night would be, or a smaller paper.",
        },
    )
    quick_margin
    return (quick_margin,)


@app.cell(hide_code=True)
def quick_check_before():
    quick_before = L.choice(
        "quick-before",
        question="The plan of Guided reproduction 1 calls a run grokked when validation reaches 99% at least 10 times as late as training. One run reaches it 8 times as late, and its curve looks like grokking to you. What does the plan say?",
        options={
            "curve": "`\"grokked\"`: the curve shows it",
            "change": "Set `DELAY` to 8: 8 is close to 10",
            "right": "`\"no delay\"`: 8 times is less than the 10 of the plan",
        },
        correct="right",
        why={
            "curve": "The curve may look like it, but the plan says at least 10 times as late, and it said so before the run. See [Guided reproduction 1](/course/open/repro-guided-claim?lang=en).",
            "change": "Changing `DELAY` after you have seen a run fits the rule to the result, and then the plan can no longer fail. See [Guided reproduction 1](/course/open/repro-guided-claim?lang=en).",
            "right": "The plan was written before the run, so 8 times as late is a finding too, not a reason to change the rule.",
        },
    )
    quick_before
    return (quick_before,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In the guided reproduction, the course wrote the plan's numbers for you: 99%, and 10 times as late. For your own paper, you write them. Before your first run, you decide what you measure, against what, how many seeds, what it costs, and what result would mean that the claim did not reproduce. A plan without that last part cannot fail, and a reproduction that cannot fail tells nobody anything, you included.

        **An everyday example.** A bet you settle after the match, with a rule you choose then, you always win. Before the match, "they win by two goals or more" can be lost, and that is what makes winning it mean something.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           1, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_extra():
    guess_extra = L.guess(
        "extra",
        question="Your plan says 3 seeds, and that the claim holds when its setting is ahead by at least 0.5 in the median over the seeds. After 3 seeds, the median is 0.4. Disappointed, you run 2 more, and over all 5 the median is 0.6. What does your plan say?",
        options={
            "yes": "Supported: over 5 seeds, the median is 0.6",
            "right": "Not supported: the plan's 3 seeds give 0.4",
            "wait": "No verdict yet: run seeds until it is clear",
        },
        reveal="**Not supported: the plan's 3 seeds give 0.4.** The 2 extra seeds were run because the result disappointed, and would not have been run otherwise: counting them is cherry-picking. Report them, as extra; the verdict comes from the plan.",
    )
    guess_extra
    return (guess_extra,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A claim as a comparison.** Each claim on the list can be tested as a comparison of two settings that differ in one thing: with and without dropout, two layers and one, random labels and the true ones. Your plan names both, what stays the same, and one number per run, the metric. A claim of *better* says which way the metric goes with its setting, higher or lower.

        **Per seed, then the median.** You run both settings with the same seeds: seed 0 of one next to seed 0 of the other, which share their luck, such as the order of the data. For each seed, the *gain* is how far the setting of the claim is ahead, the way the claim says. The median of the gains decides: one seed that goes wrong cannot move it much.

        **How big.** The plan says how big the median gain must be, at least, to count as the claim: an effect size, in the units of the metric. Too small, and noise between seeds passes for the claim; too big, and a real effect fails. A claim of *keeping up*, such as a network learning random labels about as well as the true ones, has a negative number: at least -1 means at most 1 behind. Such a claim also holds when neither setting works, so the plan says what the other one must reach, such as about 100% training accuracy with the true labels.

        **Before any run, with the seeds you plan.** Runs you add because the result disappoints do not count. Science calls writing it down beforehand a preregistration.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **effect size** | how big a difference is, in the units of what you measure: 1.2 points of test error, say |
        | **preregistration** | writing down your plan, with the rule that decides, before you see a result, and keeping it |
        | **gain** | here: for one seed, how far the setting of the claim is ahead of the other, the way the claim says |

        The picture: made-up outcomes for the dropout paper, the gain of each seed as a dot, in points of test error, and the median of each row as a ring. On top, dropout works; below, it makes no difference, and only the seeds differ. The dashed line through both rows is your *at least*: move it, or drag it.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_counts_cell():
    _labels = {"title": "What counts as the claim", "x": "gain per seed, points of test error",
               "works": "dropout works", "none": "no effect", "yes": "supported", "no": "not supported",
               "alt": "Two rows of three dots on an axis of gains from -1 to 2.5. Top row, dropout works: gains 0.9, 1.2 and 1.5, median 1.2. Bottom row, no effect: gains -0.3, 0.1 and 0.4, median 0.1. A dashed line through both rows marks the smallest gain that counts."}

    def _frame(d):
        _d = round(d, 1)
        _works, _none = C.verdicts_at(_d)
        _say = ("both count as the claim, and noise passes for dropout working." if _works and _none
                else "the plan tells them apart." if _works else "neither counts, and even dropout that works fails your plan.")
        return C.counts_picture(_d, _labels), f"At least {C.number(_d, 1, L.lang).replace('-', '−')}: {_say}"

    explore_counts = L.explore("at-least", "at least", _frame, start=0, low=-0.5, high=2, step=0.1, role="",
                               task="Find where the plan tells the two outcomes apart. Where in that range would you put it, before any run?")
    explore_counts
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        From 0.2 to 1.2, the plan tells the two outcomes apart; below, noise counts as the claim, and above, so little counts that dropout that works fails. Where you put it comes from two things you can know before your runs: how much two seeds of the same setting differ, as in [One run is not a result](/course/open/res-seeds-variance?lang=en), and how big the paper says the effect is. Put it above the noise, and no higher than the smallest effect you would still call the paper's claim.

        **The cost.** One seed of both settings takes some hours; the plan costs that times the seeds, times your margin, and that has to fit the time you have.
        """),
        L.watch_out("Changing the rule after you have seen your results is not a correction but a new plan. If you do, say so in your report, with the verdict of both."),
        mo.md(r"""
        ### Recipe

        To write the plan of your reproduction:

        1. Write the claim in one sentence, and find the figure or table of the paper that shows it.
        2. Name the two settings, and everything that stays the same in both.
        3. Choose the metric: one number per run.
        4. Say which way the claim goes, and the smallest median gain that counts, from the noise between seeds and the paper's effect.
        5. Plan 3 seeds or more, and count the cost: seeds times the hours of one seed, times a margin of 2 or 3.
        6. Write it down before any run, and do not change it afterwards.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The rule by hand, for the top row of the picture: three seeds of a test error in %, with dropout and without, and a plan that says lower by at least 0.5. Run the cell.
    """)
    return


@app.cell
def example_rule():
    _with = [10.6, 10.3, 9.9]       # test error with dropout, seeds 0, 1 and 2
    _without = [11.5, 11.5, 11.4]   # the same seeds, without dropout
    _gains = []
    for _i in range(3):
        _gains.append(_without[_i] - _with[_i])   # lower is the claim: how much lower with dropout
    print("gains:", [round(_g, 2) for _g in _gains], "median:", round(np.median(_gains), 2), "counts:", np.median(_gains) >= 0.5)
    print("cost:", round(3 * 0.3 * 2, 2), "hours")   # 3 seeds of 0.3 hours, with a margin of 2
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The gains are 0.9, 1.2 and 1.5 points, the median is 1.2, and that is at least 0.5: supported. With dropout the error is lower, so the gain is *without minus with*; for a claim of higher, it is the other way round. The plan costs 1.8 hours, which fits an evening. `round` only makes the gains easy to read: `11.5 - 10.6` is 0.9000000000000004 in floating point.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The rule of your plan

        Complete `verdict(plan, results)`. `plan["settings"]` holds the two settings, the claim's first; `plan["direction"]` is `"higher"` or `"lower"`; `plan["at_least"]` is the smallest median gain that counts; `plan["seeds"]` is the number of seeds. `results[name]` is the metric of each run of the setting `name`, seed 0 first. Give back `"too few runs"` when a setting has fewer runs than the plan's seeds, else `"supported"` when the median gain over the plan's seeds is at least `plan["at_least"]`, and `"not supported"` when it is not. The check tries both directions, a seed short, more runs than planned, a median of exactly `at_least`, a negative `at_least` and four seeds.
        """),
        L.task("Complete `verdict`: count the runs, make the gain of each of the plan's seeds, and compare its median."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=16)
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
            "First the count: `len(results[claim]) < n or len(results[other]) < n` means a seed is missing.",
            "Then the loop of the worked example, over `range(n)`: only the plan's seeds, whatever more there are. Which of the two you subtract from the other depends on `plan[\"direction\"]`.",
            "Last, `np.median(gains) >= plan[\"at_least\"]`.",
        ],
        why="The gain per seed compares runs that share their luck, the median keeps one bad seed from deciding, and `range(n)` keeps extra runs out.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.md(r"""
    ### Exercise 2. Your plan

    Fill in the plan of your own reproduction, for the paper you chose in [Your reproduction 1](/course/open/repro-own-choose?lang=en). The check below reads it the way a reviewer would, and sends back the first thing that is missing or does not add up. Once there is nothing, your plan is saved as `work/projects/own/plan.json`, for the lessons that build and run it. You can change it later, before your first run.
    """)
    return


@app.cell(hide_code=True)
def plan_inputs():
    _choice = C.saved_choice()
    _old = C.saved_plan() or {}
    _old = _old if _choice and _old.get("paper") == _choice["id"] else {}
    _s = list(_old.get("settings") or ["", ""])
    _ways = {"higher than with the other": "higher", "lower than with the other": "lower"}
    _levels = {"any computer": "any", "NVIDIA, 8 GB": "gpu8", "recommended spec": "rec"}
    plan_claim = mo.ui.text_area(label="The claim you test, in one sentence", full_width=True, rows=4, value=_old.get("claim") or C.claim_of(_choice, L.lang))
    plan_shown = mo.ui.text(label="Where the paper shows it: the figure or table", full_width=True, value=_old.get("shown_in", ""))
    plan_a = mo.ui.text(label="The setting of the claim, such as with dropout", full_width=True, value=_s[0])
    plan_b = mo.ui.text(label="The setting it is compared with, such as without dropout", full_width=True, value=_s[1])
    plan_setup = mo.ui.text_area(label="What stays the same in both: data, network, steps", full_width=True, rows=2, value=_old.get("setup", ""))
    plan_metric = mo.ui.text(label="What you measure: one number per run, such as the test error at the end, in %", full_width=True, value=_old.get("metric", ""))
    plan_seeds = mo.ui.number(start=1, stop=20, step=1, value=_old.get("seeds") or 3, label="Seeds")
    plan_direction = mo.ui.dropdown(options=_ways, value=next((_k for _k, _v in _ways.items() if _v == _old.get("direction")), None), label="With the setting of the claim, the metric is")
    plan_at_least = mo.ui.number(start=-1000, stop=1000, step=0.01, value=_old.get("at_least") or 0, label="by at least, in the median over the seeds (negative: at most this much the other way)")
    plan_per_seed = mo.ui.number(start=0, stop=100, step=0.1, value=_old.get("hours_per_seed") or 0, label="Hours for one seed, both settings")
    plan_margin = mo.ui.number(start=1, stop=5, step=0.5, value=_old.get("margin") or 2, label="Margin")
    plan_computer = mo.ui.dropdown(options=_levels, value=next(_k for _k, _v in _levels.items() if _v == (_old.get("computer") or C.machine())), label="Your computer")
    plan_hours = mo.ui.number(start=0, stop=200, step=0.5, value=_old.get("hours") or 4, label="Hours you have for it")
    _head = mo.md(f"*Your paper: {C.plain(_choice['title'])}.*") if _choice else mo.md("*No paper chosen yet: choose one in [Your reproduction 1](/course/open/repro-own-choose?lang=en) first.*")
    mo.vstack([_head,
               mo.md("**The claim**"), plan_claim, plan_shown,
               mo.md("**The experiment**"), plan_a, plan_b, plan_setup, plan_metric, plan_seeds,
               mo.md("**What counts**"), plan_direction, plan_at_least,
               mo.md("**The cost**"), plan_per_seed, plan_margin, plan_computer, plan_hours])
    return plan_a, plan_at_least, plan_b, plan_claim, plan_computer, plan_direction, plan_hours, plan_margin, plan_metric, plan_per_seed, plan_seeds, plan_setup, plan_shown


@app.cell(hide_code=True)
def plan_value(plan_a, plan_at_least, plan_b, plan_claim, plan_computer, plan_direction, plan_hours, plan_margin, plan_metric, plan_per_seed, plan_seeds, plan_setup, plan_shown):
    _choice = C.saved_choice() or {}
    plan = {"paper": _choice.get("id"), "needs": _choice.get("needs"), "claim": plan_claim.value, "shown_in": plan_shown.value,
            "metric": plan_metric.value, "settings": [plan_a.value, plan_b.value], "setup": plan_setup.value,
            "direction": plan_direction.value, "at_least": plan_at_least.value, "seeds": plan_seeds.value,
            "hours_per_seed": plan_per_seed.value, "margin": plan_margin.value, "computer": plan_computer.value, "hours": plan_hours.value}
    return (plan,)


@app.cell(hide_code=True)
def check_plan(plan):
    result_plan = L.check("plan", plan)
    result_plan
    return (result_plan,)


@app.cell(hide_code=True)
def hints_plan():
    L.hints(
        "plan",
        [
            "Start from the card of your paper in [Your reproduction 1](/course/open/repro-own-choose?lang=en): its claim names the two settings, and *You measure* the metric.",
            "For *at least*: how much do two seeds of the same setting differ in your metric? Go above that, and no higher than the smallest effect you would still call the paper's claim.",
            "For the hours: time one short run, for a paper of the list in the lesson it builds on, multiply by the steps you plan, and add both settings.",
        ],
        why=C.course_plan_text(C.saved_choice(), L.lang),
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your plan says

    Your rule on three made-up outcomes of your own plan: one where the gains are well past what counts, one where they fall short of it by as much, and the first one stopped a seed early.
    """)
    return


@app.cell(hide_code=True)
def demo_plan(plan, result_verdict, result_plan):
    result_verdict, result_plan  # run again when exercise 1 saves your rule
    _shown = C.demo(plan)
    if _shown is None:
        _out = mo.md("*Once a reviewer would send nothing back, your rule shows here what it says.*")
    else:
        _whose = {"yours": "the verdict of your `own/plan.py`", "course": "the course's verdict, until exercise 1 saves yours"}[_shown[1]]
        _out = mo.vstack([mo.Html(C.outcomes_table(plan, _shown[0], {"holds": "well past what counts", "fails": "short of it by as much",
                                                                     "early": "the first, a seed early", "outcome": "made-up outcome",
                                                                     "gains": "gain per seed", "verdict": "verdict"}, L.lang)),
                          mo.md(f"*With {_whose}.*")])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. Why that number

    Your plan's *at least* is the number that can make your reproduction fail. Say where it comes from.
    """)
    return


@app.cell(hide_code=True)
def explain_why_number():
    explain_why_number_cell = L.explain(
        "why-number",
        question="Why did you choose the number of *at least* in your plan? And what would choosing it after your runs do?",
        model_answer="**The number:** lower by at least 0.5 points of test error. If two seeds of the same network differ by about 0.3 points, a smaller number lets noise support the claim; the paper calls a clearly lower test error its result, and half a point on an error of about 11% is the least I would still call that. **Afterwards:** with my runs in front of me, I could pick a number just below the median I got, and every outcome would support the claim. Written down now, my rule can fail, and that makes it mean something when it does not.",
        points=[
            "you give your number, in the units of your metric",
            "you compare it with how much two seeds differ",
            "you compare it with what the paper calls its effect",
            "you say what choosing it after the runs would do",
        ],
    )
    explain_why_number_cell
    return (explain_why_number_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The plan of [Guided reproduction 1](/course/open/repro-guided-claim?lang=en), now with numbers you choose; the gain per seed of [Guided reproduction 6](/course/open/repro-guided-compare?lang=en), which compared runs that share a seed; and the noise between seeds of [One run is not a result](/course/open/res-seeds-variance?lang=en). In the next lesson, *Your reproduction 3: build it*, you build the method and the data of your paper, for the two settings of your plan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_verdict, result_plan, explain_why_number_cell):
    L.wrap_up(result_verdict, result_plan, explain_why_number_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
