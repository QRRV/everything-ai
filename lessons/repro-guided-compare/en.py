# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-compare. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-compare", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Is the difference real?](/course/open/res-significance?lang=en) and [Guided reproduction 5: the ablation](/course/open/repro-guided-ablation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_interval():
    quick_interval = L.choice(
        "quick-interval",
        question="Five seeds, paired: the 95% interval of the mean difference runs from −0.05 to 0.40, with the mean at 0.18. What does your report say?",
        options={
            "zero": "That there is no difference between the two settings",
            "right": "No clear difference: the interval holds 0, so it could be noise",
            "mean": "A difference of 0.18, since that is the mean of the five pairs",
        },
        correct="right",
        why={
            "zero": "An interval that holds 0 does not show that there is no difference: it holds 0.3 too. It says that five seeds cannot tell. See [Is the difference real?](/course/open/res-significance?lang=en).",
            "mean": "The mean alone hides how unsure it is: with these five pairs, the true difference could just as well be 0. See [Is the difference real?](/course/open/res-significance?lang=en).",
            "right": "An interval that holds 0 means the data cannot tell a difference from noise; more seeds would make the interval narrower.",
        },
    )
    quick_interval
    return (quick_interval,)


@app.cell(hide_code=True)
def quick_check_one_seed():
    _times = C.A.read(C.A.run(1.0, 0.5))["times"]   # with weight decay at 50%: how many times as late as training
    quick_one_seed = L.choice(
        "quick-one-seed",
        question=f"In your ablation, with weight decay at 50%, validation reached 99% {C.number(_times, 1, L.lang)} times as late as training, on one seed; your plan calls {C.DELAY} times or more \"grokked\". What may your report say about 50%?",
        options={
            "grok": f"That it grokked at 50%: {C.number(_times, 1, L.lang)} is as good as {C.DELAY}",
            "none": "That from 50% on, weight decay takes the delay away",
            "right": f"That one seed fell just under {C.DELAY} times as late, and another could land on either side",
        },
        correct="right",
        why={
            "grok": f"Your plan drew the line at {C.DELAY} before any run, so that no result could move it afterwards. {C.number(_times, 1, L.lang)} is under it, so the verdict is \"no delay\". See [Guided reproduction 1](/course/open/repro-guided-claim?lang=en).",
            "none": f"One seed per combination cannot tell a close call from a rule: at 50% it came to {C.number(_times, 1, L.lang)}, close to the line. See [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en).",
            "right": "With one seed per combination, a close call is just that: only more seeds show which side it usually falls on.",
        },
    )
    quick_one_seed
    return (quick_one_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your ablation found that weight decay brings the jump sooner: on one seed per combination. The paper says that weight decay "improves generalization the most": on another task, the product in the group $S_5$, measured as how much data it takes. Before your report can say that your reproduction agrees with the paper, two questions. Is your difference real, or one roll of the dice? Run it over seeds, and compare pairs. And where your number and the paper's differ, why? Line up the setups, and find what the paper does not say.

        **An everyday example.** You and a friend bake from the same recipe, and your loaf is flatter. Before you say the recipe fails, bake a few more, on different days: is it every time, or was it that one day? And then compare what you did: the same flour, the same oven? The recipe never said how long to knead.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           5, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_seeds():
    _f = C.factors(len(C.SEEDS))
    _got, _none = [_x for _x in _f if _x is not None], [_s for _s, _x in enumerate(_f) if _x is None]
    guess_seeds = L.guess(
        "five-seeds",
        question=f"In seed 0 at {round(100 * C.WIDE)}% of the table, validation reached 99% {C.number(C.factors(1)[0], 1, L.lang)} times as late without weight decay as with it. Over five seeds, what do you think?",
        options={
            "half": "Later without weight decay in about half of the seeds, sooner in the others",
            "all": "Later without weight decay in every seed",
            "most": "Sooner without weight decay in most seeds: seed 0 was the odd one",
        },
        reveal=f"**Later in every seed:** in {len(_got)} seeds from {C.number(min(_got), 1, L.lang)} to {C.number(max(_got), 1, L.lang)} times as late, and in seed {', '.join(str(_s) for _s in _none)}, without weight decay, validation did not reach 99% within the 12,000 steps. How sure that makes the mean, *The idea* shows.",
    )
    guess_seeds
    return (guess_seeds,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## The idea

    **Pair by seed.** In your kit the seed fixes the split, the starting weights and the batches. Two runs with the same seed and another weight decay differ in that one setting only, so compare them as a pair: what the seed does to both, the pair takes out.

    **Compare factors, as logs.** The moment $v$ at which validation reaches 99% can differ by factors of 10, so compare how many times as late, not how many steps more. For each pair, $\log_{{10}} v_a - \log_{{10}} v_b$: 0 when both are as fast, 1 when $a$ is 10 times as late. The paired bootstrap of [Is the difference real?](/course/open/res-significance?lang=en) gives the mean of those differences and a 95% interval; $10$ to the power of the mean is how many times as late, typically.

    Below are the course's runs at {round(100 * C.WIDE)}% of the table, with and without weight decay, for seeds 0 to 4: a dot for each seed, how many times as late validation reaches 99% without weight decay, or an arrow up from the least it can be when it does not get there within 12,000 steps. The dashed line is the mean, the grey band its interval, and the thin line 1, where both are as fast. Move the slider to set how many seeds the comparison uses.
    """)
    return


@app.cell(hide_code=True)
def explore_seeds_cell():
    def _frame(n):
        _n = int(n)
        _r = C.compare(C.runs_of(0.0, C.WIDE)[:_n], C.runs_of(1.0, C.WIDE)[:_n])
        _say = (f"{C.number(10 ** _r['mean'], 1, L.lang)} times as late, with an interval from {C.number(10 ** _r['low'], 1, L.lang)} to {C.number(10 ** _r['high'], 1, L.lang)}")
        _pairs = f"{_r['pairs']} {'pair' if _r['pairs'] == 1 else 'pairs'}" + (f", {_r['left_out']} left out" if _r["left_out"] else "")
        _lost = "not reached"
        return (C.factors_picture(_n, {"title": f"Without weight decay against with it, {_n} {'seed' if _n == 1 else 'seeds'}", "x": "seed", "y": "times as late",
                                       "band": "95% interval", "none": _lost,
                                       "alt": f"For seeds 0 to {_n - 1}, how many times as late validation reaches 99% without weight decay as with it, on a log axis: {', '.join(C.number(_x, 1, L.lang) if _x is not None else _lost for _x in C.factors(_n))}. The mean is {_say}."}),
                f"With {_n} {'seed' if _n == 1 else 'seeds'} ({_pairs}): without weight decay, validation reaches 99% {_say}.")

    explore_seeds = L.explore("seeds", "seeds", _frame, start=5, low=1, high=5, step=1, role="",
                              task="How does the interval change as you add seeds? And what does the last seed add?")
    explore_seeds
    return


@app.cell(hide_code=True)
def text_idea_after():
    _two, _four, _five = (C.compare(C.runs_of(0.0, C.WIDE)[:_n], C.runs_of(1.0, C.WIDE)[:_n]) for _n in (2, 4, 5))
    _lost = [_s for _s, _x in enumerate(C.factors(len(C.SEEDS))) if _x is None][0]   # the seed that is left out
    mo.vstack([
        mo.md(rf"""
        With one seed the interval is a line: a bootstrap of one pair cannot see any spread, so it looks sure while it knows nothing. With two it runs from one pair to the other, {C.number(10 ** _two['low'], 1, L.lang)} to {C.number(10 ** _two['high'], 1, L.lang)}: a bootstrap can only draw from the pairs it has, so with a few it is too narrow, as [Is the difference real?](/course/open/res-significance?lang=en) warned. With four it runs from {C.number(10 ** _four['low'], 1, L.lang)} to {C.number(10 ** _four['high'], 1, L.lang)}, and every pair lies clear of 1: without weight decay, validation reaches 99% later, in a way the seeds do not explain.

        Seed {_lost} adds no pair. Without weight decay, validation did not reach 99% within 12,000 steps, so `compare` leaves it out, and counts it. It is the slowest seed of all, more than {C.number(C.at_least(_lost), 1, L.lang)} times as late, so leaving it out makes the mean of {C.number(10 ** _five['mean'], 1, L.lang)} too low, not too high. Your report says that: {_five['pairs']} pairs, {_five['left_out']} left out, and why.

        **Then line up the setups.** The paper's claim is about $S_5$ and how much data it takes; yours is about addition at {round(100 * C.WIDE)}% and how many steps it takes. A number to match needs the same setup: every detail the paper gives, and the ones it does not give found in its code or asked of its authors.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **paired by seed** | two runs with the same seed and one setting different, compared as a pair | `by_seed[log["seed"]]` |
        | $\log_{{10}} v_a - \log_{{10}} v_b$ | how many factors of 10 later validation reaches 99% in $a$ than in $b$ | `math.log10(va) - math.log10(vb)` |
        | **undocumented detail** | a choice the paper made without writing it down, which can change the result | |
        """),
        L.watch_out("A pair where one run does not reach 99% within the budget has no factor. Leaving it out is honest only when you say so: the result counts it, and your report names it. If many pairs fall out, the comparison is about the runs that were fast enough, not about all of them."),
        mo.md(r"""
        ### Recipe

        To compare your reproduction with a claim of the paper:

        1. Run both settings over several seeds, and pair the runs by seed.
        2. For each pair, take the difference of the logs of the moments; count the pairs that do not reach them.
        3. Give the mean with its paired bootstrap interval, and say how many times as late that is.
        4. Line up your setup with the paper's: what it says, what it does not say, and what you do differently on purpose.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 2 and 3 of the recipe by hand, with seeds 0 and 1 at 90%, and seed 4, which has no pair. Run the cell.
    """)
    return


@app.cell
def example_pairs():
    import math

    _without = {0: 3548, 1: 5012, 4: None}   # v without weight decay, per seed; None: not within 12,000 steps
    _with = {0: 1259, 1: 1000, 4: 891}       # v with weight decay
    _diffs, _left_out = [], 0
    for _s in _without:
        if _without[_s] is None or _with[_s] is None:   # 2. no factor: the pair is left out, and counted
            _left_out = _left_out + 1
        else:
            _diffs.append(math.log10(_without[_s]) - math.log10(_with[_s]))   # 2. a log difference per pair
    print("log differences:", [round(_d, 2) for _d in _diffs], "left out:", _left_out)
    print("times as late, typically:", round(10 ** (sum(_diffs) / len(_diffs)), 2))   # 3. the mean, as a factor
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Seed 0 gives 0.45, a factor of 2.8, and seed 1 gives 0.70, a factor of 5.0; seed 4 has no factor, so it is left out, and counted. The mean of the two, 0.575, is a factor of 3.76: not the mean of 2.8 and 5.0, which is 3.9, because a mean of logs is the factor in the middle on a log axis. Two pairs make no interval worth having; the block above shows what more seeds do.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The comparison

        Complete `compare(runs_a, runs_b, n_boot, seed)`: for each run of `runs_a` whose seed is in `by_seed`, the run of `runs_b` with that seed, and in both the first step at which validation reaches `THRESHOLD`, with `first_step` of your plan. A pair where either is `None` is left out, and counted in `left_out`; for the others, `math.log10` of each step goes into `a` and `b`. `paired_bootstrap` and the rest are given. The check makes small runs whose validation reaches 99% at a step you can read, as in `run(0, 1000)`, and compares the results. When it is right, it is saved in your project as `repro/compare.py`.
        """),
        L.task("Complete `compare`: pair by seed, read v in both runs, count a pair without it, and keep the logs of the others."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compare():
    box_compare = L.code_box("compare", lines=30)
    box_compare
    return (box_compare,)


@app.cell(hide_code=True)
def check_compare(box_compare):
    result_compare = L.check("compare", box_compare.value)
    result_compare
    return (result_compare,)


@app.cell(hide_code=True)
def hints_compare():
    L.hints(
        "compare",
        [
            "`if log[\"seed\"] in by_seed:` finds the pair; `by_seed[log[\"seed\"]]` is the run of `runs_b` with that seed. `first_step(log[\"step\"], log[\"val\"], THRESHOLD)` gives `v` of a run.",
            "When `va is None or vb is None`, add 1 to `left_out`; otherwise `a.append(math.log10(va))` and `b.append(math.log10(vb))`.",
        ],
        why="Pairing by seed, not by place in the list, keeps the comparison right when one setting has a seed more or another order.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_details():
    mo.md(r"""
    ### Exercise 2. What does the paper say?

    Put each detail of the setup in a group: what the paper says, in the text or its appendix; what it does not say, so that you would have to look in its code or ask its authors; and what your reproduction does differently on purpose. Keep the paper's text open beside you: section 3 and Appendix A.1.
    """)
    return


@app.cell(hide_code=True)
def exercise_details():
    sort_details = L.sort("details")
    sort_details
    return (sort_details,)


@app.cell(hide_code=True)
def check_details(sort_details):
    result_details = L.check("details", sort_details)
    result_details
    return (result_details,)


@app.cell(hide_code=True)
def text_exercise_agrees():
    mo.md(r"""
    ### Exercise 3. Does it agree?
    """)
    return


@app.cell(hide_code=True)
def choice_agrees_cell():
    _r = C.compare(C.runs_of(0.0, C.WIDE), C.runs_of(1.0, C.WIDE))
    choice_agrees = L.choice(
        "agrees",
        question=f"Your compare over five seeds at {round(100 * C.WIDE)}% gives, without weight decay against with it, {_r['pairs']} pairs and {_r['left_out']} left out, a mean of {C.number(_r['mean'], 2, L.lang)} and an interval from {C.number(_r['low'], 2, L.lang)} to {C.number(_r['high'], 2, L.lang)}. The paper says weight decay \"improves generalization the most\", on S₅, measured as how much data it takes. What does your report say?",
        options={
            "right": "It agrees with the paper in direction, on another task and measure",
            "match": f"It reproduces the paper's number: weight decay makes validation {C.number(10 ** _r['mean'], 1, L.lang)} times as fast",
            "differ": "It does not agree, since the paper measured on another task",
        },
        correct="right",
        why={
            "match": "The paper gives no such number for addition at 90%: its claim is about how much data S₅ takes. Your factor is your own result; the paper's claim is the direction it points in.",
            "differ": "Another task makes the comparison weaker, not wrong: both say weight decay helps generalization, and your interval is clear of 0. Say what differs, and that the direction agrees.",
            "right": f"Without weight decay, validation reaches 99% {C.number(10 ** _r['mean'], 1, L.lang)} times as late, with an interval from {C.number(10 ** _r['low'], 1, L.lang)} to {C.number(10 ** _r['high'], 1, L.lang)}, over {_r['pairs']} of the {len(C.SEEDS)} seeds; in the other one it did not get there within 12,000 steps. That is the same direction as the paper's claim, on addition and in steps, and it is the sentence for your report.",
        },
    )
    choice_agrees
    return (choice_agrees,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your `compare` can hold your reproduction against a reference, seed by seed. The cell below compares your runs of [Guided reproduction 4](/course/open/repro-guided-figure?lang=en), in `work/projects/repro/runs.jsonl`, with the course's runs of the same seeds: if your kit does what the course's does, validation reaches 99% at about the same step, and the mean is close to 0.
    """)
    return


@app.cell(hide_code=True)
def demo_compare(result_compare):
    result_compare  # run again when the exercise saves the comparison
    _r = C.demo()
    if _r is None:
        _out = mo.md("*Finish exercise 1 first: then your `compare` is in your project, and this cell uses it.*")
    elif _r == "runs":
        _out = mo.md("*There are no runs of yours yet: train them with the button of [Guided reproduction 4](/course/open/repro-guided-figure?lang=en), then come back.*")
    elif _r == "compare":
        _out = mo.md("*Your `repro/compare.py` stops with an error. Did you change it after its check? Delete it, and check its exercise again.*")
    elif not _r["pairs"]:
        _out = mo.md(f"*None of your runs has a pair with the course's that reaches 99% in both ({_r['left_out']} left out).*")
    else:
        _same = max(abs(_r["low"]), abs(_r["high"])) < 0.06   # every pair within one logged step: 20 per factor of 10 are 0.05 apart
        _verdict = ("Your reproduction matches the course's, seed by seed." if _same
                    else "In at least one seed, your runs differ from the course's by more than a logged step: a difference to hunt for, in the device, the number of threads, or a module you changed after its check.")
        _stand = (" The course's version stood in for: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
        _out = mo.md(f"""
Your runs against the course's, seeds {', '.join(str(_s) for _s in _r['seeds'])}: {_r['pairs']} {'pair' if _r['pairs'] == 1 else 'pairs'}, {_r['left_out']} left out, a mean of **{L.number(_r['mean'], 3)}**, with an interval from {L.number(_r['low'], 3)} to {L.number(_r['high'], 3)}. {_verdict}{_stand}
""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The paired bootstrap of [Is the difference real?](/course/open/res-significance?lang=en), now over seeds of your reproduction; the grid of [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en), now repeated over seeds; and the appendix of the paper you read in [Guided reproduction 2](/course/open/repro-guided-build?lang=en), now as a checklist. Your kit now has `repro/compare.py`. In the next lesson, *Guided reproduction 7: write the report*, you write up what you found: the claim, the setup, the result, the differences, and what you could not check.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_compare, result_details, choice_agrees):
    L.wrap_up(result_compare, result_details, choice_agrees)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
