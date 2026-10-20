# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-compare. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-compare", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 6: the ablation](/course/open/repro-own-ablation?lang=en) and [Is the difference real?](/course/open/res-significance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_keeps():
    quick_keeps = L.choice(
        "quick-keeps",
        question="An ablation keeps 120% of the claim's gain. What does the rule of one half say, and what does the share mean?",
        options={
            "broken": "That the check is broken: no ablation can keep more than the whole gain",
            "right": "That it does not depend on the part: without it, the method does even a little better",
            "depends": "That the result depends on the part, as 120% lies far from the 100% of the claim's own gain",
        },
        correct="right",
        why={
            "broken": "A share above 1 is a median gain above the claim's: without the part, the method does better than with it. That can happen, and it is worth saying. See [Your reproduction 6](/course/open/repro-own-ablation?lang=en).",
            "depends": "The rule looks at less than half: 120% is more than half, so the result does not depend on the part. See [Your reproduction 6](/course/open/repro-own-ablation?lang=en).",
            "right": "At least half is left, so by the rule the part is not needed; that it does better without it is a finding of its own.",
        },
    )
    quick_keeps
    return (quick_keeps,)


@app.cell(hide_code=True)
def quick_check_paired():
    quick_paired = L.choice(
        "quick-paired",
        question="Two settings ran with the same seeds. Why does a paired bootstrap give a narrower interval than one that resamples each setting on its own?",
        options={
            "twice": "Pairs give it twice as many numbers to draw from: two runs for every seed",
            "boot": "A paired bootstrap draws fewer resamples",
            "right": "What a seed does to both runs drops out of their difference",
        },
        correct="right",
        why={
            "twice": "Pairs give it half as many numbers: one difference for each seed, not two runs. See [Is the difference real?](/course/open/res-significance?lang=en).",
            "boot": "Both draw as many resamples as you ask for. See [Is the difference real?](/course/open/res-significance?lang=en).",
            "right": "A lucky seed lowers both runs alike; their difference keeps what the setting does, and the spread between seeds falls away.",
        },
    )
    quick_paired
    return (quick_paired,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your main result says what your rule says about your runs. It does not yet say whether they match the paper. The paper has its own number, from its own setup, often one run. When yours differs, there are two questions: is the difference more than your seeds explain, and if it is, where does it come from? A difference is no failure: finding its reason is often the most useful thing a reproduction gives back.

        **An everyday example.** Your bread rises less than the photo in the recipe. One loaf could be bad luck; when loaf after loaf falls short, it is not. Then you look at the flour, the warmth of the kitchen and the time it stood.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           6, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_real():
    guess_real = L.guess(
        "real",
        question="In the course's main result, dropout lowers the test error by 0.4, 0.7 and 1.2 points for its three seeds, whose error without dropout is 12.9% to 13.6%. The paper: with dropout, the error on MNIST goes from 1.60% to 1.35%, a gain of 0.25 points. In points, the course gains more. And as a share of the error without dropout?",
        options={
            "right": "Less: its gains are small next to its large error",
            "more": "More there too: a larger drop of the error is a larger share of it",
            "same": "About as much: the same method does about the same on both",
        },
        reveal="**Less.** The paper's 0.25 points are 16% of its 1.60%; the course's 0.4 to 1.2 points are only 3% to 9% of its 12.9% to 13.6%, an error about eight times the paper's. In points the course gains more, as a share less: the same runs differ from the paper in opposite directions. Whether three seeds are enough to say so, *The idea* shows.",
    )
    guess_real
    return (guess_real,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Your gain against the paper's.** The bootstrap of [Is the difference real?](/course/open/res-significance?lang=en) gives the mean gain of your seeds and a 95% interval: the means of many resamples of your gains, without the lowest and the highest 2.5%. The paper gives one number. When it lies outside your interval, your seeds do not explain the difference: it is real. With three seeds, though, the interval is only the range of your seeds, which [Is the difference real?](/course/open/res-significance?lang=en) called much too narrow to trust: then how far outside the number lies counts as well.

    **In the unit of the claim.** A gain can be counted in points of the metric, or as a share of the metric without the method. The paper's 1.60% to 1.35% is 0.25 points, or 16% of its error. When the data differ, the two can tell opposite stories: say which unit the claim is about, and give both.

    **Then look for the reason.** Line up your setup with the paper's: the data, the network, the training and the measurement; what the paper says, what it does not say, and what you did differently on purpose, as in Guided reproduction 6. A reason is a guess until a run tests it, such as an ablation.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | relative gain | the gain as a share of the metric without the method, in % |

    The picture: the course's main result for the dropout paper at 12,800 steps against the paper's own numbers, in two panels: in points of test error, and as a share of the error without dropout. Each dot is the gain of one seed, the grey band its interval, the ring its mean; the dashed line is the paper's. Move the slider to use one, two or three seeds.
    """)
    return


@app.cell(hide_code=True)
def explore_seeds_cell():
    _labels = {"points": "Gain in points of test error", "share": "Gain as a share of the error without dropout",
               "x_points": "points", "x_share": "%", "paper": "paper",
               "alt": "The gains of the course's three seeds at 12,800 steps against the paper's. In points: 0.4, 0.7 and 1.2, an interval from 0.4 to 1.2; the paper's 0.25 lies below it. As a share of the error without dropout: 3.1%, 5.1% and 8.8%, an interval from 3.1% to 8.8%; the paper's 15.6% lies above it."}
    _c, _o = C.course_results()

    def _frame(k):
        _k = int(k)
        _p = C.interval(C.gains(_c, _o)[:_k])
        _s = C.interval(C.shares(_c, _o)[:_k])
        _pg, _ps = C.paper_gain(C.PAPER), C.paper_share(C.PAPER)
        _seeds = "1 seed" if _k == 1 else f"{_k} seeds"
        _say = (f"{_seeds}. In points: a mean of {C.short(round(_p[0], 2), L.lang)}, interval {C.short(round(_p[1], 2), L.lang)} to {C.short(round(_p[2], 2), L.lang)}; "
                f"the paper's {C.short(_pg, L.lang)} lies {'outside' if not _p[1] <= _pg <= _p[2] else 'inside'}. "
                f"As a share: {C.short(round(_s[0], 1), L.lang)}%, interval {C.short(round(_s[1], 1), L.lang)}% to {C.short(round(_s[2], 1), L.lang)}%; "
                f"the paper's {C.short(round(_ps, 1), L.lang)}% lies {'outside' if not _s[1] <= _ps <= _s[2] else 'inside'}.")
        return C.compare_picture(_k, _labels), _say

    explore_seeds = L.explore("seeds", "seeds", _frame, start=3, low=1, high=3, step=1, role="",
                              task="With how many seeds is the interval more than a point? And on which side of it does the paper's gain lie, in each unit?")
    explore_seeds
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        With one seed, the interval is a point: a bootstrap of one gain cannot see any spread, so it looks sure while it knows nothing. With three, it runs from the lowest seed to the highest: all three draws land on the lowest seed in 1 of 27 resamples, more than the 2.5% the interval cuts off. That is the interval [Is the difference real?](/course/open/res-significance?lang=en) called much too narrow to trust, so look at how far outside it the paper's number lies. As a share of the error, the paper's 15.6% lies far above the highest seed, 8.8%, further than the seeds spread: there the course's gain is clearly smaller. In points, the paper's 0.25 lies just below the lowest seed, 0.4, while the seeds spread over 0.8 points: the course's gain looks larger, but three seeds do not show that well. Dropout lowers the test error on both datasets, which is what the claim says; how much does not carry over from MNIST, with an error of 1.60%, to Fashion-MNIST, with about 13.4%.
        """),
        L.watch_out("With three seeds, a paper's number just outside your interval is weak evidence; one further outside than your seeds spread is strong, and more seeds make the interval worth more. The paper's number comes without a spread over seeds: its own noise is not in your interval. And the paper's 1.60% and 1.35% belong to two different networks: the best it names without dropout, and one with dropout that it improves on further, to below 1%. Say what each number is of."),
        mo.md(r"""
        ### Recipe

        To compare your reproduction with the paper:

        1. Find the paper's own numbers, with the method and without, and write down where they stand and what they are of.
        2. Give the mean gain of your seeds with its bootstrap interval, at your budget.
        3. Say whether the paper's gain lies outside the interval, and how far: further outside than your seeds spread, the difference is real; just outside, with few seeds, it is weak.
        4. Do the same as a share of the metric without the method, and say which unit the claim is about.
        5. Line up the setups, and name the difference you think explains it; test it with a run when you can.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The paper's gain and the course's, in points and as a share of the error without dropout. Run the cell.
    """)
    return


@app.cell
def example_units():
    _with, _without = 1.35, 1.60       # the paper, its results on MNIST
    _gain = round(_without - _with, 2)
    print("paper:", _gain, "points,", round(100 * _gain / _without, 1), "%")
    _claim = [12.5, 12.9, 12.4]        # the course: with dropout
    _other = [12.9, 13.6, 13.6]        # without
    _mine = [round(o - c, 1) for c, o in zip(_claim, _other)]
    _share = [round(100 * g / o, 1) for g, o in zip(_mine, _other)]
    print("course:", _mine, "points,", _share, "%")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The paper's 0.25 points are 15.6% of its 1.60; the course's gains are larger in points and smaller as a share, because its error without dropout is more than eight times as large. A gain as a share divides by the metric of the other setting with the same seed, so it stays paired.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your gain against the paper's

        Complete `vs_paper(gains, paper_gain, n_boot, seed)`: the comparison of your kit. It gives the mean of your gains, the 0.025 and the 0.975 quantile of the means of `n_boot` resamples of them, and `real`: whether the paper's gain lies outside that interval. The check tries four gains against a paper's 0.3, the course's gains in points and as a share, a paper whose gain lies inside, five seeds, and one.
        """),
        L.task("Complete `vs_paper`: the resamples with replacement, the two quantiles, and `real`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vs():
    box_vs = L.code_box("vs-paper", lines=14)
    box_vs
    return (box_vs,)


@app.cell(hide_code=True)
def check_vs(box_vs):
    result_vs = L.check("vs-paper", box_vs.value)
    result_vs
    return (result_vs,)


@app.cell(hide_code=True)
def hints_vs():
    L.hints(
        "vs-paper",
        [
            "The loop of `paired_bootstrap` of [Is the difference real?](/course/open/res-significance?lang=en), on your gains: they are the differences of the pairs already.",
            "`np.quantile(means, [0.025, 0.975])` gives both ends; then `real` is true when `paper_gain` is below the one or above the other.",
        ],
        why="The resamples show how much the mean of a few seeds can move; a gain of the paper outside that range is one your seeds would not give.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_numbers():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The paper's numbers

        Look up the paper's own numbers for your claim, where your plan says the paper shows it, or in the text or table that gives that result in numbers: the metric with the method and without it, in the unit of your metric. Write down where they stand, and what they are of: the same network twice, or the best of each kind. The check saves them in your kit, as `work/projects/own/paper_numbers.json`.
        """),
        L.task("Fill in the paper's two numbers, and in a sentence where they stand and what they are of."),
    ])
    return


@app.cell(hide_code=True)
def numbers_inputs():
    _old = C.saved_numbers() or {}
    numbers_with = mo.ui.number(start=-1e9, stop=1e9, step=0.01, value=_old.get("with") or 0, label="The paper's number with the method")
    numbers_without = mo.ui.number(start=-1e9, stop=1e9, step=0.01, value=_old.get("without") or 0, label="The paper's number without it")
    numbers_where = mo.ui.text_area(label="Where in the paper, and what the numbers are of", full_width=True, rows=3, value=_old.get("where", ""))
    _plan = C.P.saved_plan()
    _head = (mo.md(f"*Your plan: {_plan.get('metric')}; the paper shows it in {_plan.get('shown_in')}.*") if _plan
             else mo.md("*No plan yet: write it in [Your reproduction 2](/course/open/repro-own-claim?lang=en) first.*"))
    mo.vstack([_head, numbers_with, numbers_without, numbers_where])
    return numbers_where, numbers_with, numbers_without


@app.cell(hide_code=True)
def check_numbers(numbers_where, numbers_with, numbers_without):
    result_numbers = L.check("paper-numbers", {"with": numbers_with.value, "without": numbers_without.value, "where": numbers_where.value})
    result_numbers
    return (result_numbers,)


@app.cell(hide_code=True)
def hints_numbers():
    L.hints(
        "paper-numbers",
        [
            "Start where your plan says the paper shows the claim; the text around a table often says in words what the table shows in numbers.",
            "Check that both numbers are in the unit of your metric, such as the test error in %, and say whether they belong to one network or to two.",
        ],
        why="For the dropout paper: **1.35 with dropout, 1.60 without.** " + C.PAPER_WHERE[L.lang],
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your main result at your budget, against the paper's numbers in your kit, read with your `vs_paper`: in points and as a share of the other setting.
    """)
    return


@app.cell(hide_code=True)
def demo_compare(result_vs, result_numbers):
    result_vs, result_numbers  # run again when an exercise saves a part of the kit
    _r = C.my_comparison()
    if _r is None:
        _out = mo.md("*Your comparison shows here once your kit has your plan, your budget, the runs of your main result at it, your `vs_paper` of exercise 1 and the paper's numbers of exercise 2.*")
    elif _r == "error":
        _out = mo.md("*Your `own/compare.py` stops with an error, or gives something other than its dictionary: check exercise 1 again.*")
    else:
        _row = lambda _name, _x, _p, _d: (f"| {_name} | {C.signed(round(_x['mean'], _d), L.lang)} | {C.signed(round(_x['low'], _d), L.lang)} to {C.signed(round(_x['high'], _d), L.lang)} | "
                                           f"{C.signed(round(_p, _d), L.lang)} | {'real' if _x['real'] else 'not shown'} |")
        _out = mo.md(f"""
| unit | your mean | your interval | the paper's | the difference |
|---|---|---|---|---|
{_row("points", _r["points"], _r["paper_gain"], 2)}
{_row("share, %", _r["share"], _r["paper_share"], 1) if _r["share"] is not None else "| share, % | | | | none: without the method, the metric is not above 0 |"}
""")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. Why it differs

    Your comparison is in front of you. Say what it shows, and what you think explains it.
    """)
    return


@app.cell(hide_code=True)
def explain_why_differs():
    explain_why_differs_cell = L.explain(
        "why-differs",
        question="Is the difference between your result and the paper's real, and in which unit? What do you think explains it, and how could a run test that?",
        model_answer="**Clear as a share, weak in points:** as a share of the error, my gain runs from 3% to 9%, and the paper's 16% lies far above it. In points, my mean gain is 0.77, with an interval from 0.4 to 1.2, and the paper's 0.25 lies just below it; but with three seeds that interval is only the range of my seeds, much too narrow to trust. The claim, a clearly lower test error, holds in both. **My best guess** for the size: my data. Fashion-MNIST is harder, so my error is about eight times the paper's, and with 6000 training pictures instead of 60,000, a network can overfit more, which gives dropout more to do in points. Also, the paper's two numbers belong to two networks, and they come without a spread over seeds. **A test:** the same experiment on the course's 6000 digits of QMNIST, pictures like those of MNIST: if my gain in points falls toward the paper's, it is the kind of data, not the number of pictures. More seeds would show whether my gain in points really differs.",
        points=[
            "you say whether the paper's gain lies outside your interval, and how far, in each unit",
            "you name a difference in the setup that could explain it",
            "you say how a run could test that",
        ],
    )
    explain_why_differs_cell
    return (explain_why_differs_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The bootstrap of [Is the difference real?](/course/open/res-significance?lang=en) and the line-up of setups of [Guided reproduction 6](/course/open/repro-guided-compare?lang=en), now for your own paper; and the main result of [Your reproduction 5](/course/open/repro-own-figure?lang=en). In the next lesson, *Your reproduction 8: report and share*, you write it all down, publish your code and results, and tell the authors what you found.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_vs, result_numbers, explain_why_differs_cell):
    L.wrap_up(result_vs, result_numbers, explain_why_differs_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
