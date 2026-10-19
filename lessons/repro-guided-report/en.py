# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-report. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-report", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Guided reproduction 6: does it match the paper?](/course/open/repro-guided-compare?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_unsaid():
    quick_unsaid = L.choice(
        "quick-unsaid",
        question="Which of these does the grokking paper leave for you to choose?",
        options={
            "batch": "The batch size, when the training data is small",
            "lr": "The learning rate of AdamW",
            "right": "Which equations each seed trains on",
        },
        correct="right",
        why={
            "batch": "Appendix A.1.2 gives it: 512, or half the training data when that is smaller. You put it in your `SETTINGS` in [Guided reproduction 2](/course/open/repro-guided-build?lang=en).",
            "lr": "Appendix A.1.2 gives it: 10⁻³. You put it in your `SETTINGS` in [Guided reproduction 2](/course/open/repro-guided-build?lang=en).",
            "right": "The paper says the training data is a random fraction of the equations, but not how that draw is made; your kit makes it from the seed.",
        },
    )
    quick_unsaid
    return (quick_unsaid,)


@app.cell(hide_code=True)
def quick_check_two_pairs():
    _two = C.compare(C.runs_of(0.0, C.WIDE)[:2], C.runs_of(1.0, C.WIDE)[:2])
    quick_two_pairs = L.choice(
        "quick-two-pairs",
        question=f"With two pairs, the interval of your `compare` runs from {C.number(10 ** _two['low'], 1, L.lang)} to {C.number(10 ** _two['high'], 1, L.lang)} times as late. What is that interval worth?",
        options={
            "sure": "A lot: it lies clear of 1, so the difference is real",
            "right": "Little: a bootstrap of two pairs can only draw from those two",
            "none": "Nothing at all: an interval needs at least ten seeds to mean anything",
        },
        correct="right",
        why={
            "sure": "Clear of 1 with two pairs says little: the interval runs from one pair to the other, however far apart the seeds could fall. See [Guided reproduction 6](/course/open/repro-guided-compare?lang=en).",
            "none": "There is no such line: each pair makes the interval worth more, and with very few it is too narrow. See [Guided reproduction 6](/course/open/repro-guided-compare?lang=en).",
            "right": "With few pairs the interval looks surer than it is; more seeds make it worth having.",
        },
    )
    quick_two_pairs
    return (quick_two_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your kit has results now: the runs of the key figure, the grid of the ablation, a comparison over five seeds. They sit in files on your computer, where no one else can read them, let alone check them. A reproduction counts once someone else can read what you tested, how you tested it, what came out, and what you could not check, and could run it again. That is what a reproduction report is for. Journals such as ReScience C publish them, and a reproduction that did not hold is worth writing up too: it saves the next person from trying the same thing.

        **An everyday example.** A review of a recipe that helps: "Baked it twice. It rose as promised, but took 15 minutes longer in my oven. The recipe does not say the size of the tin; I used 24 cm. I did not try the gluten-free version." It says what was tried, how, what came out, where it differed, and what is still open.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           6, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_pick():
    _every = C.every()
    guess_pick = L.guess(
        "pick",
        question=f"Over all four pairs at {round(100 * C.WIDE)}%, your `compare` gives {C.number(_every, 1, L.lang)} times as late without weight decay. A report that gave only the one seed that suits the claim best: how many times as late would it say?",
        options={
            "double": f"Close to twice as much, about {round(C.reported(1))}",
            "same": "About the same: the seeds agree",
            "less": "Less, about 3",
        },
        reveal=f"**About {round(C.reported(1))}:** seed {C.best_first(1)[0]} alone gives {C.number(C.reported(1), 1, L.lang)}, close to twice the {C.number(_every, 1, L.lang)} of all four pairs. *The idea* shows what each choice gives.",
    )
    guess_pick
    return (guess_pick,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## The idea

    **Five parts, always in this order.** A reproduction report says what you tested, how, what came out, where it differs from the paper, and what you could not check:

    1. **Claim:** the claim in the paper's words, with where it stands, and the rule your plan fixed before any run.
    2. **Setup:** every setting, the data, the seeds, the budget, and what you do differently on purpose.
    3. **Results:** every run you planned, with its numbers, its intervals, and the runs that did not get there.
    4. **Differences:** where your setup and numbers differ from the paper's, and the details the paper does not give.
    5. **Not checked:** what you could not test, and what it would take.

    **Every run you planned.** Which runs go into the results is not yours to choose afterwards. Below are the five seeds of your comparison at {round(100 * C.WIDE)}%, as in the last lesson. Move the slider to let a report give only the seeds that suit the claim best: the filled dots are in the report, the open ones left out. The thick dashed line is the number that report would give, the thin dashed one the mean of all four pairs, and the solid line 1, where both are as fast.
    """)
    return


@app.cell(hide_code=True)
def explore_pick_cell():
    def _frame(k):
        _k = int(k)
        _seeds = ", ".join(str(_s) for _s in C.best_first(_k))
        _lost = "not reached"
        _every = C.every()
        return (C.chosen_picture(_k, {"title": "A report of the seed that suits the claim best" if _k == 1 else f"A report of the {_k} seeds that suit the claim best", "x": "seed",
                                      "y": "times as late", "none": _lost, "reported": "the report", "every": "all pairs",
                                      "alt": f"How many times as late validation reaches 99% without weight decay, seed by seed; the report gives seeds {_seeds}: {C.number(C.reported(_k), 1, L.lang)} times as late, against {C.number(_every, 1, L.lang)} over all pairs."}),
                f"A report of seed{'' if _k == 1 else 's'} {_seeds} says **{C.number(C.reported(_k), 1, L.lang)} times as late**; all four pairs give {C.number(_every, 1, L.lang)}.")

    explore_pick = L.explore("pick", "seeds in the report", _frame, start=1, low=1, high=4, step=1, role="",
                             task="How far does the number move as the report picks its seeds? And which seed gets no number, whatever the report picks?")
    explore_pick
    return


@app.cell(hide_code=True)
def text_idea_after():
    _every = C.every()
    _lost = [_s for _s, _x in enumerate(C.factors(len(C.SEEDS))) if _x is None][0]
    mo.vstack([
        mo.md(rf"""
        The one seed that suits the claim best gives {C.number(C.reported(1), 1, L.lang)}, and each seed you add pulls the number down, to {C.number(_every, 1, L.lang)} with all four pairs. A report that picks its runs can give any of those numbers, and a reader cannot tell which runs it left out: that is the **cherry-picking** of [Critical reading: claims versus evidence](/course/open/res-critical-reading?lang=en), now in your own report. So the results give every run the plan named, judged by the rule fixed before: all four pairs, and seed {_lost}, which has no number to choose. Without weight decay, its validation did not reach 99% within 12,000 steps. That is a **negative result**, and it belongs in the report as much as the others: with its budget, and with what it does to the mean.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **reproduction report** | a short write-up of a reproduction: the claim, the setup, the results, the differences with the paper, and what you could not check | `report(title, parts)` |
        | **negative result** | a result that shows something did not work or a claim did not hold; worth reporting, because it saves others from trying the same | `f"not within {{budget}}"` |
        | **Markdown** | plain text with a few marks that a page shows as layout: a line that starts with `# ` is a title, `## ` a heading, `*words*` italic, and <code>&#124; a &#124; b &#124;</code> a row of a table | `"## " + name` |
        """),
        L.watch_out("A negative result says what did not happen within what you tried. \"Not within 12,000 steps\" is a result; \"does not generalise\" is a claim the run cannot make. Write it with its budget, and say what more it would take."),
        mo.md(r"""
        ### Recipe

        To write a reproduction report:

        1. Write the claim in the paper's words, with where it stands, and the rule of your plan.
        2. Give the setup: every setting, the seeds, the budget, and what you do differently on purpose.
        3. Give every planned run in the results, with its numbers and intervals; a run that did not get there, with its budget.
        4. Name the differences with the paper, and the details it does not give.
        5. Say what you could not check, and give the verdict your results support.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 3 of the recipe by hand: a results table for two runs without weight decay at 90%, seed 0, which got there, and seed 4, which did not. Run the cell.
    """)
    return


@app.cell
def example_table():
    _budget = 12000
    _runs = [{"seed": 0, "t": 316, "v": 3548}, {"seed": 4, "t": 355, "v": None}]   # t and v of each run
    _rows = []
    for _run in _runs:
        _v = _run["v"]
        if _v is None:   # 3. a run that did not get there: in the table, with its budget
            _v = f"not within {_budget}"
        _rows.append([_run["seed"], _run["t"], _v])
    print("| seed | t | v |")
    for _row in _rows:
        print(f"| {_row[0]} | {_row[1]} | {_row[2]} |")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Both runs get a row. Seed 4's `None` becomes "not within 12000": a reader sees that the run stopped at its budget, not that validation would never get there, and not a `None` that means nothing to them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A row for every run

        Complete `result_rows(runs, budget)`: for each run of the log, in its order, a row `[seed, t, v, verdict]`. `t` and `v` come from `first_step` of your plan, with `THRESHOLD`; where one is `None`, the text `f"not within {budget}"` takes its place; the verdict is `verdict` of your plan. The check makes small runs whose training and validation reach 99% at a step you can read, as in `run(0, 10, 1000)`, and compares the rows. When it is right, it is saved in your project as `repro/results.py`.
        """),
        L.task("Complete `result_rows`: t and v of each run, the budget for a step not reached, and a row with the verdict."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_rows = L.code_box("result-rows", lines=22)
    box_rows
    return (box_rows,)


@app.cell(hide_code=True)
def check_rows(box_rows):
    result_rows_check = L.check("result-rows", box_rows.value)
    result_rows_check
    return (result_rows_check,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "result-rows",
        [
            "Follow the worked example: `first_step` once with the training accuracy and once with the validation accuracy, and for each that is `None`, the text in its place.",
        ],
        why="Every run gets a row, whatever came out: a table that drops the runs that did not get there tells only the part of the story that worked.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The template

        Complete `report(title, parts)`: the report in Markdown. First `"# "` and the title; then, for each name in `SECTIONS`, in that order, `"## "` and the name, and its text from the dictionary `parts`, or `"*Not written yet.*"` when `parts` has no text for it; all of them joined with an empty line between. The sections are in English, as most reproduction reports are, so that anyone can read yours. When it is right, it is saved in your project as `repro/report.py`.
        """),
        L.task("Complete `report`: every section of `SECTIONS`, in order, with its text or \"*Not written yet.*\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=18)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Loop over `SECTIONS`, not over `parts`, and for each name decide whether `parts` has a text for it, also when that text is empty.",
        ],
        why="With every section in its place, also the ones not written yet, a reader sees at once what the report does not say yet, and so do you.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_not_checked():
    mo.md(r"""
    ### Exercise 3. What you could not check
    """)
    return


@app.cell(hide_code=True)
def explain_not_checked():
    explain_not_checked_cell = L.explain(
        "not-checked",
        question="Your reproduction tested the grokking claim and weight decay on addition, with the course's kit. Name what it could not check about the paper's claims, and what it would take to check each.",
        model_answer="**The paper's own tasks and measure:** its weight decay claim is on the product in S₅, measured as how much data it takes, and its Figure 1 (left) on division; checking them takes those tasks, and runs over many fractions. **The paper's budgets and seeds:** 100,000 steps for most runs, 500,000 for the median over seven seeds of Figure 1 (center), and 1,000,000 for Figure 1 (left); that takes 8 to 80 times the steps per run, and more seeds. **What the paper does not say:** which parameters weight decay applies to, and how often validation was measured; that takes the paper's code, or a question to its authors. **The run that stopped:** seed 4 without weight decay at 90%, beyond 12,000 steps; that takes a longer run.",
        points=[
            "you name the paper's own task or measure (S₅, how much data it takes, division) as not checked",
            "you name the paper's budget or its number of seeds",
            "you name a detail the paper does not give, and that its code or its authors would settle it",
            "for each, you say what it would take to check it",
        ],
    )
    explain_not_checked_cell
    return (explain_not_checked_cell,)


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.md(r"""
    ### Exercise 4. The verdict
    """)
    return


@app.cell(hide_code=True)
def choice_verdict_cell():
    choice_verdict = L.choice(
        "verdict",
        question="Your figure grokked in all three seeds; your comparison found, over four of five seeds, that validation reaches 99% later without weight decay, and in the fifth not within the budget. Which verdict goes at the end of your report?",
        options={
            "full": "Reproduced: the delay and the effect of weight decay hold, so the paper's claims hold",
            "none": "Not reproduced: the numbers differ from the paper's",
            "right": "Partly reproduced: both hold on addition; the paper's own setup was not checked",
        },
        correct="right",
        why={
            "full": "What your runs show holds on addition, in steps, within 12,000 steps per run; the paper's claim about weight decay is on S₅, in data. A verdict says no more than the results show.",
            "none": "Other numbers on another task are no failure: for addition, the paper gives only the best validation accuracy after 100,000 steps (Figure 2, right), not the step at which it gets there, with or without weight decay. What your runs tested, held.",
            "right": "That is what the results support: what held, on what, and what is still open, which the section *Not checked* names.",
        },
    )
    choice_verdict
    return (choice_verdict,)


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(r"""
        ### Your report

        The lesson writes your report with your kit: your `result_rows` and `report` once they pass their checks, your `compare` on the course's runs over five seeds at 90%, your settings, and your runs of the key figure and the ablation, or the course's where yours are not there yet. The claim starts as the course's; the part *Not checked* is yours to write, in English, as the report is, and to end with your verdict of exercise 4. The button below the report saves it as `work/projects/repro/report.md`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def report_inputs():
    claim_box = mo.ui.text_area(value=C.CLAIM, label="Claim", full_width=True, rows=5)
    not_checked_box = mo.ui.text_area(value="", label="Not checked", full_width=True, rows=5)
    mo.vstack([claim_box, not_checked_box])
    return claim_box, not_checked_box


@app.cell(hide_code=True)
def demo_report(claim_box, not_checked_box, result_rows_check, result_report):
    result_rows_check, result_report  # run again when an exercise saves its module
    _r = C.demo(claim_box.value, not_checked_box.value)
    report_text = _r["text"]
    _whose = {"result_rows": "`result_rows`", "report": "`report`", "compare": "`compare`"}
    _mine = [_whose[_k] for _k in _r["theirs"]]
    _notes = [f"*Written with your {' and '.join(_mine)}; the course's stand in for the rest.*" if _mine else "*Written with the course's functions: yours stand in once they pass their checks.*",
              "*Your runs of the key figure.*" if _r["my_figure"] else "*The course's runs of the key figure: yours are not there yet.*",
              "*Your grid of the ablation.*" if _r["my_grid"] else "*The course's grid of the ablation: yours is not there yet.*"]
    _body = report_text.replace("\n## ", "\n#### ").replace("# ", "### ", 1)   # its headings below the lesson's
    mo.vstack([
        mo.Html(f"<div style='border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.2rem 1rem; overflow-x: auto'>{mo.md(_body).text}</div>"),
        mo.md(" ".join(_notes)),
    ])
    return (report_text,)


@app.cell(hide_code=True)
def save_cell():
    save_button = mo.ui.run_button(label="Save the report")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def demo_save(save_button, report_text):
    _msg = None
    if save_button.value:
        try:
            _msg = f"*Saved as `{C.save(report_text)}`. The next lesson shares it, with your code, settings and seeds.*"
        except OSError:
            _msg = "*The report could not be saved: is the folder `work/projects/repro/` there, and may the course write in it?*"
    mo.output.replace(mo.md(_msg)) if _msg else mo.output.clear()
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The plan of [Guided reproduction 1](/course/open/repro-guided-claim?lang=en), now the first section of your report; the runs of [Guided reproduction 4](/course/open/repro-guided-figure?lang=en) and the grid of [Guided reproduction 5](/course/open/repro-guided-ablation?lang=en), now its tables; and the comparison and the sort of [Guided reproduction 6](/course/open/repro-guided-compare?lang=en), now its results and differences. Your kit now has `repro/results.py` and `repro/report.py`, and your report. The next lesson, *Sharing your work so others can repeat it*, publishes it with your code, settings and seeds, so that someone else can run it again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rows_check, result_report, explain_not_checked_cell, choice_verdict):
    L.wrap_up(result_rows_check, result_report, explain_not_checked_cell, choice_verdict)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
