# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-figure. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-figure", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 4: sanity checks](/course/open/repro-own-sanity?lang=en) and [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_learns():
    quick_learns = L.choice(
        "quick-learns",
        question="Your sanity check gives `\"learns\": False` for a model of text that sees only the token before. In its batch, the token *the* comes before *cat* three times and before *dog* twice. What does that tell you about the model?",
        options={
            "frozen": "A part of it is frozen or cut off from the loss: a model that works learns any batch by heart",
            "right": "Nothing yet: no model that sees only the token before can learn that batch by heart",
            "rate": "Its learning rate is too low: a model of text needs more than `1e-3`",
        },
        correct="right",
        why={
            "frozen": "After *the*, a model that sees only *the* gives one answer, and the batch wants *cat* three times and *dog* twice: even the best answer after *the*, 3/5 *cat* and 2/5 *dog*, leaves a loss of about 0.67 at those five places. See [Your reproduction 4](/course/open/repro-own-sanity?lang=en).",
            "rate": "With any learning rate, the best it can do after *the* is 3/5 *cat* and 2/5 *dog*: the batch holds it back, not the rate. See [Your reproduction 4](/course/open/repro-own-sanity?lang=en).",
            "right": "Choose a batch in which each input tells its label apart, and the check says something about the model again.",
        },
    )
    quick_learns
    return (quick_learns,)


@app.cell(hide_code=True)
def quick_check_resume():
    quick_resume = L.choice(
        "quick-resume",
        question="Your `run_seeds` of Guided reproduction 4 stops during seed 2, when seeds 0 and 1 are in the experiment log. You press the button again. What runs?",
        options={
            "right": "Seed 2, from its first step: a run that stopped wrote no line",
            "where": "Seed 2, from the step where it stopped, since its log was kept",
            "all": "Seeds 0, 1 and 2, so that all three come from the same press",
        },
        correct="right",
        why={
            "where": "A run adds its line only when it is done: nothing of seed 2 was kept, so it starts again. See [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).",
            "all": "`run_seeds` skips every seed whose line is in the log: that is what makes a stop cost one run. See [Guided reproduction 4](/course/open/repro-guided-figure?lang=en).",
            "right": "That is what a stop costs: the run that was going. The seeds that are done stay done.",
        },
    )
    quick_resume
    return (quick_resume,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        The main result is what your reproduction is for: the figure or table of the paper, made again from your own runs, and the verdict of your plan on it. It is the longest part of your project: every seed of your plan, both settings, for hours. So every run goes into one experiment log, a line right after it, and the runner goes on where it stopped: a stop costs one run, not the evening. And every dot of your figure and every number of your verdict comes from a line of that log, which anyone can read again.

        **An everyday example.** On election night, each polling station writes down its count as soon as it is done, and sends it in. A power cut costs the count of one station, not of the country, and anyone can add up the sheets again.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           4, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_ladder():
    guess_ladder = L.guess(
        "ladder",
        question="Your plan's budget is 12,800 steps. To also see how the result grows with training, you run the budget and its halvings: 100, 200, 400, and so on up to 12,800 steps, each a run of its own. How many steps is that together, against the 12,800 of the budget alone?",
        options={
            "eight": "About eight times as many: one run for each of the eight budgets",
            "free": "Hardly more: next to the long run, the short ones cost almost nothing",
            "right": "About twice as many: all the shorter runs add up to one more budget",
        },
        reveal="**About twice as many.** 100 + 200 + 400 + 800 + 1600 + 3200 + 6400 = 12,700: all the shorter runs together are 100 steps short of the longest. The whole ladder is 25,500 steps, for each seed and setting. Each run is as long as all the shorter runs together, plus the shortest, so a ladder costs less than twice its top.",
    )
    guess_ladder
    return (guess_ladder,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The budget, first.** The rule of your plan judges the runs of one budget: the steps of your main run. You fix it before the first run, from how long the paper trains and the hours your plan has for a seed, and write it into your kit. Choosing it after a look at the results is choosing the answer.

    **A ladder of budgets.** The paper shows its result as training goes on. Your main result runs the budget and seven halvings of it, each a run of its own, from its own start to its own end: eight points of the same figure, for each seed and setting.

    **One line per run.** The runner of your kit, `run_plan`, goes through every seed, budget and setting, and adds each result to the experiment log right after its run, as one line of JSON: `{"setting": "claim", "seed": 0, "steps": 12800, "result": 12.5}`. The setting, the seed and the steps together name a run: a run whose line is there is not run again, so a stop costs the run that was going.

    **The verdict from the log.** `results_at` takes the lines of one budget and gives them to the rule of your plan as it wants them: under the names of the plan, the smallest seed first. The rule takes the first results of each list as the seeds of the plan, and compares the two settings seed by seed.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | ladder of budgets | the budget and its halvings, each a run of its own: the points of the figure as training goes on |

    The picture: the course's main result for the dropout paper, as its experiment log fills. 3 seeds, 8 budgets from 100 to 12,800 steps, both settings: 48 runs, in the order of the runner. Each dot is the test error of one run, filled with dropout and hollow without; the lines go through the medians of the budgets where every seed so far has its run, solid with dropout and dashed without; the large ring marks the newest run. The thin line is the budget of the plan. The slider starts with every run done, and shows the log as it was along the way.
    """)
    return


@app.cell(hide_code=True)
def explore_log_cell():
    _names = C.P.course_plan("dropout", L.lang)["settings"]
    _labels = {"title": "The course's main result, run by run", "x": "steps of the run", "y": "test error (%)", "budget": "budget of the plan",
               "alt": "The test error of the course's experiment for the dropout paper against the steps of each run, on a log axis from 100 to 12,800, with and without dropout, three seeds each. With every run done: with dropout the error is higher up to 800 steps, about 27.6% against 23.8% at 100 steps, and lower from 1600 steps on, 12.5% against 13.6% in the median at 12,800 steps."}
    _verdicts = {"too few runs": "too few runs", "supported": "supported", "not supported": "not supported"}
    _plan = C.P.course_plan("dropout", "en")

    def _frame(n):
        _lines = C.course_log()[:int(n)]
        _next = C.stop_at(int(n))
        _v = _verdicts[C.course_verdict(_lines, _plan)]
        _stop = "Every run is done." if _next is None else f"A stop now costs the run that is going: \"{_next['setting']}\", seed {_next['seed']}, {C.whole(_next['steps'], L.lang)} steps."
        _pic = C.figure_picture(_lines, _names, C.COURSE_STEPS, _labels, last=_lines[-1] if _lines else None, ylim=(10, 30))
        return _pic, f"{int(n)} of {len(C.course_log())} runs in the log. The plan's verdict at {C.whole(C.COURSE_STEPS, L.lang)} steps: {_v}. {_stop}"

    explore_log = L.explore("runs", "runs", _frame, start=48, low=0, high=48, step=1, role="",
                            task="Slide back from the end: which run does the verdict wait for? And what does a stop after 30 runs cost?")
    explore_log
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        With every run done, the test error at 12,800 steps is 12.5% with dropout and 13.6% without, in the median; seed by seed, dropout is ahead by 0.4, 0.7 and 1.2 points, a median of 0.7, and the rule of the plan says supported. The verdict waits for the very last run, `"other"` of seed 2 at 12,800 steps: until then the plan has too few runs, however good the figure looks. A stop after 30 runs costs one run, the 31st, `"claim"` of seed 1 at 12,800 steps; the next press goes on from there. The figure tells the paper's story at the size of the course: with dropout the error starts higher, as the network learns more slowly with it, and from 1600 steps on it is lower.
        """),
        L.watch_out("Each point of the ladder is the end of a run of its own, with its schedule over its own steps. The paper's figure follows one long run along the way. The two can differ: a short run that ends calmly can do better than a long run at the same step, still on its way. Say so under your figure."),
        mo.md(r"""
        ### Recipe

        To run the main result of your reproduction:

        1. Fix the budget from the paper and the hours of your plan, and write it into your kit.
        2. Run the budget and its seven halvings, every seed, both settings, with the runner of your kit: a line in the experiment log right after each run.
        3. After a stop, run it again: the runner goes on with the runs that have no line.
        4. Draw the figure from the log: each run a dot, each setting a line through the medians.
        5. Read the runs at the budget with `results_at`, and give them to the rule of your plan.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A log in which seed 1 ran before seed 0, as when you ran seed 1 first to try it out. New is `key=` in `sorted`: it sorts the lines by what the function gives for each one, here its seed. Run the cell.
    """)
    return


@app.cell
def example_order():
    # a log in which "claim" of seed 1 ran first, to try it out
    _rows = [("claim", 1, 17.5), ("claim", 0, 16.4),
             ("other", 0, 15.5), ("other", 1, 15.3)]
    _log = []
    for _s, _seed, _r in _rows:
        _log.append({"setting": _s, "seed": _seed,
                     "result": _r})
    _sorted = sorted(_log, key=lambda _l: _l["seed"])
    print("log:", [_l["seed"] for _l in _log])
    print("sorted:", [_l["seed"] for _l in _sorted])
    for _s in ["claim", "other"]:
        print(_s, [_l["result"] for _l in _sorted
                   if _l["setting"] == _s])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the order of the log, the lists would be `[17.5, 16.4]` and `[15.5, 15.3]`: at place 0, seed 1 of `"claim"` against seed 0 of `"other"`, two runs that saw other starting weights and other batches. Sorted by seed, they are `[16.4, 17.5]` and `[15.5, 15.3]`: place 0 of both is seed 0, and place 1 is seed 1. That is what the rule needs, as it takes the first results as the seeds of the plan and compares them place by place.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_runner():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The runner

        Complete `run_plan(run, seeds, budgets, path, progress=None, settings=("claim", "other"))`. It runs `run(setting, seed, steps)` of your experiment for each seed, each budget and each of `settings`, in the order of the loops, and adds one line of JSON to the experiment log at `path` right after each run. First it reads which runs are done: a tuple of the setting, the seed and the steps of every line in the log. The check gives it a quick stand-in for your experiment, which notes how many lines the log has at each run: with two seeds and two budgets into a new log, with a log that has three runs already, with a plan that stops in its second run and is run again, and with the setting `"ablation"` alone, as the next lesson runs it.
        """),
        L.task("Complete `run_plan`: first `done`, then each run that is not done, with its line written right after it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_runner():
    box_runner = L.code_box("run-plan", lines=26)
    box_runner
    return (box_runner,)


@app.cell(hide_code=True)
def check_runner(box_runner):
    result_runner = L.check("run-plan", box_runner.value)
    result_runner
    return (result_runner,)


@app.cell(hide_code=True)
def hints_runner():
    L.hints(
        "run-plan",
        [
            "`done` is made the way `run_seeds` of Guided reproduction 4 made it, from every line of `open(path)` with `json.loads`, but with a tuple of three for each line instead of its seed alone.",
            "In the loops: `if (setting, seed, steps) not in done:`, then the line as a dictionary with its result, written with `open(path, \"a\")` as `json.dumps(line) + \"\\n\"`, and given to `progress` when there is one.",
        ],
        why="The tuple of three names a run, so a stop between two settings, or between two budgets of a seed, costs only the run that was going; and a line written right after its run is kept, whatever comes after.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_results():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The results at the budget

        Complete `results_at(lines, steps, plan)`. It gives the rule of your plan the runs of one budget: a dictionary with the plan's two names of its settings, each with the results of that setting at `steps` steps, the smallest seed first. `names` turns the kit's `"claim"` and `"other"` into the names of the plan. The check tries a made-up log of two seeds and two budgets, the same log with seed 1 first, and another plan.
        """),
        L.task("Complete `results_at`: the lines in the order of their seeds, only those of `steps` steps, each result under the plan's name of its setting."),
    ])
    return


@app.cell(hide_code=True)
def exercise_results():
    box_results = L.code_box("results-at", lines=12)
    box_results
    return (box_results,)


@app.cell(hide_code=True)
def check_results(box_results):
    result_results = L.check("results-at", box_results.value)
    result_results
    return (result_results,)


@app.cell(hide_code=True)
def hints_results():
    L.hints(
        "results-at",
        [
            "Go through the lines the way the worked example sorts them: `sorted(lines, key=...)`, by their seed.",
            "For each line of `steps` steps, `names[line[\"setting\"]]` is the name of its setting in the plan: append `line[\"result\"]` to that list of `results`.",
        ],
        why="Sorted by seed, the first results of each list are the seeds of the plan, and both lists hold the same seed at each place, as the rule compares them.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Fix your budget

        Before the first run of your main result, fix its budget: the steps of its longest runs, from how long the paper trains and the hours your plan has for a seed. The ladder of seven halvings comes with it, at less than twice the cost. Your plan's hours are for one seed, both settings, so the ladder of a seed takes about 2 × 2 × the budget × the seconds of one step: time one step from a short run of your experiment. The check saves the budget in your kit, as `work/projects/own/budget.json`, with the date.
        """),
        L.task("Fill in the budget of your main run, and in a sentence where it comes from."),
    ])
    return


@app.cell(hide_code=True)
def budget_inputs():
    _old = C.saved_budget() or {}
    budget_steps = mo.ui.number(start=0, stop=10_000_000, step=1, value=_old.get("steps") or 0, label="The budget of your main run, in steps")
    budget_why = mo.ui.text_area(label="Where it comes from: how long the paper trains, and the hours of your plan", full_width=True, rows=3, value=_old.get("why", ""))
    _plan = C.P.saved_plan()
    _head = (mo.md(f"*Your plan: {_plan['seeds']} seeds, {C.short(_plan['hours_per_seed'], L.lang)} hours for a seed, both settings.*") if _plan
             else mo.md("*No plan yet: write it in [Your reproduction 2](/course/open/repro-own-claim?lang=en) first.*"))
    mo.vstack([_head, budget_steps, budget_why])
    return budget_steps, budget_why


@app.cell(hide_code=True)
def check_budget(budget_steps, budget_why):
    result_budget = L.check("budget", {"steps": budget_steps.value, "why": budget_why.value})
    result_budget
    return (result_budget,)


@app.cell(hide_code=True)
def hints_budget():
    L.hints(
        "budget",
        [
            "The setup of the paper says how long it trains: in steps, in epochs, or in passes over the data. One pass over N examples, in batches of B, is N / B steps.",
            "Time a run of 200 steps of your experiment and divide by 200. The ladder of a seed takes about 2 × 2 × the budget × that: the hours of your plan for a seed should hold it, with room to spare.",
        ],
        why=f"For the course's experiment for the dropout paper: **{C.whole(C.COURSE_STEPS, L.lang)} steps.** " + C._B_WHY[L.lang],
    )
    return


@app.cell(hide_code=True)
def text_main():
    mo.md(rf"""
    ### What your project can do now

    The button runs your main result with your kit: your `run_plan` with the `run` of your experiment, every seed of your plan, the ladder of your budget, into your experiment log `work/projects/own/runs.jsonl`. You see each run as its line comes in. For the course's experiment for the dropout paper, all 48 runs take about {C.COURSE_MINUTES} minutes on one thread of the course's computer; your plan says how long yours may take. Close the lesson halfway, and the runs that are done stay in your log: the next press goes on with the others. While it runs, the rest of this lesson waits: you can read on, or work in another lesson.
    """)
    return


@app.cell(hide_code=True)
def button_main():
    run_main = mo.ui.run_button(label="Run my main result")
    run_main
    return (run_main,)


@app.cell(hide_code=True)
def demo_run(run_main, result_runner, result_budget):
    import time as _time

    result_runner, result_budget  # run again when an exercise saves a part of the kit
    _plan = C.P.saved_plan()
    _names = C.P._settings(_plan) if _plan else ("claim", "other")
    _labels = {"title": "Your main result, run by run", "x": "steps of the run", "y": (_plan or {}).get("metric") or "your metric", "budget": "your budget", "alt": "Your runs so far."}
    main_lines = None
    if not run_main.value:
        _out = mo.md("*Press the button to run your main result with your kit.*")
    else:
        _began, _done = _time.perf_counter(), []
        _budget = C.saved_budget()

        def _progress(_line):
            _done.append(_line)
            _steps = sum(_l["steps"] for _l in _done)
            _todo = 2 * int(_plan["seeds"]) * sum(C.ladder(_budget["steps"])) - sum(_l["steps"] for _l in (C.my_log() or [])) if _budget else 0
            _spent = _time.perf_counter() - _began
            _left = (f", about {round(_spent / _steps * max(_todo, 0) / 60)} minutes to go" if _spent / _steps * _todo >= 90 else ", less than two minutes to go") if _steps and _todo > 0 else ""
            _all = C.my_log() or []
            mo.output.replace(mo.vstack([
                mo.md(f"**{len(_all)} runs in your log** · the newest: `\"{_line['setting']}\"`, seed {_line['seed']}, {C.whole(_line['steps'], L.lang)} steps: {C.short(_line['result'], L.lang)} · {C.short(round(_spent / 60, 1), L.lang)} minutes{_left}"),
                mo.Html(C.figure_picture(_all, _names, _budget["steps"], _labels, last=_line, role="ink")),
            ]))

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Your kit needs three parts first, and one of them is not there yet: your runner, exercise 1 here; your experiment, exercise 2 of [Your reproduction 3](/course/open/repro-own-build?lang=en); and your plan, of [Your reproduction 2](/course/open/repro-own-claim?lang=en).*")
        elif _r == "budget":
            _out = mo.md("*Fix your budget in exercise 3 first: the button runs its ladder.*")
        elif _r == "log":
            _out = mo.md("*A line of your experiment log, `work/projects/own/runs.jsonl`, is not a run as `run_plan` writes it: `\"setting\"` `\"claim\"` or `\"other\"`, a whole `\"seed\"` and `\"steps\"`, and a number as `\"result\"`. Did you change your `own/runner.py` after its check, or the log by hand?*")
        elif _r == "error":
            _out = mo.md("*Your `own/runner.py` or your `own/experiment.py` stops with an error. Did you change one of them after its check? Then check it again. The runs that were done are in your log.*")
        else:
            main_lines = _r
            _out = mo.md(f"*Done: {len(_r)} runs are in your experiment log, `work/projects/own/runs.jsonl`.*")
    _out
    return (main_lines,)


@app.cell(hide_code=True)
def demo_main(main_lines, result_results):
    main_lines, result_results  # read the log again after a run
    _lines, _plan, _budget = C.my_log(), C.P.saved_plan(), C.saved_budget()
    if not _lines or not _plan or not _budget:
        _out = mo.md("*Your main result shows here once its runs are in your experiment log.*")
    else:
        _names = C.P._settings(_plan)
        _labels = {"title": "Your main result", "x": "steps of the run", "y": _plan.get("metric") or "your metric", "budget": "your budget",
                   "alt": f"Your runs: {len(_lines)} in your experiment log, against the steps of each run, on a log axis."}
        _verdicts = {"too few runs": "too few runs", "supported": "supported", "not supported": "not supported"}
        _said = C.my_verdict(_lines, _plan, _budget["steps"])
        _rule = (f"The rule of your plan, on the runs of {C.whole(_budget['steps'], L.lang)} steps read with your `results_at`: **{_verdicts.get(_said, _said)}**." if _said
                 else "Your `own/results.py` or the rule in your `own/plan.py` stops with an error, or is not in your kit yet: check exercise 2, and the rule of Your reproduction 2.")
        _out = mo.vstack([
            mo.Html(C.figure_picture(_lines, _names, _budget["steps"], _labels, role="ink")),
            mo.md(_rule),
            mo.Html(C.main_table(_lines, _plan, _budget["steps"], {"seed": "seed", "gain": "gain"}, L.lang)),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 4. What your main result says

    Your main result is in your log. Say what it says about the claim, and where your setup differs from the paper's.
    """)
    return


@app.cell(hide_code=True)
def explain_why_main():
    explain_why_main_cell = L.explain(
        "why-main",
        question="What does your main result say about the claim? And where could your setup differ from the paper's, so that your numbers differ from its figure?",
        model_answer="**My result:** at 12,800 steps, the test error with dropout is lower than without for all three seeds, by 0.4, 0.7 and 1.2 points; the median, 0.7, is above the 0.5 of my plan, so my rule says supported. Up to 800 steps dropout is behind: the network learns more slowly with it. **Where it differs:** my data is Fashion-MNIST, 6000 training pictures, not the 60,000 of MNIST; my network has two hidden layers of 512, smaller than the paper's; I train with SGD and a cosine decay of my own choice; and I measure the test error on 1000 pictures, so one picture is 0.1 point. My errors are much higher than the paper's, about 12% against about 1%: what I can compare is the effect, not the number.",
        points=[
            "you give the verdict of your rule at your budget, with the median gain",
            "you name a difference in the data or how it is prepared",
            "you name a difference in the network or the training",
            "you name a difference in how the result is measured",
        ],
    )
    explain_why_main_cell
    return (explain_why_main_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The runner of [Guided reproduction 4](/course/open/repro-guided-figure?lang=en), now for any experiment of your kit; the contract and the parts of [Your reproduction 3](/course/open/repro-own-build?lang=en); and the rule of [Your reproduction 2](/course/open/repro-own-claim?lang=en), which now judges real runs. In the next lesson, *Your reproduction 6: the ablation*, you test which part of the method your result depends on, with the same runner.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_runner, result_results, result_budget, explain_why_main_cell):
    L.wrap_up(result_runner, result_results, result_budget, explain_why_main_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
