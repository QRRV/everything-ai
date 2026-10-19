# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-choose. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-choose", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Guided reproduction 7: write the report](/course/open/repro-guided-report?lang=en) and [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_section():
    quick_section = L.choice(
        "quick-section",
        question="Which section of your reproduction report says what you could not test, and what it would take?",
        options={
            "differences": "Differences",
            "right": "Not checked",
            "results": "Results",
        },
        correct="right",
        why={
            "differences": "*Differences* says where your setup and numbers differ from the paper's, and the details it does not give. See [Guided reproduction 7](/course/open/repro-guided-report?lang=en).",
            "results": "*Results* gives every run you planned, with its numbers. See [Guided reproduction 7](/course/open/repro-guided-report?lang=en).",
            "right": "And it ends with the verdict your results support.",
        },
    )
    quick_section
    return (quick_section,)


@app.cell(hide_code=True)
def quick_check_sums():
    quick_sums = L.choice(
        "quick-sums",
        question="Someone checks the folder you shared with `sha256sum -c SHA256SUMS`, and one line says FAILED. What does that tell them?",
        options={
            "broken": "That your code has a bug in that file",
            "right": "That their copy of that file is not the one you shared",
            "order": "That the lines of SHA256SUMS are in another order",
        },
        correct="right",
        why={
            "broken": "A checksum says nothing about whether code is right, only whether a file is the same one. See [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en).",
            "order": "Each line is checked on its own, against the file it names. See [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en).",
            "right": "One changed character in a file gives it a completely different SHA-256, so they know which file to get again.",
        },
    )
    quick_sums
    return (quick_sums,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        From here on, the reproduction is yours: you choose the paper, and the course helps less. Most of whether it works is decided now. A paper that needs more than your computer has, a claim no figure can show false, or data you may not use, fails before the first run, however well you build it. [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en) gave you the checks; this lesson applies them to a list the course has made, and to your own computer.

        **An everyday example.** Choosing what to bake for guests: a cake that needs an oven you do not have, or a day you do not have, or something you cannot buy, is no choice, however good it looks in the cookbook.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           0, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_cpu():
    guess_cpu = L.guess(
        "cpu",
        question=f"Of the {len(C.papers())} papers on the course's list, how many does the course expect to fit a computer without an NVIDIA card?",
        options={
            "none": "None: every paper on the list needs a GPU",
            "right": f"{sum(_p['needs'] == 'any' for _p in C.papers())}",
            "all": "All of them, if you wait long enough",
        },
        reveal=f"**{sum(_p['needs'] == 'any' for _p in C.papers())}:** " + " and ".join(f"*{_p['title']}*" for _p in C.papers() if _p['needs'] == 'any') + ", both reproduced with an MLP on Fashion-MNIST, pictures of clothes. The other six, two CNNs on the same pictures among them, are made for an NVIDIA card with 8 GB, or the recommended spec. *The idea* shows the whole list.",
    )
    guess_cpu
    return (guess_cpu,)


@app.cell(hide_code=True)
def text_idea():
    _here = {"any": "a computer without an NVIDIA card", "gpu8": "an NVIDIA card with 8 GB or more", "rec": "the recommended spec, an NVIDIA card with 12 GB or more"}[C.machine()]
    mo.md(rf"""
    ## The idea

    **The list.** Each paper on it passed the checks of [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en): one claim that a figure or a table can show false, data with a license that lets you use it, and a core result that the course estimates to fit an evening or a night on the recommended spec. Those estimates are the course's, not yet measured on its test machine: plan with a margin. Each paper builds on lessons you have done, which the card names.

    **Your computer.** What a paper needs is one of three levels: any computer, an NVIDIA card with 8 GB or more, or the recommended spec, an NVIDIA card with 12 GB. A computer of a higher level runs what a lower one does. The course sees {_here} here; a Mac counts as a computer without an NVIDIA card, because the estimates are for an NVIDIA card. The hours are those of the recommended spec: on a computer of a lower level, a paper it can run may take longer, so time it before you choose, as *Watch out* below says.

    Below is the list, with the estimate of each paper and, for each level of computer, whether it fits one evening of 4 hours. Move the slider to the margin you plan with: the estimate times the margin has to fit the evening. The columns are the three levels: any computer, an NVIDIA card with 8 GB, and 12 GB. The column of your computer is marked; the papers that fit it come first, the others are pale.
    """)
    return


@app.cell(hide_code=True)
def explore_margin_cell():
    _mine = C.machine()
    _yours = {"any": "one without an NVIDIA card", "gpu8": "one with an NVIDIA card of 8 GB", "rec": "one of the recommended spec"}[_mine]
    _labels = {"paper": "paper", "hours": "hours", "yes": "yes", "more": "needs more", "long": "too long", "yours": "(yours)",
               "any": "any", "gpu8": "8 GB", "rec": "12 GB"}

    def _frame(m):
        _m = int(m)
        _n = sum(C.course_fits(_p, _mine, C.EVENING / _m) for _p in C.papers())
        return (C.levels_picture(_m, _mine, _labels),
                f"With a margin of {_m}, an evening of {C.EVENING} hours holds an estimate of up to {C.number(C.EVENING / _m, 1, L.lang)} hours. On your computer, {_yours}, **{_n} of the {len(C.papers())} papers {'fits' if _n == 1 else 'fit'}**.")

    explore_margin = L.explore("margin", "margin", _frame, start=2, low=1, high=3, step=1, role="",
                               task="How many papers fit your computer in one evening with a margin of 2, and with 3? And what would a computer of a higher level add?")
    explore_margin
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(rf"""
        Without a margin, one evening fits the two papers with an MLP on pictures of clothes on any computer; an NVIDIA card with 8 GB adds the small transformer of the induction heads and the two CNNs, and the recommended spec the other three, on language models: the mirage of emergence, the reversal curse and LoRA. With a margin of 3, only {" and ".join(_p["title"].split(":")[0] for _p in C.papers() if C.course_fits(_p, "rec", C.EVENING / 3))} still fits an evening, on any computer. A paper that does not fit one evening may still fit a night, or a weekend: that is your call.

        **A paper of your own** is welcome, as in the ML Reproducibility Challenge, if it passes the same checks: write its card yourself, with its claim, what you would measure, its data and license, what it needs, and your estimate of the hours.

        Here are the cards of the list:
        """),
        mo.md("\n\n".join(f"**{_p['title']}** ({_p['year']}), {_p['authors']}, {_p['ref']}. *Claim:* {C.CARDS[_p['id']]['claim']['en']} *You measure* {C.CARDS[_p['id']]['measure']['en']}. *Data:* {C.data_text(_p, L.lang)}. *Builds on:* {C.builds_on_text(_p, L.lang)}."
                          for _p in C.papers())),
        L.watch_out("An estimate is not a measurement. Before you choose a paper that only just fits, time one short run of it in the lesson it builds on, and multiply: steps, seeds and settings, with a margin of 2 or 3."),
        mo.md(r"""
        ### Recipe

        To choose the paper of your reproduction:

        1. Find the level of your computer, and the hours you have.
        2. Keep the papers that need no more than your level and fit your hours.
        3. Read the claim of each, and in the paper the figure or table it rests on.
        4. Check that you may use its data, and which lessons it builds on.
        5. Choose one, and say why: the claim, what you measure, and what it costs.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 2 of the recipe for one paper by hand: a paper that needs an NVIDIA card with 8 GB and about 3 hours, on a computer of the recommended spec, with one evening of 4 hours. New is the method `.index` of a list: `_levels.index("gpu8")` gives the index of `"gpu8"`, its place in the list counted from 0, here 1. Run the cell.
    """)
    return


@app.cell
def example_fits():
    _levels = ["any", "gpu8", "rec"]   # from least to most
    _paper = {"needs": "gpu8", "hours": 3}
    _machine, _hours = "rec", 4
    _enough_machine = _levels.index(_paper["needs"]) <= _levels.index(_machine)   # 2. no more than your level
    _enough_time = _paper["hours"] <= _hours                                      # 2. within your hours
    print("machine:", _enough_machine, "time:", _enough_time, "fits:", _enough_machine and _enough_time)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `"gpu8"` is at index 1 of the levels and `"rec"` at index 2, so the machine is enough; 3 hours fit in 4. Compare the indexes, not the names: `"gpu8" <= "rec"` compares their spelling, letter by letter, and gives the right answer here only because a, g and r happen to come in that order in the alphabet. The index says which is more, whatever the names.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Does it fit?

        Complete `fits(paper, machine, hours)`: `True` when the paper needs no more than `machine`, and its estimate `paper["hours"]` is no more than `hours`; else `False`. `paper["needs"]` and `machine` are names of `LEVELS`, from least to most. The check tries papers that need more, the same, and less than the machine, with too few hours and exactly enough.
        """),
        L.task("Complete `fits`: compare the places in `LEVELS`, and the estimate with your hours."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fits():
    box_fits = L.code_box("fits", lines=8)
    box_fits
    return (box_fits,)


@app.cell(hide_code=True)
def check_fits(box_fits):
    result_fits = L.check("fits", box_fits.value)
    result_fits
    return (result_fits,)


@app.cell(hide_code=True)
def hints_fits():
    L.hints(
        "fits",
        [
            "Put the two lines of the worked example into one `return`: `LEVELS.index(...)` for both levels, and `<=` for both comparisons, joined by `and`.",
        ],
        why="The place in `LEVELS` says how much a level has; the name itself does not.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.md(r"""
    ### Exercise 2. Your paper

    Choose your paper below, from the list or one of your own, and save it: it goes into your project as `work/projects/own/paper.json`, and the lessons of your reproduction start from it. You can change it later by saving again. Then say why you chose it.
    """)
    return


@app.cell(hide_code=True)
def choose_inputs():
    _options = {f"{_p['title']} ({_p['year']})": _p["id"] for _p in C.papers()}
    _options["A paper of my own"] = "own"
    _saved = C.saved_choice()
    _start = next((_k for _k, _v in _options.items() if _saved and _v == _saved["id"]), None)
    paper_dropdown = mo.ui.dropdown(options=_options, value=_start, label="Your paper", full_width=True)
    _own = _saved if _saved and _saved["id"] == "own" else {}
    own_title = mo.ui.text(label="Title", full_width=True, value=_own.get("title", ""))
    own_ref = mo.ui.text(label="Reference, such as arXiv:2201.02177", full_width=True, value=_own.get("ref", ""))
    own_claim = mo.ui.text_area(label="Its claim, in one or two sentences", full_width=True, rows=3, value=_own.get("claim", ""))
    own_measure = mo.ui.text_area(label="What you would measure, and the figure or table of the paper it reproduces", full_width=True, rows=2, value=_own.get("measure", ""))
    own_data = mo.ui.text(label="Its data, and the license that lets you use it", full_width=True, value=_own.get("data", ""))
    _needs = {"any computer": "any", "NVIDIA, 8 GB": "gpu8", "recommended spec": "rec"}
    own_needs = mo.ui.dropdown(options=_needs, value=next((_k for _k, _v in _needs.items() if _v == _own.get("needs")), "recommended spec"), label="What it needs")
    own_hours = mo.ui.number(start=0, stop=200, step=0.5, value=_own.get("hours") or 4, label="Your estimate, in hours")
    save_choice_button = mo.ui.run_button(label="Save my choice")
    mo.vstack([paper_dropdown, mo.accordion({"Only for a paper of your own": mo.vstack([own_title, own_ref, own_claim, own_measure, own_data, own_needs, own_hours])}), save_choice_button])
    return own_claim, own_data, own_hours, own_measure, own_needs, own_ref, own_title, paper_dropdown, save_choice_button


@app.cell(hide_code=True)
def demo_choice(paper_dropdown, own_title, own_ref, own_claim, own_measure, own_data, own_needs, own_hours, save_choice_button, box_fits, result_fits):
    _out = None
    if save_choice_button.value and paper_dropdown.value:
        _own = {"title": own_title.value, "ref": own_ref.value, "claim": own_claim.value, "measure": own_measure.value,
                "data": own_data.value, "needs": own_needs.value, "hours": own_hours.value}
        if paper_dropdown.value == "own" and not (own_title.value.strip() and own_claim.value.strip() and (own_hours.value or 0) > 0):
            _out = mo.md("*For a paper of your own, write at least its title, its claim and your estimate of the hours, then save again.*")
        else:
            _c = C.save_choice(paper_dropdown.value, _own)
            _fn = C.fits_of(box_fits.value) if result_fits.state == "correct" else None
            _ok = (_fn or C.course_fits)(_c, C.machine(), 4)
            _say = "It fits your computer in one evening." if _ok else "By its estimate, it does not fit your computer in one evening: plan a night or a weekend, or choose a smaller one." if C.level_fits(_c, C.machine()) else "It needs more than your computer has, and more hours do not help: choose a paper your computer can run, or run it on a computer that has what it needs."
            _out = mo.md(f"*Saved as `work/projects/own/{C.CHOICE}`: {C.plain(_c['title'])}. {_say}*")
    _saved = C.saved_choice()
    _out = _out or (mo.md(f"*Your choice so far: {C.plain(_saved['title'])}.*") if _saved else mo.md("*No choice saved yet.*"))
    _out
    return


@app.cell(hide_code=True)
def explain_why_this():
    explain_why_this_cell = L.explain(
        "why-this",
        question="Why is the paper you chose one you can reproduce? Say what you will test, on what, and what it costs.",
        model_answer="**The claim:** a network trained with dropout has a clearly lower test error than the same network without; it rests on the figure of the paper that puts the test error on MNIST of networks with and without dropout side by side. **What I measure:** the test error of the same network with and without dropout, over three seeds, on Fashion-MNIST, which the course hosts under the MIT license. **What it costs:** the course estimates less than an hour on the recommended spec, and it runs on any computer. Mine has no NVIDIA card, so it may take longer: I time one short run first. I have an evening, so even at three times the estimate it fits. **What I reuse:** the network of *Project: make a deep network train*, and dropout from its own lesson.",
        points=[
            "you name the claim, and the figure or table of the paper it rests on",
            "you say what you will measure, on which data, and that you may use it",
            "you compare the cost with your computer and your hours, with a margin",
            "you name what you can reuse from the lessons or your kit",
        ],
    )
    explain_why_this_cell
    return (explain_why_this_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The checks of [Choosing a paper you can reproduce](/course/open/res-choose-paper?lang=en), now on a list and on your own computer; and the plan of [Guided reproduction 1](/course/open/repro-guided-claim?lang=en), which comes next for your paper. In the next lesson, *Your reproduction 2: claim and plan*, you write down your claim, the experiment, its cost and what would count as a failed reproduction, in the form of your kit, before any run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_fits, explain_why_this_cell):
    L.wrap_up(result_fits, explain_why_this_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
