# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-report. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-report", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Your reproduction 7: compare with the paper](/course/open/repro-own-compare?lang=en) and [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_failed():
    quick_failed = L.choice(
        "quick-failed",
        question="Your method lowers the error for every seed, by 2 to 3 points, and the paper's gain is 6 points, above all of them. A friend says: so your reproduction failed. What do you say?",
        options={
            "failed": "That it did: a reproduction should give the paper's number, within its interval",
            "right": "That the claim held, a lower error; the size differs, and that is a finding to explain",
            "paper": "That the paper's number must be wrong, as all your seeds agree with each other",
        },
        correct="right",
        why={
            "failed": "A reproduction tests the claim, here a clearly lower test error, and that held. A size that differs in a setup that differs is a finding, with its reasons. See [Your reproduction 7](/course/open/repro-own-compare?lang=en).",
            "paper": "That your seeds agree says how sure your own number is, not that the paper's is wrong: its setup differs from yours. Look at what differs, as in [Your reproduction 7](/course/open/repro-own-compare?lang=en).",
            "right": "Your report says both: what held, and what differs, with the reasons you can name.",
        },
    )
    quick_failed
    return (quick_failed,)


@app.cell(hide_code=True)
def quick_check_hash():
    quick_hash = L.choice(
        "quick-hash",
        question="A reader downloads your `runs.jsonl`, and its SHA-256 is not the one in your manifest. What does that tell them?",
        options={
            "right": "That their file is not yours: at least one character differs",
            "older": "That their copy is older than yours, by at least a day",
            "close": "That the results differ, but only a little, as two hashes of almost the same file are close",
        },
        correct="right",
        why={
            "older": "A hash says nothing about time: it changes with every change to the text, whenever it was made. See [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en).",
            "close": "Two texts that differ in one character give hashes that look nothing alike: a hash tells same or not, not how close. See [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en).",
            "right": "The same text gives the same SHA-256; any other text, another one.",
        },
    )
    quick_hash
    return (quick_hash,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A reproduction that nobody can read might as well not have happened. Your report says what you tested, how, what held and what differed, and what you did not check, so that a reader can weigh it without asking you. Publishing your code, your plan and your logs lets them run it again and find your numbers. And the authors of the paper are the readers who can learn the most from it: tell them, briefly and kindly, what you found.

        **An everyday example.** You tried a recipe for a cooking club, with another flour and a smaller oven. You write down what you changed, how it came out and what you did not try, add the photos, and send a note to the author of the recipe.
        """),
        draw.project_steps(["Choose a paper", "Claim and plan", "Build it", "Sanity checks", "The main result", "Ablation", "Compare", "Report and share"],
                           7, here_label="← today", title="Where your project stands: your own reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_sentence():
    guess_sentence = L.guess(
        "sentence",
        question="The course's reproduction of the dropout paper supports the claim by its rule. Which sentence belongs in the Results of its report?",
        options={
            "general": "Dropout lowers the test error of neural networks, as the paper says, so the claim holds in general",
            "right": "With dropout, the test error at 12,800 steps is lower for all three seeds, by 0.7 points in the median",
            "works": "Dropout clearly works: the reproduction confirms the paper beyond any doubt, with a gain larger than its own",
        },
        reveal="**With dropout, the test error at 12,800 steps is lower for all three seeds, by 0.7 points in the median.** It says what was measured, at which budget and over which seeds, and nothing more: one network, on 6000 pictures of Fashion-MNIST. \"In general\" and \"beyond any doubt\" claim what three seeds of one setup cannot show.",
    )
    guess_sentence
    return (guess_sentence,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The report of your kit.** The `report` of Guided reproduction 7 puts five parts together: *Claim*, *Setup*, *Results*, *Differences* and *Not checked*. You write four of them; the Results come from your kit: the rule of your plan and its verdict at your budget, the runs of every seed in a table, the share your ablation keeps, and your comparison with the paper. The report is in English, the language of the paper and of its authors.

    **Numbers at their precision.** A test error over 1000 pictures moves in steps of 0.1 point: one decimal says all it knows. A gain that is a difference of two floats, such as `12.9 - 12.5`, comes out as `0.40000000000000036`; rounded, it is 0.4. Your table rounds every number to the decimals the metric has.

    **What goes public, and its manifest.** Your code, your plan with its date, your budget, your logs and your report go public; a token or a password never does. A manifest lists the SHA-256 of every file, as the `SHA256SUMS` of [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en) did, so that a reader can check that theirs are yours.

    **Tell the authors.** A short message: what you reproduced and how, what held, what differed and what you think explains it, and where your code and report are. Kind and specific: they made the paper you learned from.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | manifest | a list of the files that go public, each with its SHA-256 |

    Below is the course's own report of its reproduction of the dropout paper, with its five parts as tabs, the Results from its runs. Click through the parts.
    """)
    return


@app.cell(hide_code=True)
def course_report_cell():
    _parts = {**C.COURSE_PARTS, "Results": C.course_results_text()}
    mo.vstack([
        mo.md(f"**{C.COURSE_PARTS['title']}**"),
        mo.ui.tabs({_name: mo.md(_parts[_name]) for _name in C.SECTIONS}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Every number of the Results can be found again in `runs.jsonl` and the other files of the kit. The Differences say what was changed on purpose, and what the paper's numbers are of: two networks, without a spread over runs. Not checked names what a reader might think was tested and was not: MNIST itself, larger networks, more seeds.
        """),
        L.watch_out("A report says what the runs showed, at the budget and over the seeds you ran, and no more. Leave out a run because it did not fit, and the report is no longer about your runs. A result that did not hold goes in just like one that did: it is just as useful to the next reader."),
        mo.md(r"""
        ### Recipe

        To report and share your reproduction:

        1. Write the Claim as the paper makes it, with where it stands; the Setup so that a reader could build it again.
        2. Let your kit write the Results: the rule and its verdict, the runs as a table at their precision, the ablation, the comparison.
        3. Write the Differences, on purpose and not, and what you did Not check.
        4. Publish your code, plan, budget, logs and report, with a manifest of their SHA-256, and what Sharing your work so others can repeat it put with them: the versions you ran with, where the data come from, a README and a license. No tokens or passwords.
        5. Tell the authors what you found, kindly and specifically, with the link.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One gain of the course's main result, as Python computes it and as a report shows it, in a table of one row. Run the cell.
    """)
    return


@app.cell
def example_round():
    _gain = 13.6 - 12.4          # without dropout minus with, seed 2
    print("computed:", _gain)
    print("rounded:", round(_gain, 1))
    _table = "| seed | gain |\n|---|---|\n| 2 | " + str(round(_gain, 1)) + " |"
    print(_table)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `13.6 - 12.4` is not exactly 1.2 in a float: the last decimals are noise of the computer, not of the runs. Rounded to the one decimal the test error has, it is 1.2. The second line of the table, `|---|---|`, tells Markdown that the line above it holds the names of the columns.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_table():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The table of your report

        Complete `markdown_table(head, rows, digits)`: a table in Markdown, with a line of names, a line of `---` for each column, and a line for each row, every number rounded to `digits` decimals. `"---|" * 3` repeats a text three times: `"---|---|---|"`. The Results of your report use it for the runs of your seeds, with the decimals your runs have. The check tries two of the course's runs, a table with three rows and two decimals, a table without rows, and one with three decimals.
        """),
        L.task("Complete `markdown_table`: the first two lines, then a line for each row, rounded."),
    ])
    return


@app.cell(hide_code=True)
def exercise_table():
    box_table = L.code_box("table", lines=10)
    box_table
    return (box_table,)


@app.cell(hide_code=True)
def check_table(box_table):
    result_table = L.check("table", box_table.value)
    result_table
    return (result_table,)


@app.cell(hide_code=True)
def hints_table():
    L.hints(
        "table",
        [
            "A line is `\"| \" + \" | \".join(cells) + \" |\"`, with the cells as text; the second line is `\"|\" + \"---|\"` once for each column.",
            "For each row: round every number to `digits` decimals and make it text with `str`, as its cells; at the end, join the lines with `\"\\n\"`.",
        ],
        why="The line of `---` makes Markdown draw a table; rounding shows each number at the precision its metric has, without the noise of floats.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your report

        Write the four parts of your report that your kit cannot write for you, in English, and its title. The Claim as the paper makes it, with where it stands; the Setup so that a reader could build it again; the Differences, on purpose and not; and what you did Not check. The check puts them together with the Results from your kit, with the `report` of your kit, and saves the report as `work/projects/own/REPORT.md`.
        """),
        L.task("Write the title and the four parts of your report."),
    ])
    return


@app.cell(hide_code=True)
def report_inputs():
    _old = C.saved_parts() or {}
    _start = C.start_parts()   # the claim and the setup of your plan, in English
    report_title = mo.ui.text(label="Title", full_width=True, value=_old.get("title", ""))
    report_claim = mo.ui.text_area(label="Claim", full_width=True, rows=4, value=_old.get("Claim") or _start.get("Claim", ""))
    report_setup = mo.ui.text_area(label="Setup", full_width=True, rows=5, value=_old.get("Setup") or _start.get("Setup", ""))
    report_differences = mo.ui.text_area(label="Differences", full_width=True, rows=4, value=_old.get("Differences", ""))
    report_not_checked = mo.ui.text_area(label="Not checked", full_width=True, rows=3, value=_old.get("Not checked", ""))
    mo.vstack([report_title, report_claim, report_setup, report_differences, report_not_checked])
    return report_claim, report_differences, report_not_checked, report_setup, report_title


@app.cell(hide_code=True)
def check_report(report_claim, report_differences, report_not_checked, report_setup, report_title):
    result_report = L.check("report", {"title": report_title.value, "Claim": report_claim.value, "Setup": report_setup.value,
                                       "Differences": report_differences.value, "Not checked": report_not_checked.value})
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Start the Claim and the Setup from your plan of Your reproduction 2: its claim, where the paper shows it, and what stays the same in both settings.",
            "For the Differences, look at Your reproduction 7: what you changed on purpose, what the paper's numbers are of, and what you think explains the difference.",
        ],
        why="The course's own parts are in the tabs above, under *The idea*: a Claim with where it stands, a Setup a reader could build again, Differences on purpose and not, and what was Not checked.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The button writes your report anew, with the Results your kit has now, and then the manifest of the files that go public, `work/projects/own/MANIFEST.txt`. Put the folder `work/projects/own/` in a public repository with your license, as in Sharing your work so others can repeat it, and the manifest with it.
    """)
    return


@app.cell(hide_code=True)
def button_share():
    share_button = mo.ui.run_button(label="Write my report and its manifest")
    share_button
    return (share_button,)


@app.cell(hide_code=True)
def demo_share(share_button, result_table, result_report):
    result_table, result_report  # run again when an exercise saves a part of the kit
    if not share_button.value:
        _out = mo.md("*Press the button to write your report and its manifest.*")
    else:
        _lines = C.manifest()
        if _lines is None:
            _out = mo.md("*Write your report in exercise 2 first: then the button can put it together.*")
        elif _lines == "report":
            _out = mo.md("*The `report` of your kit stops with an error, `work/projects/repro/report.py` of [Guided reproduction 7](/course/open/repro-guided-report?lang=en). Check it again there.*")
        else:
            _title, _parts = C.report_sections(C.report_path().read_text(encoding="utf-8"))
            _out = mo.vstack([
                mo.md("**Your report, `work/projects/own/REPORT.md`:**"),
                mo.md(f"**{_title}**"),
                mo.ui.tabs({_name: mo.md(_body) for _name, _body in _parts.items()}),
                mo.md("**Its manifest, `work/projects/own/MANIFEST.txt`:**"),
                mo.md("| file | SHA-256, its first 16 of 64 hex digits |\n|---|---|\n" + "\n".join(f"| `{_l.split()[1]}` | `{_l.split()[0][:16]}…` |" for _l in _lines) + "\n\nThe file has all 64."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_authors():
    mo.md(r"""
    ### Exercise 3. Tell the authors

    Your report and your code are public. Write the message to the authors of the paper.
    """)
    return


@app.cell(hide_code=True)
def explain_to_authors():
    explain_to_authors_cell = L.explain(
        "to-authors",
        question="Write the message to the authors of your paper: what you reproduced and how, what held, what differed and why you think so, and where your report and code are.",
        model_answer="Dear Dr. Srivastava and coauthors, I reproduced the claim of your dropout paper as part of a course: a smaller network (784-512-512-10, ReLU) on 6000 pictures of Fashion-MNIST, with dropout 0.2 at the input and 0.5 in the hidden layers, three seeds, 12,800 steps. With dropout, the test error was lower for all three seeds, by 0.7 points in the median, so the claim held. The gain is larger in points than your 1.60% to 1.35% on MNIST and smaller as a share of the error; I think the harder data and the smaller training set explain that. An ablation without the dropout at the input kept none of the gain. My report, code and logs: <link>. Thank you for a paper that was a pleasure to reproduce.",
        points=[
            "you say what you reproduced, on what, and how many seeds",
            "you say what held and what differed, with your guess why",
            "you give the link to your report and code",
            "your tone is kind and specific",
        ],
    )
    explain_to_authors_cell
    return (explain_to_authors_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The report of [Guided reproduction 7](/course/open/repro-guided-report?lang=en) and the sharing of [Sharing your work so others can repeat it](/course/open/res-share-work?lang=en), now for your own paper; and every part of your kit, from the plan of [Your reproduction 2](/course/open/repro-own-claim?lang=en) to the comparison of [Your reproduction 7](/course/open/repro-own-compare?lang=en). Your own reproduction is done. The next lesson, *Where to go next*, shows where the other branches of the course connect to what you know now.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_table, result_report, explain_to_authors_cell):
    L.wrap_up(result_table, result_report, explain_to_authors_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
