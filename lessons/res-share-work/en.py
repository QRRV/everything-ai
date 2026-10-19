# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-share-work. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-share-work", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Also from outside the course:** a few words. **Git** is a program that keeps every version of the files in a folder, its **repository**; to **push** is to copy those versions to a website such as **GitHub**, where others can get them. A **model hub**, such as Hugging Face, is a website where people share trained models. A **hex digit** is one of the 16 digits 0 to 9 and a to f in which a hash is written. A **terminal** is a window in which you type commands that run programs, such as `python rerun.py`; a **script** is a Python file you run that way, as a whole. [The terminal: running a script](/course/open/py-terminal?lang=en) shows both. If Git is new, read the first chapter of [Pro Git](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control).
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Keeping an experiment log](/course/open/res-experiment-log?lang=en) and [Where does this data come from?](/course/open/data-provenance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_version():
    quick_version = L.choice(
        "quick-version",
        question="A run in a colleague's experiment log has code version `4f9a1c2e`, and you have a copy of their code. How do you find out whether it is the code that ran?",
        options={
            "right": "Compute the hash of your copy the same way, and compare it with `4f9a1c2e`",
            "date": "Look at when the file was last changed: the code of that day is the one that ran",
            "seed": "Run it with the same seed: if the result is the same, so is the code",
        },
        correct="right",
        why={
            "date": "A date says when a file changed, not what is in it: a copy made later, or a change undone, has another date and the same code. See [Keeping an experiment log](/course/open/res-experiment-log?lang=en).",
            "seed": "The same result does not mean the same code: a changed comment, or a change this run never reaches, gives the same numbers with other code. See [Keeping an experiment log](/course/open/res-experiment-log?lang=en).",
            "right": "The same text gives the same hash, and one changed character another: if yours starts with `4f9a1c2e`, it is, almost surely, the code that ran. This lesson does the same for every file of a folder.",
        },
    )
    quick_version
    return (quick_version,)


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="You found a dataset on a website that says nothing about a license. May you put a copy of it in your public repository?",
        options={
            "online": "Yes: it is on the internet already",
            "right": "No: without a license you may not pass it on; link to where it is",
            "credit": "Yes, as long as you name where it comes from",
        },
        correct="right",
        why={
            "online": "Being on the internet does not make it free to copy: without a license, all rights stay with whoever made it. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "credit": "Naming the source is what a license such as CC BY asks; without a license there is nothing that lets you copy it at all. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "Others can still get it from its source: your folder says where, and which version.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your report says what you found. To run it again, someone needs more than that: your code, the settings and the seeds, the versions of Python and PyTorch the runs used, the data or the way to make it, and a license that says they may. Without them, a reader has your word for it, and you yourself, a year from now, may not know any more what you ran. Asking the author by email is no way to rerun it: by then the code has changed, or is gone.

    **An everyday example.** A recipe you pass on: "a bit of flour, bake until done" helps no one. The amounts, the oven at 180 degrees for 35 minutes, the size of the tin, and that they may pass it on themselves: then someone else bakes the same cake.
    """)
    return


@app.cell(hide_code=True)
def predict_same_seed():
    guess_same_seed = L.guess(
        "same-seed",
        question="Someone gets your kit and runs it on their computer, with your settings and seed 0. Do they get your accuracies?",
        options={
            "exact": "Exactly the same on every computer, always: the seed fixes every random number there is",
            "right": "With your versions on one CPU thread usually the same; elsewhere the run can come out a little different",
            "random": "Different every time you run it, even on your own computer: training is random",
        },
        reveal=f"**With your versions on one CPU thread usually the same; elsewhere a little different.** The seed fixes the split, the starting weights and the batches. But the order in which a sum is added up depends on the CPU or GPU, the number of threads and the version of PyTorch, and that can change the last bit of a number. Such a tiny difference grows while the model trains. The course ran seed 0 again with the last bit of one starting weight changed: validation reached 99% one logged step later, at step {C.number(C.ONE_BIT['v'], 0, L.lang)} instead of {C.number(C.ONE_BIT['v_before'], 0, L.lang)}, and at step {C.number(C.ONE_BIT['v_before'], 0, L.lang)} it was at {round(100 * C.ONE_BIT['val_at'])}% instead of {C.number(100 * C.ONE_BIT['val_before'], 1, L.lang)}%. So the folder says which versions the runs used, and `rerun.py` runs on one thread.",
    )
    guess_same_seed
    return (guess_same_seed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **An open artifact** is one folder with everything that is needed to run your work again, and a license that says others may. For your reproduction it looks like this:

        ```
        share/
          repro/             your kit, with its settings
          llm/               the modules of your GPT the kit imports
          runs.jsonl         the logs of your runs: seed and accuracies
          ablation.jsonl     the same for the ablation
          seeds.jsonl        the same for the comparison over seeds
          report.md          your report, with its negative results
          rerun.py           a script that runs the key figure again
          requirements.txt   the versions of PyTorch and NumPy
          LICENSE            what others may do with the code
          README.md          what it is, and how to run it again
          SHA256SUMS         a checksum of every file
        ```

        What is not yours to hand out, such as the paper's PDF or a dataset whose license does not let you pass it on, gets a link and its version, not a copy. Secrets, such as a file with a login token, and what nobody needs to run it, stay out. If a run saved a checkpoint, share it too, under a license such as CC BY 4.0, the one the course gives its own models; one that is too big for the repository goes to a model hub such as Hugging Face, with its link and checksum in the folder.

        **Checksums.** A **checksum** of a file is the hash of its contents, as the code version of your experiment log was the hash of your code. To tell your own versions apart, 8 hex digits were enough; to show that a file is exactly the one you shared, a checksum takes all 64 of its SHA-256. The program `sha256sum` writes one line per file, the hash, two spaces and the name, and `sha256sum -c SHA256SUMS` checks every file against its line.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **open artifact** | everything that is needed to run a piece of work again, shared with a license: code, settings, seeds, versions, data or how to make it, results | the folder `share/` |
        | **checksum** | the hash of a file's contents; whoever has the same checksum has the same file | `sha256(...).hexdigest()` |
        """),
        L.watch_out("Look for secrets before you share: a login token or a password in a file or a notebook is public the moment you push it to a public repository, and it stays in the history of Git even after you delete the file. If one got out, revoke it, and make a new one."),
        mo.md(r"""
        ### Recipe

        To share your work so that others can run it again:

        1. Put the code, the settings, the seeds, the logs and the report in one folder.
        2. Write down the versions it ran with, and add a script that runs it again.
        3. Leave out secrets and what nobody needs; give what is not yours a link and its version.
        4. Add a license and a README that says how to run it.
        5. Add the checksums of every file, and run the script once from the folder alone.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The checksum lines of two small files by hand: step 5 of the recipe. Run the cell, then change one character of a file and run it again.
    """)
    return


@app.cell
def example_sums():
    import hashlib

    _files = {"settings.py": "SEED = 0\n", "LICENSE": "MIT License\n"}   # the name of each file, and its text
    for _name in sorted(_files):   # 5. in sorted order, so that the list is the same for everyone
        _hash = hashlib.sha256(_files[_name].encode()).hexdigest()
        print(_hash + "  " + _name)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `LICENSE` comes before `settings.py`: in sorted order, capitals come before small letters. Each hash is 64 hex digits, and one changed character gives a completely different one, so the list shows at once which file is not the one you shared.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_checksums():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The checksums

        Complete `checksums(files)`: for each name in `sorted(files)`, a line with the SHA-256 of the file's text as 64 hex digits, two spaces, and its name. `files` is a dictionary that maps each name to its text. The demo below uses your `checksums` for the file `SHA256SUMS` of your folder once it passes its check.
        """),
        L.task("Complete `checksums`: one line per file, in sorted order, with the hash of its text, two spaces and its name."),
    ])
    return


@app.cell(hide_code=True)
def exercise_checksums():
    box_checksums = L.code_box("checksums", lines=12)
    box_checksums
    return (box_checksums,)


@app.cell(hide_code=True)
def check_checksums(box_checksums):
    result_checksums = L.check("checksums", box_checksums.value)
    result_checksums
    return (result_checksums,)


@app.cell(hide_code=True)
def hints_checksums():
    L.hints(
        "checksums",
        [
            "Follow the worked example, with `files[name]` as the text and `lines.append(...)` in place of `print`.",
        ],
        why="The hash of the text, not of the name, changes when the file does; and sorted order makes the list itself the same for everyone who has the same files.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_what_goes():
    mo.md(r"""
    ### Exercise 2. What goes into the folder?

    Put each item in a group: in the folder, a link to it instead of the file, or it stays out.
    """)
    return


@app.cell(hide_code=True)
def exercise_what_goes():
    sort_what_goes = L.sort("what-goes")
    sort_what_goes
    return (sort_what_goes,)


@app.cell(hide_code=True)
def check_what_goes(sort_what_goes):
    result_what_goes = L.check("what-goes", sort_what_goes)
    result_what_goes
    return (result_what_goes,)


@app.cell(hide_code=True)
def text_exercise_license():
    mo.md(r"""
    ### Exercise 3. A license for the code
    """)
    return


@app.cell(hide_code=True)
def choice_license_cell():
    choice_license = L.choice(
        "license",
        question="Which license lets anyone run, change and share the code of your kit, as long as your name stays on it?",
        options={
            "none": "None: the repository is public, so anyone may use it",
            "nc": "CC BY-NC 4.0, so that no company makes money with it",
            "right": "The MIT license",
        },
        correct="right",
        why={
            "none": "Without a license, all rights stay with you: others may read your code, but not copy, change or share it.",
            "nc": "Creative Commons itself advises against its licenses for code: they are made for text, pictures and data. And NC keeps out more than companies: a school or a lab that gets money may not be sure it is allowed.",
            "right": "MIT lets anyone use, change and share the code, also in their own work, as long as the license with your name goes with it. For the report and the logs, CC BY 4.0 does the same for text and data.",
        },
    )
    choice_license
    return (choice_license,)


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(r"""
        ### Your folder

        The lesson puts your reproduction together in `work/projects/repro/share/`: every module of your kit, your logs and your report, a script that runs the key figure again, the versions of this computer, the MIT license with your name, a README, and the checksums, made with your `checksums` once it passes its check. Where a module or a log of yours is missing, the course's goes in, as in your report, and the README says so; the comparison over seeds always used the course's runs. Write the name that goes in the license, then press the button. It writes the folder anew each time; nothing leaves your computer.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def share_inputs():
    name_box = mo.ui.text(value="", label="Your name, for the license", full_width=True)
    build_button = mo.ui.run_button(label="Put the folder together")
    mo.vstack([name_box, build_button])
    return build_button, name_box


@app.cell(hide_code=True)
def demo_share(build_button, name_box, box_checksums, result_checksums):
    _out = mo.md("*Press the button to put the folder together.*")
    if build_button.value:
        _mine = C.checksums_of(box_checksums.value) if result_checksums.state == "correct" else None
        try:
            _r = C.build(name_box.value, _mine)
        except OSError:
            _r = None
        if _r is None:
            _out = mo.md("*The folder could not be written: is `work/projects/repro/` there, and may the course write in it?*")
        else:
            _notes = [f"*The folder is in `{_r['where']}`, with {len(_r['files'])} files.*",
                      "*The checksums are made with your `checksums`.*" if _r["mine"] else "*The checksums are made with the course's `checksums`: yours takes over once it passes its check.*"]
            if _r["missing"]:
                _notes.append("*Not there yet, so not in the folder: " + ", ".join(f"`{_m}`" for _m in _r["missing"]) + ". Make it in the lesson before, then put the folder together again.*")
            _notes.append("*The runs in " + ", ".join(f"`{_m}`" for _m in _r["course_logs"]) + " are the course's; the README says so.*")
            if _r["course"]:
                _notes.append("*The course's versions of " + ", ".join(f"`{_m}`" for _m in _r["course"]) + " are in it, as you have no version of your own; the README says so.*")
            _sums = "\n".join(_r["files"]["SHA256SUMS"].splitlines()[:4]) + "\n…"
            _out = mo.vstack([
                mo.md(" ".join(_notes)),
                mo.Html(f"<div style='border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.2rem 1rem; overflow-x: auto'>{mo.md(_r['files']['README.md'].replace(chr(10) + '## ', chr(10) + '#### ').replace('# ', '### ', 1)).text}</div>"),
                mo.md(f"The first lines of `SHA256SUMS`:\n\n```text\n{_sums}\n```"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    To put it out, make the folder a Git repository and push it to GitHub, or upload it to Zenodo, which gives it a DOI: a fixed link that a report or a paper can cite, also when the repository moves. Before you do, open every file once: is there nothing in it that you would not want to be public?
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The code version of [Keeping an experiment log](/course/open/res-experiment-log?lang=en), now as a checksum of every file; the license of [Where does this data come from?](/course/open/data-provenance?lang=en), now on your own work; and the report of [Guided reproduction 7](/course/open/repro-guided-report?lang=en), now with all that is needed to check it. In the next lesson, *Your reproduction 1: choose a paper*, you start a reproduction of your own, with the same steps and less help, and at its end, a folder like this one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_checksums, result_what_goes, choice_license):
    L.wrap_up(result_checksums, result_what_goes, choice_license)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
