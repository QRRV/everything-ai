# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-package-squatting. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-package-squatting", "en")
    C = L.checks   # the course's run: Qwen3's package names for ten tasks, and what PyPI had on the day it looked


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [Is this the model you meant?](/course/open/rai-hub-integrity?lang=en) and [Using code from modules](/course/open/py-imports?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pin():
    quick_pin = L.choice(
        "quick-pin",
        question="Your pin holds a commit id and a SHA-256, and a file matches both. What does that tell you?",
        options={
            "maker": "That the organization in its name made it",
            "right": "That it is the file you pinned",
            "safe": "That no one put an attack in the file",
        },
        correct="right",
        why={
            "right": "A pin keeps you on one file, good or bad, whoever made it. This lesson asks the question before the pin: whether the name you chose is the one you meant.",
            "maker": "A name can change hands, and a SHA-256 taken from the new owner's file matches that file. See *What a hash does not tell you* in [Is this the model you meant?](/course/open/rai-hub-integrity?lang=en).",
            "safe": "A SHA-256 shows a change made after the pin, not an attack that was there before it, as in the releases of xz. See *What a hash does not tell you* in [Is this the model you meant?](/course/open/rai-hub-integrity?lang=en).",
        },
    )
    quick_pin
    return (quick_pin,)


@app.cell(hide_code=True)
def quick_check_import():
    quick_import = L.choice(
        "quick-import",
        question="What does `import math` do?",
        options={
            "download": "It downloads math from the internet each time",
            "copy": "It writes the code of math into your file",
            "right": "It brings the module math into your program",
        },
        correct="right",
        why={
            "right": "`import` brings in a module that is already on your computer: `math` comes with Python. A package that someone else wrote, such as NumPy, was put there once, by its name. That name is where this lesson starts.",
            "download": "`import` downloads nothing: `math` comes with Python, and `import` brings in what is already on your computer. See *The idea* in [Using code from modules](/course/open/py-imports?lang=en).",
            "copy": "Your file stays as it is: `import` makes the module's names usable with `math.` in front. See *The idea* in [Using code from modules](/course/open/py-imports?lang=en).",
        },
    )
    quick_import
    return (quick_import,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Spoke & Chain, the bike shop of the previous lesson, pins its model now. Its scripts need Python packages too, and the shop asks an LLM which ones. For the barcode on a photo of a bike part, the course's LLM, Qwen3-0.6B, answers `pip install barcode_reader` or `pip install barcode-reader`, in all ten of the course's tries. pip, which installs Python packages, finds a package by its name on PyPI, the Python Package Index, where anyone can publish one. On 19 November 2026, PyPI had no project of that name, so pip would have found nothing to install. Once someone registers the name, the same command installs whatever they put in it.

    Two kinds of names invite that. **Typosquatting** is publishing a package under a name that looks like a popular one, such as `urlib3` for `urllib3`, in the hope that people install it by mistake. **Slopsquatting** is registering a name that language models make up, such as `barcode-reader`, so that whoever trusts the model installs the attacker's package. This lesson checks a name before it is installed; the names it shows are there to be checked, never to be installed or registered.

    **An everyday example.** A friend tells you to buy a part at "Wheel World on Station Street". If there is no such shop, whoever opens one there under that name gets your visit, whatever it sells.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In research on package names that LLMs make up, Bar Lanyado of the security company Lasso Security came across `huggingface-cli`, a Python package that did not exist. Lanyado published an empty package under that name. How many downloads did it get in three months?",
        options={
            "small": "About 300",
            "middle": "Around 3,000",
            "right": "Over 30,000",
        },
        reveal="**Over 30,000.** In Lanyado's words: \"In three months the fake and empty package got more than 30k authentic downloads! (and still counting).\" Instructions to install it turned up in the README of a repository for research by Alibaba. Downloads are not victims: the package was empty, and Lanyado published a second dummy package to estimate how many downloads came from scanners and not from people.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Names that look alike

    The trick is older than packages. CAPEC, MITRE's catalog of attack patterns, describes it for web addresses as CAPEC-630, *TypoSquatting*: "An adversary registers a domain name with at least one character different than a trusted domain." For packages, Joseph Spracklen and five co-authors put it as "creating packages with names similar to popular ones to deceive users".

    In September 2017, the National Security Authority of Slovakia told the Python Security Response Team that PyPI hosted malicious packages. Python's security announcement lists 11, among them `setup-tools` and `urlib3`, published in June and online for about three months (Vu et al. count ten). Once reported, "all identified packages were taken down, within 70 minutes", and the packages "also installed the expected module so it wasn't easy to notice the attack." People do mistype: by the same announcement, a project that held the names of the standard library, the modules that come with Python, "blocked around 7500 attempted standard library package installations over a period of 4 days", about 0.006% of the roughly 123 million downloads from PyPI in those days.

    In 2020, Duc-Ly Vu and four co-authors studied 36 known attacks: "Attackers prefer to delete characters from the legitimate packages to generate squatting names." Other forms they list: a singular for a plural (`setuptool`), an added hyphen (`setup-tools`), the British spelling (`colourama` for `colorama`), a capital I for a small l (`jeIlyfish`, downloaded 119 times when it was found), and *combosquatting*, which reorders the words of a name: `nmap-python` for `python-nmap`. In March 2024, Check Point found "over 500 malicious packages" on PyPI, "Each package originated from a unique maintainer account", with names from "simple misspellings (requestss instead of requests) to more conceptual (pandas-sdk instead of pandas) or visually (request5 instead of requests) deceptive variations." On 28 March 2024, PyPI stopped new projects and new accounts for about ten and a half hours, by its status page, "to mitigate an ongoing malware upload campaign".

    ### When two names are one

    PyPI does not compare names letter for letter. The rule, in the Python Packaging User Guide: "The name should be lowercased with all runs of the characters ., -, or _ replaced with a single - character." So `setup_tools`, `Setup.Tools` and `setup-tools` are one name, and `setuptools` is another. Exercise 1 writes that rule.

    You can ask PyPI whether it has a project of a name: its JSON API answers `GET /pypi/<project>/json` with the project's details, or with the code 404, not found. A 404 says only that there is no project now, not that anyone may take the name: PyPI's rules, PEP 541, let its maintainers "pre-emptively declare certain package names as unavailable for security reasons", and since the announcement of 2017, PyPI "dynamically prohibits the use of any standard library package names".

    ### How far apart two names are

    Spracklen et al. measure how alike two names are with the **Levenshtein distance**: "a measure of how many insertions, deletions, and substitutions are required for two strings to match". From `urllib3` to `urlib3` takes one deletion: distance 1. From `requests` to `reqeusts` takes two substitutions: distance 2. The course's `C.distance(a, b)` computes it, and Exercises 2 and 3 use it. Vu et al. set their threshold for how alike two names are at two, because it "allows us to identify the majority of known attacks (21 out of 36) and reduce the number of false positives": the other 15 lay beyond it. `pandas-sdk` is 4 edits from `pandas`.

    ### Names a model makes up

    Ask an LLM for code, and it may name a package that does not exist. Spracklen et al. call it *package hallucination*: "Package hallucination occurs when an LLM generates code that recommends or contains a reference to a package that does not actually exist." They generated 576,000 code samples in Python and JavaScript with 16 LLMs, and compared each Python name with the list of PyPI and each JavaScript name with that of npm, the index for JavaScript, both of 10 January 2024. Of 2.23 million package names, 440,445 (19.7%) were not on the list for their language: 205,474 different names. The average was at least 5.2% for the commercial models, OpenAI's GPT models, and 21.7% for the open-source models; GPT-4 Turbo had the lowest rate, 3.59%. Made-up names come back: of 500 prompts that had given a made-up name, each asked 10 more times, "43% of hallucinated packages were repeated in all 10 queries, while 39% did not repeat at all", and "81% of hallucinated packages are generated by only one model". And they rarely look like a typo: of the made-up Python names, "only 13.4% (10,263 of 76,489) have a Levenshtein distance of 1 or 2" to a real package, and 48.6% are 6 or more edits away. Spracklen et al. did not publish their list of names.

    The word slopsquatting came in April 2025: "Slopsquatting – when an LLM hallucinates a non-existent package name, and a bad actor registers it maliciously. The AI brother of typosquatting." Andrew Nesbitt posted that on Mastodon on 8 April 2025, with "Credit to @sethmlarson for the name": Seth Larson, security developer-in-residence at the Python Software Foundation, had suggested it in a conversation. *Slop* is a common scornful word for what AI models produce, as The Register put it. Larson, quoted there: "Users of LLM generated code, packages, and information should be double-checking LLM outputs against reality before putting any of that information into operation". How many installs come from made-up names is "likely impossible" to count, in Larson's words, without more openness from the makers of LLMs; the well-known case is the empty test package of the guess above.

    ### Before you install

    A name on PyPI means only that someone registered it. PyPI is, in the words of Python's security announcement of 2017, "an open publication platform with deliberately minimal barriers to participation": it removes a project after a report, when the project "is malware" or "is name squatting (package has no functionality or is empty)", in the words of PEP 541, and it does not check projects before they are published. Spracklen et al.: "the mere presence of a package in an open-source repository does not confirm its credibility." So a name from a model gets the checks of a name from a stranger: is it the package you meant, does it exist, and, when it does, what do its page and its code say.

    ### What the course did

    The course asked Qwen3-0.6B, which runs on your own computer, for the packages of ten tasks of Spoke & Chain, ten times each, with the seeds 0 to 9: "Which Python packages do I need to ...? Answer with one pip install command and nothing else.", with the task in place of the dots. Then it asked PyPI's JSON API, once, on 19 November 2026, whether it had a project of each name. Nothing was installed. Spracklen et al. kept their list of names to themselves; the names in this lesson come from a small LLM that anyone can run with these ten questions, and they are here to be checked. The course itself installs its packages from `uv.lock`, a file in the course that lists 101 packages, one version of each and the SHA-256 of each of their files, 323 in all: pins, as in the previous lesson. Those 101 names are `C.known()`.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **typosquatting** | publishing a package under a name that looks like a popular one, in the hope that people install it by mistake |
        | **slopsquatting** | registering a name that language models make up, so that whoever trusts the model installs the attacker's package |
        | **PyPI** | the Python Package Index, where pip and uv find a package by its name, and where anyone can publish one |
        | **normalized name** | a name as PyPI compares it: small letters, and every run of `.`, `-` and `_` as one `-` |
        | **Levenshtein distance** | how many letters must be put in, taken out or changed to turn one name into another |
        | `C.distance(a, b)` | the Levenshtein distance between `a` and `b` |
        """),
        L.watch_out("Never install a name from this lesson, or from an LLM's answer, to see what it is, and never register one. A name that PyPI did not have can be registered by anyone, with anything in it; and a name that it had is someone's package, good or bad."),
        mo.Html(C.names_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Normalize the name, and compare it with the packages you already use: the same name is the package you know.
    2. A name 1 or 2 edits from one you know is probably a typo: use the one you know.
    3. A name that PyPI does not have was made up, by a person or a model: find the package for the job another way.
    4. A name that PyPI has: read its page and its code before you install it, and then pin it, with a version and the hashes of its files.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell counts the course's run. `C.recorded_answers()` gives, for each task, the texts of Qwen3's ten answers; `C.names_in(text)` the package names in the pip install commands of an answer, normalized; `C.recorded_names()` every name in the 100 answers, each once; and `C.pypi()["names"]`, for each of those names, whether PyPI had it on 19 November 2026.
    """)
    return


@app.cell
def example_count():
    answers = C.recorded_answers()
    print(answers["barcode"][1], C.names_in(answers["barcode"][1]))

    names = C.recorded_names()
    on_pypi = C.pypi()["names"]   # True when PyPI had the name
    missing = []
    for name in names:
        if not on_pypi[name]:
            missing.append(name)
    print(len(names), "different names,", len(missing), "not on PyPI")
    print(missing[:5])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        `barcode_reader` becomes `barcode-reader`, the name PyPI compares. 98 of the 100 answers had a pip install command, and they gave 33 different names; 15 of those names were not on PyPI on 19 November 2026, and 31 answers named one of them. For a heart rate monitor, Qwen3 gave six names in ten answers, and PyPI had none of them. Spracklen et al. counted made-up names that came back in all ten tries, and this run has one: `barcode-reader` came every time.
        """),
        mo.Html(C.run_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_normalize():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. normalize: the name PyPI compares

        Write `normalize(name)`: `name` as PyPI compares it, in small letters, with every run of `.`, `-` and `_` replaced by a single `-`. The box goes through the letters of `name.lower()` and builds the result in `out`; `out.endswith("-")` is `True` when the last character of `out` is a `-`. Replace the `...`. The check calls `normalize("barcode_reader")`, then the names in the picture under it.
        """),
        L.task("Add a letter as it is; add a \"-\" for a separator only when out does not end with one already."),
    ])
    return


@app.cell(hide_code=True)
def exercise_normalize():
    box_normalize = L.code_box("normalize", lines=10)
    box_normalize
    return (box_normalize,)


@app.cell(hide_code=True)
def check_normalize(box_normalize):
    result_normalize = L.check("normalize", box_normalize.value)
    result_normalize
    return (result_normalize,)


@app.cell(hide_code=True)
def hints_normalize():
    L.hints(
        "normalize",
        [
            "*When two names are one* in *The idea*: small letters, and a run of separators counts as one.",
            "An `if` on whether `ch` is one of the three separators; under it, a second `if` that adds a `\"-\"` only when `out` does not end with one. Under `else`, add `ch` itself.",
            "`        if ch in ___:`",
        ],
        why="`if ch in \".-_\":` with `if not out.endswith(\"-\"):` and `out += \"-\"` under it, and `else:` with `out += ch`. `normalize(\"barcode_reader\")` gives `\"barcode-reader\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_normalize():
    mo.md(r"""
    `zope.interface` and `zope-interface` are one project to PyPI, and `my-._package` shows why a run counts as one: it becomes `my-package`. Every check in the rest of this lesson starts from the normalized name.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_near():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. near: names that are almost the same

        Write `near(name, known)`: the names in the list `known` that are 1 or 2 edits from `name` once it is normalized, in the order of `known`. `known` holds normalized names, and `found` collects the names. The box may use the course's `normalize` and `distance`. Python can chain two comparisons: `a <= b <= c` is `True` when `b` is at least `a` and at most `c`. Replace the `...`. The check calls `near("urlib3", KNOWN)`, where `KNOWN` holds the 101 names in the course's `uv.lock`, then the names in the picture under it.
        """),
        L.task("Keep each name in known that is 1 or 2 edits away, and only those."),
    ])
    return


@app.cell(hide_code=True)
def exercise_near():
    box_near = L.code_box("near", lines=8)
    box_near
    return (box_near,)


@app.cell(hide_code=True)
def check_near(box_near):
    result_near = L.check("near", box_near.value)
    result_near
    return (result_near,)


@app.cell(hide_code=True)
def hints_near():
    L.hints(
        "near",
        [
            "*How far apart two names are* in *The idea*: a typo of a name is a small distance away from it, and a name is no distance from itself.",
            "One `if` on `distance(name, k)`, with a comparison on each side of it, as in `a <= b <= c`; under it, `found.append(k)`.",
            "`        if ___ <= distance(name, k) <= 2:`",
        ],
        why="`if 1 <= distance(name, k) <= 2:` with `found.append(k)` under it. `near(\"urlib3\", KNOWN)` gives `[\"urllib3\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_near():
    mo.md(r"""
    `near` finds `urlib3` and `setup-tools`, two of the 11 names of 2017. Even with `pandas` in `known`, it would not find `pandas-sdk`, 4 edits away: a distance of 1 or 2 catches typos, not added words. And `numpy` is close to `sympy`: these two names, both in `uv.lock`, are 2 edits apart, so a name that is close to a known one can be a known one itself. Exercise 3 looks in `known` first.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. verdict: what to do with a name

        Write `verdict(name, known, on_pypi)`: what to do with a name, as one of four texts, checked in this order. `"known"` when the normalized name is in `known`; `"close to "` with the first name that `near` gives, when it gives one; `"not on PyPI"` when the name is not in the list `on_pypi`; and `"read its page first"` for every other name. The box may use the course's `normalize`, `distance` and `near`; put what `near` gives in `close`, with `close = near(name, known)`, and `if close:` is true when the list `close` is not empty. Replace the `...`. The check calls `verdict("barcode_reader", KNOWN, ON_PYPI)`, where `ON_PYPI` holds the names of the course's run and of this lesson's examples that PyPI had on 19 November 2026, then the names in the picture under it.
        """),
        L.task("Known first, then a close name, then PyPI; the last line of the box is for every other name."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=11)
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
            "Steps 1 to 4 of the recipe, in that order: each step that applies ends the function with its text.",
            "Three `if`s, each with a `return` under it: on whether `name` is in `known`; on `close`, after `close = near(name, known)`; and on whether `name` is not in `on_pypi`.",
            "`        return \"close to \" + ___`",
        ],
        why="`if name in known:` with `return \"known\"`, then `close = near(name, known)` and `if close:` with `return \"close to \" + close[0]`, then `if name not in on_pypi:` with `return \"not on PyPI\"`. `verdict(\"barcode_reader\", KNOWN, ON_PYPI)` gives `\"not on PyPI\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_verdict():
    mo.md(r"""
    `urlib3` gets `"close to urllib3"` before PyPI is even asked: that it is not on PyPI today matters less than what it tries to look like. `pdfbox` gets `"read its page first"`: it is on PyPI and close to nothing in `uv.lock`, and only its page and its code can say more.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Qwen3's answers, judged

    Pick a task of Spoke & Chain. The table shows the ten answers that the course recorded, the names in each, their verdicts, and whether PyPI had each name on 19 November 2026. The verdicts are your `verdict`'s once Exercise 3 is right, and the course's until then.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    demo_task = mo.ui.dropdown(options=C.task_options(L.lang), value=C.task_text("barcode", L.lang), label="The task")
    demo_task
    return (demo_task,)


@app.cell(hide_code=True)
def demo_look(box_verdict, demo_task, result_verdict):
    _verdict = C.mine(box_verdict.value, result_verdict.state == "correct", "verdict")
    _out = mo.Html(C.task_picture(demo_task.value, L.lang, _verdict))
    _out
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Ask Qwen3 yourself

    Write a task of your own, and let Qwen3 answer, sampled as in the course's run; another seed gives another try. Under the answer you read the verdict of each name, by your `near` and `verdict` once their exercises are right. The course has PyPI's answer only for the names it recorded: for a new name, the verdict says whether it is in `uv.lock` or close to a name in it, and otherwise that you look it up yourself, on pypi.org, without installing it.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*The LLM Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the answers the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="read the speed sensor of an e-bike over Bluetooth", label="Which Python packages do I need to",
                           full_width=True)
    seed_box = mo.ui.number(start=0, stop=99, step=1, value=0, label="Seed")
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([demo_text, mo.hstack([seed_box, ask_button], justify="start", gap=1.5)])
    return ask_button, demo_text, seed_box


@app.cell(hide_code=True)
def demo_ask(ask_button, box_near, box_verdict, demo_text, result_near, result_verdict, seed_box):
    _msgs = C.demo_messages(demo_text.value)
    _near = C.mine(box_near.value, result_near.state == "correct", "near")
    _verdict = C.mine(box_verdict.value, result_verdict.state == "correct", "verdict")
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to let Qwen3 answer.*")
    elif not demo_text.value.strip():
        _out = mo.md("*Write a task in the box first.*")
    else:
        _words = {"title": "What Qwen3 answers", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds.",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(chat.demo(_msgs, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang,
                                 max_tokens=C.MAX_TOKENS, seed=int(seed_box.value),
                                 note=lambda _text: C.live_note(_text, L.lang, _near, _verdict), **chat.SAMPLING))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_exists():
    mo.md(r"""
    ### Exercise 4. A name that exists

    For a Garmin FIT file, Qwen3 named `pygears` once in ten answers, and PyPI had a project of that name on 19 November 2026.
    """)
    return


@app.cell(hide_code=True)
def choice_exists_cell():
    choice_exists = L.choice(
        "exists",
        question="What does that tell you?",
        options={
            "right": "That someone published a project under that name",
            "reads": "That pygears reads FIT files, since it is a real package",
            "safe": "That it is safe: PyPI removes malware",
        },
        correct="right",
        why={
            "right": "A project on PyPI says only that someone registered the name: PyPI checks no project before it is published. What pygears does, and who makes it, is on its page and in its code: that is the verdict `\"read its page first\"`. In the words of Spracklen et al.: \"the mere presence of a package in an open-source repository does not confirm its credibility.\"",
            "reads": "That a project exists says nothing about what it does: only its page and its code can tell. For the same task, Qwen3 also named `fitpy` and `fitbit`. See *Before you install* in *The idea*.",
            "safe": "PyPI removes malware after a report, as with the over 500 packages of 2024; it does not check projects before they are published. See *Before you install* in *The idea*.",
        },
    )
    choice_exists
    return (choice_exists,)


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.md(r"""
    ### Exercise 5. A check on distance

    Spoke & Chain's scripts install the names that an LLM gives, but first they warn when a name is 1 or 2 edits from a name in `KNOWN`, the 101 names in the course's `uv.lock`. One day, someone registers `bluetooth-hr-monitor` on PyPI, a name that Qwen3 gave for a heart rate monitor.
    """)
    return


@app.cell(hide_code=True)
def choice_distance_cell():
    choice_distance = L.choice(
        "distance-check",
        question="What happens the next time Qwen3 gives that name?",
        options={
            "typo": "It warns: a made-up name is a typo of a real package",
            "right": "The script installs it without any warning",
            "pip": "pip refuses it: PyPI checks every new project first",
        },
        correct="right",
        why={
            "right": "`bluetooth-hr-monitor` is more than 2 edits from every name in `KNOWN`, so the check stays quiet, and pip installs whatever the new owner put in it. Of the made-up names that Spracklen et al. counted, only 13.4% were within distance 2 of a real package; of the 15 names in the course's run that PyPI did not have, none was within 2 of a name in the course's `uv.lock`. A distance check catches typos; a name from a model needs looking up, and reading.",
            "typo": "Most made-up names are no typo: only 13.4% were within distance 2 of a real package in the count of Spracklen et al., and `bluetooth-hr-monitor` is more than 2 edits from every name in `KNOWN`. See *Names a model makes up* in *The idea*.",
            "pip": "In Python's security announcement of 2017, PyPI is \"an open publication platform with deliberately minimal barriers to participation\": it removes projects after a report, not before they are published. See *Before you install* in *The idea*.",
        },
    )
    choice_distance
    return (choice_distance,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Is this the model you meant?](/course/open/rai-hub-integrity?lang=en) pinned a model file to a commit and a SHA-256, and a lock file such as the course's `uv.lock` does the same for packages, once the names are right. This lesson was about the names: one close to a package you know, and one that a model made up, which PyPI may not have today and anyone may register tomorrow. [Using code from modules](/course/open/py-imports?lang=en) brought in modules with `import`; every package you import got onto your computer by its name. The next lesson, *Red teaming*, attacks a system on purpose, to find its weak spots before someone else does.

    **An everyday example.** A note says to buy "Kwik-Fix" glue. A tube called "Kwick-Fix" on the next shelf is not that glue, and a tube with exactly the name on the note is only as good as whoever made it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_distance, choice_exists, result_near, result_normalize, result_verdict):
    L.wrap_up(result_normalize, result_near, result_verdict, choice_exists, choice_distance)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
