# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-ship-app. Route: course/route/lessen/bouwen.yaml. Checks, the bank's settings files, the log of two days and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-ship-app", "en")
    C = L.checks   # the bank's settings files, the log of two days and the pictures
    DEFAULTS = C.DEFAULTS             # what the app uses for a setting that a settings file leaves out
    HISTORY = C.HISTORY               # the bank's settings files, oldest first, each with its "version"
    SETTINGS_TEXT = C.SETTINGS_TEXT   # the settings file of 2026-11-12, as text
    LOG = C.LOG                       # the log of two days: one line of JSON text per request


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_threshold():
    _f = C.findings()
    _lo, _mid = _f["cache"][0.75], _f["cache"][0.85]
    quick_threshold = L.choice(
        "quick-threshold",
        question="A semantic cache gives an earlier answer from a similarity of 0.85 on. What happens when you lower that to 0.75?",
        options={
            "fewer": "Fewer hits, and each one fits better",
            "more": "More hits, more of them wrong",
            "same": "The same hits, only found faster",
        },
        correct="more",
        why={
            "more": "From 0.75 on the course's cache answered {h} of the {n} questions instead of {h0}, and {o} got the answer for another topic instead of {o0}.".format(
                h=_lo[0], n=_f["n"], h0=_mid[0], o=_lo[1], o0=_mid[1]),
            "fewer": "A lower threshold lets more earlier answers through, not fewer: from 0.75 on the course's cache answered {h} questions instead of {h0}. See [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en).".format(
                h=_lo[0], h0=_mid[0]),
            "same": "The threshold decides which earlier answers count as close enough, so the hits change: from 0.75 on the course's cache answered {h} questions instead of {h0}. See [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en).".format(
                h=_lo[0], h0=_mid[0]),
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    _v1, _v2 = (s for s, _, _ in C.DAYS)
    mo.md(r"""
    ## Why this matters

    In this module you looked at the bank's assistant from every side: an eval set, users, traces, guardrails, cost and a cache. Each of those changes the app, and an app that customers use keeps changing: a new prompt, a new model, a cache. Version 2 of the prompt passed the eval set for {p2} of the {n} test questions against {p1} for version 1, and still made something up in {m2} replies against {m1}, by the course's notes in [Offline, online and what it is worth](/course/open/build-eval-online?lang=en). The eval set has no check for that.

    Once such a version is live, you want to know which version answered which customer, and you want to go back in one step, before you know the cause.

    **An everyday example.** A bakery tries a new bread recipe. The old recipe card stays in the drawer, and every order slip says which recipe the bread was baked from. When customers complain, the bakery bakes the old recipe again the next morning, and only then looks for what went wrong with the new one.
    """.format(n=_f["n"], p1=_f["passed"][_v1], p2=_f["passed"][_v2], m1=_f["made_up"][_v1], m2=_f["made_up"][_v2]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **AI engineering** | building an application on a model someone else trained: the work is in the prompts, the data, the evals and the code around the model, not in training it |
    | **rollback** | going back to an earlier version that worked; here, loading its settings file again |

    An LLM app that customers use has four parts: its settings, your code, the model and the log. The settings are everything you can change without changing code: the model, the version of the prompt, `max_tokens`, the temperature, the cache and its threshold. They live in a file with a version. Every change is a new file, and the old ones stay.

    Your code reads the settings when it starts, and uses a default for anything the file leaves out. Every request writes one line to the log: the time, the version of the settings, the question, the seconds, the tokens and what the checks said. A line is JSON text, so a program can read it back and count.

    A rollback loads an older settings file. No code changes and no model is trained; one file changes. Then you look for the cause, and ship the fix as a new version.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: shipping a change

    1. **A file per version:** put every setting in a file with a version: the model, the prompt version, `max_tokens`, the temperature, the cache. Keep the old files.
    2. **Before customers:** run the eval set on the new version and look at its regressions, as in [An eval set and regression tests](/course/open/build-eval-set?lang=en).
    3. **Then some customers:** give the new version to part of the customers first, as in [Offline, online and what it is worth](/course/open/build-eval-online?lang=en), and log every request with the version of its settings.
    4. **Count per version:** from the log, count for each version what the checks said, what people flagged, the seconds and the tokens.
    5. **Back first, then the cause:** when a version does worse, roll back to the last good one, and look for the cause after. The fix is a new version that goes through the same steps.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _first = C.DAYS[0]
    mo.md(r"""
    ### Worked example

    `SETTINGS_TEXT` is the settings file of 2026-11-12 as text: `{text}`. `json.loads(SETTINGS_TEXT)` turns it into a dictionary. The file names only what this version sets itself; for everything else the app uses `DEFAULTS`: {defaults}.

    `HISTORY` holds the bank's {k} settings files, oldest first: version 1 of the prompt on 2026-11-02, version 2 on 2026-11-09, and on 2026-11-12 the semantic cache of [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en), from a similarity of 0.75 on.

    `LOG` holds {lines} lines of the log, one per request. On {d1} the {n} test questions came in under the settings of {s1}, and on {d2} the same {n} under those of {s2}. `LOG[0]` is:

    ```json
    {line}
    ```

    The seconds and tokens are what the course measured when it ran each version; the times of day are the course's own. `passed` is what the eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en) said of the reply, and `made_up` is the course's note on it from [Offline, online and what it is worth](/course/open/build-eval-online?lang=en). A real app does not know that at once: it learns it from thumbs and from people who read a sample of the replies.
    """.format(text=SETTINGS_TEXT, defaults=", ".join(f"`{k}` {json.dumps(v)}" for k, v in DEFAULTS.items()), k=len(HISTORY),
               lines=_f["lines"], n=_f["n"], d1=_first[2], s1=_first[0], d2=C.DAYS[1][2], s2=C.DAYS[1][0], line=LOG[0]))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _v1, _v2 = (s for s, _, _ in C.DAYS)
    _h, _o, _cp, _fp = _f["cache"][0.75]
    mo.md(r"""
    ### What the course measured

    Counted by the version of the settings, the log says: under {v1} the eval set passed {p1} of the {n} replies, and {m1} made something up; under {v2} it passed {p2}, and {m2} made something up. Version 2 does better on the eval set and makes more up; by the course's notes it also had another problem in {o2} replies against {o1}, so the two versions helped about as many customers, {h2} against {h1}. Of the questions that version 1 passed, version 2 failed {reg}. Speed is not the difference: the {n} replies of each version took {s1} and {s2} seconds in all.

    The settings of 2026-11-12 turned on the cache from 0.75. In [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en) that cache answered {h} of the {n} questions, and {o} of them got the answer for another topic; the eval set passed {cp} of its answers against {fp} of the model's own. So on 2026-11-12 the bank goes back to 2026-11-09 first. Whether it then goes back further, to 2026-11-02, depends on what weighs more for it: fewer other problems and a better eval set, or fewer replies that make something up.
    """.format(v1=_v1, v2=_v2, n=_f["n"], p1=_f["passed"][_v1], p2=_f["passed"][_v2], m1=_f["made_up"][_v1], m2=_f["made_up"][_v2],
               o1=_f["other"][_v1], o2=_f["other"][_v2], h1=_f["helped"]["v1"], h2=_f["helped"]["v2"],
               reg=_f["regressions"], s1=L.number(_f["seconds"][_v1], 1), s2=L.number(_f["seconds"][_v2], 1), h=_h, o=_o, cp=_cp, fp=_fp))
    return


@app.cell(hide_code=True)
def picture_measured():
    L.watch_out("Keep the old settings files, and write the version of the settings in every line of the log. Without the version, a log cannot tell you which change made things worse; without the old file, there is nothing to go back to.")
    return


@app.cell(hide_code=True)
def text_settings():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Settings with defaults

        `settings_from(text)` should turn the text of a settings file into a dictionary with every setting the app needs: what the file says, and for each key of `DEFAULTS` that the file leaves out, its default. The start code reads the file with `json.loads`. `for key in DEFAULTS:` goes over the keys of `DEFAULTS`, and `key in a_dict` is `True` when the dictionary has that key. The picture shows what your function gives for `SETTINGS_TEXT`.
        """),
        L.task("Add each key of `DEFAULTS` that `settings` does not have yet, with its default."),
    ])
    return


@app.cell(hide_code=True)
def exercise_settings():
    box_settings = L.code_box("settings_from", lines=7)
    box_settings
    return (box_settings,)


@app.cell(hide_code=True)
def check_settings(box_settings):
    result_settings = L.check("settings_from", box_settings.value)
    result_settings
    return (result_settings,)


@app.cell(hide_code=True)
def hints_settings():
    L.hints(
        "settings_from",
        [
            "The file names only what this version sets itself; the app still needs every setting.",
            "Go over the keys of `DEFAULTS` with `for`, and add a key only when it is not in `settings`.",
            "Under that `if`, write `settings[key] = ___`.",
        ],
        why="""```python
def settings_from(text):
    settings = json.loads(text)
    for key in DEFAULTS:
        if key not in settings:
            settings[key] = DEFAULTS[key]
    return settings
```

The file stays short and says only what this version sets. When two versions differ, that is what you compare.""",
    )
    return


@app.cell(hide_code=True)
def text_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Counted per version

        `count_by_settings(lines, field)` should count, for each version of the settings in the log, the lines where `field` is `True`. Each line is JSON text: the start code reads it with `json.loads` and takes the version from `entry["settings"]`. Give every version in the log a count, also when it is 0. The picture counts `made_up` and `passed` in `LOG`.
        """),
        L.task("Start each new version at 0, and add 1 when `entry[field]` is `True`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_by_settings", lines=11)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_by_settings", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_by_settings",
        [
            "`counts` maps a version to its count: a version gets its key the first time it comes by, with 0.",
            "Two `if`s inside the loop: one for a version that is not in `counts` yet, and one for `entry[field]`.",
            "Under the second `if`, write `counts[version] = ___`.",
        ],
        why="""```python
def count_by_settings(lines, field):
    counts = {}
    for line in lines:
        entry = json.loads(line)
        version = entry["settings"]
        if version not in counts:
            counts[version] = 0
        if entry[field]:
            counts[version] = counts[version] + 1
    return counts
```

The same function counts any field that is `True` or `False`: the eval set, the notes, and with one more field also the thumbs down.""",
    )
    return


@app.cell(hide_code=True)
def text_good():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The way back

        `bad` is a set of versions that did worse. `last_good(history, bad)` should give the newest settings in `history` whose version is not in `bad`, or `None` when there is none. `history` is oldest first, like `HISTORY`. The start code goes through it and keeps every settings dictionary, so it gives the newest, bad or not. The picture shows two rollbacks: with 2026-11-12 bad, and with 2026-11-09 bad as well.
        """),
        L.task("Keep a settings dictionary only when its version is not in `bad`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_good():
    box_good = L.code_box("last_good", lines=6)
    box_good
    return (box_good,)


@app.cell(hide_code=True)
def check_good(box_good):
    result_good = L.check("last_good", box_good.value)
    result_good
    return (result_good,)


@app.cell(hide_code=True)
def hints_good():
    L.hints(
        "last_good",
        [
            "The loop goes from old to new, so what it keeps last is the newest it kept.",
            "Put an `if` before `good = settings` that looks at the version of `settings`.",
            "Write `if ___ not in bad:`.",
        ],
        why="""```python
def last_good(history, bad):
    good = None
    for settings in history:
        if settings["version"] not in bad:
            good = settings
    return good
```

Go back as little as you can: every version you skip also takes away what it did well.""",
    )
    return


@app.cell(hide_code=True)
def text_first_step():
    mo.md(r"""
    ### Exercise 4. The first step
    """)
    return


@app.cell(hide_code=True)
def choice_first_step_cell():
    _f = C.findings()
    _h, _o = _f["cache"][0.75][:2]
    choice_first_step = L.choice(
        "first-step",
        question="The settings of 2026-11-12 are live, and customers get answers meant for another topic. What do you do first?",
        options={
            "back": "Load the settings of 2026-11-09 again",
            "fix": "Find a better threshold with the eval set, and ship that",
            "wait": "Keep it running and read the log next week",
        },
        correct="back",
        why={
            "back": "A rollback changes one file and stops the wrong answers now. Then look for a better threshold, and ship it as a new version that goes through the eval set first.",
            "fix": "That is a good next step, but it takes time, and in the meantime customers keep getting answers for another topic. Roll back first: load the settings of 2026-11-09 again.",
            "wait": "Every day costs customers wrong answers: from 0.75 on, {o} of the {h} answers from the cache were for another topic. Roll back first.".format(o=_o, h=_h),
        },
    )
    choice_first_step
    return (choice_first_step,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A line of the log is a small trace, as in [Seeing what happened](/course/open/build-tracing?lang=en), and counting it per version is the online evaluation of [Offline, online and what it is worth](/course/open/build-eval-online?lang=en). The settings file holds the prompt versions of [Prompts as code: templates and versions](/course/open/build-prompt-templates?lang=en) and the cache of [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en). [Ask before acting](/course/open/build-human-in-loop?lang=en) adds a person to the app, for the steps it should not take alone.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first_step, result_count, result_good, result_settings):
    L.wrap_up(result_settings, result_count, result_good, choice_first_step)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
