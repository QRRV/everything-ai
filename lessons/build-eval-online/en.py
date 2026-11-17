# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-eval-online. Route: course/route/lessen/bouwen.yaml. Checks, the notes on version 2, the week of chats and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-eval-online", "en")
    C = L.checks   # versions 1 and 2 of the bank's assistant, the course's notes on both, a week of chats, and the pictures
    WEEK = C.WEEK
    WEEK_2 = C.WEEK_2   # weeks with other coin flips, which the checks also try
    WEEK_3 = C.WEEK_3


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [An eval set and regression tests](/course/open/build-eval-set?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_regression():
    quick_regression = L.choice(
        "quick-regression",
        question="What does a regression test between two versions show?",
        options={
            "total": "How many more inputs pass than with the old version",
            "broke": "The inputs that passed before and fail now",
            "both": "The inputs that fail with both versions",
        },
        correct="broke",
        why={
            "broke": "A regression is an input that went from right to wrong. The total can go up while some inputs break: the regression test shows which ones.",
            "total": "The total can go up while inputs break. A regression test lists the inputs that passed with the old version and fail with the new one. See [An eval set and regression tests](/course/open/build-eval-set?lang=en).",
            "both": "Those were wrong already. A regression is an input that passed with the old version and fails with the new one. See [An eval set and regression tests](/course/open/build-eval-set?lang=en).",
        },
    )
    quick_regression
    return (quick_regression,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    In [An eval set and regression tests](/course/open/build-eval-set?lang=en), version 2 of the bank's assistant passed every check for {e2} of the {n} inputs, against {e1} for version 1. Then the course read the replies of version 2 too, with the rules of [Look at the mistakes first](/course/open/build-error-analysis?lang=en): version 2 made something up in {m2} replies, version 1 in {m1}. The eval set has no check for *made up*, so it could not see that. Customers would feel it: a made-up deadline is what they act on.

    No eval set holds every message customers will send, or knows what they do with a reply. That you only find out with real customers, while the app runs, and even then you can measure only some of it.

    **An everyday example.** A restaurant tries a new recipe in its own kitchen first: the cooks taste it. Then it serves the new dish at half of the tables and the old one at the rest, and looks at what comes back on the plates. A few guests fill in the card on the table; most say nothing.
    """.format(n=_f["n"], e1=_f["eval"]["v1"], e2=_f["eval"]["v2"], m1=_f["kinds"]["v1"]["made-up"], m2=_f["kinds"]["v2"]["made-up"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **offline evaluation** | testing a change on a fixed set of examples before users get it, such as with an eval set |
    | **online evaluation** | measuring how the app does with real users while it runs: what they click, whether their task got done, what each chat cost |
    | **user feedback** | what users tell you about a reply: on purpose, such as a thumb up or down or a comment, or by what they do, such as asking again or asking for a person |
    | **A/B test** | each user gets version A or version B by chance, such as a coin flip, and you compare the same number for the two groups |
    | **data flywheel** | the chats that went wrong are read and become new inputs for the eval set, so that every next version is tested on them |

    What a version is worth shows in what it does for the business: how many customers got their task done, and what that cost. A version that costs less per chat but helps fewer customers can cost more for each task done.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.loop_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: from offline to online

    1. **Offline:** run the eval set on the new version. With regressions you cannot accept, it goes no further.
    2. **A/B test:** give the new version to customers chosen by chance, and the old one to the rest.
    3. **Measure:** for each version, the share of thumbs up, the share of tasks done, and the cost per task done.
    4. **Feed back:** read the chats that went wrong, above all the ones the eval set passed, and add them to the eval set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The bank has no customers, so the course plays them. The {n} test messages come in as {n} chats in one week, and a coin flip gives each customer version 1 or version 2. A customer is helped when the course's note on their reply has no *another problem*, nothing *made up* and no *sends elsewhere*; a wrong topic field they never see, a reply that is too long still helps, and they do not mind the bank admitting a mistake. A task counts as done when the reply helped. One customer in {t}, picked by chance, clicks a thumb: up when helped, down when not.

    These played customers know the facts, so they notice every made-up reply, and they click by chance. Real customers often cannot tell that a reply is made up, and whether they click can depend on how the chat went. The week shows how the numbers of an online evaluation work, not how real customers behave.

    `WEEK` holds the chats, each a dictionary: `"message"`, `"version"` (`"v1"` or `"v2"`), `"reply"`, `"prompt_tokens"` and `"output_tokens"`, `"seconds"`, `"thumb"` (`"up"`, `"down"` or `None` when the customer clicked nothing), `"done"` (`True` when the task got done) and `"passed"` (`True` when the reply passed every check of the eval set).
    """.format(n=len(WEEK), t=C.THUMB))
    return


@app.cell(hide_code=True)
def picture_week():
    mo.Html(C.week_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_read():
    _f = C.findings()
    mo.md(r"""
    1. **Who got what:** the coin flips gave {w1} customers version 1 and {w2} version 2.
    2. **Feedback:** {t} customers clicked a thumb, {u} of them up: {up}% up. Tasks got done in {d} of the {n} chats, {dp}%.
    3. **Cost:** at the made-up prices of [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en), {pi} and {po} dollars for a million tokens in and out, 1,000 chats of version 1 cost {c1} dollars, and 1,000 of version 2, {c2}: its replies are shorter.
    """.format(w1=_f["week_n"]["v1"], w2=_f["week_n"]["v2"], t=_f["thumbs"], u=_f["ups"], up=C.pct(_f["up_share"]),
               d=sum(c["done"] for c in WEEK), n=_f["n"], dp=C.pct(_f["done_all"]), pi=C.PRICE_IN, po=C.PRICE_OUT,
               c1=C.money(_f["cost_1000"]["v1"], L.lang), c2=C.money(_f["cost_1000"]["v2"], L.lang)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _k = _f["kinds"]
    mo.md(r"""
    ### What the course measured

    The notes on version 2 point both ways: *another problem* went from {o1} replies to {o2} and *admits a mistake* from {a1} to {a2}, but *made up* went from {m1} to {m2}. If every customer got version 1, it would help {h1} of the {n}; version 2, {h2}. That gap rests on few replies: {only1} replies helped only with version 1, {only2} only with version 2. On these {n} messages the two are about even, and how the course noted a few replies could tip it either way.

    In the week the course played, the task got done for {d1}% of the customers with version 1 and {d2}% with version 2. Other coin flips give other weeks: in {r} weeks, version 1 came out ahead {a} times and version 2 {b} times, with version 2 anywhere from {lo} points worse to {hi} points better. With {n} customers the coin flips decide more than the versions do: an A/B test needs far more customers before a few points mean something.

    What it is worth: 1,000 chats cost {c1} dollars with version 1 and {c2} with version 2, but 1,000 tasks done cost {k1} against {k2}, almost the same. Version 2 is cheaper per chat, but it helps {dh} customers fewer, so per task done the saving is almost gone.
    """.format(o1=_k["v1"]["other-problem"], o2=_k["v2"]["other-problem"], a1=_k["v1"]["admits"], a2=_k["v2"]["admits"],
               m1=_k["v1"]["made-up"], m2=_k["v2"]["made-up"], h1=_f["helped"]["v1"], h2=_f["helped"]["v2"], n=_f["n"],
               e1=_f["eval"]["v1"], e2=_f["eval"]["v2"], d1=C.pct(_f["done"]["v1"]), d2=C.pct(_f["done"]["v2"]), r=C.RUNS,
               a=_f["v1_ahead"], b=_f["v2_ahead"], lo=-_f["spread"][0], hi=_f["spread"][1], only1=_f["only"]["v1"], only2=_f["only"]["v2"], dh=_f["helped"]["v1"] - _f["helped"]["v2"],
               c1=C.money(_f["cost_1000"]["v1"], L.lang), c2=C.money(_f["cost_1000"]["v2"], L.lang),
               k1=C.money(_f["cost_done"]["v1"], L.lang), k2=C.money(_f["cost_done"]["v2"], L.lang)))
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.notes_picture(L.lang)),
        L.watch_out("Feedback and chats hold what customers wrote. Tell them you use it, keep it safe, and let a person check what goes back into the eval set."),
    ])
    return


@app.cell(hide_code=True)
def text_up():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The share of thumbs up

        `up_share(chats)` should give the share of thumbs up among the chats that got a thumb: the thumbs up divided by the thumbs up and down together. Chats where the customer clicked nothing do not count. The start code divides by all chats. The picture under the check puts your share beside the share of tasks done, for the whole week and for each version.
        """),
        L.task("Divide by the chats that got a thumb."),
    ])
    return


@app.cell(hide_code=True)
def exercise_up():
    box_up = L.code_box("up_share", lines=10)
    box_up
    return (box_up,)


@app.cell(hide_code=True)
def check_up(box_up):
    result_up = L.check("up_share", box_up.value)
    result_up
    return (result_up,)


@app.cell(hide_code=True)
def hints_up():
    L.hints(
        "up_share",
        [
            "Most customers click nothing. A share of thumbs up is out of the thumbs, not out of the chats.",
            "The start code already counts `ups` and `downs`. Divide by the number of chats that got a thumb.",
            "Make the last line `return ups / (___)`.",
        ],
        why="""```python
def up_share(chats):
    ups = 0
    downs = 0
    for chat in chats:
        if chat["thumb"] == "up":
            ups = ups + 1
        if chat["thumb"] == "down":
            downs = downs + 1
    return ups / (ups + downs)
```

Only some customers click, so this share rests on few chats. Compare it with the share of tasks done in the picture.""",
    )
    return


@app.cell(hide_code=True)
def text_done():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Tasks done for one version

        `done_share(chats, version)` should give, of the chats that got `version`, the share where the task got done: `chat["done"]` is `True`. That is the number an A/B test compares. The start code counts the chats of both versions.
        """),
        L.task("Count only the chats of `version`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_done():
    box_done = L.code_box("done_share", lines=10)
    box_done
    return (box_done,)


@app.cell(hide_code=True)
def check_done(box_done):
    result_done = L.check("done_share", box_done.value)
    result_done
    return (result_done,)


@app.cell(hide_code=True)
def hints_done():
    L.hints(
        "done_share",
        [
            "Two counts, both only for the chats of one version: how many chats, and how many of them got their task done.",
            "Put an `if` around the two counts in the loop that asks whether the chat got the version you count.",
            "In the loop: `if ___ == version:`, with the `if` on `chat[\"done\"]` and `count = count + 1` one step further in.",
        ],
        why="""```python
def done_share(chats, version):
    done = 0
    count = 0
    for chat in chats:
        if chat["version"] == version:
            if chat["done"]:
                done = done + 1
            count = count + 1
    return done / count
```

Each customer saw one version, so each version has its own group. The coin flip makes the two groups alike in everything but the version, as far as chance allows.""",
    )
    return


@app.cell(hide_code=True)
def text_passed_down():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. What the eval set missed

        `passed_down(chats)` should give the messages of the chats with a thumb down that the eval set passed, in their order. The eval set needs these inputs most: it passed the reply, and the customer gave it a thumb down. The start code gives every chat with a thumb down.
        """),
        L.task("Keep a chat only when the eval set passed it too."),
    ])
    return


@app.cell(hide_code=True)
def exercise_passed_down():
    box_passed_down = L.code_box("passed_down", lines=8)
    box_passed_down
    return (box_passed_down,)


@app.cell(hide_code=True)
def check_passed_down(box_passed_down):
    result_passed_down = L.check("passed_down", box_passed_down.value)
    result_passed_down
    return (result_passed_down,)


@app.cell(hide_code=True)
def hints_passed_down():
    L.hints(
        "passed_down",
        [
            "Two conditions: the thumb is down, and the eval set passed the reply.",
            "A second `if` inside the first, which asks whether the eval set passed the chat.",
            "Under `if chat[\"thumb\"] == \"down\":`, write `if ___:`, with the `append` one step further in.",
        ],
        why="""```python
def passed_down(chats):
    messages = []
    for chat in chats:
        if chat["thumb"] == "down":
            if chat["passed"]:
                messages.append(chat["message"])
    return messages
```

These messages go into the eval set, with a note of what a good reply needs. That is the data flywheel: what goes wrong online is tested offline from then on.""",
    )
    return


@app.cell(hide_code=True)
def text_worth():
    mo.md(r"""
    ### Exercise 4. What a version is worth
    """)
    return


@app.cell(hide_code=True)
def choice_worth_cell():
    _f = C.findings()
    _c = {v: C.money(_f["cost_1000"][v], L.lang) for v in ("v1", "v2")}
    _k = {v: C.money(_f["cost_done"][v], L.lang) for v in ("v1", "v2")}
    choice_worth = L.choice(
        "worth",
        question="Which number tells you best what a version is worth to the bank?",
        options={
            "chat": "What 1,000 chats cost",
            "eval": "How many inputs pass the eval set",
            "done": "What 1,000 tasks done cost",
        },
        correct="done",
        why={
            "done": "A chat that does not help costs tokens too, and the customer is not helped. Version 2 costs {c2} dollars for 1,000 chats against {c1}, but for 1,000 tasks done {k2} against {k1}: almost the same, because it helps fewer customers.".format(
                c1=_c["v1"], c2=_c["v2"], k1=_k["v1"], k2=_k["v2"]),
            "chat": "Version 2 is cheaper per chat, {c2} against {c1} dollars for 1,000 chats, but it helps fewer customers, {h2} against {h1} of the {n}. For 1,000 tasks done the two cost almost the same, {k2} against {k1}.".format(
                c1=_c["v1"], c2=_c["v2"], k1=_k["v1"], k2=_k["v2"], h1=_f["helped"]["v1"], h2=_f["helped"]["v2"], n=_f["n"]),
            "eval": "The eval set passed more replies of version 2, {e2} against {e1}, but it has no check for made up. What the bank gets is tasks done, and what it pays for them.".format(
                e1=_f["eval"]["v1"], e2=_f["eval"]["v2"]),
        },
    )
    choice_worth
    return (choice_worth,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try other weeks

    Each number is a week with other coin flips: other customers get version 1 and version 2. Watch how much the result of the A/B test moves.
    """)
    return


@app.cell(hide_code=True)
def settings_runs():
    runs_slider = mo.ui.slider(start=1, stop=C.RUNS, value=C.SEED, label="Week", show_value=True)
    runs_slider
    return (runs_slider,)


@app.cell(hide_code=True)
def demo_runs(runs_slider):
    mo.Html(C.runs_picture(runs_slider.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en) is the offline half of this loop; the chats that come back from the online half are what [Look at the mistakes first](/course/open/build-error-analysis?lang=en) reads. A check for *made up* needs someone who reads the meaning, as in [Who grades the answer](/course/open/build-judges?lang=en). [Seeing what happened](/course/open/build-tracing?lang=en) shows what happened inside one chat, step by step, and [Cost, latency and caching answers](/course/open/build-cost-latency-cache?lang=en) makes each chat cheaper.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_worth, result_done, result_passed_down, result_up):
    L.wrap_up(result_up, result_done, result_passed_down, choice_worth)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
