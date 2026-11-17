# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-eval-set. Route: course/route/lessen/bouwen.yaml. Checks, the two versions, the course's checks and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-eval-set", "en")
    C = L.checks   # the eval set, both versions of the bank's assistant, the course's checks and the pictures
    reply_of = C.reply_of
    RATE = C.RATE
    PASSED_V1 = C.passed("v1")
    PASSED_V2 = C.passed("v2")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Checking code that AI wrote](/course/open/build-check-ai-code?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="All eight of Qwen3's count_vowels counted only small letters. Which test catches that bug?",
        options={
            "banana": 'assert count_vowels("banana") == 3',
            "empty": 'assert count_vowels("") == 0',
            "apple": 'assert count_vowels("Apple") == 2',
        },
        correct="apple",
        why={
            "apple": "The bug shows only with a capital: \"Apple\" has two vowels, and a function that counts only small letters gives 1. A test catches a bug only when its case lets the bug show.",
            "banana": "All its letters are small: a function that counts only small letters gives 3 as well, so the test passes with the bug. See [Checking code that AI wrote](/course/open/build-check-ai-code?lang=en).",
            "empty": "An empty text has no letters at all: the test passes with the bug. See [Checking code that AI wrote](/course/open/build-check-ai-code?lang=en).",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Look at the mistakes first](/course/open/build-error-analysis?lang=en), the biggest kind of mistake of the bank's assistant was the wrong topic. You change the system prompt to fix it. Did the change help? And did it break something that worked? Reading all 120 replies again takes an hour, and you will change the prompt many more times. Checks in code answer in a second, the same way every time.

    **An everyday example.** After repairing the brakes, a garage goes through the same checklist as after every repair: lights, horn, wipers, brakes. The list shows that the brakes work now, and also that the repair did not break anything else, such as a cable of the lights that came loose.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **eval set** | a fixed list of inputs, each with what you expect, such as the topic it is about, and checks you run on your app's answers to them after every change |
    | **assertion-based test** | a check in code that says yes or no about one answer, such as: is it valid JSON, is its topic the one the eval set expects, does it have at most three sentences |
    | **regression test** | running the eval set on the old and the new version of your app; it fails when an input that passed with the old version fails with the new one, a *regression* |

    A check in code sees only what code can see: the form, a word, the length, a topic you know in advance. Before you trust a check, try it on the answers you already read: it should fail what you noted for its kind, and nothing else. A check that fails a good answer gives a false alarm; one that passes a bad answer misses it.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: an eval set

    1. **Inputs:** a fixed list of inputs, each with what you expect. Keep the list the same from version to version.
    2. **Checks:** for each kind of mistake that code can see, a function that gives `True` or `False` for one answer.
    3. **Test the checks:** run them on the answers you read, and compare with your notes.
    4. **Run:** the eval set on the old version and on the new one.
    5. **Compare:** what the new version fixed, and its regressions. Read the regressions before you keep the new version.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    1. **Inputs:** the {n} questions of the error analysis, each with the topic it is about in BANKING77.
    2. **Checks:** one for the form, and one for each kind that code can see: *wrong topic*, *another problem* and *too long*. This one is for *too long*:

    ```python
    def short_ok(text, topic):
        answer = reply_of(text)
        if answer is None:
            return False
        return sentences(answer["reply"]) <= 3
    ```

    Every check gets the same two things: `text`, the answer of the app, and `topic`, the topic the eval set expects. `reply_of(text)` gives the answer as a dictionary with `"topic"` and `"reply"`, or `None` when the text is not JSON of that form; `sentences` counts the sentences of a text. `json_ok` gives `False` only when `reply_of` gives `None`.

    3. **Test the checks:** on the {n} replies of version 1, next to the course's notes:
    """.format(n=C.findings()["n"]))
    return


@app.cell(hide_code=True)
def picture_tested():
    mo.Html(C.tested_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_tested():
    _f = C.findings()
    _a = _f["notes"]["admits_ok"]
    _s = _f["notes"]["steps_ok"]
    mo.md(r"""
    `topic_ok` and `short_ok` fail exactly the replies the course noted for their kind. `steps_ok` fails {sc} of the {sn} replies noted as *another problem*, and no other reply. The {sm} it misses ask for the steps of another problem in other words, such as freezing a card that has not arrived yet; a check that looks for fixed words cannot see that. A check that misses a few but gives no false alarm still earns its place: what it fails is wrong.

    The course also tried a check for the kind *admits a mistake*: a reply fails it when it has the word *incorrect*, *wrong* or *too low*. It fails all {caught} noted replies, and {alarms} more: *"If the rate is incorrect, please provide the date and amount"* admits nothing, and *"blocked after 3 wrong attempts"* is about the PIN. Whether a reply admits a mistake takes someone who reads the meaning, so `admits_ok` stays out of the eval set.
    """.format(caught=len(_a["caught"]), alarms=len(_a["alarms"]), sc=len(_s["caught"]), sn=len(_s["noted"]), sm=len(_s["missed"])))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.vstack([
        mo.md(r"""
        4. **Run:** version 2 of the system prompt puts each fact next to its topic, and asks the model to choose the topic first:
        """),
        mo.Html(C.E.prompt_html(C.SYSTEM_V2)),
        mo.md(r"""
        The eval set on both versions:
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_versions():
    mo.Html(C.versions_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_compare():
    _f = C.findings()
    mo.md(r"""
    5. **Compare:** {all1} of the {n} inputs passed every check with version 1, and {all2} with version 2. For what it aimed at, the change worked: {r2} of the {rn} questions about a refund and {p2} of the {pn} about a blocked PIN got the right topic, against {r1} and {p1} with version 1. Yet `topic_ok` passed {t1} inputs with version 1 and {t2} with version 2: the same number, but not the same inputs.
    """.format(n=_f["n"], all1=_f["v1"]["all"], all2=_f["v2"]["all"], r1=_f["right"]["v1"]["request_refund"], r2=_f["right"]["v2"]["request_refund"],
               p1=_f["right"]["v1"]["pin_blocked"], p2=_f["right"]["v2"]["pin_blocked"], rn=C.LABELS.count("request_refund"),
               pn=C.LABELS.count("pin_blocked"), t1=_f["v1"]["topic_ok"], t2=_f["v2"]["topic_ok"]))
    return


@app.cell(hide_code=True)
def text_topic():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The topic check

        `topic_ok(text, topic)` should give `True` when the answer is JSON of the right form and its topic is `topic`, the topic the eval set expects. The start code gives `True` for every answer that is JSON. The picture under the check shows what your check says about each reply of version 1.
        """),
        L.task("Give whether the topic of `answer` is the same as `topic`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_topic():
    box_topic = L.code_box("topic_ok", lines=6)
    box_topic
    return (box_topic,)


@app.cell(hide_code=True)
def check_topic(box_topic):
    result_topic = L.check("topic_ok", box_topic.value)
    result_topic
    return (result_topic,)


@app.cell(hide_code=True)
def hints_topic():
    L.hints(
        "topic_ok",
        [
            "`answer` is a dictionary with the keys `\"topic\"` and `\"reply\"`. The check should compare one of them with `topic`.",
            "`answer[\"topic\"]` is the topic the model chose. Compare it with `==`.",
            "Replace `return True` by `return ___ == topic`, with the topic the model chose in the gap.",
        ],
        why="""```python
def topic_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return answer["topic"] == topic
```

The check knows the right topic because the eval set gives it for every input: without what you expect, code cannot tell a right topic from a wrong one.""",
    )
    return


@app.cell(hide_code=True)
def text_steps():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The steps of another problem

        Many replies of version 1 asked for the date and amount of a payment, the steps for a wrong exchange rate, while the customer asked about something else. `steps_ok(text, topic)` should give `False` for an answer whose reply contains `"date and amount"`, unless `topic` is in `RATE`, the two topics about exchange rates where those steps fit. It should also give `False` for a text that is not JSON. `a or b` is `True` when at least one of the two is `True`, and `x not in text` is `True` when `x` does not occur in `text`.
        """),
        L.task("Give `True` when the reply does not contain \"date and amount\", or when `topic` is in `RATE`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_steps():
    box_steps = L.code_box("steps_ok", lines=6)
    box_steps
    return (box_steps,)


@app.cell(hide_code=True)
def check_steps(box_steps):
    result_steps = L.check("steps_ok", box_steps.value)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps_ok",
        [
            "Two things can make an answer pass: its reply does not ask for the date and amount, or the question is about exchange rates. One of the two is enough.",
            "`\"date and amount\" not in answer[\"reply\"]` is the first; `topic in RATE` is the second. Join them with `or`.",
            "Replace `return True` by `return \"date and amount\" ___ answer[\"reply\"] or topic ___ RATE`.",
        ],
        why="""```python
def steps_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return "date and amount" not in answer["reply"] or topic in RATE
```

Use `topic`, not `answer["topic"]`: when the model chose a wrong exchange rate for a refund, the steps fit its own topic, and only the eval set knows the question was about a refund.""",
    )
    return


@app.cell(hide_code=True)
def text_regressions():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The regressions

        `PASSED_V1` and `PASSED_V2` hold, for each of the 120 inputs, `True` when the answer passes every check of the eval set, with version 1 and with version 2. `regressions(old, new)` should give the numbers of the inputs that passed with `old` and fail with `new`. The start code gives every input that fails with `new`: `not new[i]` is `True` when `new[i]` is `False`.
        """),
        L.task("Keep only the inputs that also passed with `old`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_regressions():
    box_regressions = L.code_box("regressions", lines=6)
    box_regressions
    return (box_regressions,)


@app.cell(hide_code=True)
def check_regressions(box_regressions):
    result_regressions = L.check("regressions", box_regressions.value)
    result_regressions
    return (result_regressions,)


@app.cell(hide_code=True)
def hints_regressions():
    L.hints(
        "regressions",
        [
            "An input that failed with `old` already is not a regression: it was never right. Only an input that went from passing to failing counts.",
            "The `if` should ask two things at once: `old[i]` is `True`, and `new[i]` is not. Join them with `and`.",
            "Change the `if` line into `if old[i] ___ not new[i]:`.",
        ],
        why="""```python
def regressions(old, new):
    found = []
    for i in range(len(old)):
        if old[i] and not new[i]:
            found.append(i)
    return found
```

A regression test fails when this list is not empty. Then you read those answers before you keep the new version.""",
    )
    return


@app.cell(hide_code=True)
def text_keep():
    _f = C.findings()
    mo.md(r"""
    ### Exercise 4. Keep version 2?

    Version 2 fixed {fixed} inputs and broke {broke}, and every one of the {broke} regressions fails `topic_ok`.
    """.format(fixed=len(_f["fixed"]), broke=len(_f["broke"])))
    return


@app.cell(hide_code=True)
def choice_keep_cell():
    _f = C.findings()
    _an = C.LABELS.count("card_arrival")
    choice_keep = L.choice(
        "keep-v2",
        question="What do you do with version 2?",
        options={
            "keep": "Keep it: {a2} inputs pass every check, not {a1}".format(a1=_f["v1"]["all"], a2=_f["v2"]["all"]),
            "read": "Read the {b} regressions before you decide".format(b=len(_f["broke"])),
            "drop": "Drop it: a new version must not break anything",
        },
        correct="read",
        why={
            "read": "The regressions are {ba} questions about when a card arrives and {br} about a wrong exchange rate. {al} of the {an} questions about a card's arrival now get the topic of a lost or stolen card, and {af} of their replies tell the customer to freeze the card; of the {wn} about a wrong exchange rate, {w2} got the right topic, against {w1} with version 1. The fix for refunds made new mistakes: the regressions show them, the totals hide them. Fix them in a version 3, and run the eval set again.".format(
                ba=_f["broke_arrival"], br=_f["broke_rate"], al=_f["arrival_lost"], an=_an, af=_f["arrival_freeze"],
                wn=C.LABELS.count("card_payment_wrong_exchange_rate"), w2=_f["right"]["v2"]["card_payment_wrong_exchange_rate"],
                w1=_f["right"]["v1"]["card_payment_wrong_exchange_rate"]),
            "keep": "The total went up by {up}, but {b} inputs that were right are wrong now: {al} of the {an} questions about when a card arrives get the topic of a lost or stolen card. A total can hide what broke.".format(
                up=_f["v2"]["all"] - _f["v1"]["all"], b=len(_f["broke"]), al=_f["arrival_lost"], an=_an),
            "drop": "Version 2 gave {r2} of the {rn} questions about a refund the right topic, against {r1}. Dropping it throws that away: almost every change breaks something, and the regression test tells you what. Read what broke, and fix that in the next version.".format(
                r2=_f["right"]["v2"]["request_refund"], r1=_f["right"]["v1"]["request_refund"], rn=C.LABELS.count("request_refund")),
        },
    )
    choice_keep
    return (choice_keep,)


@app.cell(hide_code=True)
def text_read():
    mo.md(r"""
    ### Compare the versions yourself

    Choose a group, and go through its inputs one at a time, with the answer of each version and what each check says about it.
    """)
    return


@app.cell(hide_code=True)
def settings_group():
    group_pick = mo.ui.dropdown(options=C.group_options(L.lang), value=list(C.group_options(L.lang))[0], label="Group", full_width=True)
    group_pick
    return (group_pick,)


@app.cell(hide_code=True)
def settings_input(group_pick):
    input_pick = mo.ui.slider(start=1, stop=max(1, len(C.group(group_pick.value))), value=1, label="Input", show_value=True)
    input_pick
    return (input_pick,)


@app.cell(hide_code=True)
def demo_compare(group_pick, input_pick):
    _numbers = C.group(group_pick.value)
    mo.Html(C.compare_card(_numbers[min(input_pick.value, len(_numbers)) - 1], L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Checks in code cover the kinds that code can see. For *made up* and *admits a mistake*, [Who grades the answer](/course/open/build-judges?lang=en) lets a model grade the replies, and measures it against the course's notes the way you measured your checks. [Offline, online and what it is worth](/course/open/build-eval-online?lang=en) adds what happens with real users: what they report becomes new inputs for the eval set.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keep, result_regressions, result_steps, result_topic):
    L.wrap_up(result_topic, result_steps, result_regressions, choice_keep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
