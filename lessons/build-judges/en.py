# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-judges. Route: course/route/lessen/bouwen.yaml. Checks, the judges' recorded verdicts and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-judges", "en")
    C = L.checks   # the replies of version 1, what two judges wrote about them, the course's notes and the pictures
    words = C.words
    JUDGE_PLAIN = C.JUDGE_PLAIN
    JUDGE_RUBRIC = C.JUDGE_RUBRIC
    PEOPLE_GOOD = C.PEOPLE_GOOD
    PEOPLE_MADE_UP = C.PEOPLE_MADE_UP


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
def quick_check_alarm():
    quick_alarm = L.choice(
        "quick-alarm",
        question="The course's admits_ok failed all 18 replies noted as *admits a mistake*, and 21 more. Why did it stay out of the eval set?",
        options={
            "alarms": "Its words also come in replies that admit nothing",
            "missed": "It missed several of the replies that admitted a mistake",
            "slow": "A check in code is too slow to run after every change",
        },
        correct="alarms",
        why={
            "alarms": "*\"If the rate is incorrect, please provide the date and amount\"* admits nothing, and *\"blocked after 3 wrong attempts\"* is about the PIN. Whether a reply admits a mistake takes someone who reads what it means: that is what this lesson is about.",
            "missed": "It missed none: it failed all 18. Its trouble was the other way round, 21 false alarms. See [An eval set and regression tests](/course/open/build-eval-set?lang=en).",
            "slow": "A check in code goes through all 120 replies in a moment. Its trouble was 21 false alarms. See [An eval set and regression tests](/course/open/build-eval-set?lang=en).",
        },
    )
    quick_alarm
    return (quick_alarm,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Of the six kinds of mistake the course found in the bank's assistant, the eval set checks three in code: *wrong topic*, *another problem* and *too long*. *Made up*, in 23 replies, and *admits a mistake*, in 18, take someone who reads what a reply means. The course read all 120 replies, but nobody reads every answer after every change, let alone every chat of real customers. A model can read, quickly: you give it the customer's message, the reply and what to look for, and it writes a verdict. Whether that verdict is right, only measuring tells you.

    **An everyday example.** A teacher gets an assistant to mark the exams. Before the assistant marks alone, the teacher marks thirty exams as well and compares: does the assistant fail the same ones? An assistant who fails every exam agrees with the teacher on every bad one, and is still of no use.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **LLM as a judge** | a model that grades the answers of an app: it gets the input, the answer and what to look for, and writes a verdict, such as pass or fail |
    | **rubric** | a written description of one thing to grade: what counts as a pass and what as a fail, sometimes with examples; people and judges grade with the same rubric |

    Three can grade an answer: code, people and a judge. A judge reads like a person and is fast like code, but its verdicts are a measurement too, and you check them against people's, on answers people graded. Two shares tell you how a judge does: of the answers people failed, the share the judge failed too, and of the answers people passed, the share the judge passed too. Agreement, the share of answers where the two give the same verdict, puts both in one number, and can hide that one of them is zero.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.graders_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a judge you can trust

    1. **Criterion:** choose one thing to grade, and write its rubric.
    2. **People:** grade a sample of answers yourself, with the rubric.
    3. **Judge:** let the model grade the same answers, with the same rubric.
    4. **Read:** turn what the judge wrote into a verdict, in code.
    5. **Measure:** the share of people's fails it catches and of people's passes it passes too, beside the agreement. Use the judge only when both shares are high; otherwise change the rubric or take a larger model, and measure again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _p = _f["plain"]
    mo.vstack([
        mo.md(r"""
    ### Worked example

    The judge is Qwen3, the model of the assistant itself, at temperature 0, so that the same request always gets the same answer. Each request holds the customer's message and the reply of version 1.

    1. **Criterion:** first without a rubric: is the reply good?
    2. **People:** a reply is good when the course's notes on it are empty; {pf} of the {n} replies were not good.
    3. **Judge:** this system prompt:
    """.format(n=_f["n"], pf=_p["people_fail"])),
        mo.Html(C.E.prompt_html(C.PLAIN)),
        mo.md(r"""
    4. **Read:** it answered with one word every time: *fail* {f} times, *pass* {p} times.
    5. **Measure:** it failed {c} of the {pf} replies the course failed, and passed {pp} of the {ppass} the course passed. Its agreement is {agree:.0%}: it fails nearly every reply, good or bad.
    """.format(n=_f["n"], pf=_p["people_fail"], f=_p["fail"], p=_p["pass"], c=_p["caught"], pp=_p["passed"],
               ppass=_p["people_pass"], agree=_p["agree"])),
    ])
    return


@app.cell(hide_code=True)
def text_measured():
    _r = C.findings()["rubric"]
    mo.vstack([
        mo.md(r"""
    ### What the course measured

    Then one criterion with a rubric, for the kind *made up*:

    """),
        mo.Html(C.E.prompt_html(C.RUBRIC)),
        mo.md(r"""
    The rubric asks for a sentence first and the verdict last; in {nl} of the {n} judgments the verdict came first, before a sentence of its own, and in the others it wrote only the verdict. It wrote *pass* every time, also for the {pf} replies the course noted as *made up*.
    """.format(nl=_r["not_last"], n=C.findings()["n"], pf=_r["people_fail"])),
    ])
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.measured_picture(L.lang)),
        L.watch_out("Neither judge could tell a good reply from a bad one: one fails nearly everything, the other passes everything. A model as small as Qwen3-0.6B is not a judge for this. A larger model often judges better, and you measure it in the same way, on answers people graded; until a judge passes that test, people grade."),
    ])
    return


@app.cell(hide_code=True)
def text_verdict():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Read the verdict

        A judge does not always write one word: the judge with a rubric wrote *pass. All facts are included in the reply.* `verdict(text)` should give `True` for pass and `False` for fail, from the first word of the text that is pass or fail, and `None` when neither comes. `words(text)` gives the words of a text in small letters, without punctuation. The start code reads only a text that is exactly `pass` or `fail`.
        """),
        L.task("Go through `words(text)`, and give the verdict of the first word that is pass or fail."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=7)
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
            "Look at the words one by one, and stop at the first one that is pass or fail. Only when no word was, the function gives `None`.",
            "`for word in words(text):` with the two `if` lines of the start code inside it, comparing `word` instead of `text`. The `return None` comes after the loop.",
            "Above the two `if` lines, write `for word in ___:`, move the two `if` lines one step in, and compare `word` instead of `text` in both. Keep `return None` at the end, not inside the loop.",
        ],
        why="""```python
def verdict(text):
    for word in words(text):
        if word == "pass":
            return True
        if word == "fail":
            return False
    return None
```

`None` keeps apart what the judge did not say: a judgment without a verdict is neither a pass nor a fail.

This reads the first pass or fail, which fits these judges: they put the verdict first. A judge that follows the rubric writes its sentence first, and that sentence can hold the word, as in *nothing in it would make it fail*; for such a judge, read the last pass or fail instead. Which one fits, you see by reading what your judge writes.""",
    )
    return


@app.cell(hide_code=True)
def text_agreement():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Agreement

        `JUDGE_PLAIN` holds the verdicts of the plain judge, one for each of the 120 replies, and `PEOPLE_GOOD` the course's: `True` for a reply without any mistake. `agreement(judge, people)` should give the share of the replies where the two verdicts are the same. The start code counts every reply.
        """),
        L.task("Count a reply only when `judge[i]` is the same as `people[i]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agreement():
    box_agreement = L.code_box("agreement", lines=6)
    box_agreement
    return (box_agreement,)


@app.cell(hide_code=True)
def check_agreement(box_agreement):
    result_agreement = L.check("agreement", box_agreement.value)
    result_agreement
    return (result_agreement,)


@app.cell(hide_code=True)
def hints_agreement():
    L.hints(
        "agreement",
        [
            "The two agree on a reply when they give it the same verdict: both pass, or both fail.",
            "Put an `if` around `same = same + 1` that compares `judge[i]` with `people[i]`.",
            "Above `same = same + 1`, write `if ___ == ___:`, and move `same = same + 1` one step in.",
        ],
        why="""```python
def agreement(judge, people):
    same = 0
    for i in range(len(people)):
        if judge[i] == people[i]:
            same = same + 1
    return same / len(people)
```

The plain judge agrees on more than half of the replies, while it never passes a good one: agreement alone does not tell you that.""",
    )
    return


@app.cell(hide_code=True)
def text_caught():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The fails it catches

        `caught(judge, people)` should give, of the replies people failed, the share the judge failed too. A verdict `None` is not a fail. The start code counts the replies people failed in `failed`, but never counts `both`. The picture shows your share beside the share of good replies each judge passed.
        """),
        L.task("Inside the `if`, count the reply in `both` when the judge failed it too."),
    ])
    return


@app.cell(hide_code=True)
def exercise_caught():
    box_caught = L.code_box("caught", lines=9)
    box_caught
    return (box_caught,)


@app.cell(hide_code=True)
def check_caught(box_caught):
    result_caught = L.check("caught", box_caught.value)
    result_caught
    return (result_caught,)


@app.cell(hide_code=True)
def hints_caught():
    L.hints(
        "caught",
        [
            "Only the replies people failed count here. Of those, count the ones the judge failed as well.",
            "A second `if` inside the first one: when `judge[i] == False`, add 1 to `both`.",
            "Under `failed = failed + 1`, at the same depth, write `if ___:` with `both = both + 1` one step further in.",
        ],
        why="""```python
def caught(judge, people):
    failed = 0
    both = 0
    for i in range(len(people)):
        if people[i] == False:
            failed = failed + 1
            if judge[i] == False:
                both = both + 1
    return both / failed
```

`judge[i] == False` and not `not judge[i]`: `not None` is `True`, and a judgment without a verdict is no fail.""",
    )
    return


@app.cell(hide_code=True)
def text_what():
    _r = C.findings()["rubric"]
    mo.md(r"""
    ### Exercise 4. What does it tell you?

    The judge with a rubric agrees with the course on {agree:.0%} of the replies.
    """.format(agree=_r["agree"]))
    return


@app.cell(hide_code=True)
def choice_what_cell():
    _f = C.findings()
    _r = _f["rubric"]
    choice_what = L.choice(
        "what-agreement",
        question="What does that agreement tell you?",
        options={
            "good": "It catches most of the replies that make something up",
            "little": "Little: it passed all 120, the 23 made up too",
            "fifth": "It fails about one in five replies, as people do",
        },
        correct="little",
        why={
            "little": "{pp} of the {n} replies make nothing up, and a judge that passes everything agrees on all of them: {agree:.0%} without catching one of the {pf}. Read the share it catches beside the agreement.".format(
                pp=_r["people_pass"], n=_f["n"], agree=_r["agree"], pf=_r["people_fail"]),
            "good": "It caught {c} of the {pf}: it wrote pass for all {n} replies. The {agree:.0%} are the {pp} replies that make nothing up, which a judge that always passes gets right.".format(
                c=_r["caught"], pf=_r["people_fail"], n=_f["n"], agree=_r["agree"], pp=_r["people_pass"]),
            "fifth": "It failed none: it wrote pass for all {n} replies. People failed {pf}, and those are exactly where it disagrees.".format(
                n=_f["n"], pf=_r["people_fail"]),
        },
    )
    choice_what
    return (choice_what,)


@app.cell(hide_code=True)
def text_read():
    mo.md(r"""
    ### Read the judges yourself

    Choose a judge, and go through the replies one at a time: the reply, the course's notes, what the judge wrote and the verdict read from it.
    """)
    return


@app.cell(hide_code=True)
def settings_judge():
    judge_pick = mo.ui.dropdown(options={"plain judge": "plain", "judge with a rubric": "rubric"}, value="plain judge", label="Judge", full_width=True)
    judge_pick
    return (judge_pick,)


@app.cell(hide_code=True)
def settings_reply():
    reply_pick = mo.ui.slider(start=1, stop=120, value=1, label="Reply", show_value=True)
    reply_pick
    return (reply_pick,)


@app.cell(hide_code=True)
def demo_read(judge_pick, reply_pick):
    mo.Html(C.judged_card(judge_pick.value, reply_pick.value - 1, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The eval set grades what code can see, and a judge, once measured, can grade what it cannot. [Offline, online and what it is worth](/course/open/build-eval-online?lang=en) adds what you learn from real users, whose reports become new inputs and new graded answers. In [Guardrails](/course/open/build-guardrails?lang=en), checks like these run while the app answers, before a reply reaches a customer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_what, result_agreement, result_caught, result_verdict):
    L.wrap_up(result_verdict, result_agreement, result_caught, choice_what)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
