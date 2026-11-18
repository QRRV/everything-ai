# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ethics-washing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ethics-washing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Principles and guidelines](/course/open/rai-principles?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="Which principle did the most of the 84 documents name?",
        options={
            "sustainability": "Sustainability",
            "right": "Transparency",
            "solidarity": "Solidarity",
        },
        correct="right",
        why={
            "sustainability": "Sustainability was among the least named: 14 of the 84. See the picture in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "right": "Transparency, in 73 of the 84 documents, more than any other principle.",
            "solidarity": "Solidarity was the least named of all: 6 of the 84. See the picture in [Principles and guidelines](/course/open/rai-principles?lang=en).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def quick_check_meaning():
    quick_meaning = L.choice(
        "quick-meaning",
        question="A guideline asks for transparency and says nothing more. What does that tell you?",
        options={
            "explain": "That every decision must be explained to the person it is about",
            "code": "That the code of the system must be published",
            "right": "Not yet which: saying AI is used, opening it up or explaining",
        },
        correct="right",
        why={
            "explain": "That is one meaning, but others call a notice that AI is used, or a published dataset, transparency too. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "code": "That is one meaning among several. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "right": "Transparency can ask for a notice that AI is at work, a look inside the system, or an explanation of one decision. The word alone does not say which.",
        },
    )
    quick_meaning
    return (quick_meaning,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Writing down a principle is cheap. In March 2019, Google announced an outside council to advise it on the ethics of AI; after protests about who was on it, Google dissolved it less than two weeks later. A principle, a board or a statement can make an organization look responsible without changing what it builds. This lesson gives you a short test to tell a promise that costs something from one that costs nothing.

    **An everyday example.** A shop promises "satisfaction guaranteed". One shop takes anything back within 30 days, no questions asked. Another has no returns desk, no form and no address. Both made the same promise.
    """)
    return


@app.cell(hide_code=True)
def predict_code():
    guess_code = L.guess(
        "code",
        question="Researchers showed one group of software developers the code of ethics of the ACM, the largest society for computing, and then asked them what they would do in a series of tricky situations. Another group did not see the code. How much did the code change the decisions, do you think?",
        options={
            "lot": "A lot",
            "some": "A little",
            "right": "No difference found",
        },
        reveal="**No difference found.** Students and professional developers who had read the code decided no differently from those who had not (McNamara, Smith and Murphy-Hill, 2018). A text alone did not change what people did.",
    )
    guess_code
    return (guess_code,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Ethics washing** is using principles, boards or statements to look responsible while avoiding real change or binding rules. The name follows greenwashing. In 2018 the researcher Ben Wagner warned that ethics could become an escape from regulation: a company that writes its own principles may hope that lawmakers leave it alone. In 2019 the philosopher Thomas Metzinger, one of the members of the EU expert group of the previous lesson, warned of ethics washing in its work: he wrote that the *red lines*, uses of AI that should never be allowed, had been watered down and taken out of the final text under pressure from industry.

        A principle costs nothing as long as nothing happens when it is broken. Three questions give it a price:

        - **A check**: how would anyone find out that it was broken? A test, an audit, a number that is published.
        - **An owner**: who answers for it? A named person or team, not just "we".
        - **A consequence**: what happens when it is broken? A model goes offline, a launch waits, someone who complains gets an answer.

        In one sentence: what would change, and who would notice, if the promise were broken? A promise that passes none of the three is only words. That does not make every principle washing: a principle can be a first step, and the question is what comes after it.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **ethics washing** | using principles, boards or statements to look responsible while avoiding real change or binding rules, named after greenwashing |
        """),
        L.watch_out("An owner alone is not enough. A board whose advice is secret and need not be followed has members who answer for it, but with no check and no consequence, nothing changes when it is ignored."),
        mo.Html(C.picture({"title": "Six promises of a company, against three questions", "tests": ["check", "owner", "consequence"],
                           "words": "only words",
                           "alt": "Six promises of a made-up company. Fair AI and planet have no check, no owner and no consequence: only words. The bias test has a check and an owner, the outside test only a check, the ethics board only an owner, and the complaints promise has all three."})),
        mo.md(r"""
        A company made up by the course, with six promises. *Fair AI*: we are committed to fair AI. *Bias test*: every model is tested for errors per group, and the head of the data team signs the test off. *Outside test*: outside experts try to break each model before it is released. *Ethics board*: a board of named members advises us. *Complaints*: anyone can complain, a named team answers within 30 days, and a model that keeps failing is taken offline. *Planet*: we care about the planet. A filled dot means that the promise has it, an open dot that it does not.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To test a promise about ethics:

    1. Write the promise down in one sentence.
    2. Ask how anyone would find out that it was broken: is there a check?
    3. Ask who answers for it: is there an owner with a name?
    4. Ask what happens when it is broken: is there a consequence?
    5. A promise that passes none of the three is only words; say what it would need.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A school, made up by the course, grades essays with AI and makes three promises. A teacher reviews every grade, and the grade the teacher sets is the one that counts. Students can ask why they got a grade, and the head of the school answers. And the grading is fair. Three dictionaries hold 1 where a promise has a check, an owner or a consequence, and 0 where it does not. The loop adds them up for each promise. Run the cell.
    """)
    return


@app.cell
def example_school():
    _check = {"teacher review": 1, "ask why": 0, "fair grading": 0}
    _owner = {"teacher review": 1, "ask why": 1, "fair grading": 0}
    _consequence = {"teacher review": 1, "ask why": 0, "fair grading": 0}

    for _promise in _check:
        print(_promise, _check[_promise] + _owner[_promise] + _consequence[_promise])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The teacher review passes all three. Asking why has an owner, but nobody checks that the answers come, and nothing happens if they do not. Fair grading passes none: only words. Step 5 of the recipe asks which promises of a whole company are only words: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_passed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Three tests per promise

        The company of the picture. `check`, `owner` and `consequence` hold the three tests, 1 when a promise passes and 0 when it does not, and the loop goes through the promises with `promise`. Replace the `...` with one line that adds up the three tests for the promise, as in the worked example, and puts the total in `passed` instead of printing it. A promise with a total of 0 is only words. The last line is `passed`, so the check sees it. The check also tries other consequences.
        """),
        L.task("Put in `passed`, under each promise, how many of the three tests it passes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_passed():
    box_passed = L.code_box("passed", lines=13)
    box_passed
    return (box_passed,)


@app.cell(hide_code=True)
def check_passed(box_passed):
    result_passed = L.check("passed", box_passed.value)
    result_passed
    return (result_passed,)


@app.cell(hide_code=True)
def hints_passed():
    L.hints(
        "passed",
        [
            "The sum of the worked example, with its answer stored under the promise.",
            "Which three values belong to this promise? And how do you put a value under a key in a dictionary?",
            "`passed[promise] = check[promise] + ___ + ___`",
        ],
        why="`passed[promise] = check[promise] + owner[promise] + consequence[promise]` gives `{'fair AI': 0, 'bias test': 2, 'outside test': 1, 'ethics board': 1, 'complaints': 3, 'planet': 0}`: fair AI and planet pass none, so nobody checks them, nobody answers for them, and nothing happens when they are broken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_notice():
    mo.md(r"""
    ### Exercise 2. Would anyone notice?

    Six promises from companies that build AI, made up by the course. Put each one with what happens when it is broken.
    """)
    return


@app.cell(hide_code=True)
def sort_notice_cell():
    sort_notice = L.sort("notice")
    sort_notice
    return (sort_notice,)


@app.cell(hide_code=True)
def check_notice(sort_notice):
    result_notice = L.check("notice", sort_notice)
    result_notice
    return (result_notice,)


@app.cell(hide_code=True)
def hints_notice():
    L.hints(
        "notice",
        [
            "Ask for each promise the three questions of the recipe: a check, an owner, a consequence?",
            "Three of the promises name something you could count or see: a number, a test, a deadline. Which three?",
            "Costs nothing: people first, \\_\\_\\_ and \\_\\_\\_. Someone would notice: the error rates, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Costs nothing:** people first, we strive and the board. **Someone would notice:** the error rates (a published number), no launch (a test, a person and a launch that waits) and appeal (a person and a deadline).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_code_of_ethics():
    mo.md(r"""
    ### Exercise 3. A code that changed nothing
    """)
    return


@app.cell(hide_code=True)
def choice_code_of_ethics_cell():
    choice_code_of_ethics = L.choice(
        "code-of-ethics",
        question="Developers who had read a code of ethics decided no differently from developers who had not. What does that suggest?",
        options={
            "useless": "That developers cannot learn anything about ethics at all",
            "right": "That a text on its own changes little in what people do",
            "bad": "That this code of ethics must have been badly written",
        },
        correct="right",
        why={
            "useless": "The study tested reading a code, not teaching ethics in other ways. What did the code lack? Think of the three questions in *The idea*.",
            "right": "A code with no check, no owner and no consequence is a text people can read and forget. What changes behavior is what comes with it.",
            "bad": "The study says nothing about how well the code was written. What did the code lack? Think of the three questions in *The idea*.",
        },
    )
    choice_code_of_ethics
    return (choice_code_of_ethics,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Principles and guidelines](/course/open/rai-principles?lang=en) you saw that most documents share a few words and mean different things by them. Here you tested whether a principle costs anything at all: a check, an owner and a consequence turn a word into a promise. In the next lesson, *Designing with values*, you go one step further: from a value, through rules, to requirements an engineer can check.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_code_of_ethics, result_notice, result_passed):
    L.wrap_up(result_passed, result_notice, choice_code_of_ethics)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
