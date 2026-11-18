# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-disability-screen-out. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-disability-screen-out", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="An online test is the same for everyone and names no ground, but people with a disability fail it much more often. What is it at first sight?",
        options={
            "direct": "Direct discrimination, as the effect is large",
            "right": "Possibly indirect discrimination",
            "none": "No distinction",
        },
        correct="right",
        why={
            "direct": "The test names no ground, so it is not direct. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
            "right": "A neutral rule that hits one group particularly hard is indirect, unless it is objectively justified.",
            "none": "Being the same for everyone is not enough: the effect counts. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_college():
    quick_college = L.choice(
        "quick-college",
        question="What can the Netherlands Institute for Human Rights do when it finds discrimination?",
        options={
            "fine": "Impose a fine",
            "right": "Give a judgment that carries weight",
            "ban": "Ban the organization from hiring anyone",
        },
        correct="right",
        why={
            "fine": "The Institute cannot impose a fine or damages; for that you go to court. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
            "right": "Its judgments are not binding, but measures follow in more than 80 percent of the cases where it finds discrimination.",
            "ban": "The Institute gives a judgment; it cannot ban anything. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
        },
    )
    quick_college
    return (quick_college,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Automatic filters are cheap: reject every CV with a long gap, require a score on a timed game, let software rate how someone speaks in a video. Each can remove a person who could do the job, because of a disability or a chronic illness. Comparing groups, as in the last lessons, does not find this well, because no two disabilities are the same. You have to look at who a rule removes, and why.

    **An everyday example.** A restaurant only takes reservations by phone. Someone who is deaf wants to eat there just as much, but the rule keeps them out, though nobody meant it to. An email address would be enough.
    """)
    return


@app.cell(hide_code=True)
def predict_vendor():
    guess_vendor = L.guess(
        "vendor",
        question="A vendor checked its hiring test: people with a disability, as one group, pass it as often as others. Can the test still unlawfully screen out someone with a disability?",
        options={
            "no": "No, the check on the group shows it is safe",
            "right": "Yes, as every disability is different",
            "intent": "Only if the vendor meant it to",
        },
        reveal="**Yes.** The US Equal Employment Opportunity Commission (EEOC) wrote in 2022 that each disability is unique: a person can do badly on a test because of a disability, and be screened out, however well other people with disabilities do. A test that works for wheelchair users can still fail a blind applicant.",
    )
    guess_vendor
    return (guess_vendor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Screen out.** In May 2022 the EEOC and the US Department of Justice each published guidance on hiring software and disability. Screen out, the EEOC wrote, "occurs when a disability prevents a job applicant or employee from meeting—or lowers their performance on—a selection criterion, and the applicant or employee loses a job opportunity as a result." Its examples: a chatbot that rejects applicants with long gaps in their employment history, while a gap can come from treatment for an illness; a 90 percent score required on a memory game, which a blind applicant cannot play, though the applicant "still might have a very good memory"; software that rates speech patterns in a video interview, which is unlikely to score someone with a speech impediment fairly; and counting keystrokes per minute, which disadvantages someone who uses voice recognition instead of a keyboard.

        **When it is unlawful.** Under the US disability law, the ADA, screen out is unlawful when the person could do the essential functions of the job with a reasonable accommodation: a change in the way things are done that lets someone with a disability apply for a job or do it. The employer is usually responsible, even when a vendor made or runs the tool. The EEOC advised telling applicants which traits a tool measures and that other formats and other tests are available, and measuring only abilities the job truly needs. The guidance was never binding, and in January 2025 it was removed from the EEOC website; the ADA itself still applies.

        **Every disability is different.** A vendor that calls its tool free of bias has usually checked the averages for race and sex. But a person can do badly on a test because of a disability "regardless of how well other individuals with disabilities fare". The US guidelines with the four-fifths rule of [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) do not apply to disability. So you look at a rule itself: count who it removes, and ask whether those people could do the job, perhaps with an adjustment.

        **In the Netherlands.** The Equal Treatment (Disability or Chronic Illness) Act of 2003, the WGBH/CZ, forbids direct and indirect distinction on these grounds, also in offering a job and in recruitment. It adds a duty: whoever the ban applies to must make effective adjustments as needed, unless they are a disproportionate burden. The Netherlands Institute for Human Rights asks whether the adjustment is suitable and necessary, and the duty starts once the person has made the need known. The Institute warns employers that people who are partially sighted or hard of hearing may be unable to use a game assessment or an online form: make digital forms, tests and games accessible. The EU directive of 2000 on work asks the same, and under the UN Convention on the Rights of Persons with Disabilities, in force in the Netherlands since 14 July 2016, denying a reasonable accommodation is itself discrimination.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **screen out** | a disability keeps someone from meeting a selection criterion, or lowers their result, and they lose the job opportunity |
        | **reasonable accommodation** | a change in how things are done that lets someone with a disability apply, work or learn on an equal footing; Dutch law calls it a *doeltreffende aanpassing* |
        | **disproportionate burden** | the limit of the duty: an adjustment that would cost too much or be too hard for this organization is not required |
        """),
        L.watch_out("Passing a check on groups does not mean nobody is screened out. Disability is not one group with one average: avoiding discrimination against blind applicants does not prevent it against applicants with autism or epilepsy, the Department of Justice wrote."),
        mo.Html(C.picture({"title": "A filter on gaps in the CV",
                           "xlabel": "months without work in the CV", "yes": "can do the job",
                           "no": "cannot do the job", "limit": "removed: more than 6 months", "months": "months",
                           "key": "filled: removed by the filter, open: kept"})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out whether a filter or a test screens people out:

    1. List what it looks at: gaps, speed, speech, keystrokes, a game, a form.
    2. For each, ask which disabilities or illnesses could lower the result, while the person can still do the job, perhaps with an adjustment.
    3. Count who the rule removes, and look at those people, not only at the rate of a group.
    4. Tell applicants what the tool measures and how to ask for an adjustment, and offer another format or another test.
    5. Keep only the criteria the job truly needs.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A filter on gaps, with applicants made up by the course. Each applicant in `_applicants` has a name, the months without work in their CV, and whether they can do the job, perhaps with an adjustment. The filter removes everyone with a gap of more than `_limit` months. The loop collects their names in `_removed`. Run the cell.
    """)
    return


@app.cell
def example_removed():
    # made-up applicants: months without work in their CV, and whether they can do the job
    _applicants = [
        {"name": "Ada", "gap": 0, "can_do_job": "yes"},
        {"name": "Bram", "gap": 14, "can_do_job": "yes"},
        {"name": "Chen", "gap": 3, "can_do_job": "no"},
        {"name": "Dewi", "gap": 9, "can_do_job": "yes"},
        {"name": "Emre", "gap": 1, "can_do_job": "no"},
        {"name": "Fleur", "gap": 24, "can_do_job": "no"},
        {"name": "Gijs", "gap": 7, "can_do_job": "yes"},
        {"name": "Jesse", "gap": 6, "can_do_job": "yes"},
    ]
    # the filter: nobody with a gap of more than this many months
    _limit = 6

    _removed = []
    for _applicant in _applicants:
        if _applicant["gap"] > _limit:
            _removed.append(_applicant["name"])
    print(_removed)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Four are removed: Bram, Dewi, Fleur and Gijs. In this made-up list, Bram's gap came from treatment for an illness. But how many of the four could do the job? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_wrongly():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Removed, but able

        Count the applicants the filter removes while they can do the job. Replace the `...` with a loop over the applicants, as in the worked example, with the names without their underscore: `applicant`. Start with `wrongly = 0`, and add 1 when the gap is more than `limit` and `can_do_job` is `"yes"`: put a second `if` inside the first. The last line is `wrongly`, so the check sees it. The check also tries another limit.
        """),
        L.task("Work out `wrongly`: how many applicants the filter removes while they can do the job."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wrongly():
    box_wrongly = L.code_box("wrongly", lines=20)
    box_wrongly
    return (box_wrongly,)


@app.cell(hide_code=True)
def check_wrongly(box_wrongly):
    result_wrongly = L.check("wrongly", box_wrongly.value)
    result_wrongly
    return (result_wrongly,)


@app.cell(hide_code=True)
def hints_wrongly():
    L.hints(
        "wrongly",
        [
            "Use the loop and the first `if` of the worked example. Instead of appending a name, count: a number that starts at 0 and goes up by 1.",
            "Inside the first `if`, a second `if` asks `applicant[\"can_do_job\"] == \"yes\"`. Only then does the count go up: `wrongly = wrongly + 1`.",
            "Inside the second `if`: `wrongly = ___`.",
        ],
        why="`wrongly = 0`, then `for applicant in applicants:` with `if applicant[\"gap\"] > limit:` under it, inside that `if applicant[\"can_do_job\"] == \"yes\":`, and inside that `wrongly = wrongly + 1`. That gives `3`: Bram, Dewi and Gijs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_concerns():
    mo.md(r"""
    ### Exercise 2. What did the EEOC warn about?

    The EEOC warned about three things: screen out, adjustments that should be offered, and questions about health that a tool makes applicants answer. Sort each case by the warning it belongs to.
    """)
    return


@app.cell(hide_code=True)
def sort_concerns_cell():
    sort_concerns = L.sort("concerns")
    sort_concerns
    return (sort_concerns,)


@app.cell(hide_code=True)
def check_concerns(sort_concerns):
    result_concerns = L.check("concerns", sort_concerns)
    result_concerns
    return (result_concerns,)


@app.cell(hide_code=True)
def hints_concerns():
    L.hints(
        "concerns",
        [
            "Ask of each case: does a rule remove someone, does a change let someone take part, or does a tool make someone tell about their health?",
            "The gap and the memory game are examples of the EEOC of screen out; the lower register is its example of an adjustment.",
            "Screen out: the gap and \\_\\_\\_. An adjustment: out loud and \\_\\_\\_.",
        ],
        why="**Screen out:** the gap and the memory game. **An adjustment:** out loud and the lower register. **A question about health:** the form and medicines.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_adjust():
    mo.md(r"""
    ### Exercise 3. Captions in an interview
    """)
    return


@app.cell(hide_code=True)
def choice_adjust_cell():
    choice_adjust = L.choice(
        "adjust",
        question="An applicant who is hard of hearing asks a Dutch employer for captions in an online video interview. What does Dutch law ask of the employer?",
        options={
            "nothing": "Nothing, as the interview is the same for everyone",
            "right": "Adjust, unless that is a disproportionate burden",
            "any": "Whatever the applicant asks, at any cost",
        },
        correct="right",
        why={
            "nothing": "The same interview for everyone can still keep someone out. The law asks for effective adjustments as needed. See *In the Netherlands* in *The idea*.",
            "right": "The applicant made the need known, and captions are suitable and necessary for taking part; only a disproportionate burden lifts the duty.",
            "any": "The duty has a limit: an adjustment that is a disproportionate burden is not required, and it must be suitable and necessary. See *In the Netherlands* in *The idea*.",
        },
    )
    choice_adjust
    return (choice_adjust,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en) you saw that a neutral rule can discriminate through its effect. A filter on gaps is such a rule, and disability adds a duty: not only to stop a rule that keeps people out, but to make adjustments so they can take part. Counting group rates, as in [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en), is not enough here; your count of who a rule removes, and why, is. The next lesson, *Bias at every step*, follows where bias can enter, from the world to a model in use.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_adjust, result_concerns, result_wrongly):
    L.wrap_up(result_wrongly, result_concerns, choice_adjust)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
