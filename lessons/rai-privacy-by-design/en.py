# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-privacy-by-design. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-privacy-by-design", "en")
    C = L.checks   # the member file of a made-up library, and what each purpose needs


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Finding and hiding personal data](/course/open/rai-find-pii?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_typo():
    quick_typo = L.choice(
        "quick-typo",
        question="A filter hides nine digits when they pass the eleven test. Someone types their real BSN with one digit wrong. What does the filter do?",
        options={
            "close": "It hides it: one digit off is still close",
            "right": "It leaves it in the text: the test fails",
            "stop": "It stops: the number is not a BSN",
        },
        correct="right",
        why={
            "right": "One wrong digit always makes the eleven test fail, so the filter sees no possible BSN and leaves the number as it is, although it is nearly someone's real number. A fail does not make a number harmless.",
            "close": "The eleven test does not look at how close a number is: one wrong digit makes it fail, and the filter leaves the number in the text. See *What a pass tells you* in [Finding and hiding personal data](/course/open/rai-find-pii?lang=en).",
            "stop": "The filter asks the test, gets `False` and goes on: the number stays in the text. See *What a pass tells you* in [Finding and hiding personal data](/course/open/rai-find-pii?lang=en).",
        },
    )
    quick_typo
    return (quick_typo,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="On ten messages, a filter hides 23 of the 32 pieces of personal data. What does that tell you?",
        options={
            "any": "It will hide 72% of the personal data in any text",
            "right": "Its recall on these messages is 0.71875",
            "wrong": "9 of the things it hid were not personal data",
        },
        correct="right",
        why={
            "right": "23 of 32 is 0.71875: the share of the personal data in these messages that it finds, its recall. On other texts it can find more or less.",
            "any": "Ten made-up messages say little about other texts: on those, the filter can find more or less. See Exercise 3 in [Finding and hiding personal data](/course/open/rai-find-pii?lang=en).",
            "wrong": "The 9 are the pieces of personal data it missed: they stay in the text. What it hid by mistake is counted separately, among the other numbers in the messages. See Exercise 3 in [Finding and hiding personal data](/course/open/rai-find-pii?lang=en).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A made-up library wants to train a model that recommends books from what its members borrowed. Its member file holds, for each of ten members, a card number, a name, an e-mail address, a phone number (one member never gave one), a date of birth, a postcode, the kinds of books they borrowed, whether they want the newsletter, and the month their membership ended, if it did. In [Finding and hiding personal data](/course/open/rai-find-pii?lang=en), you hid names and numbers that were already in the text, and something always slipped through. Here the question comes first: what does each purpose need, and when can the data go? What the pipeline never takes in, no filter has to find, and what has been deleted cannot leak. In this lesson you write those rules for the library: keep only what a purpose needs, let go of the members who left, and count what is left.

    **An everyday example.** A shop asks for your date of birth and your phone number to sell you a book. Two years after you left a gym, it still has a copy of your passport. In both cases the data serves no purpose any more, and in both cases it can still leak.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="The Dutch tax administration should have deleted the dual nationality of Dutch citizens in January 2014: it plays no role in assessing an application for childcare benefit. Of how many people was it still registered in the tax administration's systems in May 2018?",
        options={
            "few": "14 thousand",
            "mid": "140 thousand",
            "right": "1.4 million",
        },
        reveal="**1.4 million**, according to the Dutch data protection authority, the Autoriteit Persoonsgegevens (AP). The tax administration kept the data and used it; only in the summer of 2020 had it removed it completely. *Data that should have been deleted* in *The idea* tells what the AP did about it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Before the fact, not after

    In [The GDPR for AI](/course/open/rai-gdpr?lang=en) you met purpose limitation: data collected for one purpose may not be used later for a purpose that does not fit with it. **Privacy by design** goes a step further: build the protection into a system before it collects anything. Ann Cavoukian, then the Information and Privacy Commissioner of Ontario in Canada, set it out in seven principles, years before the GDPR: "In short, Privacy by Design comes before-the-fact, not after." The seven, in Cavoukian's words:

    1. "Proactive not Reactive; Preventative not Remedial"
    2. "Privacy as the Default"
    3. "Privacy Embedded into Design"
    4. "Full Functionality – Positive-Sum, not Zero-Sum"
    5. "End-to-End Security – Lifecycle Protection"
    6. "Visibility and Transparency"
    7. "Respect for User Privacy"

    Privacy embedded into design means, in Cavoukian's words, that "It is not bolted on as an add-on, after the fact." And on collecting data, Cavoukian writes: "the collection of personally identifiable information should be kept to a strict minimum."

    ### What the law asks

    The GDPR has its own term, and its article 25 does not list the seven principles. The article's title is "Data protection by design and by default", and it asks for measures, such as pseudonymization, that put the principles of article 5 into practice. Three of those principles carry this lesson. Purpose limitation: data is collected for specified purposes, and not used later in a way that does not fit with them. **Data minimization**: data is "adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed". And storage limitation: data is "kept in a form which permits identification of data subjects for no longer than is necessary". Article 25 then says what "by default" covers: "only personal data which are necessary for each specific purpose of the processing are processed. That obligation applies to the amount of personal data collected, the extent of their processing, the period of their storage and their accessibility."

    Note the first word of data minimization: adequate. It does not ask for as little data as possible, but for what the purpose needs, and no more.

    ### By default

    The European Data Protection Board, the EDPB, in which the European data protection authorities work together, explains article 25 in guidelines. A default is "the pre-existing or preselected value of a configurable setting", as in computer science: what holds when nobody changes anything. So, in the EDPB's words, "by default, the controller shall not collect more data than is necessary, they shall not process the data collected more than is necessary for their purposes, nor shall they store the data for longer than necessary." Cavoukian put the same thing from the other side: "If an individual does nothing, their privacy still remains intact."

    ### Less detail, or none at all

    The EDPB starts earlier still: "Controllers should first of all determine whether they even need to process personal data for their relevant purposes." And when personal data is needed, less detail may be enough: counting members per decade of birth needs no dates of birth. The EDPB's own example is a bookshop whose order form asks for a date of birth and a phone number, which buying a book does not need; for e-books, which are not sent by post, the form needs no address either.

    ### Delete when it is done

    "If personal data is no longer necessary for the purpose of the processing, then it shall by default be deleted or anonymized," the EDPB writes, and "Deletion of certain personal data should be automated", so that it does not wait until someone remembers. For a membership, the EDPB's example: the data is deleted when the membership ends and no lawful basis is left to store it longer. The GDPR sets no fixed period; in the EDPB's words: "The length of the period of retention will therefore depend on the purpose of the processing in question." And anonymizing instead of deleting is only enough when nobody can find the person again, as in [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en): the EDPB asks the controller to check that.

    ### Data that should have been deleted

    The Dutch data protection authority, the AP, described a case in 2021. The tax administration should have deleted the dual nationality of Dutch citizens in January 2014, because it plays no role in assessing an application for childcare benefit. It kept and used the data anyway: in May 2018, the dual nationality of 1.4 million people was still registered in its systems, and only in the summer of 2020 had it been removed completely. The AP fined the tax administration €2.75 million in total, €750,000 of it for keeping this data. Its decision fined this as processing without a lawful basis, not under storage limitation; the breach ran from 2014 to 2020, and the fine counted only the years from 1 January 2016, when the AP got the power to fine such violations.

    ### Why this comes before a filter

    In [Finding and hiding personal data](/course/open/rai-find-pii?lang=en), every filter missed something. A key the form never asks for needs no filter, and a member who has been deleted cannot leak. What the pipeline does keep is often still personal data: less to leak is not nothing to leak.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **privacy by design** | building privacy protection into a system from the start, with privacy-friendly default settings, instead of adding it at the end |
        | **data minimization** (*minimale gegevensverwerking*) | processing only the personal data a purpose needs: adequate, relevant and limited to what is necessary |
        | `x is None` | `True` when `x` is `None`, Python's word for nothing; `x is not None` is the opposite |
        """),
        L.watch_out("Data minimization is not \"as little as possible\": the data must be adequate for the purpose, so too little is a problem too. And what a pipeline keeps after minimizing is often still personal data: less to leak is not nothing to leak."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a pipeline that keeps only what its purposes need:

    1. Write down each purpose first, and per purpose the keys it needs. Ask first whether the purpose needs personal data at all, then whether less detail is enough, such as a decade instead of a date of birth.
    2. Keep of each record only those keys. That is `minimize`.
    3. Decide when data goes, and let it go without waiting for someone to remember: here, when a membership ends. That is `to_delete`.
    4. Count what is left: the values that would show if the pipeline's data leaked. That is `values_kept`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell shows the first member of the file, key by key, and then what a count of members per decade of birth needs: the decade, not the date. `C.MEMBERS` is the file, a list of ten dicts. `birth_date[:3]` takes the first three characters, such as `"195"`, and `+ "0s"` makes `"1950s"`; `postcode[:4]` takes the area, `"3511"`. A dict counts the members per decade.
    """)
    return


@app.cell
def example_decades():
    # the first member of the made-up library, as the file has it
    _member = C.MEMBERS[0]
    for _key in _member:
        print(_key, _member[_key])

    # for a count per decade of birth and per area, less detail is enough
    print("decade:", _member["birth_date"][:3] + "0s", "area:", _member["postcode"][:4])

    # how many members were born in each decade
    _count = {}
    for _m in C.MEMBERS:
        _decade = _m["birth_date"][:3] + "0s"
        if _decade in _count:
            _count[_decade] = _count[_decade] + 1
        else:
            _count[_decade] = 1
    print(_count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Anouk de Vries has nine values in the file, from the card number `L-1001` to `ended`, which is `None`: Anouk de Vries is still a member. For a count per decade, `1950s` is enough, and the area `3511` says less about where someone lives than `3511 KL`. The count gives two members for each decade from the 1970s to the 2000s, and one each for the 1950s and the 1960s: ten members, and for the count the decade of each is enough: a form could ask for that instead of the date of birth. In the exercises you build the same idea into the pipeline: per purpose, only the keys it needs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minimize():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. minimize: only what the purpose needs

        `minimize(record, needed)` gets one member as a dict `record`, and a list `needed` with the keys a purpose needs. It gives back a new dict with only the keys of `record` that are in `needed`, each with its value. A key in `needed` that the member does not have, such as the phone number Lina Postma never gave, does not go in. `PURPOSES` holds what each purpose needs: `PURPOSES["lending"]` is `["card", "name", "email"]`, for lending with a reminder by e-mail when a book is late, and `PURPOSES["recommend"]` is `["books"]`, for the recommendation model. Inside, `kept` starts as an empty dict, and the loop goes over the keys of `record`. Replace the `...`. The check calls `minimize(MEMBERS[0], PURPOSES["lending"])`, then other members and keys.
        """),
        L.task("Give back a dict with only the keys the purpose needs, each with its value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minimize():
    box_minimize = L.code_box("minimize", lines=7)
    box_minimize
    return (box_minimize,)


@app.cell(hide_code=True)
def check_minimize(box_minimize):
    result_minimize = L.check("minimize", box_minimize.value)
    result_minimize
    return (result_minimize,)


@app.cell(hide_code=True)
def hints_minimize():
    L.hints(
        "minimize",
        [
            "Step 2 of the recipe. In [Dictionaries](/course/open/py-dicts?lang=en), `for fruit in prices:` gives one key at a time, and `prices[fruit]` its value.",
            "For each key, ask whether it is in `needed`. If it is, put it in `kept`, with the same value it has in `record`.",
            "`if key in needed:` and under it `kept[key] = ___`",
        ],
        why='`if key in needed:` with `kept[key] = record[key]` under it keeps a key with its value when the purpose needs it. `minimize(MEMBERS[0], PURPOSES["lending"])` gives `{"card": "L-1001", "name": "Anouk de Vries", "email": "anouk.devries@example.org"}`: the other six values of the member stay out.',
    )
    return


@app.cell(hide_code=True)
def text_after_minimize():
    mo.md(r"""
    With a right `minimize`, 30 of the 89 squares in the picture are filled: three keys for each of the ten members. Lending needs no date of birth and no phone number, so the pipeline for lending never takes them in. But it still keeps the three members who left.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_delete():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. to_delete: who has to go

        `to_delete(members)` gets a list of members, and gives back the card numbers of those whose membership ended, in the order of the list: the members whose data the library deletes. A member who left has a month in `"ended"`, such as `"2019-03"`; a member who still is one has `None`. Inside, `cards` starts as an empty list, and the loop goes over the members. Replace the `...`. The check calls `to_delete(MEMBERS)`, then other lists.
        """),
        L.task("Give back the card numbers of the members whose membership ended."),
    ])
    return


@app.cell(hide_code=True)
def exercise_delete():
    box_delete = L.code_box("to-delete", lines=7)
    box_delete
    return (box_delete,)


@app.cell(hide_code=True)
def check_delete(box_delete):
    result_delete = L.check("to-delete", box_delete.value)
    result_delete
    return (result_delete,)


@app.cell(hide_code=True)
def hints_delete():
    L.hints(
        "to-delete",
        [
            "Step 3 of the recipe. `x is None` asks whether `x` is `None`; `x is not None` asks the opposite.",
            'A member left when `"ended"` holds a month, so when it is not `None`. For each such member, add the card number to `cards` with `append`.',
            '`if member["ended"] is not None:` and under it `cards.append(___)`',
        ],
        why='`if member["ended"] is not None:` with `cards.append(member["card"])` under it collects the card numbers of the members who left. `to_delete(MEMBERS)` gives `["L-1003", "L-1006", "L-1009"]`: Yara El Amrani left in March 2019, Sem Brouwer in November 2023 and Nora Willems in June 2025.',
    )
    return


@app.cell(hide_code=True)
def text_after_delete():
    mo.md(r"""
    Seven members stay. In a real pipeline, such a step runs by itself, as the EDPB asks. Copies count too: the EDPB asks controllers to decide which personal data backups and logs keep, and for how long. The GDPR sets no fixed period; this library deletes a member's data when the membership ends, as in the EDPB's example.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kept():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. values_kept: what is left to leak

        `values_kept(members, needed)` gives back how many values the pipeline keeps for a purpose: for each member who stays, the number of keys that `minimize` keeps for that member, added up. `len` of a dict gives its number of keys. A member whose `"ended"` is not `None` counts for nothing: that data is deleted. The box knows the course's `minimize`. Inside, `count` starts at 0. Replace the `...`. The check calls `values_kept(MEMBERS, PURPOSES["recommend"])`, then other lists and purposes.
        """),
        L.task("Give back how many values the pipeline keeps of the members who stay."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kept():
    box_kept = L.code_box("values-kept", lines=7)
    box_kept
    return (box_kept,)


@app.cell(hide_code=True)
def check_kept(box_kept):
    result_kept = L.check("values-kept", box_kept.value)
    result_kept
    return (result_kept,)


@app.cell(hide_code=True)
def hints_kept():
    L.hints(
        "values-kept",
        [
            'Step 4 of the recipe. `len({"a": 1, "b": 2})` gives `2`.',
            "Skip a member whose membership ended. For each other member, let the course's function from Exercise 1 do the keeping, and add to `count` how many keys it kept.",
            '`if member["ended"] is None:` and under it `count = count + len(___)`',
        ],
        why='`if member["ended"] is None:` with `count = count + len(minimize(member, needed))` under it adds up what the pipeline keeps. `values_kept(MEMBERS, PURPOSES["recommend"])` gives `7`: one value for each of the 7 members who stay. For lending it gives `21`, and with every key, `values_kept(MEMBERS, FIELDS)` gives `62`: 7 members times 9 keys, minus the phone number Lina Postma never gave.',
    )
    return


@app.cell(hide_code=True)
def text_after_kept():
    mo.md(r"""
    From 89 values to 7: what the recommendation model gets is less than a tenth of the file. If its data leaked, it would show which kinds of books seven members borrowed, without their names, e-mail addresses or dates of birth. That can still be personal data: a reading list can point to one person. And lending still needs names and e-mail addresses. Privacy by design does not make a pipeline safe; it makes what can go wrong smaller.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### A purpose of your own

    Choose the keys a purpose of your own needs. The picture shows what the pipeline keeps of the seven members who stay: with your `minimize` once Exercise 1 is right, and with the course's until then. Try a newsletter: which keys does it need? And which of the members would you send it to?
    """)
    return


@app.cell(hide_code=True)
def try_keys_cell():
    try_keys = mo.ui.multiselect(options=list(C.FIELDS), value=["books"], label="Keys your purpose needs")
    try_keys
    return (try_keys,)


@app.cell(hide_code=True)
def demo_try(box_minimize, result_minimize, try_keys):
    _yours = C.mine(box_minimize.value, result_minimize.state == "correct")
    _out = mo.Html(C.try_picture(try_keys.value, _yours, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_minimum():
    mo.md(r"""
    ### Exercise 4. Less than needed

    To let even less leak, the library takes the e-mail address out of `PURPOSES["lending"]`. Now it can no longer send a reminder when a book is late.
    """)
    return


@app.cell(hide_code=True)
def choice_minimum_cell():
    choice_minimum = L.choice(
        "minimum",
        question="What does data minimization ask of the library?",
        options={
            "few": "As little as possible, whatever the purpose",
            "right": "What the purpose needs, and nothing more",
            "filter": "Only what a filter can hide afterwards",
        },
        correct="right",
        why={
            "right": "The GDPR asks for data that is adequate, relevant and limited to what is necessary for the purpose. For lending with a reminder, the e-mail address is necessary: without it, the library has too little for its purpose.",
            "few": "The first word of the rule is adequate: enough for the purpose. A library that cannot send its reminder has too little. See *What the law asks* in *The idea*.",
            "filter": "A filter comes after the data is in; data minimization decides, by the purpose, what comes in at all. See *What the law asks* in *The idea*.",
        },
    )
    choice_minimum
    return (choice_minimum,)


@app.cell(hide_code=True)
def text_exercise_default():
    mo.md(r"""
    ### Exercise 5. The newsletter

    A new member signs up on the library's website. The library sends a newsletter to the members who want it.
    """)
    return


@app.cell(hide_code=True)
def choice_default_cell():
    choice_default = L.choice(
        "default",
        question="Which sign-up form follows privacy as the default?",
        options={
            "right": "The newsletter box starts empty; members can tick it",
            "ticked": "The newsletter box starts ticked; members can untick it",
            "link": "No box: every newsletter has a link to stop it",
        },
        correct="right",
        why={
            "right": "Privacy as the default: \"If an individual does nothing, their privacy still remains intact.\" A member who does nothing gets no newsletter, and the library does not use the e-mail address for it.",
            "ticked": "A member who does nothing then gets the newsletter: the default works against their privacy. See *By default* in *The idea*.",
            "link": "A link to stop asks the member to act after the fact; by default, nothing should happen when the member does nothing. See *By default* in *The idea*.",
        },
    )
    choice_default
    return (choice_default,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Finding and hiding personal data](/course/open/rai-find-pii?lang=en), you hid names and numbers that were already in the text, and something always slipped through. Here you decided before anything came in: per purpose which keys the pipeline takes, and when a member's data goes. In the next lesson, *Who looks after the data?*, the question becomes who in an organization decides that, who may see what, and how anyone knows where all the copies of someone's data are.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_default, choice_minimum, result_delete, result_kept, result_minimize):
    L.wrap_up(result_minimize, result_delete, result_kept, choice_minimum, choice_default)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
