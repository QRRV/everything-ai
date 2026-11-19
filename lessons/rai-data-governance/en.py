# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-data-governance. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-data-governance", "en")
    C = L.checks   # the catalog and the staff of a made-up library


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Privacy by design](/course/open/rai-privacy-by-design?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kept():
    quick_kept = L.choice(
        "quick-kept",
        question="In the made-up library, `values_kept(MEMBERS, PURPOSES[\"recommend\"])` gives `7`. What does the 7 count?",
        options={
            "members": "The members who want recommendations",
            "right": "The values kept of the members who stay",
            "keys": "The keys that the recommendation model needs",
        },
        correct="right",
        why={
            "right": "For each of the 7 members who stay, the pipeline keeps one value for the recommendation model: the kinds of books that member borrowed. 7 members, 1 value each.",
            "members": "Nobody was asked whether they want recommendations. The 7 counts values: per member who stays, the keys that `minimize` keeps. See Exercise 3 in [Privacy by design](/course/open/rai-privacy-by-design?lang=en).",
            "keys": "The recommendation model needs one key, `\"books\"`. The 7 is that one key for each of the 7 members who stay. See Exercise 3 in [Privacy by design](/course/open/rai-privacy-by-design?lang=en).",
        },
    )
    quick_kept
    return (quick_kept,)


@app.cell(hide_code=True)
def quick_check_left():
    quick_left = L.choice(
        "quick-left",
        question="A member left the library in 2019. What does storage limitation ask of the data about that member?",
        options={
            "keep": "Keep it in case the member comes back",
            "right": "Delete it when it is no longer needed",
            "filter": "Keep it, with the name hidden by a filter",
        },
        correct="right",
        why={
            "right": "Storage limitation: data is kept in a form that identifies people \"for no longer than is necessary\". The membership ended, and with it the purpose: the data goes, or is anonymized so that nobody can find the person again.",
            "keep": "That the member might come back is no purpose: the membership has ended. See *Delete when it is done* in [Privacy by design](/course/open/rai-privacy-by-design?lang=en).",
            "filter": "Hiding the name keeps the rest of the record, which can still point to the member. See *Delete when it is done* in [Privacy by design](/course/open/rai-privacy-by-design?lang=en).",
        },
    )
    quick_left
    return (quick_left,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Joris van Dam, a member of the made-up library from [Privacy by design](/course/open/rai-privacy-by-design?lang=en), ends the membership and asks the library to erase all data about Joris. The desk can delete the row in the member file. But the library has more files than that: the loans, the data its recommendation model was trained on, the model itself, a newsletter list, a backup from October 2026 and a survey from 2024 in a shared folder. Which of them hold data of Joris van Dam? Who decides, for each of them, whether the data goes? And who may read it in the meantime? Without answers, a request to erase reaches the file someone remembers, and misses the rest. In this lesson you give the library those answers in code: who may read which dataset, where someone's data is, and where nobody is in charge.

    **An everyday example.** You leave your sports club and ask it to delete your data. The secretary deletes you from the member list. But your name is still in last year's tournament spreadsheet on someone's laptop and in the club's mailing tool, and nobody at the club knows who is supposed to decide about those.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Researchers split the training data of a model into 20 parts and trained one model per part, so that forgetting one person's data means training only one part again. For 8 such requests on one dataset, how much faster was that than training again from scratch?",
        options={
            "few": "about 1.5 times",
            "right": "about 4.6 times",
            "many": "about 20 times",
        },
        reveal="**4.63 times** faster, on the dataset Purchase, with each part further divided into 50 slices, according to Bourtoule and others, who call the method SISA. It is not free: *And the model?* in *The idea* says what it costs, and why a model needs it at all.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Someone has to decide

    In [Privacy by design](/course/open/rai-privacy-by-design?lang=en) you decided per purpose what the pipeline keeps and when data goes. In an organization, someone has to make those decisions for every dataset, and keep making them. That is **data governance**. A definition in the glossary of NIST, the US standards institute: "A set of processes that ensures that data assets are formally managed throughout the enterprise." It is not about who happens to have the data. The UK government's guidance on data ownership says: "Data ownership does not deal with ‘possession of data’. It’s about formalising the roles of people responsible for the management of data throughout its life cycle."

    ### Owner, steward, custodian

    The UK guidance names three roles around a dataset. The **owner** is "a senior individual with dedicated accountabilities for data". Owners "do not need a granular understanding of the data", but they are accountable, among other things, for a retention schedule and for limiting access to those who are authorized. The steward knows what is in the dataset and runs it from day to day: "Data stewards are responsible for day-to-day operational activities in their data domain that support data owners’ decisions". And the custodian looks after the system that holds it: "Data custodians are responsible for capturing, storing and disposing of data in line with the data owner’s requirements." In the guidance's words, owners are "accountable" and stewards "responsible". These role names are not a fixed standard, and other sources divide the work differently; this lesson follows the UK guidance. It also recommends reviewing who owns what "on an annual (once a year) basis", and whenever roles or responsibilities change significantly.

    ### Who may see what

    **Access control** is, in a NIST definition, "The decision to permit or deny a subject access to system objects (network, data, application, service, etc.)." Which access to give follows the principle of least privilege: "a system should restrict the access privileges of users (or processes acting on behalf of users) to the minimum necessary to accomplish assigned tasks." Least privilege does for access what data minimization does for collecting. And it is not set once: NIST's catalog of security controls asks for "A periodic review of assigned user privileges", to check that the reason for each one still holds. In the made-up library, access goes by role: the desk, lending, the data team, communications and IT may each read what their tasks need, and a person may read what one of their roles allows.

    ### A catalog of what exists

    Nobody can look after data that nobody knows about. A **data catalog** lists the datasets of an organization, with who looks after each one and who may use it. Here, copies count as datasets too: a backup, an export in a shared folder, a model trained on the data. For one dataset, the datasheet of [Where does this data come from?](/course/open/data-provenance?lang=en) goes deeper. Gebru and others, who proposed datasheets, ask among other things "Who will be supporting/hosting/maintaining the dataset?" and, for a dataset about people, whether there are "applicable limits on the retention of the data associated with the instances".

    ### A request to erase

    In [The GDPR for AI](/course/open/rai-gdpr?lang=en) you met the right to erasure: on one of six grounds, a person can have their personal data erased "without undue delay", with exceptions. Article 12 sets the deadline: the controller tells the person what it did with the request "without undue delay and in any event within one month of receipt of the request". "That period may be extended by two further months where necessary, taking into account the complexity and number of the requests", and the person has to hear about that within the first month, "together with the reasons for the delay". At most three months in all, then. To answer in time, the library has to know where the data is, and who decides for each copy whether it goes or, under one of the exceptions, stays. A catalog with owners gives it both.

    ### And the model?

    The recommender was trained on the rows of the members who stay. Deleting a row from the training data does not change a model that has already been trained. The EDPB's opinion on AI models, which you met in [The GDPR for AI](/course/open/rai-gdpr?lang=en), says that "information from the training dataset, including personal data, may still remain ‘absorbed’ in the parameters of the model", and concludes that "AI models trained on personal data cannot, in all cases, be considered anonymous": whether a model is anonymous should be judged case by case. The sure way to forget is to train again without the data, which Bourtoule and others call "A naive way": it works, but it costs a whole training run every time. Their SISA training, "short for Sharded, Isolated, Sliced, and Aggregated training", splits the training data into parts, shards, trains one model per shard and combines their answers, for instance with "a majority vote over predicted labels". A request to forget then means retraining "only the affected model". With 20 shards, each divided into 50 slices, that was 4.63 times faster than retraining from scratch for 8 requests on the dataset Purchase, and 2.45 times faster for 18 requests on SVHN, at "a nominal degradation in accuracy of less than 2 percentage points". It costs extra storage, it only saves time while there are fewer requests than three times the number of shards, and on ImageNet the speed-up was 1.36 times, for 19.45 percentage points less top-5 accuracy. Hiding a name from the model's answers is not the same thing: in their words, forgetting "requires that a particular training point have zero contribution to the model".
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **data governance** | the processes and roles that decide, for every dataset, who is accountable for it, who may use it and how long it is kept |
        | **access control** (*toegangsbeheer*) | deciding and enforcing who may read or change which data and systems |
        | **data catalog** | an overview of the datasets of an organization, with for each one who looks after it and who may use it |
        """),
        L.watch_out("The owner of a dataset is not whoever happens to hold it: in the UK guidance, storing and disposing of data is the custodian's work, and the owner is a senior person who decides. And deleting rows from the training data does not make a trained model forget them."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To know who looks after the data, and to answer a request to erase:

    1. Put every dataset in a catalog, copies included, each with an owner, a steward and a custodian, and the people whose data it holds.
    2. Give each role access to what its tasks need, and nothing more, and review it regularly. That is `may_read`.
    3. When someone asks to have their data erased, find every dataset that holds it. That is `holding`.
    4. Look for the gaps: datasets that hold the data and that nobody owns. That is `no_owner`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell prints the catalog of the library: per dataset its owner, steward and custodian, how many members' data it holds and which roles may read it. `C.CATALOG` is a list of eight dicts; `None` means that nobody has that role. Then it counts the datasets that nobody owns.
    """)
    return


@app.cell
def example_catalog():
    # the catalog of the made-up library: every dataset with who looks after it
    for _d in C.CATALOG:
        print(f'{_d["name"]}: owner {_d["owner"]}, steward {_d["steward"]}, custodian {_d["custodian"]}')
        print("    cards:", len(_d["cards"]), "| may read:", _d["access"])

    # how many datasets nobody owns
    _count = 0
    for _d in C.CATALOG:
        if _d["owner"] is None:
            _count = _count + 1
    print("datasets without an owner:", _count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Six datasets have an owner, a steward and a custodian. Two have only a custodian: the backup of October 2026 and the survey of 2024. Both hold members' data, of ten and of three members, and nobody decides what happens to it. The opening hours hold no member data, and every role may read them. For the recommender, the catalog lists the cards of the seven members it was trained on: their data may still be in its parameters, the EDPB's point in *And the model?*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. may_read: who may see what

        `may_read(person, dataset)` gets a staff member as a dict `person`, with a `"name"` and a list `"roles"`, and a dataset from the catalog as a dict `dataset`, whose list `"access"` holds the roles that may read it. It gives back `True` when at least one of the person's roles is in that list, and `False` otherwise. `STAFF` is the list of the library's staff; `STAFF[0]` is Noor Jansen, at the desk. The loop goes over the person's roles, and the last line gives back `False` for when no role matched. Replace the `...`. The check calls `may_read(STAFF[0], CATALOG[0])`, then other people and datasets.
        """),
        L.task("Give back True as soon as one of the person's roles may read the dataset."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("may-read", lines=6)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("may-read", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "may-read",
        [
            "Step 2 of the recipe. `return` ends a function straight away, also in the middle of a loop.",
            "As soon as one role of the person may read the dataset, the answer is `True`, and the function can stop. Only when the loop has gone through every role without a match is the answer `False`.",
            '`if ___ in dataset["access"]:` and under it `return True`',
        ],
        why='`if role in dataset["access"]:` with `return True` under it gives `True` at the first role that may read the dataset; after the loop, `return False` gives the rest. `may_read(STAFF[0], CATALOG[0])` gives `True`: Noor Jansen has the role `"desk"`, and the desk may read the member file.',
    )
    return


@app.cell(hide_code=True)
def text_after_read():
    mo.md(r"""
    With a right `may_read`, 25 of the 72 squares are filled. Noor Jansen at the desk may read the member file, the loans and the opening hours, and nothing the data team works on. Daan Smit has two roles, and may read what either of them allows. Mees Peters, a new volunteer with no role yet, may read nothing at all. And the survey of 2024, which nobody owns, can be read by four people: the access was given once, and nobody has reviewed it since.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_holding():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. holding: where someone's data is

        `holding(catalog, card)` gets the catalog, a list of datasets, and a card number `card`. It gives back the names of the datasets whose `"cards"` include `card`, in the order of the catalog: every place a request to erase has to reach. Inside, `names` starts as an empty list, and the loop goes over the datasets. Replace the `...`. The check calls `holding(CATALOG, "L-1004")`, with the card of Joris van Dam, then other cards and lists.
        """),
        L.task("Give back the names of the datasets that hold the card."),
    ])
    return


@app.cell(hide_code=True)
def exercise_holding():
    box_holding = L.code_box("holding", lines=7)
    box_holding
    return (box_holding,)


@app.cell(hide_code=True)
def check_holding(box_holding):
    result_holding = L.check("holding", box_holding.value)
    result_holding
    return (result_holding,)


@app.cell(hide_code=True)
def hints_holding():
    L.hints(
        "holding",
        [
            'Step 3 of the recipe. `"b" in ["a", "b"]` gives `True`: `in` asks whether a list holds a value.',
            "For each dataset, ask whether the card is among its card numbers. If it is, add the name of the dataset to `names`.",
            '`if card in ___:` and under it `names.append(dataset["name"])`',
        ],
        why='`if card in dataset["cards"]:` with `names.append(dataset["name"])` under it collects every dataset that holds the card. `holding(CATALOG, "L-1004")` gives `["members", "loans", "recommend-train", "recommender", "backup-2026-10", "survey-2024"]`: six of the eight datasets.',
    )
    return


@app.cell(hide_code=True)
def text_after_holding():
    mo.md(r"""
    The catalog lists the card of Joris van Dam in six of the eight datasets: not only in the member file and the loans, but also in the training data, the backup, the survey and the model, which may still hold the data in its parameters. And Nora Willems, who left in June 2025, is still in two: the backup of October 2026 was made before the pipeline of [Privacy by design](/course/open/rai-privacy-by-design?lang=en) deleted the members who left, and the survey of 2024 was never part of that pipeline. A request to erase that only reaches the member file misses those copies.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_owner():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. no_owner: where nobody decides

        `no_owner(catalog, card)` gives back the names of the datasets that hold `card` and have no owner: their `"owner"` is `None`. For those, nobody is accountable for the request, and nobody decides whether the data may stay. Inside, `names` starts as an empty list, and the loop goes over the datasets. A dataset has to pass two questions, so you need one `if` inside another. Replace the `...`. The check calls `no_owner(CATALOG, "L-1004")`, then other cards and lists.
        """),
        L.task("Give back the names of the datasets that hold the card and have no owner."),
    ])
    return


@app.cell(hide_code=True)
def exercise_owner():
    box_owner = L.code_box("no-owner", lines=8)
    box_owner
    return (box_owner,)


@app.cell(hide_code=True)
def check_owner(box_owner):
    result_owner = L.check("no-owner", box_owner.value)
    result_owner
    return (result_owner,)


@app.cell(hide_code=True)
def hints_owner():
    L.hints(
        "no-owner",
        [
            "Step 4 of the recipe. An `if` inside another `if` runs only when both conditions are `True`; the inner `if` has four more spaces in front of it.",
            "First ask whether the dataset holds the card, as in Exercise 2. Only then ask whether nobody owns it, and if so, add its name to `names`.",
            '`if card in dataset["cards"]:` and under it `if ___ is None:`',
        ],
        why='`if card in dataset["cards"]:`, with `if dataset["owner"] is None:` under it and `names.append(dataset["name"])` under that, keeps the datasets that hold the card and have no owner. `no_owner(CATALOG, "L-1004")` gives `["backup-2026-10", "survey-2024"]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_owner():
    mo.md(r"""
    Two of the six datasets that hold the data of Joris van Dam have no owner: the backup and the survey. Kees Mulder keeps both as their custodian, but a custodian disposes of data "in line with the data owner’s requirements", and here no owner sets any. The fix is not more code: someone has to become the owner of each, and decide what happens to it. Every member of the library is in the backup, so for each of the ten, `no_owner` finds at least that one.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Whose data, and where

    Choose a member. The picture shows which datasets hold that member's data, with who looks after each: with your `holding` once Exercise 2 is right, and with the course's until then. Try Sem Brouwer, who left in November 2023.
    """)
    return


@app.cell(hide_code=True)
def try_member_cell():
    try_member = mo.ui.dropdown(options=C.MEMBER_CARDS, value="Joris van Dam", label="Whose data?")
    try_member
    return (try_member,)


@app.cell(hide_code=True)
def demo_whose(box_holding, result_holding, try_member):
    _yours = C.mine(box_holding.value, result_holding.state == "correct")
    _out = mo.Html(C.whose_picture(try_member.value, _yours, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_deadline():
    mo.md(r"""
    ### Exercise 4. The deadline

    Joris van Dam asks the library on 3 March to erase all data about Joris. Finding every copy takes time: the backup, the survey, the model.
    """)
    return


@app.cell(hide_code=True)
def choice_deadline_cell():
    choice_deadline = L.choice(
        "deadline",
        question="What does the GDPR ask of the library?",
        options={
            "one": "One month, and never longer than that",
            "right": "One month, or three if needed and announced",
            "free": "No deadline, as long as it is done in the end",
        },
        correct="right",
        why={
            "right": "Article 12: an answer \"within one month of receipt of the request\", a period that \"may be extended by two further months where necessary\", if Joris van Dam hears so within that first month, with the reasons. At most three months in all.",
            "one": "Article 12 allows two more months when the requests are complex or many, if the person hears so within the first month, with the reasons. See *A request to erase* in *The idea*.",
            "free": "Article 12 gives one month, and at most two more. See *A request to erase* in *The idea*.",
        },
    )
    choice_deadline
    return (choice_deadline,)


@app.cell(hide_code=True)
def text_exercise_model():
    mo.md(r"""
    ### Exercise 5. And the model?

    The library deletes the rows of Joris van Dam from `recommend-train`. Before that, the recommender was trained on those rows.
    """)
    return


@app.cell(hide_code=True)
def choice_model_cell():
    choice_model = L.choice(
        "model",
        question="What does that mean for the recommender?",
        options={
            "numbers": "Joris is not in it: a model only holds numbers",
            "right": "Joris may still be in it, in its parameters",
            "gone": "It forgets Joris along with the deleted rows",
        },
        correct="right",
        why={
            "right": "The EDPB: information from the training data \"may still remain ‘absorbed’ in the parameters of the model\". Deleting rows does not change a model that has been trained. To be certain the data is gone, the library trains the model again without those rows, or trains it in shards, as SISA does, so that this costs less.",
            "numbers": "Numbers can hold personal data: in the EDPB's words, AI models trained on personal data \"cannot, in all cases, be considered anonymous\". See *And the model?* in *The idea*.",
            "gone": "A trained model does not change when its training data changes. See *And the model?* in *The idea*.",
        },
    )
    choice_model
    return (choice_model,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Privacy by design](/course/open/rai-privacy-by-design?lang=en), you decided per purpose what a pipeline keeps and when data goes. Here you saw who in an organization makes those decisions, who may read what, and how to find every copy of someone's data, including the copies nobody looks after. This is the last lesson of the module privacy. The module security, starting with *Thinking like an attacker*, asks what happens when someone tries to get at the data, or at the model, on purpose.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_deadline, choice_model, result_holding, result_owner, result_read):
    L.wrap_up(result_read, result_holding, result_owner, choice_deadline, choice_model)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
