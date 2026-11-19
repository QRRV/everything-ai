# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-what-is-privacy. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-what-is-privacy", "en")
    C = L.checks   # Solove's taxonomy, three made-up tables about the same made-up people, and a made-up survey


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where does this data come from?](/course/open/data-provenance?lang=en) and [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_consent():
    quick_consent = L.choice(
        "quick-consent",
        question="A dataset of messages has the license CC BY. In its datasheet, the field `consent` is empty. What does the license not tell you?",
        options={
            "right": "Whether the writers agreed",
            "credit": "Whether you must name its maker",
            "copy": "Whether you may share copies",
        },
        correct="right",
        why={
            "right": "A license comes from whoever holds the rights and says what you may do with the data. Whether the people who wrote the messages agreed is a separate question: consent.",
            "credit": "CC BY says that: you may use and share the data if you name the maker. See *The idea* in [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "copy": "CC BY allows sharing copies, as long as you name the maker. See *The idea* in [Where does this data come from?](/course/open/data-provenance?lang=en).",
        },
    )
    quick_consent
    return (quick_consent,)


@app.cell(hide_code=True)
def quick_check_stakeholders():
    quick_stakeholders = L.choice(
        "quick-stakeholders",
        question="A model decides which tenants get a reminder about late rent. Is a tenant who never gets a reminder one of its stakeholders?",
        options={
            "right": "Yes: it decides about them too",
            "flagged": "No: only those it flags count",
            "landlord": "No: only the landlord who uses it",
        },
        correct="right",
        why={
            "right": "Stakeholders are everyone with an interest in the system or affected by it, including every person it decides about, also when it decides to do nothing.",
            "flagged": "Deciding not to flag someone is a decision about them too. See *The idea* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).",
            "landlord": "The landlord is one stakeholder, but not the only one: the people the model decides about count too. See *The idea* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en).",
        },
    )
    quick_stakeholders
    return (quick_stakeholders,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    AI systems are built from information about people: what they write, buy, search for and where they go. A model can also combine such information, and work out things that nobody told it. Whether that is a privacy problem depends on what you take privacy to be, and there is no single answer. This lesson gives you three views of privacy, a list of sixteen kinds of privacy problem to name what goes wrong, and a question that runs through the whole debate: if people say they care, why do they give their data away?

    **An everyday example.** You tell your doctor that you sleep badly. You would mind if your employer heard it. It is the same information, and the doctor knowing it is no problem; where it goes next is.
    """)
    return


@app.cell(hide_code=True)
def predict_card():
    guess_card = L.guess(
        "card",
        question="In a study by Acquisti and Grossklags, some people had high concern about the collection of offline identifying information, such as name and address. What share of them do you think signed up for a loyalty card using their real identifying information?",
        options={
            "quarter": "About 1 in 4",
            "half": "About half",
            "right": "Almost 9 in 10",
        },
        reveal="**Almost 9 in 10: 87.5%**, as Solove reports the study. Such a gap between what people say and what they do is called the privacy paradox. Whether it shows that people do not really care is another question: this lesson comes back to it.",
    )
    guess_card
    return (guess_card,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Everyone uses the word privacy, and hardly anyone means the same by it. The law professor Daniel Solove wrote in 2006: "Privacy is a concept in disarray. Nobody can articulate what it means." Instead of one definition, this lesson gives you three views, and a list of problems that you can name.

        ### Three views of privacy

        In an earlier paper, from 2002, Solove sorts what scholars have written about privacy under six headings, and notes that they only track how scholars chose to think about it. The course takes three of them, joins two of those, and adds one later idea: three views.

        **Being let alone.** In 1890, Samuel Warren and Louis Brandeis wrote about a "right to be let alone", a phrase they took from an 1880 book by Judge Thomas Cooley on torts, the law on harm that one person does to another. Close to it is privacy as secrecy: privacy is broken when something that was hidden becomes known.

        **Control.** Alan Westin wrote in 1967, in words that Solove quotes: "Privacy is the claim of individuals, groups, or institutions to determine for themselves when, how, and to what extent information about them is communicated to others."

        **Appropriate flow.** Helen Nissenbaum calls privacy the appropriate flow of personal information. What counts is whether a flow of information fits the context it comes from. Nissenbaum writes that this view "cares only whether the flow is appropriate, not whether it takes place at all". Your doctor knowing that you sleep badly fits health care; your employer knowing it does not. Nissenbaum's name for this view is contextual integrity, the subject of *Information in the wrong place*, two lessons on.

        The views can disagree about the same case. Something you posted yourself is no longer secret, so secrecy sees no problem when others pass it on. Control asks whether you decided. Appropriate flow asks where it went next, and whether that fits where it came from.

        ### Solove's taxonomy

        A taxonomy is a sorting into groups and kinds. Solove's taxonomy of 2006 does not define privacy: it sorts the activities that can cause privacy problems. Solove finds four groups and sixteen kinds. The first three groups follow information as it moves further away from the control of the person it is about: it is collected, processed and spread. The fourth, invasion, comes toward the person, and need not involve information. Solove's processing is narrower than processing in the GDPR of the next lesson, where collecting counts as processing too.
        """),
        mo.Html(C.taxonomy_picture("en")),
        mo.md(r"""
        Many of the kinds fit AI systems. A camera in a doorbell that records the street is surveillance. When a company collects chats to answer questions and then trains a model on them without asking, that is secondary use. A model that links an account to the person behind it is identification. And a data broker that joins shopping, location and posts into one profile is aggregation, about which Solove writes: "A piece of information here or there is not very telling. But when combined together, bits and pieces of data begin to form a portrait of a person. The whole becomes greater than the parts."

        ### The privacy paradox

        The privacy paradox is the observation that people say they value privacy highly, yet give their personal data away for very little or do not protect it. The name spread in the mid-2000s. As Solove reports the studies, in the study by Acquisti and Grossklags from the guess, 87.5% of the people with high concern signed up for a loyalty card using their real identifying information; in another, 95% said they were "interested in the protection of their personal information", and nearly all then "chose the store that offered the cheaper price but required more personal data".

        Not everyone agrees on what that shows. Susanne Barth and Menno de Jong reviewed 32 papers in 2017 and found that people weigh what they risk against what they get, or judge the risk to be negligible. They also warn that what people think and plan to do is not the same as what they do, and that the studies seldom measure what people actually do. Solove, in 2021, calls the paradox "a myth created by faulty logic": a choice to share one piece of data in one context, for one benefit, says little about how much someone values privacy in general. In Solove's words: "A person does not surrender all privacy when sharing data with others." Barth and de Jong, writing four years earlier, treat the paradox as real and try to explain it: the sources disagree about whether it exists.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **privacy** | what a person can keep from others, and how information about them may be gathered, used and passed on; there is no single definition |
        | **taxonomy** | a sorting of things into groups and kinds; Solove's sorts the activities that can cause privacy problems |
        | **aggregation** | combining pieces of information about one person, so that together they say more than each piece alone |
        | **privacy paradox** | the observation that people say they value privacy, yet give their personal data away for very little |
        """),
        L.watch_out("Solove's taxonomy is not a definition of privacy. It sorts activities that can cause privacy problems; privacy itself, Solove says, has no single essence."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To name a privacy problem in an AI system:

    1. Ask what the system does with information about people: collects it, processes it, spreads it, or intrudes on the people themselves.
    2. Find the group and the kind in Solove's taxonomy. That is `group_of`.
    3. Look at what pieces of information say together, not only one at a time: aggregation. That is `facts`.
    4. Ask the three views: does something hidden become known, did the people decide, and does the flow fit the context the information came from?
    5. When people say they care and still share, measure the gap, and ask what they got in return before you conclude that they do not care. That is `share_who_shared`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four events in AI systems, made up by the course, each with the kind of privacy problem it is. `_group` gives the group of each of those kinds. The cell prints, for each event, the group, the kind and the event.
    """)
    return


@app.cell
def example_events():
    # four made-up events in AI systems, each with the kind of privacy problem it is
    _events = [("a doorbell camera records the street all day", "surveillance"),
               ("a chat app trains a model on its chats without asking", "secondary use"),
               ("a data broker joins shopping and location data", "aggregation"),
               ("a chatbot repeats an address from a public record", "increased accessibility")]
    _group = {"surveillance": "collection", "secondary use": "processing", "aggregation": "processing",
              "increased accessibility": "dissemination"}

    for _event, _kind in _events:
        print(_group[_kind], "|", _kind, "|", _event)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two of the four events are processing. The address in a public record was already there to find; the chatbot makes it easier to find, which Solove calls increased accessibility. `_group` goes from a kind straight to its group. Solove's taxonomy goes the other way, from each group to the list of its kinds: to find the group of a kind there, you go over the groups. That is Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_group():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. group_of: which group does a kind belong to?

        `group_of(kind, taxonomy)` gets a kind of privacy problem `kind`, such as `"aggregation"`, and a taxonomy `taxonomy`: a dictionary from each group to the list of its kinds. `TAXONOMY` is Solove's, with the four groups and sixteen kinds of the picture, such as `"processing": ["aggregation", "identification", "insecurity", "secondary use", "exclusion"]`. The `for` loop goes over `taxonomy.items()`: every pair of a group's name `group` and its list `kinds`. Replace the `...` with two lines: when `kind` is in `kinds`, give back `group`. After the loop, the function gives back `None`: no group has the kind. The check calls `group_of("aggregation", TAXONOMY)`, then other kinds, a kind that is not in the taxonomy, and a smaller taxonomy.
        """),
        L.task("Give back the group whose list of kinds holds `kind`, or `None`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_group():
    box_group = L.code_box("group", lines=6)
    box_group
    return (box_group,)


@app.cell(hide_code=True)
def check_group(box_group):
    result_group = L.check("group", box_group.value)
    result_group
    return (result_group,)


@app.cell(hide_code=True)
def hints_group():
    L.hints(
        "group",
        [
            "Step 2 of the recipe. Each time through the loop, `kinds` is the list of one group: ask whether `kind` is in it.",
            "An `if` with `in`, and under it a `return`. Which of the two loop variables is the name of a group, and which is its list?",
            "`if kind in kinds:` and under it `return ___`",
        ],
        why="`if kind in kinds:` and under it `return group`. `group_of(\"aggregation\", TAXONOMY)` is `\"processing\"`, and a kind that no group has, such as `\"profiling\"`, gives `None`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_facts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. facts: what do the tables say together?

        Three made-up apps each keep a table about the same made-up people, by email address: `RUNS`, what a running app knows from their runs, `PHARMACY`, what a pharmacy card knows from their purchases, and `FORUM`, what a forum knows from their posts. `TABLES` is the list of the three dictionaries. Four of the six people:

        | person | running app | pharmacy card | forum |
        |---|---|---|---|
        | sam@example.org | runs from Harbor Street at 6:30 on weekdays | buys allergy tablets every April | writes about night shifts at the hospital |
        | noor@example.org | runs in the dunes on Sunday mornings | | writes about moving house in May |
        | lior@example.org | | buys reading glasses every two years | |
        | ash@example.org | runs to the station at 7:15 on weekdays | buys sleeping tablets every week | writes about looking for a new job |

        `facts(key, tables)` gets an email address `key` and a list `tables` of dictionaries, and gives back a list of what every table that has `key` says about it. `found` starts as an empty list, and the `for` loop goes over every `table`. Replace the `...` with two lines: when `key` is in `table`, add `table[key]` to `found`. The function gives back `found`. The check calls `facts("sam@example.org", TABLES)`, then other people, an address that is in no table, and a short list.
        """),
        L.task("Give back a list of what every table that has `key` says about it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_facts():
    box_facts = L.code_box("facts", lines=7)
    box_facts
    return (box_facts,)


@app.cell(hide_code=True)
def check_facts(box_facts):
    result_facts = L.check("facts", box_facts.value)
    result_facts
    return (result_facts,)


@app.cell(hide_code=True)
def hints_facts():
    L.hints(
        "facts",
        [
            "Step 3 of the recipe. Not every table has every person, so ask first whether `key` is in `table`.",
            "An `if` with `in`, and under it `found.append(...)`. How do you look up in a dictionary what it holds for a key?",
            "`if key in table:` and under it `found.append(___)`",
        ],
        why="`if key in table:` and under it `found.append(table[key])`. `facts(\"sam@example.org\", TABLES)` gives three facts, one from each app.",
    )
    return


@app.cell(hide_code=True)
def text_after_facts():
    mo.md(r"""
    Each app alone knows one thing about sam. Together they tell where sam starts running on weekday mornings, and so perhaps where sam lives, that sam has an allergy, and that sam works night shifts at a hospital. Each app alone holds little, and still the whole says more than the parts: that is aggregation. A system that joins such tables knows it too, and can also guess wrong: sam may start from the hospital rather than from home.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_views():
    mo.md(r"""
    ### Exercise 3. Three views, one case
    """)
    return


@app.cell(hide_code=True)
def choice_views_cell():
    choice_views = L.choice(
        "views",
        question="Jules agreed, in an app's terms, that it may pass on Jules's step counts to partners. The app sends them to Jules's doctor, and also to an insurer, which raises Jules's premium. Which view sees a problem in the flow to the insurer, although Jules agreed?",
        options={
            "secret": "Secrecy: the data were hidden first",
            "control": "Control: Jules had no say in it",
            "right": "Flow: it does not fit its context",
        },
        correct="right",
        why={
            "secret": "The step counts were not hidden: Jules shared them with the app and agreed that it may pass them on. Once information is shared, secrecy sees no problem. See *Three views of privacy* in *The idea*.",
            "control": "Jules did have a say, at least on paper, and agreed in the terms. In the control view the person decides, so it sees no problem in a flow the person agreed to. See *Three views of privacy* in *The idea*.",
            "right": "Step counts that go to a doctor fit the context of health care; the same numbers used to set a premium at an insurer do not, and a tick in the app's terms does not settle that by itself. In Nissenbaum's view, consent is not the only judge of whether a flow is appropriate.",
        },
    )
    choice_views
    return (choice_views,)


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. share_who_shared: say one thing, do another?

        `SURVEY` is a survey of 20 people, made up by the course. Each person is a dictionary with `"concern"`, how concerned they say they are about privacy, from 1 (not at all) to 5 (very), and `"shared"`, `True` when they signed up for a store card with their name and address, for a discount. `share_who_shared(people, least)` gets such a list `people` and a number `least`, and gives back, of the people whose concern is `least` or more, the share who shared. `count` and `shared` start at 0, and the `for` loop goes over every `person`. Replace the `...` with four lines: when `person["concern"]` is at least `least`, add 1 to `count`, and when `person["shared"]` is also `True`, add 1 to `shared`. The function gives back `shared / count`. The check calls `share_who_shared(SURVEY, 4)`, then other values of `least`, and a short list.
        """),
        L.task("Give back, of the people whose concern is `least` or more, the share who shared."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=10)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Step 5 of the recipe. Count two things: in `count` the people who are concerned enough, and in `shared` those of them who shared.",
            "Two `if`s, one inside the other: the outer one about the person's concern, the inner one about `person[\"shared\"]`. At least means `>=`.",
            "`if person[\"concern\"] >= ___:`",
        ],
        why="`if person[\"concern\"] >= least:` with under it `count = count + 1` and `if person[\"shared\"]:` with under it `shared = shared + 1`. `share_who_shared(SURVEY, 4)` is 0.875: 7 of the 8 people with concern 4 or more shared.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    In the made-up survey, 7 of the 8 people with concern 4 or more signed up for the card: 0.875, the same share as in the real study from the guess; the course chose it so. Of the 12 others, 10 did: 0.833. Read as the privacy paradox, the concerned people act as if they did not care. Read the way Solove reads such studies, each of them made one choice, about a name and an address, for a discount, at one store; that says little about how much they value privacy in general. A survey like this one measures what people say, and only one thing they do.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Exercise 5. A headline
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Suppose a news site writes about the study from the guess: \"Nobody cares about privacy: 87.5% hand over their address for a discount.\" What does the study, as Solove reports it, support?",
        options={
            "exact": "The headline as it stands",
            "right": "Less: one choice, not what they value",
            "more": "More: concerned people lie in surveys too",
        },
        correct="right",
        why={
            "exact": "The 87.5% are of the people with high concern, not of everyone, and they signed up for a loyalty card. Solove calls it a leap in logic to go from such a choice in one context to how people value privacy. See *The privacy paradox* in *The idea*.",
            "right": "The study saw people with high concern sign up for a loyalty card using their real identifying information: what they did in one case, for one benefit. Solove calls it a leap in logic to conclude from that how people value privacy.",
            "more": "The study set what people said beside what they did; it did not show that they lied. See *The privacy paradox* in *The idea*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Where does this data come from?](/course/open/data-provenance?lang=en) you wrote down whose consent a dataset has, and in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en) you asked whom a system can harm. Privacy asks about the people in the data themselves: which information about them goes where, and what that can do to them. Solove's taxonomy gives each problem a name, and the three views say why it is one. In the next lesson, *The GDPR for AI*, you read what the law in Europe says about information about people.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_headline, choice_views, result_facts, result_group, result_share):
    L.wrap_up(result_group, result_facts, choice_views, result_share, choice_headline)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
