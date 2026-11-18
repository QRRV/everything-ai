# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-labels-tay-cambridge. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-labels-tay-cambridge", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en), and one about [Dark patterns, nudges and manipulation](/course/open/rai-dark-patterns?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="An image search for *engineer* shows almost only men. What kind of harm is that?",
        options={
            "allocative": "Allocative harm: it hands out fewer jobs to women",
            "right": "Representational harm: in how it shows a group",
            "none": "No harm, as nobody is refused anything",
        },
        correct="right",
        why={
            "allocative": "The search hands out nothing: no job, no loan, no check. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "It decides nothing about anyone, but it shows a group in a skewed way, and that is harm too.",
            "none": "Harm does not need a refusal. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_hypernudge():
    quick_hypernudge = L.choice(
        "quick-hypernudge",
        question="What makes a hypernudge different from an ordinary nudge?",
        options={
            "law": "It is always forbidden, by the AI Act and the DSA",
            "right": "It adapts to each person, from data about them",
            "price": "It makes one of the choices cost more money",
        },
        correct="right",
        why={
            "law": "The law forbids manipulation that causes harm, not every hypernudge. See *The idea* in [Dark patterns, nudges and manipulation](/course/open/rai-dark-patterns?lang=en).",
            "right": "It changes all the time and differs per person, so others may not see what you see, which makes it hard to notice.",
            "price": "A nudge, hyper or not, hardly changes the costs. See *The idea* in [Dark patterns, nudges and manipulation](/course/open/rai-dark-patterns?lang=en).",
        },
    )
    quick_hypernudge
    return (quick_hypernudge,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2015 a photo app labeled two Black friends with the name of an ape. In 2016 a chatbot posted racist messages within hours of going online. In 2018 it came out that the data of up to 87 million Facebook users may have been shared with a company that targeted political ads. The first two cases did harm with what a system said; the third with what was done with data about people, most of whom had never used the app.

    **An everyday example.** A guest book at a party lies open all night, and anyone can write in it. By morning, the page that everyone reads first may say more about the worst guests than about the party.
    """)
    return


@app.cell(hide_code=True)
def predict_tay():
    guess_tay = L.guess(
        "tay",
        question="Microsoft put its chatbot Tay on Twitter on 23 March 2016, to learn from the people it talked with. How long do you think it stayed online?",
        options={
            "week": "About a full week",
            "month": "About a month",
            "right": "About 16 hours",
        },
        reveal="**About 16 hours**, and in that time Tay posted more than 95,000 tweets. Microsoft wrote that it had made a critical oversight for this specific attack.",
    )
    guess_tay
    return (guess_tay,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The photo labels.** In June 2015 the programmer Jacky Alciné found that Google Photos had labeled photos of Alciné and a friend, both Black, with the name of an ape, a label with a long racist history that this course does not repeat. Google said it was appalled and genuinely sorry. In 2018 the magazine *Wired* tested the app with 40,000 photos full of animals: searches for apes and monkeys found nothing. Google had not fixed the model; it had removed the labels. In 2023 *The New York Times* found the same, and Google said that the benefit of such labels does not outweigh the risk of harm.

        **Tay.** On 23 March 2016 Microsoft put Tay on Twitter, a chatbot that learned from the people it talked with. A group of users used a *repeat after me* function to make it post racist and hateful messages, and Tay also took up such language by itself. After 16 hours and more than 95,000 tweets, Microsoft took Tay offline. It apologized, and wrote: *We take full responsibility for not seeing this possibility ahead of time.*

        Both are representational harm: what the system said decided nothing about anyone, and still hurt people.

        **Microtargeting** is showing tailored political or commercial messages to small groups of people, selected with data about their traits and behavior. Others do not see those messages, as with a hypernudge.

        **Cambridge Analytica.** In 2014 the researcher Aleksandr Kogan built a Facebook app with a personality quiz. Some 300,000 people worldwide used it, and through the rules Facebook had then, the app also collected data of their friends: up to 87 million people, most of them in the United States, Facebook estimated in 2018. Users were told the app would not collect their name or other identifying information; according to the US Federal Trade Commission, the FTC, that was false. Kogan's company sold data from the app to SCL Elections and Cambridge Analytica: according to the British regulator, data on about 30 million American voters. They gave voters personality scores and aimed political ads at groups of them. *The Observer* and *The Guardian* revealed it in March 2018. The British Information Commissioner fined Facebook 500,000 pounds, the most the law of the time allowed, and in 2019 the FTC fined Facebook 5 billion dollars for breaking an earlier order about privacy. In 2020 the Commissioner concluded that the company's claims, such as 5,000 data points on each of 230 million American adults, may have been an exaggeration: its methods were common, and its predictions for new people were likely much less accurate. That the ads changed any votes has not been shown.

        That is harm in what was done with data: people's data was used for a purpose they did not agree to, and most of them had never used the app.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **microtargeting** | showing tailored political or commercial messages to small groups of people, selected with data about their traits and behavior |
        """),
        L.watch_out("Big claims about what a model can do are not evidence that it works. Cambridge Analytica said a lot about its power; the investigators found common methods and predictions that were likely much weaker on new people."),
        mo.Html(C.picture({"title": "Of every 1,000 people whose data was taken, how many used the app",
                           "installed": "used the quiz app", "others": "data taken through a friend",
                           "alt": "1,000 small squares, of which 3 are marked: of every 1,000 people whose data was taken, about 3 used the quiz app themselves."})),
        mo.md(r"""
        About 300,000 people used the app, and the data of up to 87 million people was taken: about 3 in every 1,000. The other 997 had never used the app.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look for this harm before a system goes live:

    1. Collect what the system says about people and groups, and look for insults and stereotypes, as the lesson *Two kinds of harm* showed.
    2. Ask what happens when people try to make it say something hateful, as Tay's users did.
    3. List whose data the system collects, and count who of them agreed to it.
    4. Check what the people in the data were told, against what happens to it.
    5. Be wary of claims about what a model can do, until someone has tested them on new people.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A quiz app made up by the course, installed by three people, A, B and C. `_friends` holds how many friends each of them has; the course counts them as if no two of the three have a friend in common. The app collects data of each person who installs it and of all their friends. The loop goes through the people with `_person` and adds up in `_reached` everyone whose data the app collects: the person, 1, and their friends. `_share` is the number of people who installed the app, `len(_friends)`, divided by `_reached`. Run the cell.
    """)
    return


@app.cell
def example_friends():
    _friends = {"A": 200, "B": 350, "C": 150}

    _reached = 0
    for _person in _friends:
        _reached = _reached + 1 + _friends[_person]
    _share = len(_friends) / _reached
    print(_reached, round(_share, 5))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The app collects the data of 703 people, and 3 of them installed it: a share of 0.00427, or about 4 in every 1,000. Step 3 of the recipe for another app: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_friends():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who agreed?

        Another quiz app, made up by the course, installed by four people. `friends` holds how many friends each of them has. Replace the `...` with the lines of the worked example, with the names without their underscore: `reached`, `person` and `share`. The loop goes through `friends` with `person`. Leave out the `print`: the last line is `share`, so the check sees it. The check also tries other people.
        """),
        L.task("Work out `share`: the number of people who installed the app, divided by everyone whose data it collects."),
    ])
    return


@app.cell(hide_code=True)
def exercise_friends():
    box_friends = L.code_box("friends", lines=10)
    box_friends
    return (box_friends,)


@app.cell(hide_code=True)
def check_friends(box_friends):
    result_friends = L.check("friends", box_friends.value)
    result_friends
    return (result_friends,)


@app.cell(hide_code=True)
def hints_friends():
    L.hints(
        "friends",
        [
            "The worked example does this for three people. Copy its lines, without the `print`, and take the underscore off every name.",
            "Start `reached` at 0. For each person who installed the app, the data of that person and of their friends is collected. Then divide.",
            "In the loop: `reached = reached + 1 + ___[person]`; after it: `share = len(friends) / ___`.",
        ],
        why="`reached = 0`, then `for person in friends:` with `reached = reached + 1 + friends[person]` inside it, and after the loop `share = len(friends) / reached`. That gives `0.00286`: 4 of 1,399 people.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 2. Harm and answer

    Six facts from the three cases. Put each one under what it is about.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Ask of each fact: is it something a system said, something done with data, or something done afterwards to answer the harm?",
            "The photo labels give two facts, Tay one and Cambridge Analytica three.",
            "Said: the label and \\_\\_\\_. Data: the friends and \\_\\_\\_. Answer: \\_\\_\\_ and the fine.",
        ],
        why="**Harm in what a system says or shows about people:** the label and repeat after me. **Harm in what is done with data:** the friends and the promise. **What was done about it:** the missing labels and the fine.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_claims():
    mo.md(r"""
    ### Exercise 3. What did the investigators find?
    """)
    return


@app.cell(hide_code=True)
def choice_claims_cell():
    choice_claims = L.choice(
        "claims",
        question="What did the British Information Commissioner conclude in 2020 about Cambridge Analytica?",
        options={
            "brexit": "Its ads were what decided the result of the Brexit vote",
            "right": "Its claims about its power may have been exaggerated",
            "votes": "Its ads changed the votes of millions of Americans",
        },
        correct="right",
        why={
            "brexit": "The Commissioner found no part in the EU referendum beyond some first enquiries, and the Facebook data was about American voters. See *The idea*.",
            "right": "The company claimed 5,000 data points on each of 230 million American adults; the Commissioner found that this may have been an exaggeration, that its methods were common, and that its predictions for new people were likely much less accurate.",
            "votes": "That the ads changed votes has not been shown. What did the Commissioner say about the company's own claims? See *The idea*.",
        },
    )
    choice_claims
    return (choice_claims,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) you saw that harm in how a system shows people decides nothing and is still harm: the photo labels and Tay are such cases. In [Dark patterns, nudges and manipulation](/course/open/rai-dark-patterns?lang=en) you met the hypernudge, which others do not see; microtargeting works the same way, and Cambridge Analytica built it on data that people had not agreed to give. In the next lesson, *The crash of a self-driving test car*, you look at a crash that no single failure caused.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claims, result_friends, result_kinds):
    L.wrap_up(result_friends, result_kinds, choice_claims)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
