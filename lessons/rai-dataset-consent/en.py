# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-dataset-consent. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dataset-consent", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Codes of conduct and values in research](/course/open/rai-professional-ethics?lang=en) and [Where does this data come from?](/course/open/data-provenance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_paramount():
    quick_paramount = L.choice(
        "quick-paramount",
        question="The ACM code names one consideration as the paramount one. Which?",
        options={
            "right": "The public good",
            "employer": "What your employer asks of you",
            "science": "The progress of science",
        },
        correct="right",
        why={
            "right": "The code calls the public good the paramount consideration, whoever you work for.",
            "employer": "A code of conduct applies whoever you work for. See *The idea* in [Codes of conduct and values in research](/course/open/rai-professional-ethics?lang=en).",
            "science": "Progress matters, but the code puts something else first. See *The idea* in [Codes of conduct and values in research](/course/open/rai-professional-ethics?lang=en).",
        },
    )
    quick_paramount
    return (quick_paramount,)


@app.cell(hide_code=True)
def quick_check_no_license():
    quick_no_license = L.choice(
        "quick-no-license",
        question="A dataset you found online has no license. What do you assume?",
        options={
            "free": "You may do anything with it, as with CC0",
            "public": "It is in the public domain, free for all",
            "right": "You may not copy or share it: ask first",
        },
        correct="right",
        why={
            "free": "CC0 is a license in which the maker gives up their rights. Without any license, nothing is given. See *The idea* in [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "public": "Something is in the public domain because its rights have run out or were given up, not because nobody wrote a license. See *The idea* in [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "Without a license, assume you may not copy or share it, and ask the maker before you build on it.",
        },
    )
    quick_no_license
    return (quick_no_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Models are trained on huge amounts of text, photos and recordings, much of it simply downloaded. That a file can be downloaded says nothing about whether you may train on it, or whether the people in it would agree. This lesson keeps those two questions apart, and shows when research with people needs a check before it starts.

    **An everyday example.** A photo you took of a friend is yours as far as copyright goes. You would still ask your friend before putting it on a billboard.
    """)
    return


@app.cell(hide_code=True)
def predict_missing():
    guess_missing = L.guess(
        "missing",
        question="Researchers traced more than 1,800 text datasets that are used to train models. On the big websites that host such datasets, how often do you think the license was missing?",
        options={
            "rare": "About 1 in 20",
            "third": "About 1 in 3",
            "right": "More than 7 in 10",
        },
        reveal="**More than 7 in 10.** And many of the licenses those websites did show were wrong (Longpre and colleagues, 2024).",
    )
    guess_missing
    return (guess_missing,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Two questions decide whether you may train on a dataset, and they are about different people.

        **The license** comes from whoever holds the rights to the data, such as the maker of the dataset or of the photos in it. It says what you may copy, change and share, as in [Where does this data come from?](/course/open/data-provenance?lang=en). But a license only gives rights that its giver has. In 2019 IBM published *Diversity in Faces*, a list of about a million photos from Flickr that their photographers had shared under a Creative Commons license. Copyright was not the problem there: the people in the photos had never been asked whether they wanted to help train face recognition. Creative Commons itself said that copyright is not a good tool to protect privacy.

        **Consent** comes from the people the data are about, or who made it: they agree, freely and knowingly, to a stated use. Public is not the same as agreed: a post on an open forum was written for its readers, not to train a model. Under the GDPR, information about a person stays personal data even when anyone can see it. Whether the law allows training without a license differs between countries, and courts are still deciding; this lesson is about what you can account for, not only about what is allowed.

        **Ethics review.** Research with people starts with a check by an ethics committee: in the US an institutional review board, in the Netherlands a medical ethics committee for medical research, and often a faculty committee for other research. User studies and some labeling work need it too. In January 2012, Facebook changed for one week what 689,003 users saw in their news feed, to see whether emotions spread. The users were not asked beyond agreeing to Facebook's terms of use, and the journal that published the study in 2014 added an *editorial expression of concern*.

        **Data you cannot take back.** In 2020, MIT took its dataset *80 Million Tiny Images* offline after Vinay Prabhu and Abeba Birhane found offensive labels and images in it, all gathered from the web without asking anyone. Copies, and models trained on it, did not go away. The course keeps a list of every dataset it uses, in `data/README.md` and `course/route/bronnen.yaml`: the source, the license, and what the course may do with it. CIFAR-10, a small labeled part of Tiny Images, has no license, so the course only lets you download it; the photos of Caltech-101 come from the web, so the course shares no model trained on them.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **ethics review** | a check by an ethics committee, before research with people starts, that the risks are acceptable and that the people who take part give informed consent |
        """),
        L.watch_out("Consent is for a purpose. People who agreed to one study did not agree to every later use of their data: when the use changes, ask again."),
        mo.Html(C.picture({"title": "Two questions, four cases", "columns": ["has a license", "has no license"],
                           "rows": ["the people agreed", "the people were not asked"],
                           "cells": ["Mozilla's Common Voice: people read sentences aloud to donate their voice, and the recordings come out under CC0.",
                                     "Messages that volunteers gave to one research team, which never gave the set a license.",
                                     "IBM's Diversity in Faces: photos shared under a Creative Commons license, gathered for face recognition without asking the people in them.",
                                     "Posts copied from a forum whose rules say nothing about it."],
                           "both": "both questions pass",
                           "alt": "A grid of four cases. With a license and the people's consent, both questions pass: Mozilla's Common Voice. Without a license but with consent: messages given to one research team that gave them no license. With a license but without asking the people: IBM's Diversity in Faces. Without either: posts copied from a forum whose rules say nothing about it."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To decide whether you may train on a dataset:

    1. Find out who made it, and under which license.
    2. Check that the license allows what you plan: train, share a model, earn money.
    3. Find out who the data are about or who made them, and whether they agreed to this use.
    4. If your work involves people directly, ask whether it needs ethics review.
    5. Write the answers in the datasheet, and leave the data out when a question has no good answer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A team, made up by the course, wants to train a model on text that people wrote. Its rule: use a set only when it has a license, and only when its writers agreed. `licenses` holds the license of each set, `"none"` when it has none. Run the cell: it checks the first question.
    """)
    return


@app.cell
def example_first_question():
    _licenses = {"forum": "none", "survey": "CC BY", "reviews": "CC BY-SA"}

    for _name in _licenses:
        if _licenses[_name] != "none":
            print(_name, "has a license")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `survey` and `reviews` pass the first question. But the people who wrote the reviews were never asked: Exercise 1 adds the second question.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_usable():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Both questions

        The team's five sets. `licenses` holds the license of each, `agreed` whether its writers agreed to training. The loop goes through the sets with `name`, and the first question, the license, is already in it. Replace the `...` so that a set goes into `usable` only when its writers agreed too. The last line is `usable`, so the check sees it. The check also tries other answers.
        """),
        L.task("Put in `usable` the names of the sets that have a license and whose writers agreed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_usable():
    box_usable = L.code_box("usable", lines=12)
    box_usable
    return (box_usable,)


@app.cell(hide_code=True)
def check_usable(box_usable):
    result_usable = L.check("usable", box_usable.value)
    result_usable
    return (result_usable,)


@app.cell(hide_code=True)
def hints_usable():
    L.hints(
        "usable",
        [
            "Inside the first `if`, a second one: only a set that passes both gets to the `append`.",
            "A value that is already `True` or `False` can stand after `if` on its own. Which dictionary holds such values? And what goes into the list: a name or a license?",
            "`if ___[name]:` and under it `usable.append(___)`",
        ],
        why="`if agreed[name]:` and under it `usable.append(name)` gives `['survey', 'interviews']`: `reviews` has a license but no consent, `chats` consent but no license.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_questions():
    mo.md(r"""
    ### Exercise 2. Which question fails?

    Six cases. Put each one with the question it fails, or with *both questions pass*.
    """)
    return


@app.cell(hide_code=True)
def sort_questions_cell():
    sort_questions = L.sort("questions")
    sort_questions
    return (sort_questions,)


@app.cell(hide_code=True)
def check_questions(sort_questions):
    result_questions = L.check("questions", sort_questions)
    result_questions
    return (result_questions,)


@app.cell(hide_code=True)
def hints_questions():
    L.hints(
        "questions",
        [
            "Ask two things for each case: is there a license, and did the people agree to this use?",
            "In two cases the people agreed, but whoever holds the rights gave no license. In two others a company holds the rights and gives a license, but the people whose words they are were never asked.",
            "Both pass: the translations and \\_\\_\\_. License missing: \\_\\_\\_ and the game. Consent missing: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Both pass:** the translations and the recipes. **License missing:** the school photos and the game. **Consent missing:** the support chats and the podcasts.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_review():
    mo.md(r"""
    ### Exercise 3. A test with fellow students
    """)
    return


@app.cell(hide_code=True)
def choice_review_cell():
    choice_review = L.choice(
        "review",
        question="A student wants to test a chatbot with 30 fellow students and keep everything they type. What comes first?",
        options={
            "publish": "Publishing all the chats, so others can check the results",
            "right": "Their consent, and asking if ethics review is needed",
            "nothing": "Nothing: tests with fellow students do not count as research",
        },
        correct="right",
        why={
            "publish": "Publishing makes it worse: the fellow students did not agree to that either. See *The idea*.",
            "right": "The fellow students must know what is kept and agree to it, and many universities ask for ethics review even of student projects.",
            "nothing": "A test with people is research with people, in a student project too: they type things about themselves that are kept. See *The idea*.",
        },
    )
    choice_review
    return (choice_review,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Where does this data come from?](/course/open/data-provenance?lang=en) you wrote down where a dataset comes from; in [Codes of conduct and values in research](/course/open/rai-professional-ethics?lang=en) you saw what a professional owes the public. Here the two meet: a license says what the maker allows, consent what the people in the data agreed to, and ethics review checks research with people before it starts. In the next lesson, *COMPAS and a hiring tool*, you look at two real systems that went wrong.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_review, result_questions, result_usable):
    L.wrap_up(result_usable, result_questions, choice_review)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
