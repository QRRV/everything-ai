# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-provenance. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("data-provenance", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Asking questions of data](/course/open/data-explore?lang=en) and [Dictionaries](/course/open/py-dicts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_missing():
    quick_missing = L.choice(
        "quick-missing",
        question="`df.isna().sum()` gives `3` for the column `temp`. What does that mean?",
        options={
            "missing": "3 temperatures are missing",
            "zero": "3 temperatures are 0",
            "low": "3 temperatures are below the mean",
        },
        correct="missing",
        why={
            "zero": "`df.isna()` is not about the value 0. What does it mark with `True`? See the table of new words in [Asking questions of data](/course/open/data-explore?lang=en).",
            "low": "`df.isna()` does not compare with the mean. What does it mark with `True`? See the table of new words in [Asking questions of data](/course/open/data-explore?lang=en).",
            "missing": "`df.isna()` is `True` for each missing value, and `.sum()` counts them per column: 3 in `temp`.",
        },
    )
    quick_missing
    return (quick_missing,)


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="`card = {\"license\": \"CC0\"}`. What does `\"source\" in card` give?",
        options={
            "true": "`True`",
            "false": "`False`",
            "error": "An error",
        },
        correct="false",
        why={
            "true": "`in` looks at the keys of `card`. Is `\"source\"` one of them? See the table of new words in [Dictionaries](/course/open/py-dicts?lang=en).",
            "error": "`in` does not stop with an error when a key is not there. What does it give then? See the table of new words in [Dictionaries](/course/open/py-dicts?lang=en).",
            "false": "`card` has one key, `\"license\"`; `\"source\"` is not there, so `in` gives `False`.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A dataset can look perfect and still be wrong to use. Perhaps you may not use it at all, perhaps the people in it never agreed to it, or perhaps it covers only one place in one month. A model trained on it takes all of that along. So before you use data, you find out where it comes from, and you write it down, so the next person does not have to find out again.

    **An everyday example.** The label on a packet of food: where it comes from, what is in it, and until when it is good to eat. A dataset needs a label like that too.
    """)
    return


@app.cell(hide_code=True)
def predict_faces():
    guess_faces = L.guess(
        "faces",
        question="A website offers a dataset of 10,000 photos of faces, free to download. Do you think you may train a model on it and share that model?",
        options={
            "free": "Yes: it is free to download",
            "depends": "Only if its license allows it, and the people in the photos agreed",
            "no-sale": "Yes, as long as you do not sell the model",
        },
        reveal="**Only if its license allows it, and the people in the photos agreed.** Free to download is not the same as free to use: the license says what you may do with the data. And photos of faces are about people, so it also matters whether they agreed to this use.",
    )
    guess_faces
    return (guess_faces,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Every dataset has a story: who made it, how and when, and what you may do with it. That story is its provenance. Four questions cover most of it. Where does it come from? Under which license? Did the people in it agree? And what does it not cover? You answer them before you use the data, and you write the answers down next to it, in a datasheet.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **data provenance** | where a dataset comes from: who made it, from which sources, when, and under which license and consent | |
    | **license** | the terms under which you may use, change and share a dataset, a model or code | `"CC BY 4.0"` |
    | **CC0** | the maker gives up their rights: you may do anything with it, without naming anyone; it says nothing about the people in the data | `"CC0 1.0"` |
    | **CC BY** | you may use, change and share it, even to earn money, if you name the maker, link to the license and say what you changed; with SA, whatever you make from it and share gets the same license; with NC, not to earn money | `"CC BY-SA 4.0"` |
    | **no license** | if a dataset has no license, assume you may not copy or share it: ask the maker before you share it or build on it | |
    | **consent** | permission that people give freely and knowingly for their data to be used for a stated purpose | |
    | **limits** | what the data does not cover: other places, other times, people or cases that are missing | |
    | **datasheet** | a document that comes with a dataset and answers these questions; here, a dictionary | `card["license"]` |
    """)
    return


@app.cell(hide_code=True)
def picture_card():
    mo.vstack([
        mo.md("**The datasheet of Palmer penguins**"),
        draw.dict_boxes([("source", "Palmer Station, Antarctica"), ("collected", "2007 to 2009"), ("license", "CC0 1.0"),
                         ("consent", "no people in it"), ("limits", "only 3 islands, 2007–2009")],
                        key_label="field", value_label="answer"),
        mo.md(r"""
        A real dataset, short: a few body measurements of 344 penguins of three species, on three islands near Palmer Station in Antarctica. Each field answers one question. The license, CC0, lets you do anything with it; there are no people in it to ask; and its limits say what you cannot learn from it, such as anything about penguins elsewhere.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Free to download is not free to use. Without a license, assume you may not copy or share the data, and ask the maker. And with data about people, check what they agreed to: agreeing to one use is not agreeing to every use."),
        mo.md(r"""
        ### Recipe

        Before you use a dataset:

        1. Find the source: who made it, from what, and when. Look for a README or a datasheet next to the data.
        2. Find the license, and read what it allows: using, changing, sharing, earning money with it.
        3. If the data is about people: check that they agreed to this use; if not, find out whether you may use it at all.
        4. Write down the limits: which places, which times, who or what is in it, and who or what is missing.
        5. Keep the answers next to the data, as a datasheet, so that whoever uses it next can read them.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The datasheet of Palmer penguins as a dictionary, with a field for each question. The comments with `#` say which step of the recipe each part is. The course keeps such notes itself: `data/README.md` says where each dataset it hosts comes from, such as the plays of Shakespeare that a later lesson uses. Run the cell.
    """)
    return


@app.cell
def example_card():
    _card = {
        # 1. the source, and when
        "source": "Kristen Gorman and the Palmer Station Long Term Ecological Research program, Antarctica",
        "collected": "2007 to 2009",
        # 2. the license
        "license": "CC0 1.0",
        # 3. people who had to agree
        "consent": "not needed: there are no people in it",
        # 4. the limits
        "limits": "only 344 penguins, on 3 islands near one station, 2007 to 2009; some values are missing",
    }
    # 5. read it back, field by field
    for _field in ["source", "collected", "license", "consent", "limits"]:
        print(f"{_field}: {_card[_field]}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each line answers one question. The license, CC0 1.0, is the most open there is. Consent is not needed, as there are no people in the data. The limits warn you: a model trained on these penguins knows nothing of penguins on other islands or from other years, and the missing values need a decision before you train on them, as in [Asking questions of data](/course/open/data-explore?lang=en). For the photos of faces from the guess, the field `consent` would have to say what the people agreed to; if nobody knows, that is a reason not to use them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_license():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What does CC BY allow?
    """)
    return


@app.cell(hide_code=True)
def choice_license_cell():
    choice_license = L.choice(
        "cc-by",
        question="A dataset of bird songs has the license CC BY 4.0. You want to train a model on it, change the recordings a little, and share them on your own website. May you?",
        options={
            "never": "No: you may only use it for yourself",
            "silent": "Yes, and you do not have to name anyone",
            "credit": "Yes, if you name the maker, link to the license and say what you changed",
        },
        correct="credit",
        why={
            "never": "Look up CC BY in the table of new words: what may you do with it?",
            "silent": "That is CC0. What do the letters BY ask for? See the table of new words.",
            "credit": "CC BY allows using, changing and sharing, even to earn money, as long as you name the maker, link to the license and say what you changed. BY stands for the naming: made by whom.",
        },
    )
    choice_license
    return (choice_license,)


@app.cell(hide_code=True)
def text_exercise_card():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The datasheet of the weather station

        In [Asking questions of data](/course/open/data-explore?lang=en) you explored the month of a weather station. Its datasheet has two fields already. Add the other three: `license`, `consent` and `limits`, each with a text. The facts: the course made the data up and shares the file under CC BY-SA 4.0, like the text of its lessons; there are no people in the data; it is one made-up station for one month, with two temperatures missing and -999 for a broken wind meter. Replace the `...`. The last line is `card`, so the check sees it.
        """),
        L.task("Complete `card`: add `license`, `consent` and `limits`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_card():
    box_card = L.code_box("station-card", lines=9)
    box_card
    return (box_card,)


@app.cell(hide_code=True)
def check_card(box_card):
    result_card = L.check("station-card", box_card.value)
    result_card
    return (result_card,)


@app.cell(hide_code=True)
def hints_card():
    L.hints(
        "station-card",
        [
            "Steps 2 to 4 of the recipe: the license, the consent and the limits, each as one pair of the dictionary.",
            "Each pair is a key between quotes, a colon, and a text between quotes, with a comma after it, like the two pairs that are there.",
            "`\"license\": \"___\",` on a line of its own, then the same for `consent` and `limits`.",
        ],
        why="The license of the station data is CC BY-SA 4.0, the same as the text of the course. Consent is not needed, as there are no people in it. The limits say what a model could not learn from it: one made-up station, one month, two temperatures missing, and -999 where the wind meter broke.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Which fields are missing?

        Finish `missing_fields(card)`. It gets a datasheet as a dictionary, and gives back a list of the fields from `FIELDS` that the datasheet lacks: a field counts as missing when it is not there, or when its text is empty, `""`. Keep the order of `FIELDS`. Replace the `...`. The check calls `missing_fields({"source": "a weather station", "license": "CC0 1.0"})`: that should give `["collected", "consent", "limits"]`. It also tries other datasheets.
        """),
        L.task("Make `missing_fields(card)` give back the fields from `FIELDS` that `card` lacks."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing-fields", lines=12)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing-fields", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing-fields",
        [
            "Go through `FIELDS` one field at a time, and keep the ones that are missing in a list.",
            "A field is missing when it is not one of the keys of `card`, or when its text is empty. Check first with `in` whether it is there: only then may you look at its text.",
            "`if field in card:`, under it `if card[field] == ___:` and `missing.append(field)`; under `else:` also `missing.append(field)`. After the loop, `return missing`.",
        ],
        why="The loop goes through `FIELDS`, so the list keeps their order. `if field in card:` looks only at the text of a field that is there, so a missing key gives no error; `else:` catches the fields that are not there at all.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Asking questions of data](/course/open/data-explore?lang=en) you asked a table what is in it; now you also ask where it comes from, which no table can tell you. In [Dictionaries](/course/open/py-dicts?lang=en) you looked up values by key; a datasheet is such a dictionary. Later in the course, you check the source and the license of every dataset you train on, and a trained model gets a card of its own, a model card. In the next lesson, *Examples, features and labels*, you learn the words for what is in a dataset that a model learns from.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_license, result_card, result_missing):
    L.wrap_up(choice_license, result_card, result_missing)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
