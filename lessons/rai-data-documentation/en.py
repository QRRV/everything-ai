# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-data-documentation. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rai-data-documentation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where does this data come from?](/course/open/data-provenance?lang=en) and [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="A dataset on a website has no license. What do you assume?",
        options={
            "anything": "Anything goes",
            "right": "You may not copy or share it",
            "name": "You may use it if you name its maker",
        },
        correct="right",
        why={
            "anything": "Free to download is not free to use. Without a license, assume you may not copy or share the data, and ask the maker. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "Without a license, assume you may not copy or share the data: ask the maker before you share it or build on it.",
            "name": "Naming the maker is a condition of CC BY, a license this dataset does not have. Without a license, assume you may not copy or share it. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def quick_check_sample():
    quick_sample = L.choice(
        "quick-sample",
        question="A speech recognizer is trained on recordings with hardly any older speakers, and it works worse for them. Which bias is this?",
        options={
            "right": "Representation bias",
            "historical": "Historical bias from an unequal world",
            "measurement": "Measurement bias in the labels",
        },
        correct="right",
        why={
            "right": "Too few examples of part of the people the model is used for: representation bias, at the step of the sample.",
            "historical": "Historical bias is in the world itself, even with a perfect sample. Here the sample holds too few older speakers: representation bias. See [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).",
            "measurement": "Measurement bias comes from features or labels that measure something else for some groups. Here older speakers are rare in the data: representation bias. See [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).",
        },
    )
    quick_sample
    return (quick_sample,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    UCI Adult holds 48,842 people from a US survey of 1994. Frances Ding and colleagues (2021) found it in "more than three hundred research papers related to fairness", and named among its problems "its age, limited documentation, and outdated feature encodings". What its page on the UCI repository says about how it was made fits in two sentences: "Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))". Read slowly, the conditions say who is not in it. Ding and colleagues filtered their replacement in much the same way: people "above the age of 16, who reported usual working hours of at least 1 hour per week in the past year, and an income of at least $100". And despite the word Census, the data come from a survey, the Current Population Survey. A model trained on Adult has seen nobody outside these limits, and only documentation tells the next user so. Emily Bender and Batya Friedman (2018) wrote about the problems that come from "the use of data from certain populations in the development of technology for other populations".

    **An everyday example.** The leaflet in a box of medicine says what is in it, for whom it is meant, and when you should not take it. Nobody can see that by looking at a pill. A dataset needs the same: who is in it, how it was made, and when not to use it.
    """)
    return


@app.cell(hide_code=True)
def predict_women():
    guess_women = L.guess(
        "women",
        question="UCI Adult comes from a US survey of 1994. What share of its 48,842 people are women?",
        options={
            "half": "About half",
            "right": "About a third",
            "quarter": "About a quarter",
        },
        reveal="**About a third: 16,192 of the 48,842, or 0.33.** Who is in a dataset is not the same as who lives in a country. A data card says in numbers who is in the data, so that nobody has to guess, or assume half.",
    )
    guess_women
    return (guess_women,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Three questions.** In [Where does this data come from?](/course/open/data-provenance?lang=en) you wrote a datasheet with five fields. The datasheets of Timnit Gebru and colleagues (2021) group their questions in seven sections that roughly follow a dataset's life: "motivation, composition, collection process, preprocessing/cleaning/labeling, uses, distribution, and maintenance." The questions are "not intended to be prescriptive": those that do not apply are skipped. Other formats ask much the same in their own shape. For fairness three questions matter most: who is in the data, how it was collected and labeled, and what it is not fit for.

        **Data statements.** Emily Bender and Batya Friedman (2018) wrote data statements for language data: why these texts were chosen, the language variety with a tag such as `en-US`, the age, gender and native language of the speakers and of the annotators, the situation the speech comes from, and more. A long form belongs in the paper that presents a dataset; every paper that uses it should give a short form of 60 to 100 words that points to the long one. Gaps are allowed: "full specification of all of this information may not be feasible in all cases", and ranges may be better than exact values. Bender and Friedman write data statements for language data; for a table like Adult, a datasheet or a data card fits better.

        **Data Cards.** Mahima Pushkarna, Andrew Zaldivar and Oddur Kjartansson (2022) worked with 12 teams in a large technology company to create 22 Data Cards: "structured summaries of essential facts about various aspects of ML datasets needed by stakeholders across a dataset's lifecycle". A card asks at three levels. Telescopes give an overview, such as "Does this dataset contain Sensitive Human Attributes?"; periscopes add technical detail; microscopes ask about the "unobservable" human processes, decisions, assumptions and policies that shape the dataset, and those were the hardest to answer. Participants in their study said "known unknowns" were as important as what is known. Among what went wrong: people copied a finished card, so its errors spread to the copies, and they wrote "N/A" when they were unsure.

        **Nutrition labels.** Sarah Holland and colleagues (2018) made the Dataset Nutrition Label, after the label on food: sample modules such as Metadata, the only one required, Provenance, Variables and Statistics. In their survey, 47 percent of the respondents said they did some analysis of the data before building a model, but 74 percent said their organization had no explicit best practices for it. A second generation, presented in 2020, first asks what you want to use the data for, and then shows the alerts that matter for that use.

        **On the Hugging Face Hub**, a dataset card is the `README.md` of a dataset, which the Hub shows on its page. Its template has a section *Uses* with *Direct Use* and *Out-of-Scope Use*, for "misuse, malicious use, and uses that the dataset will not work well for", and a section *Bias, Risks, and Limitations*, "meant to convey both technical and sociotechnical limitations". A field nobody filled in says "[More Information Needed]".

        | Question | Datasheet | Data Card | Hugging Face card |
        |---|---|---|---|
        | Who is in it? | *Composition* | *Breakdown of subgroups* | *Who are the source data producers?* |
        | How was it made? | *Collection process*; *labeling* | *Collection methods* | *Data Collection and Processing* |
        | What is it not fit for? | *Uses* | *Unacceptable uses* | *Out-of-Scope Use* |

        **FAIR is something else.** In 2014 a workshop in Leiden, "Jointly Designing a Data Fairport", led to the FAIR principles of Mark Wilkinson and colleagues (2016): data should be Findable, Accessible, Interoperable and Reusable, "both for machines and for people". There are fifteen, such as F1, a "globally unique and persistent identifier" (Adult has the DOI 10.24432/C5XW20), and R1.1, "a clear and accessible data usage license". They are "not, themselves, a standard or a specification", and data need not be open to be FAIR: for sensitive data, rich metadata with clear rules for access gives "a high degree of 'FAIRness'". FAIR says nothing about bias. A dataset can meet all fifteen and still leave out a group.

        **The law.** Under the EU AI Act, the data to train, validate and test a high-risk AI system on need data governance that includes an "examination in view of possible biases" that are likely to affect health and safety, harm fundamental rights or lead to discrimination prohibited under EU law (Article 10), and they must be "relevant, sufficiently representative, and to the best extent possible, free of errors and complete in view of the intended purpose". The technical documentation asks, where relevant, for "datasheets" that describe the training data (Annex IV). After a change in 2026, the Digital Omnibus on AI, these rules apply from 2 December 2027 to the high-risk systems listed in Annex III, and from 2 August 2028 to the others. Since 2 August 2025, providers of general-purpose AI models must publish "a sufficiently detailed summary about the content used for training", on a template the European Commission published in 2025; for models already on the market before that date, by 2 August 2027.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **data statement** | a description of a language dataset: why these texts, which language variety, who spoke or wrote them and who annotated them |
        | **data card** | a structured summary of the essential facts about a dataset, for everyone who works with it, including what is not known |
        | **Dataset Nutrition Label** | a label for a dataset, after the label on food, with modules such as metadata, provenance and statistics |
        | **dataset card** | on the Hugging Face Hub, the `README.md` of a dataset, shown on its page |
        | **out-of-scope use** | a use a dataset is not meant for, or will not work well for |
        | **known unknown** | something you know you do not know about a dataset; you write it down instead of leaving it out |
        | **FAIR principles** | findable, accessible, interoperable and reusable: how to share data so that people and machines can find and reuse it; not about fairness |
        """),
        L.watch_out("FAIR is not fair: data can be findable, accessible, interoperable and reusable and still leave out a group. And where you do not know something, write that it is unknown: an empty field or N/A hides what the next user needs to know."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To write a data card for a dataset you use, also one you did not make:

    1. Why it was made, by whom and from what. Where nobody wrote it down, write that it is unknown to you.
    2. Who is in it: count each group, and its share of all rows.
    3. What is unknown: per column how often a value is missing, also per group.
    4. What it is not fit for: the people and situations outside its range, and what follows from how it was made.
    5. Where to find it and on which terms: a persistent identifier, such as a DOI, and the license.
    """)
    return


@app.cell(hide_code=True)
def picture_card():
    mo.vstack([
        mo.md("**A data card for UCI Adult, as this lesson fills it**"),
        draw.dict_boxes([("made by", "Barry Becker, Ronny Kohavi"), ("from", "Current Population Survey, 1994"),
                         ("who is in it", "exercise 1"), ("unknown", "exercise 3"),
                         ("not fit for", "the worked example"), ("consent", "unknown to us"),
                         ("license", "CC BY 4.0"), ("identifier", "DOI 10.24432/C5XW20")],
                        key_label="field", value_label="answer"),
        mo.md(r"""
        Each field answers one question of the recipe. Three of them come from the data itself, with a few lines of code: the worked example and exercises 1 and 3. Consent is a known unknown: nothing in Adult's documentation says what the people in the survey agreed to.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4 for Adult: the lowest and highest value of two columns, `age` and `hours_per_week`. Outside that range the table has nobody to learn from. Run the cell.
    """)
    return


@app.cell
def example_ranges():
    import pandas as pd

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")

    # 4. what it is not fit for: the range of two columns
    for _column in ["age", "hours_per_week"]:
        print(_column, _people[_column].min(), "to", _people[_column].max())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Nobody in Adult is younger than 17, and nobody worked 0 hours a week: the conditions AAGE>16 and HRSWK>0 at work. So under out-of-scope use, the card says: not for people younger than 17, or for people who do not work. Who is inside the range is the next question: exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who is in it?

        `people` is UCI Adult, and `column` the column with the groups, `"race"`. Work out `shares`: for each group, the number of its people divided by the number of all rows. `people[column].value_counts()` counts the people per group, as in [Asking questions of data](/course/open/data-explore?lang=en), and `len(people)` gives the number of rows. The check also tries `column = "sex"`.
        """),
        L.task("Work out `shares`: per group of `column`, its share of all the people in the table."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=12)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Two numbers per group: how many people it has, and how many people there are in all.",
            "`people[column].value_counts()` gives the number per group; `len(people)` the number of all rows.",
            "`shares = people[column].value_counts() / ___`",
        ],
        why="`counts = people[column].value_counts()`, then `shares = counts / len(people)`. For race that gives `0.855` for White, `0.096` for Black, `0.031` for Asian-Pac-Islander, `0.010` for Amer-Indian-Eskimo and `0.008` for Other.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Exercise 2. What does FAIR tell you?
    """)
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair",
        question="A dataset meets all fifteen FAIR principles. What does that tell you?",
        options={
            "group": "No group is missing",
            "right": "It can be found and reused",
            "open": "Anyone may download the data, for free",
        },
        correct="right",
        why={
            "group": "FAIR is about finding, accessing, exchanging and reusing data, not about who is in it. A dataset can meet all fifteen principles and still leave out a group. See *FAIR is something else* in *The idea*.",
            "right": "Findable, accessible, interoperable and reusable, by people and by machines: FAIR says how data can be found and reused, and nothing about bias.",
            "open": "FAIR data need not be open: one principle allows \"an authentication and authorization procedure, where necessary\", and sensitive data can be FAIR through rich metadata and clear rules for access. See *FAIR is something else* in *The idea*.",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_unknown():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. What is unknown?

        For some people in Adult the `occupation` is unknown. Work out `unknown`: for each group of `column`, the share of its people whose occupation is unknown. Keep the rows where it is unknown, `people[people["occupation"].isna()]`, count them per group, and divide by the number of people in each group. Dividing one count per group by another divides group by group, by name. The check also tries `column = "race"`.
        """),
        L.task("Work out `unknown`: per group of `column`, the share of its people whose occupation is unknown."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unknown():
    box_unknown = L.code_box("unknown", lines=12)
    box_unknown
    return (box_unknown,)


@app.cell(hide_code=True)
def check_unknown(box_unknown):
    result_unknown = L.check("unknown", box_unknown.value)
    result_unknown
    return (result_unknown,)


@app.cell(hide_code=True)
def hints_unknown():
    L.hints(
        "unknown",
        [
            "First the people whose occupation is unknown; then, per group, their number divided by the number of people in the group.",
            "`missing = people[people[\"occupation\"].isna()]` keeps those rows; `missing[column].value_counts()` counts them per group.",
            "`unknown = missing[column].value_counts() / ___`",
        ],
        why="`missing = people[people[\"occupation\"].isna()]`, then `unknown = missing[column].value_counts() / people[column].value_counts()`. pandas divides group by group, by name. That gives `0.079` for women and `0.047` for men.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_write():
    mo.md(r"""
    ### Exercise 4. What do you write?
    """)
    return


@app.cell(hide_code=True)
def choice_write_cell():
    choice_write = L.choice(
        "write",
        question="You write a datasheet for UCI Adult. Nowhere is it written whether the people in the survey agreed to this use. What do you write under consent?",
        options={
            "na": "N/A: not applicable",
            "right": "That it is unknown to you",
            "empty": "Nothing; leave the field empty",
        },
        correct="right",
        why={
            "na": "N/A says the question does not apply, and consent does apply to data about people. Teams in the Data Cards study wrote \"N/A\" when they were unsure, which hid what they did not know. See *Data Cards* in *The idea*.",
            "right": "Write what you do not know. In their own example, Gebru and colleagues answered such questions with \"Unknown to the authors of the datasheet\", because they had not made the dataset themselves.",
            "empty": "An empty field looks as if nobody asked. Write that it is unknown to you: the next user then knows the question was asked and has no answer. See step 1 of the *Recipe*.",
        },
    )
    choice_write
    return (choice_write,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Where does this data come from?](/course/open/data-provenance?lang=en) you wrote a datasheet with five fields of text. A data card adds numbers: who is in the data, as the share of each group, and what is unknown, per group. [Bias at every step](/course/open/rai-bias-lifecycle?lang=en) named representation bias at the step of the sample; a card is where a team writes it down, so that the next team does not have to find it again. Exercise 3 showed one more way it starts: a team that drops the rows with an unknown value drops a larger share of the women than of the men.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, choice_write, result_shares, result_unknown):
    L.wrap_up(result_shares, choice_fair, result_unknown, choice_write)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
