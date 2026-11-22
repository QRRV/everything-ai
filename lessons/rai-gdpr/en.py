# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-gdpr. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-gdpr", "en")
    C = L.checks   # the six lawful bases, and a register, uses and forum posts of a made-up language-learning app


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What is privacy?](/course/open/rai-what-is-privacy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="In Solove's taxonomy of privacy problems, which group does aggregation belong to?",
        options={
            "collection": "Collection",
            "right": "Processing",
            "dissemination": "Dissemination",
        },
        correct="right",
        why={
            "right": "Aggregation combines pieces of information about one person that were already collected: it is one of the five kinds of processing.",
            "collection": "Collection holds surveillance and interrogation. Aggregation combines what was already collected: processing. See *Solove's taxonomy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
            "dissemination": "Dissemination spreads information further. Aggregation combines pieces of information about one person: processing. See *Solove's taxonomy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def quick_check_flow():
    quick_flow = L.choice(
        "quick-flow",
        question="Which view of privacy asks whether a flow of information fits the context it came from?",
        options={
            "alone": "Being let alone",
            "control": "Control over information",
            "right": "Appropriate flow",
        },
        correct="right",
        why={
            "right": "Helen Nissenbaum's view: what counts is whether a flow fits its context, not whether it takes place at all.",
            "alone": "Being let alone, or keeping secrets, sees a problem when something hidden becomes known. See *Three views of privacy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
            "control": "Control asks whether the person decided. See *Three views of privacy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
        },
    )
    quick_flow
    return (quick_flow,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Models are often trained on information about people: what they wrote, said, bought or searched for. In the European Union, the GDPR sets the rules for that. It does not forbid training on personal data, and it does not allow everything either: it asks for a reason the law accepts, a clear purpose, and room for the people in the data to object or to have their data erased. This lesson gives you the GDPR's main words, its six lawful bases, its rule about purposes, two of the rights it gives people, and what regulators have said about AI so far. It explains the law; it is not legal advice.

    **An everyday example.** A shop asks for your email address to send you a receipt. A month later, it sends you its newsletter. You gave the address for one thing, and it is used for another. The GDPR has a rule about that, called purpose limitation.
    """)
    return


@app.cell(hide_code=True)
def predict_bases():
    guess_bases = L.guess(
        "bases",
        question="Under the GDPR, processing personal data needs a lawful basis: a reason the law accepts. The consent of the person is one. How many are there in all?",
        options={
            "two": "Two",
            "four": "Four",
            "right": "Six",
        },
        reveal="**Six**, in Article 6(1): consent, a contract, a legal obligation, vital interests, a task in the public interest, and legitimate interests. The European Data Protection Board (EDPB) writes that \"there is no hierarchy between the legal bases\": consent is one of six, not the default.",
    )
    guess_bases
    return (guess_bases,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The General Data Protection Regulation, the GDPR, is a law of the European Union: Regulation (EU) 2016/679. It applies from 25 May 2018, and it is "binding in its entirety and directly applicable in all Member States": it applies in every country of the EU as it stands, without each country first making a law of its own. In Dutch it is the *algemene verordening gegevensbescherming*, the AVG.

        ### Personal data, and who is responsible

        The GDPR is about **personal data**: "any information relating to an identified or identifiable natural person". A person is identifiable when they can be identified directly or indirectly, for instance by a name, an identification number, location data or an online identifier. The person the data are about is the **data subject**. Processing is almost anything you can do with personal data: the GDPR names, among others, collecting, storing, using and erasing them. Collecting texts for a dataset is processing, and so is training a model on them, when they hold personal data.

        The **data controller** is whoever "determines the purposes and means of the processing of personal data": who decides why and how. That can be a company, a public body or a single person, alone or together with others. The controller is responsible for following the GDPR. An organization that processes the data on the controller's behalf, such as a company that trains a model on a client's instructions, is a processor.

        ### Six lawful bases

        Processing is "lawful only if and to the extent that at least one of the following applies", says Article 6(1), and then come six conditions. The course calls them the six **lawful bases**; the Regulation itself speaks of the "lawfulness of processing" and of a "legal basis".
        """),
        mo.Html(C.bases_picture("en")),
        mo.md(r"""
        The last basis has a limit in the law itself: it does not hold where the interests are "overridden by the interests or fundamental rights and freedoms of the data subject which require protection of personal data, in particular where the data subject is a child." The EDPB, the board of the EU's data protection authorities, describes the check as a test of three steps: is there a legitimate interest, is the processing necessary for it, and does it hold up against the interests and rights of the people in the data? The interest must be "real and present (i.e. not speculative)"; one of the EDPB's examples of an interest that may be legitimate is "developing the service of a conversational agent to assist users". Which basis fits is for the controller to work out; the EDPB writes that "it is for controllers to identify the appropriate legal basis".

        Two misunderstandings are common. You do not always need consent: it is one basis of six, and "there is no hierarchy between the legal bases". And public data are not free to use: public is not a basis. Whether data were publicly available is one of the things that shape what people can reasonably expect, which counts in the third step of the test, and so does "whether data subjects are actually aware that their personal data is online at all". A line in a privacy policy does not settle it either: it "does not necessarily mean that the data subjects can reasonably expect it to happen".

        Some personal data get extra protection: data about health, religious beliefs, political opinions, sexual orientation and a few more special categories. Their processing "shall be prohibited", says Article 9(1), unless one of the exceptions in Article 9(2) applies.

        ### Purpose limitation

        Personal data must be "collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes" (Article 5(1)(b)). That is **purpose limitation**. A new purpose is not forbidden as such, but it must not be incompatible with the first one. To judge that, Article 6(4) lists what to take into account, among others "the context in which the personal data have been collected" and "the existence of appropriate safeguards, which may include encryption or pseudonymisation." Training a model on chats that were collected to answer questions is such a further purpose: what Solove calls secondary use.

        ### Erasure and objection

        The GDPR gives data subjects rights over data about them. Two of them matter a lot for AI. The **right to erasure** (Article 17, which adds "right to be forgotten" in brackets) lets a person have their personal data erased "without undue delay", but only on one of six grounds: for instance when they withdraw the consent the processing was based on and there is no other legal ground for it, or when the data "have been unlawfully processed". And it has exceptions, among them the freedom of expression and information, and archiving in the public interest, scientific or historical research and statistics, in so far as erasure is likely to make the aims of that processing impossible or seriously impair them. The right to object (Article 21) lets a person object, for reasons that have to do with their own situation, to processing based on a public task or on legitimate interests. The controller must then stop, "unless the controller demonstrates compelling legitimate grounds for the processing" that override the interests, rights and freedoms of the person, or needs the data for legal claims.

        ### The GDPR and AI

        On 17 December 2024, at the request of the Irish authority, the EDPB adopted Opinion 28/2024 on AI models. It is an opinion, not a court ruling: the authorities "should take utmost account" of it when they investigate. Some of what it says:

        - "AI models trained with personal data cannot, in all cases, be considered anonymous." A model counts as anonymous only when the chance of getting the personal data it was trained on out of it, by extracting them or by asking it questions, is insignificant.
        - "Information scraped from, for example, services such as news outlets, social media, forum discussions and personal websites, may contain personal data." The Dutch authority, the Autoriteit Persoonsgegevens, writes that scraping almost always collects personal data.
        - Examples of measures that limit the impact on people: "Proposing an unconditional 'opt-out' from the outset", letting people have their data erased also when none of the six grounds of Article 17(1) applies, and, for scraping, "respecting robots.txt or ai.txt files".
        - When a model was trained unlawfully, an authority may order the erasure of part of the dataset or, where that is not possible, of "the whole dataset used to develop the AI model and/or the AI model itself."

        National authorities act on their own too. On 30 March 2023, the Italian authority, the Garante, ordered OpenAI, the maker of ChatGPT, to temporarily limit the processing of the data of people in Italy; among its reasons was the lack of a suitable legal basis for collecting personal data and processing them to train the algorithms behind ChatGPT. On 28 April 2023, after changes, OpenAI made ChatGPT available in Italy again; one of the changes was that everyone in Europe, also people who do not use ChatGPT, could opt out of the use of their data for training. On 20 December 2024, the Garante announced a fine of EUR 15 million for several findings: among them that OpenAI processed users' personal data to train ChatGPT without first identifying an appropriate legal basis, did not notify the authority of a data breach of March 2023, and had no age verification. In 2026 the Court of Rome upheld the appeal against that decision, in a judgment published on 18 March 2026: the side that appealed won. The course has not read the judgment, so it cannot say on which grounds, or whether it is final. The GDPR allows fines of up to EUR 20 million or, for an undertaking, 4% of its total worldwide annual turnover of the preceding financial year, whichever is higher.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **GDPR** | the EU law that protects personal data: when organizations may process it, and the rights people have over data about them |
        | **personal data** | any information about a person who is identified or can be identified, directly or indirectly |
        | **data subject** | the person the personal data are about |
        | **data controller** | whoever decides why and how personal data are processed, and is responsible for it under the GDPR |
        | **lawful basis** | one of the six reasons in Article 6(1), such as consent or legitimate interests: processing is lawful only if at least one applies |
        | **purpose limitation** | the rule that personal data are collected for a clear purpose and not used later in a way that is incompatible with it |
        | **right to erasure** | the right of a person to have personal data about them erased, on one of six grounds and with exceptions (Article 17) |
        """),
        L.watch_out("The GDPR does not forbid training a model on personal data, and it does not allow it because the data are public. Training is processing: it needs one of the six lawful bases and a purpose, and for health data and the other special categories also an exception in Article 9(2)."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check what an AI system may do with data about people:

    1. Ask whether the data relate to a person who can be identified, directly or indirectly. Then they are personal data, which the GDPR is about. Look out for special categories, such as health.
    2. Find the controller: who decides why and how the data are processed.
    3. Give every activity one of the six lawful bases, and check that it holds; for legitimate interests, with the three-step test. That is `has_basis`.
    4. Compare what the data are used for with what they were collected for. A new purpose must not be incompatible with the first. That is `new_purposes`.
    5. When a person asks for erasure and one of the grounds applies, find all data about them, also where others write about them, check the exceptions, and remember the models already trained on them. That is `erase`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four fields of a dataset, made up by the course, each with what it says about a person. `_special` holds four of the special categories of Article 9(1). The cell prints, for each field, its name, what it is about, and whether that is a special category.
    """)
    return


@app.cell
def example_fields():
    # four made-up fields of a made-up dataset, each with what it says about a person
    _fields = [{"field": "email", "about": "contact"},
               {"field": "diagnosis", "about": "health"},
               {"field": "church", "about": "religious beliefs"},
               {"field": "streak", "about": "use of the app"}]
    # four of the special categories of Article 9(1)
    _special = ["health", "religious beliefs", "political opinions", "sexual orientation"]

    for _row in _fields:
        print(_row["field"], "|", _row["about"], "|", _row["about"] in _special)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two of the four fields are special categories: a diagnosis is about health, a church about religious beliefs. Processing them is prohibited unless one of the exceptions in Article 9(2) applies. Article 9(1) lists more, such as trade union membership and genetic data. All four fields are personal data, so processing any of them needs one of the six lawful bases; the two special categories also need an exception. `in` gives `True` or `False`; in Exercise 1 you store such an answer for each activity in a dictionary.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_basis():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. has_basis: does each activity have a lawful basis?

        A made-up language-learning app keeps a register of what it does with personal data, with the basis it names for each activity. `REGISTER` is a list of dictionaries, each with `"activity"` and `"basis"`:

        | activity | basis |
        |---|---|
        | send lesson reminders | consent |
        | bill subscriptions | contract |
        | keep invoices for the tax office | legal obligation |
        | detect fraud in payments | legitimate interests |
        | train a model on posts scraped from other sites | public data |
        | share progress with advertisers | none |
        | train the tutor bot on users' chats | unknown |

        `BASES` is the list of the six lawful bases as the course writes them: `["consent", "contract", "legal obligation", "vital interests", "public task", "legitimate interests"]`. `has_basis(register, bases)` gets such a list `register` and a list `bases`, and gives back a dictionary with, for each activity, `True` when its basis is in `bases` and `False` when it is not. `result` starts as an empty dictionary, and the `for` loop goes over every `row`. Replace the `...` with one line that stores in `result`, under `row["activity"]`, whether `row["basis"]` is in `bases`. The function gives back `result`. The check calls `has_basis(REGISTER, BASES)`, then fewer bases, a short register, and an empty one.
        """),
        L.task("Give back a dictionary with, for each activity, whether its basis is one of `bases`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_basis():
    box_basis = L.code_box("basis", lines=6)
    box_basis
    return (box_basis,)


@app.cell(hide_code=True)
def check_basis(box_basis):
    result_basis = L.check("basis", box_basis.value)
    result_basis
    return (result_basis,)


@app.cell(hide_code=True)
def hints_basis():
    L.hints(
        "basis",
        [
            "Step 3 of the recipe. A dictionary gets a value with `result[key] = value`. Which key do you need, and which value?",
            "The key is the activity of the row. The value is `True` or `False`, and a test with `in` gives exactly one of the two.",
            "`result[row[\"activity\"]] = ___`",
        ],
        why="`result[row[\"activity\"]] = row[\"basis\"] in bases`. `has_basis(REGISTER, BASES)` gives `True` for the first four activities and `False` for the last three.",
    )
    return


@app.cell(hide_code=True)
def text_after_basis():
    mo.md(r"""
    Three activities have no lawful basis in the register. `"public data"` is not one of the six: that the posts were public shapes what people can expect, and is no basis by itself. `"none"` and `"unknown"` are honest, and still mean that the app may not do it until it has identified a basis. A `True` is not the end either: your function checks only that the name of a basis is written down, not that it holds. Fraud detection rests on legitimate interests, so the app still has to pass the three-step test.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_purpose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. new_purposes: used for something else?

        The same app also writes down, for each kind of data, what it collected the data for and what it uses them for now. `USES` is a list of dictionaries, each with `"data"`, `"collected_for"` and `"used_for"`:

        | data | collected for | used for |
        |---|---|---|
        | email addresses | lesson reminders | lesson reminders |
        | voice recordings | pronunciation feedback | training a speech model |
        | lesson scores | showing progress | showing progress |
        | chats with the tutor bot | answering questions | training the tutor bot |
        | payment details | billing | detecting fraud |

        `new_purposes(uses)` gets such a list `uses`, and gives back a list of the data whose `"used_for"` differs from their `"collected_for"`. `found` starts as an empty list, and the `for` loop goes over every `use`. Replace the `...` with two lines: when `use["used_for"]` differs from `use["collected_for"]`, add `use["data"]` to `found`. The function gives back `found`. The check calls `new_purposes(USES)`, then a short list and an empty one.
        """),
        L.task("Give back a list of the data that are used for another purpose than the one they were collected for."),
    ])
    return


@app.cell(hide_code=True)
def exercise_purpose():
    box_purpose = L.code_box("purpose", lines=7)
    box_purpose
    return (box_purpose,)


@app.cell(hide_code=True)
def check_purpose(box_purpose):
    result_purpose = L.check("purpose", box_purpose.value)
    result_purpose
    return (result_purpose,)


@app.cell(hide_code=True)
def hints_purpose():
    L.hints(
        "purpose",
        [
            "Step 4 of the recipe. Compare two values of the same row: what the data were collected for, and what they are used for.",
            "An `if` that asks whether the two differ, and under it `found.append(...)`. Should the list hold the whole row, or which data it is?",
            "`if use[\"used_for\"] != use[\"collected_for\"]:` and under it `found.append(___)`",
        ],
        why="`if use[\"used_for\"] != use[\"collected_for\"]:` and under it `found.append(use[\"data\"])`. `new_purposes(USES)` gives `[\"voice recordings\", \"chats with the tutor bot\", \"payment details\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_purpose():
    mo.md(r"""
    Three kinds of data have a new purpose. That does not make the new use forbidden: it must not be incompatible with the purpose the data were collected for. Your function cannot judge that. Comparing two texts does not weigh the context in which the data were collected, or the safeguards, as Article 6(4) asks; it shows where someone has to look. The chats with the tutor bot are the case from the register: collected to answer questions, now used for training, and with no basis named in the register.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_controller():
    mo.md(r"""
    ### Exercise 3. Who decides?
    """)
    return


@app.cell(hide_code=True)
def choice_controller_cell():
    choice_controller = L.choice(
        "controller",
        question="A hospital wants a model that sorts the letters it gets from patients. The hospital decides what the model is for and how it is made, and pays a company to train it on the letters, following the hospital's instructions. Who is the data controller?",
        options={
            "right": "The hospital: it decides why and how",
            "company": "The company: it does the training",
            "patients": "The patients: the letters are about them",
        },
        correct="right",
        why={
            "right": "The controller is whoever \"determines the purposes and means of the processing of personal data\". That is the hospital; the company processes the data on its behalf, and is a processor. Letters about the health of patients also hold special categories of data: processing them is prohibited unless one of the exceptions in Article 9(2) applies.",
            "company": "Doing the work does not make the company the controller: it trains the model on the hospital's behalf, following its instructions, which makes it a processor. See *Personal data, and who is responsible* in *The idea*.",
            "patients": "The patients are the data subjects: the people the data are about. The controller is whoever decides why and how the data are processed. See *Personal data, and who is responsible* in *The idea*.",
        },
    )
    choice_controller
    return (choice_controller,)


@app.cell(hide_code=True)
def text_exercise_erase():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. erase: take a person out of the dataset

        The app has a forum, and it trains a model on the posts. `POSTS` is a list of seven made-up posts, each a dictionary with `"id"`, `"author"` (who wrote it), `"mentions"` (a list of the people the post names) and `"text"`:

        | id | author | mentions | text |
        |---|---|---|---|
        | 1 | sam | | Does anyone know a good podcast in Spanish? |
        | 2 | noor | sam | Try the one about cooking, sam. |
        | 3 | kai | | Day 30 of my streak! |
        | 4 | sam | kai | Well done, kai! |
        | 5 | ash | | I keep mixing up ser and estar. |
        | 6 | lior | sam, ash | sam and ash, shall we practice on Friday? |
        | 7 | remi | noor | Thanks for the tip, noor. |

        Say the app trains on the posts with the consent of their writers, and sam withdraws it. Then sam can ask for erasure: withdrawn consent, with no other legal ground, is one of the six grounds of Article 17(1). A post by someone else that names sam is information about sam too, but sam's consent never covered it, and it is also its writer's own words. Here the app chooses to go further, and takes out every post by sam and every post that names sam. `erase(posts, name)` gets such a list `posts` and a name `name`, and gives back a list of the posts that are not by `name` and do not mention `name`. `kept` starts as an empty list, and the `for` loop goes over every `post`. Replace the `...` with two lines: when `post["author"]` is not `name` and `name` is not in `post["mentions"]`, add `post` to `kept`. The function gives back `kept`. The check calls `erase(POSTS, "sam")`, then other names, a name with no posts, and a short list.
        """),
        L.task("Give back a list of the posts that are not by `name` and do not mention `name`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_erase():
    box_erase = L.code_box("erase", lines=7)
    box_erase
    return (box_erase,)


@app.cell(hide_code=True)
def check_erase(box_erase):
    result_erase = L.check("erase", box_erase.value)
    result_erase
    return (result_erase,)


@app.cell(hide_code=True)
def hints_erase():
    L.hints(
        "erase",
        [
            "Step 5 of the recipe. A post stays only when two things hold at once: it is not by `name`, and it does not mention `name`.",
            "Two conditions in one `if`, joined by `and`. How do you ask whether a name is not in a list?",
            "`if post[\"author\"] != name and ___:`",
        ],
        why="`if post[\"author\"] != name and name not in post[\"mentions\"]:` and under it `kept.append(post)`. `erase(POSTS, \"sam\")` keeps the posts with id 3, 5 and 7.",
    )
    return


@app.cell(hide_code=True)
def text_after_erase():
    mo.md(r"""
    Four of the seven posts are gone: two by sam, and two that name sam. The dataset no longer holds them, but a model that was already trained on the old dataset does not change when rows leave the dataset. The EDPB writes that models trained on personal data "cannot, in all cases, be considered anonymous": a model can sometimes give back what it was trained on, the subject of *What a model remembers*, later in this module. And where a model was trained unlawfully, an authority may order the erasure of the model itself.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_blog():
    mo.md(r"""
    ### Exercise 5. A blog post
    """)
    return


@app.cell(hide_code=True)
def choice_blog_cell():
    choice_blog = L.choice(
        "blog",
        question="A blog writes: \"In 2024, the EDPB ruled that training AI on scraped data is illegal.\" What does Opinion 28/2024 support?",
        options={
            "exact": "The blog post as it stands",
            "right": "Less: an opinion, and no ban",
            "more": "More: its ruling binds every court",
        },
        correct="right",
        why={
            "exact": "The EDPB did not rule: it gave an opinion, which the authorities \"should take utmost account\" of. And it does not call scraping illegal: it says that scraped information \"may contain personal data\", and discusses how a basis such as legitimate interests can apply. See *The GDPR and AI* in *The idea*.",
            "right": "Opinion 28/2024 is an opinion, not a court ruling. It does not ban training on scraped data: it says such data may contain personal data, so their processing needs a lawful basis, and it names safeguards, such as respecting robots.txt or ai.txt files.",
            "more": "Rulings come from courts, not from the EDPB. Its opinion guides the data protection authorities, which \"should take utmost account\" of it. See *The GDPR and AI* in *The idea*.",
        },
    )
    choice_blog
    return (choice_blog,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What is privacy?](/course/open/rai-what-is-privacy?lang=en) you named privacy problems with Solove's taxonomy. Some of them have a counterpart in the GDPR: secondary use meets purpose limitation, and the people in the data get rights such as erasure and objection. Whoever decides why and how the data are processed is responsible, also when a model is trained on them. In the next lesson, *Information in the wrong place*, you look more closely at the third view of privacy: whether a flow of information fits its context.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_blog, choice_controller, result_basis, result_erase, result_purpose):
    L.wrap_up(result_basis, result_purpose, choice_controller, result_erase, choice_blog)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
