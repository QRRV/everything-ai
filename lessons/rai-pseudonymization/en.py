# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-pseudonymization. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-pseudonymization", "en")
    C = L.checks   # the twelve users of a made-up language-learning app, the same rows with codes, and a café's list


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [The GDPR for AI](/course/open/rai-gdpr?lang=en) and [Dictionaries](/course/open/py-dicts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_identifiable():
    quick_identifiable = L.choice(
        "quick-identifiable",
        question="Under the GDPR, information is personal data when it relates to a person who is…",
        options={
            "named": "named in it",
            "right": "identified or identifiable",
            "eu": "living in the European Union",
        },
        correct="right",
        why={
            "right": "Personal data is \"any information relating to an identified or identifiable natural person\". A person is identifiable when they can be identified directly or indirectly, also without a name.",
            "named": "A name is not needed: a person who can be identified indirectly, by a number or by facts about them, is identifiable too. See *Personal data, and who is responsible* in *The idea* of [The GDPR for AI](/course/open/rai-gdpr?lang=en).",
            "eu": "Personal data is about whether a person is identified or identifiable, not about where they live. See *Personal data, and who is responsible* in *The idea* of [The GDPR for AI](/course/open/rai-gdpr?lang=en).",
        },
    )
    quick_identifiable
    return (quick_identifiable,)


@app.cell(hide_code=True)
def quick_check_controller():
    quick_controller = L.choice(
        "quick-controller",
        question="A company runs a language-learning app and decides what it does with its users' data, and why. A hosting company keeps the data on its servers, and does only what the app's company asks. Who is the data controller?",
        options={
            "host": "The hosting company, which keeps the data",
            "right": "The company behind the app",
            "users": "Each user, for their own data",
        },
        correct="right",
        why={
            "right": "The controller is whoever \"determines the purposes and means of the processing of personal data\": the app's company. The hosting company processes the data on its behalf, and is a processor.",
            "host": "Keeping the data does not make the hosting company the controller: it follows the app's company, which decides why and how. That makes it a processor. See *Personal data, and who is responsible* in *The idea* of [The GDPR for AI](/course/open/rai-gdpr?lang=en).",
            "users": "The users are the data subjects: the data is about them. The controller is who decides why and how the data is processed: the app's company. See *Personal data, and who is responsible* in *The idea* of [The GDPR for AI](/course/open/rai-gdpr?lang=en).",
        },
    )
    quick_controller
    return (quick_controller,)


@app.cell(hide_code=True)
def quick_check_keys():
    quick_keys = L.choice(
        "quick-keys",
        question="`row` is the dictionary `{\"email\": \"ana@mail.example\", \"hour\": 21}`. What does `key` go through in `for key in row:`?",
        options={
            "right": "The keys",
            "values": "The values",
            "pairs": "Pairs of a key and its value",
        },
        correct="right",
        why={
            "right": "A loop over a dictionary goes over its keys: here `\"email\"` and then `\"hour\"`. The value of each is `row[key]`.",
            "values": "A loop over a dictionary goes over its keys; you get a value with `row[key]`. See *Exercise 2. A loop over a dictionary* in [Dictionaries](/course/open/py-dicts?lang=en).",
            "pairs": "A loop over a dictionary goes over its keys only, here `\"email\"` and `\"hour\"`; you get a value with `row[key]`. See *Exercise 2. A loop over a dictionary* in [Dictionaries](/course/open/py-dicts?lang=en).",
        },
    )
    quick_keys
    return (quick_keys,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Data about people is worth a lot for research and for training models, and it often comes with a promise: the names have been removed. That promise can mean two very different things. Replacing a name or an e-mail address with a code is pseudonymization: the data is still personal data, at least for whoever holds the key, and the GDPR still applies. Anonymous means much more: nobody can find the person again, by any means reasonably likely to be used. In 2006 AOL replaced the user IDs in its search logs with numbers, and journalists still identified people from their searches; Netflix said it had removed all customer identifying information, and two researchers showed how few ratings and dates it takes to pick out a subscriber's record. In this lesson you pseudonymize the rows of a made-up app yourself, and then you undo it twice: once with a list of e-mail addresses, once with three facts about one person.

    **An everyday example.** A class hands in essays with a number instead of a name, so that the teacher grades without knowing whose essay it is. But the teacher knows the handwriting, knows who always writes about horses, and knows who was the only one in Japan last summer. The names are gone; the pupils can still be found.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="For the Netflix Prize, Netflix published the movie ratings of 480,189 subscribers, without their names. Suppose you know 8 ratings of one subscriber, 2 of which may be completely wrong, and their dates to within 14 days. For what share of the subscribers' records is that enough to pick out exactly one?",
        options={
            "tenth": "About 1 in 10",
            "half": "About half",
            "right": "99%",
        },
        reveal="**99%**: Arvind Narayanan and Vitaly Shmatikov (2008) found that, with 8 ratings of which 2 may be wrong and dates that may be 14 days off, \"99% of records can be uniquely identified in the dataset\". For 68%, two ratings with dates within 3 days were enough. Below you see why the GDPR would not call such data anonymous.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### A code instead of an address

        The made-up language-learning app of [The GDPR for AI](/course/open/rai-gdpr?lang=en) wants to let a research team study how people learn. It takes the e-mail addresses out of its rows, and puts a code in their place. The GDPR calls such a step **pseudonymization**: processing personal data so "that the personal data can no longer be attributed to a specific data subject without the use of additional information, provided that such additional information is kept separately and is subject to technical and organisational measures", in the words of Article 4(5). The additional information can be a list that says which code belongs to which address: the key.

        The result is still personal data: "Personal data which have undergone pseudonymisation, which could be attributed to a natural person by the use of additional information should be considered to be information on an identifiable natural person" (recital 26). That does not make pseudonymization useless. It "can reduce the risks to the data subjects concerned and help controllers and processors to meet their data-protection obligations" (recital 28). And it is not required everywhere: "The GDPR does not impose a general obligation to use pseudonymisation", writes the European Data Protection Board (EDPB) in its guidelines on pseudonymization of January 2025, a draft version for public consultation.
        """),
        mo.Html(C.shared_picture("en")),
        mo.md(r"""
        ### Anonymous: nobody can be found again

        The GDPR does not apply to anonymous information: "information which does not relate to an identified or identifiable natural person or to personal data rendered anonymous in such a manner that the data subject is not or no longer identifiable" (recital 26). **Anonymization** is processing data so that this holds. Whether someone is still identifiable is a wide question: "account should be taken of all the means reasonably likely to be used, such as singling out, either by the controller or by another person", with "all objective factors, such as the costs of and the amount of time required for identification". So it is not only about the key, and not only about the app: anyone, with any means reasonably likely to be used, counts. **Re-identification** is finding out who is behind data that seemed anonymous. Neither that word nor *anonymisation* occurs in the GDPR; *anonymous* occurs only in recital 26.

        ### Three risks

        In 2014, under the law that came before the GDPR, the Article 29 Working Party wrote an opinion on anonymization techniques. It asks three questions of any technique: "(i) is it still possible to single out an individual, (ii) is it still possible to link records relating to an individual, and (iii) can information be inferred concerning an individual?" **Singling out** is "the possibility to isolate some or all records which identify an individual in the dataset". **Linkability** is "the ability to link, at least, two records concerning the same data subject or a group of data subjects", in one database or in two. Inference is "the possibility to deduce, with significant probability, the value of an attribute from the values of a set of other attributes", as a language model guessed where a writer lives in [Information in the wrong place](/course/open/rai-contextual-integrity?lang=en).

        For pseudonymization, the opinion's table answers all three questions with yes: each risk remains. Its verdict: "pseudonymisation is not a method of anonymisation. It merely reduces the linkability of a dataset with the original identity of a data subject, and is accordingly a useful security measure." Anonymization, in contrast, must be permanent: it "results from processing personal data in order to irreversibly prevent identification."

        ### A hash is not a hiding place

        A **hash function** turns any text into a short string of letters and digits: the same text always gives the same hash, and a text that differs by one character gives a completely different one. SHA-256 is such a function. The Working Party lists the hash function among "The most used pseudonymisation techniques", and warns: "if the range of input values the hash function are known they can be replayed through the hash function in order to derive the correct value for a particular record." The EDPB draft gives an example: "if you know the transformation is merely a SHA256 hash of a name, you could apply this to all names you have elsewhere and then see which hashes match in the pseudonymised dataset." It wants a secret in the transformation: "Only possession of the secret information should enable the computation of the pseudonym given the identifier." Neither text names e-mail addresses; that a hashed address can be found in the same way follows from the same argument, and you try it in Exercise 2.

        ### Names gone, people found

        In 2006, AOL published "a database containing twenty million search keywords for over 650,000 users over a 3-month period", with "the only privacy preserving measure consisting in replacing AOL user ID by a numerical attribute". "This led to the public identification and location of some of them," the Working Party writes; The New York Times reported it as "A Face Is Exposed for AOL Searcher No. 4417749".

        For the Netflix Prize, Netflix published 100,480,507 movie ratings by 480,189 subscribers, from December 1999 to December 2005. In its FAQ, Netflix wrote: "No, all customer identifying information has been removed; all that remains are ratings and dates." Arvind Narayanan and Vitaly Shmatikov (2008) showed what ratings and dates are worth: "With 8 movie ratings (of which 2 may be completely wrong) and dates that may have a 14-day error, 99% of records can be uniquely identified in the dataset. For 68%, two ratings and dates (with a 3-day error) are sufficient". Without any dates, 84% of the subscribers could be singled out by someone who knows 6 of 8 of their movies outside the top 500. To try it on real people, they took "a very small sample of around 50 IMDb users", and found the records of two of them in Netflix's data.

        Luc Rocher, Julien Hendrickx and Yves-Alexandre de Montjoye (2019) estimated with a model that "15 demographic attributes would render 99.98% of people in Massachusetts unique." Their conclusion: such datasets "are unlikely to satisfy the modern standards for anonymization set forth by GDPR".

        ### Pseudonymous for whom?

        In September 2025, the Court of Justice of the European Union ruled on exactly this question. The Single Resolution Board (SRB), an EU body, had sent 1,104 comments by shareholders and creditors of a bank to the consultancy Deloitte, each with a code instead of the name of its writer; "it was only the SRB who was able, via that code, to link the comments to the data". The case was about the rules for EU bodies, which the Court reads in the same way as the GDPR on this point. The Court: "pseudonymised data must not be regarded as constituting, in all cases and for every person, personal data", "in so far as pseudonymisation may, depending on the circumstances of the case, effectively prevent persons other than the controller from identifying the data subject". For that, the recipient must not be able to undo the measures, also not "by recourse to other means of identification such as cross-checking with other factors". For the SRB, which holds the key, the comments stayed personal data. The European Data Protection Supervisor (EDPS) had argued that pseudonymized data are personal data for everyone, and the Court rejected that as unfounded. Still, it set aside the General Court's judgment, which had annulled the EDPS's decision, and sent the case back to the General Court, which still has to weigh the facts: for the duty to tell people who receives their data, "the identifiable nature of the data subject must be assessed at the time of collection of the data and from the point of view of the controller."

        The EDPB draft, from before the judgment, goes further: pseudonymized data remain personal data, and "This statement also holds true if pseudonymised data and additional information are not in the hands of the same person." On the recipient's side, the judgment is the later and the binding text.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **pseudonymization** | replacing who someone is, such as a name or an e-mail address, with a code; with extra information kept apart, the code leads back to the person, so the data is still personal data, at least for whoever holds that information |
        | **anonymization** | processing data so that nobody can be identified from it any more, by any means reasonably likely to be used; anonymous data falls outside the GDPR |
        | **re-identification** | finding out who is behind data that seemed anonymous |
        | **singling out** | picking out the records of one person in a dataset, also without knowing their name |
        | **linkability** | being able to link two records about the same person, in one dataset or in two |
        | **hash function** | a function that turns a text into a short string of letters and digits, always the same for the same text; SHA-256 is one |
        """),
        L.watch_out("Pseudonymization is a good measure, not a bad one: it can lower the risks, and the GDPR says it can help to meet its obligations. The mistake is only to call the result anonymous. At best, a code hides who someone is from whoever lacks the key; it does not hide the rest of the row."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out whether data without names is anonymous:

    1. Find what says directly who someone is, such as a name or an e-mail address, and replace it with a code. That is `pseudonymize`.
    2. Ask who can turn a code back into a person: whoever holds the key, and, with a plain hash, anyone who can guess the inputs and hash them. That is `unmask`.
    3. Look at what is left in the rows. Count the rows that fit what someone may know about a person; when one row is left, that person is singled out. That is `candidates`.
    4. Call the data anonymous only when nobody, with means reasonably likely to be used, can single out a person, link records about them, or infer something about them. Otherwise it is still personal data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell hashes three e-mail addresses with SHA-256, and keeps the first 8 characters of each hash as the code. `hashlib.sha256` computes the hash; it works on bytes, so `.encode("utf-8")` turns the text into bytes first, `.hexdigest()` writes the hash as letters and digits, and `[:8]` keeps the first 8 of them.
    """)
    return


@app.cell
def example_hash():
    import hashlib

    for _email in ["ana@mail.example", "bo@mail.example", "ana@mail.example"]:
        # the code of an address: the first 8 characters of its SHA-256 hash
        _code = hashlib.sha256(_email.encode("utf-8")).hexdigest()[:8]
        print(_email, "→", _code)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first and the last address are the same, and so are their codes. That is what makes such a code useful: a research team can follow one user through all their rows, without ever seeing the address. It is also what makes it weak: anyone who hashes the same address gets the same code. In the exercises, `code_of(text)` does what the loop does with each address: the course's version is ready.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pseudonymize():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. pseudonymize: a code instead of the address

        `USERS` is a list of the twelve users of the made-up app. Each user is a dictionary with the keys `"email"`, `"language"` (the language they learn), `"joined"` (the year they signed up), `"hour"` (the hour of the day they usually practice), `"streak"` (the number of days in a row they practiced) and `"goal"`. The first three:

        | | email | language | joined | hour | streak | goal |
        |---|---|---|---|---|---|---|
        | `USERS[0]` | ana@mail.example | Spanish | 2022 | 21 | 15 | travel through Peru |
        | `USERS[1]` | bo@mail.example | Japanese | 2023 | 7 | 41 | a new job in Osaka |
        | `USERS[2]` | chen@mail.example | Japanese | 2023 | 22 | 3 | films without subtitles |

        `pseudonymize(row)` gets one such dictionary `row`, and gives back a new dictionary with the same keys and values, except that `"email"` is gone and `"code"` holds `code_of` of the address. The box may use `code_of`. `new` starts as an empty dictionary, and the `for` loop goes over every `key` of `row`. Replace the `...` with an `if` and an `else`: when `key` is `"email"`, put `code_of(row[key])` in `new` under `"code"`; otherwise put `row[key]` in `new` under `key`. The function gives back `new`. The check calls `pseudonymize(USERS[0])`, then other rows, also with other keys.
        """),
        L.task("Give back a copy of the row, with the e-mail address replaced by its code under the key \"code\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pseudonymize():
    box_pseudonymize = L.code_box("pseudonymize", lines=9)
    box_pseudonymize
    return (box_pseudonymize,)


@app.cell(hide_code=True)
def check_pseudonymize(box_pseudonymize):
    result_pseudonymize = L.check("pseudonymize", box_pseudonymize.value)
    result_pseudonymize
    return (result_pseudonymize,)


@app.cell(hide_code=True)
def hints_pseudonymize():
    L.hints(
        "pseudonymize",
        [
            "Step 1 of the recipe. Every key of the row ends up in `new`, except that the address goes in as its code, under another key.",
            "An `if key == \"email\":` with one line under it, and an `else:` with one line under it. Under which key does the code go, and under which key does each other value go?",
            "Under the `if`: `new[\"code\"] = code_of(row[key])`; under the `else`: `new[___] = row[key]`",
        ],
        why="`if key == \"email\":` with `new[\"code\"] = code_of(row[key])` under it, and `else:` with `new[key] = row[key]` under it. `pseudonymize(USERS[0])` gives `{\"code\": \"7f0d4910\", \"language\": \"Spanish\", \"joined\": 2022, \"hour\": 21, \"streak\": 15, \"goal\": \"travel through Peru\"}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unmask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. unmask: who is behind a code?

        `SHARED` is the list of the twelve rows after pseudonymization, as the research team gets them, and `CODES` the list of their twelve codes, in the same order. One of the researchers also runs a language café, and has its sign-up list: `CAFE` is a list of ten e-mail addresses.

        `unmask(codes, emails)` gets a list of codes `codes` and a list of addresses `emails`. It gives back a dictionary with, for every address whose code is in `codes`, that code as the key and the address as the value. The box may use `code_of`. `found` starts as an empty dictionary, and the `for` loop goes over every address `email`. Replace the `...` with two lines: when `code_of(email)` is in `codes`, put `email` in `found` under that code. The function gives back `found`. The check calls `unmask(CODES, CAFE)`, then shorter lists.
        """),
        L.task("Give back a dictionary of the codes that belong to one of the addresses, each with its address."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unmask():
    box_unmask = L.code_box("unmask", lines=7)
    box_unmask
    return (box_unmask,)


@app.cell(hide_code=True)
def check_unmask(box_unmask):
    result_unmask = L.check("unmask", box_unmask.value)
    result_unmask
    return (result_unmask,)


@app.cell(hide_code=True)
def hints_unmask():
    L.hints(
        "unmask",
        [
            "Step 2 of the recipe, as in the worked example: the same address always gives the same code, so hash every address on the list and look for its code.",
            "An `if` with `in codes`, and under it one line that adds to the dictionary. What goes on the left of `in`: the address, or its code?",
            "`if code_of(email) in codes:` and under it `found[___] = email`",
        ],
        why="`if code_of(email) in codes:` and under it `found[code_of(email)] = email`. `unmask(CODES, CAFE)` gives four addresses: those of ana, chen, ivo and lou.",
    )
    return


@app.cell(hide_code=True)
def text_after_unmask():
    mo.md(r"""
    Four of the ten addresses on the café's list belong to users of the app: ana, chen, ivo and lou. For those four, the research team can now read everything in their rows: the language, the hour, the goal. It needed no key. The code was a plain hash, and the café's list gave the inputs to try: the replay the Working Party warns about. With a secret in the transformation, as the EDPB asks, the café's list would have found nothing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_candidates():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. candidates: singled out without a name

        Bo uses the app, and a colleague of Bo knows three things: Bo learns Japanese, signed up in 2023, and practices at 7 in the morning. `KNOWN` is the dictionary `{"language": "Japanese", "joined": 2023, "hour": 7}`.

        `candidates(known, rows)` gets a dictionary of facts `known` and a list of rows `rows`, like `SHARED`. It gives back a list of the codes of the rows that match every fact: for every key in `known`, the row has the same value. `found` starts as an empty list, and the `for` loop goes over every `row`. Replace the `...` with your own lines: start with `fits = True`; go over every `key` of `known`, and when `row[key]` differs from `known[key]`, set `fits` to `False`; after that loop, when `fits` is still `True`, add `row["code"]` to `found`. The function gives back `found`. The check calls `candidates(KNOWN, SHARED)`, then fewer facts, other facts and other rows.
        """),
        L.task("Give back the codes of the rows that match every fact in known."),
    ])
    return


@app.cell(hide_code=True)
def exercise_candidates():
    box_candidates = L.code_box("candidates", lines=11)
    box_candidates
    return (box_candidates,)


@app.cell(hide_code=True)
def check_candidates(box_candidates):
    result_candidates = L.check("candidates", box_candidates.value)
    result_candidates
    return (result_candidates,)


@app.cell(hide_code=True)
def hints_candidates():
    L.hints(
        "candidates",
        [
            "Step 3 of the recipe. A row stays only when none of the facts in `known` speaks against it.",
            "Inside the loop over the rows: `fits = True`, then a loop over the keys of `known` with an `if` that marks the row as not fitting, and after that loop an `if fits:`. What goes into `found`: the row, or its code?",
            "`if row[key] ___ known[key]:` and under it `fits = False`",
        ],
        why="`fits = True`; `for key in known:` with `if row[key] != known[key]:` and `fits = False` under it; then `if fits:` with `found.append(row[\"code\"])` under it. `candidates(KNOWN, SHARED)` gives `[\"0f631e11\"]`: one row, Bo's.",
    )
    return


@app.cell(hide_code=True)
def text_after_candidates():
    mo.md(r"""
    With one fact, three rows are left; with two, two rows; with all three, one: Bo's row, singled out without a name, an address or a key. The colleague can now read the rest of that row too, such as Bo's goal: a new job in Osaka. That is singling out, the first of the Working Party's three risks, and it is what Narayanan and Shmatikov did with a few ratings and dates. Twelve rows make it easy; their 480,189 subscribers did not make it hard.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_recipient():
    mo.md(r"""
    ### Exercise 4. Personal data for whom?

    The app now makes its codes with a secret key that only the app has, instead of a plain hash, and gives the rows to a research team. The team has no key, and no other way to find out who is who: no café list, and no colleague who knows Bo.
    """)
    return


@app.cell(hide_code=True)
def choice_recipient_cell():
    choice_recipient = L.choice(
        "recipient",
        question="Are these rows personal data, according to the ruling of the Court of Justice in 2025?",
        options={
            "none": "Not for anyone, since the addresses are gone",
            "right": "For the app yes, for the team maybe not",
            "always": "Yes, for everyone who holds them, always",
        },
        correct="right",
        why={
            "right": "The Court: pseudonymized data are not personal data \"in all cases and for every person\". For the app, which holds the key, the rows stay personal data. For a team that cannot undo the pseudonymization, also not by cross-checking with other information, they may not be. The Court set aside the General Court's judgment and sent the case back, so the General Court still has to weigh the facts.",
            "none": "For the app, which holds the key, the rows stay personal data: the Court said so of the SRB, which could link the codes to the writers. See *Pseudonymous for whom?* in *The idea*.",
            "always": "The EDPB's draft of January 2025 comes close: pseudonymized data stay personal data also when the rows and the key are in different hands. Before the Court, the EDPS argued that they are personal data for everyone, and the Court rejected that as unfounded: not \"in all cases and for every person\". See *Pseudonymous for whom?* in *The idea*.",
        },
    )
    choice_recipient
    return (choice_recipient,)


@app.cell(hide_code=True)
def text_exercise_deleted():
    mo.md(r"""
    ### Exercise 5. The key is gone

    Suppose the app had given every user a random code, and kept the list of which code belongs to which address apart. Now it deletes that list, so that nobody can turn a code back into an address.
    """)
    return


@app.cell(hide_code=True)
def choice_deleted_cell():
    choice_deleted = L.choice(
        "deleted",
        question="Are the research team's rows anonymous now?",
        options={
            "yes": "Yes: without the list no one can find anyone",
            "right": "Only if nobody can be found again some other way",
            "never": "No, pseudonymized data stay personal data forever",
        },
        correct="right",
        why={
            "right": "The EDPB draft: even when all the additional information has been erased, the data \"can be considered anonymous only if the conditions for anonymity are met\". Bo's row shows why: three facts were enough, without any key. Anonymous means that nobody, with means reasonably likely to be used, can single out a person, link records about them, or infer something about them.",
            "yes": "Exercise 3 found Bo's row without any key: three facts were enough to single it out. Deleting the list takes away one way back, not every way. See *Three risks* in *The idea*.",
            "never": "Pseudonymized data can become anonymous, but only when the conditions for anonymity are met, also after the key is deleted, says the EDPB draft. See *Anonymous: nobody can be found again* in *The idea*.",
        },
    )
    choice_deleted
    return (choice_deleted,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The GDPR for AI](/course/open/rai-gdpr?lang=en) you met personal data: information about a person who is identified or identifiable. Here you saw how wide *identifiable* is. A code instead of an address leaves a person identifiable: by whoever holds the key, by anyone who can guess the inputs of a plain hash, and by anyone who knows a few facts about them. In the next lesson, *Hiding in a group: k-anonymity*, you make sure that every combination of such facts occurs in several rows, so that no row stands alone.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_deleted, choice_recipient, result_candidates, result_pseudonymize, result_unmask):
    L.wrap_up(result_pseudonymize, result_unmask, result_candidates, choice_recipient, choice_deleted)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
