# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-find-pii. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-find-pii", "en")
    C = L.checks   # ten made-up messages to a made-up municipality, and the course's is_number and is_phone


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en), [Text: strings](/course/open/py-strings?lang=en) and [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="A company replaces every name in its customer data with a code, and keeps the list that links each code to its name. What is the data now?",
        options={
            "anon": "Anonymous data: the names are gone",
            "right": "Still personal data, for the company at least",
            "law": "Outside the GDPR, as long as the list stays secret",
        },
        correct="right",
        why={
            "right": "Replacing names with codes is pseudonymization: with the list kept apart, each code leads back to a person, so under the GDPR the data is still personal data, at least for whoever holds that list.",
            "anon": "With the list, every code leads back to a name. Anonymous means that nobody can find the person again, by any means reasonably likely to be used. See *The idea* of [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en).",
            "law": "Keeping the list apart can lower the risks, but the company holds it, so for the company the data is still personal data and the GDPR applies. See *The idea* of [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en).",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def quick_check_index():
    quick_index = L.choice(
        "quick-index",
        question='With `number = "123456782"`, what does `number[8]` give?',
        options={
            "right": '`"2"`: the last character',
            "eighth": '`"8"`: the eighth character',
            "error": "An error: a string has no place 8",
        },
        correct="right",
        why={
            "right": 'Python counts places from 0, so the nine characters have the places 0 to 8, and `number[8]` is the last one, `"2"`. It is a character, a string; `int(number[8])` makes it the number 2.',
            "eighth": "That is `number[7]`: Python counts places from 0. See *The idea* in [Text: strings](/course/open/py-strings?lang=en).",
            "error": "Nine characters have the places 0 to 8, so place 8 is there: it is the last one. See *The idea* in [Text: strings](/course/open/py-strings?lang=en).",
        },
    )
    quick_index
    return (quick_index,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="A pipeline has two steps: `StandardScaler()` and then a model. What does `pipe.fit(X_train, y_train)` do?",
        options={
            "model": "It fits the model only: a scaler learns nothing",
            "right": "It fits the scaler on X_train, then the model",
            "all": "It fits the scaler on every example, test set included",
        },
        correct="right",
        why={
            "right": "`pipe.fit` learns every step from the training examples: the scaler learns the mean and standard deviation of each column of X_train, and the model then learns from X_train scaled with them.",
            "model": "The scaler learns too: the mean and standard deviation of each column, with `fit`. See [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "all": "`pipe.fit` sees only what you give it, X_train; that is what keeps the test set out. See [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A municipality wants to train a model that sorts the messages people send it by topic, with a pipeline like the ones in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en). But the messages are full of personal data: names, phone numbers, and in the Netherlands the citizen service number, the BSN, with which records from different files are easily linked together. Before such text goes into a training set, or into a model, a step finds the personal data and replaces it with a placeholder such as `[NAME]`. In this lesson you write that step for ten made-up messages, with the check that tells a possible BSN from a mistyped one, and then you measure what your filter misses. It will miss something: every filter does.

    **An everyday example.** Before you hand someone a copy of a letter, you black out your name and your account number with a marker. You find the name at the top and the number in the middle, but you miss the name in the footer, and the phone number someone wrote out as "oh six, twelve, thirty-four".
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Jan Trienes and colleagues (2020) compared DEDUCE, a tool with rules specifically developed for Dutch, with other methods on Dutch medical records from a dataset of 1,260 patients. Which share of the personal data in them did it find?",
        options={
            "most": "97%: nearly all of it",
            "right": "56%",
            "few": "12%: hardly any of it",
        },
        reveal="**56%**: DEDUCE had a recall of 0.564, and of what it marked, 80.7% was personal data (a precision of 0.807). It found 80.5% of the names and 81.2% of the phone and fax numbers, but none of the IDs. In the same comparison, a neural network, a BiLSTM-CRF, reached a recall of 0.871. In the authors' words, \"an existing rule-based method specifically developed for the Dutch language fails to generalize to this new data.\" Rules find what they were written for.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Finding personal data in text

    Personal data is any information about a person who can be identified, as in [The GDPR for AI](/course/open/rai-gdpr?lang=en). In text it sits in names, phone numbers, e-mail addresses and identification numbers. **PII redaction**, after personally identifiable information, finds such pieces and replaces each one with a placeholder that says what was there, such as `[NAME]` or `[PHONE]`: the text still reads, but no longer says who. Presidio, an open-source tool started at Microsoft and now run by a community, describes how its own recognizers work: "Predefined and custom recognizers leverage regex, Named Entity Recognition and other types of logic to detect PII in unstructured text." The simplest way is a list of names you know. A regular expression, regex for short, is a pattern for text, such as "ten digits starting with 06". Named entity recognition is a model that recognizes names, also names that are on no list. In this lesson you use a list and a few rules in plain Python.

    ### A digit that checks the others

    Many numbers carry a check of their own. Hans P. Luhn, at IBM, described it in a patent filed in 1954: "a single digit, called the check digit, is appended to the righthand end of the original or true number." That **check digit** is computed from the other digits, so that a typing error, such as "a transposition of the digits", makes the check fail. Luhn's formula is still in the standard for the numbers on cards, ISO/IEC 7812-1.

    ### The eleven test of the BSN

    The Dutch citizen service number, the BSN, has nine digits and a check of its own: the **eleven test**, *elfproef* in Dutch. The government's design for the BSN, the Logisch Ontwerp BSN of the RvIG, gives the rule: multiply the first digit by 9, the second by 8, and so on down to 2 for the eighth digit, and the ninth digit by −1; add it all up; the sum must be divisible by 11. For the made-up number 111222333: 9 · 1 + 8 · 1 + 7 · 1 + 6 · 2 + 5 · 2 + 4 · 2 + 3 · 3 + 2 · 3 − 1 · 3 = 66 = 6 · 11, so it passes. A BSN written with eight digits gets a 0 in front for the test. One wrong digit always makes the test fail, and so does a swap of any two digits: the course checked every case.

    ### What a pass tells you

    A pass does not make a number a BSN. Of the 1,000,000,000 strings of nine digits, 90,909,091 pass: 1 in 11. New numbers are made to say nothing about the person: the design asks for numbers that are *informatieloos*, without information, and that show no order. The government's own check service answers *Nummer is geen BSN*, number is not a BSN, both for a number that fails the eleven test and for one that passes but is not in the number register. So a pass means: this could be a BSN. And a fail does not make a number harmless: a real BSN with one typing error fails too.

    ### Phone numbers

    The ACM, which hands out phone numbers in the Netherlands, writes that numbers starting with 06 are meant for mobile phones, except the ranges starting with 066 and 067, and that mobile numbers have ten digits. The course's `is_phone`, in the worked example, checks just that. But people write the same number as 06-12345678, 06 12 34 56 78 or +31 6 12345678, with 31, the country code of the Netherlands, in place of the 0; a rule for one form misses the others.

    ### Who may use a BSN

    A filter is not the only rule. In the Netherlands, a number that the law prescribes to identify a person may, when personal data are processed, be used only to carry out that law, or for purposes set by law (article 46 of the Dutch GDPR implementation act). The Dutch data protection authority, the Autoriteit Persoonsgegevens, sums it up: organizations outside the government may use the BSN only where the law says so, as in healthcare and education, and consent does not change that. It calls the BSN sensitive, because with it, information from different files is easily linked together.

    ### Every filter misses something

    Presidio warns in its own documentation: "there is no guarantee that Presidio will find all sensitive information. Consequently, additional systems and protections should be employed." It also names the other side: "there is a trade-off between false positives (falsely detected text) and false negatives (PII entities which are not detected)." A rule that hides every number of nine digits also hides case numbers; a stricter rule misses more. Ildikó Pilán and colleagues (2022) measured Presidio on the test set of TAB, a benchmark of 1,268 court cases of the European Court of Human Rights: it had a recall of 0.707 on the mentions of personal information, and, counted per entity instead of per mention, of 0.460 on the direct identifiers. Milad Nasr and colleagues (2023), looking for personal data in text that ChatGPT wrote, used both regexes and a language model, and found malformed contact details too, such as "sam AT gmail DOT com": a rule that looks for an @ never sees that one.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **PII redaction** | finding personal data in text, such as names and numbers, and replacing each piece with a placeholder such as `[NAME]` |
        | **check digit** | a digit computed from the other digits of a number, so that a typing error makes the check fail |
        | **eleven test** (*elfproef*) | the check of a BSN: the nine digits times 9, 8, 7, 6, 5, 4, 3, 2 and −1, added up, must be divisible by 11 |
        | `x in y` | `True` when `x` is one of the items of the list `y`, or a piece of the string `y`; `not in` is the opposite |
        """),
        L.watch_out("A pass is not a BSN, and a fail is not harmless: 1 in 11 numbers of nine digits passes, and a real BSN with one typing error fails. And a text in which a filter leaves no name or number is not anonymous: what the text says can still point to one person."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To hide personal data in messages before they go into a training set:

    1. Cut each message into words, and take the punctuation off the front and the end of each word. The course's `C.redact` does that, and puts the message back together afterwards.
    2. Check each word against the rules: a name you know gives `[NAME]`, a mobile number `[PHONE]`, nine digits that pass the eleven test `[BSN]`, and every other word stays as it is. That is `hide`, with `eleven_test` for the last rule.
    3. Mark by hand, in a sample of the messages, which words are personal data and which other numbers are in them.
    4. Measure: the share of the personal data that your filter hides is its recall; the share of the other numbers that it hides, it hides by mistake. That is `share_found`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([mo.md(r"""
    ### Worked example

    The cell computes the eleven test of the made-up number 111222333 by hand, digit by digit, and runs the course's two helpers on four words. `C.is_number(word)` says whether a word is digits only, and `C.is_phone(word)` whether it is a Dutch mobile number, as the ACM describes it. A loop over a string, `for ch in word`, takes one character at a time. Their code:

    ```python
    def is_number(word):
        # True when word is not empty and every character of it is a digit
        if word == "":
            return False
        for ch in word:
            if ch not in "0123456789":
                return False
        return True

    def is_phone(word):
        # a Dutch mobile number: ten digits, starting with 06, but not with 066 or 067
        if len(word) != 10:
            return False
        if word[:2] != "06":
            return False
        if word[2] == "6":
            return False
        if word[2] == "7":
            return False
        return is_number(word)
    ```
    """)])
    return


@app.cell
def example_eleven():
    # the eleven test on the made-up number 111222333, digit by digit
    _number = "111222333"
    _total = (9 * int(_number[0]) + 8 * int(_number[1]) + 7 * int(_number[2]) + 6 * int(_number[3])
              + 5 * int(_number[4]) + 4 * int(_number[5]) + 3 * int(_number[6]) + 2 * int(_number[7])
              - 1 * int(_number[8]))
    print("total:", _total, "remainder after dividing by 11:", _total % 11)

    # the course's two helpers on four words
    for _word in ["0612345678", "06-12345678", "0671234567", "111222333"]:
        print(_word, C.is_number(_word), C.is_phone(_word))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The total is `66`, and 66 divided by 11 is exactly 6, so the remainder is `0`: the number passes. `0612345678` is a mobile number. `06-12345678` is the same number with a dash, and `is_number` says `False` because of the dash. `0671234567` has ten digits, but starts with 067, which is not for mobile phones. And `111222333` has only nine digits. In Exercise 1 you write the eleven test as a loop over the nine places.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_eleven():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. eleven_test: the check of a BSN

        `eleven_test(number)` gets a string `number` of nine digits. It gives back `True` when the number passes the eleven test, and `False` when it does not. `WEIGHTS` holds the nine weights, `[9, 8, 7, 6, 5, 4, 3, 2, -1]`, so digit i gets the weight `WEIGHTS[i]`. Inside, `total` starts at 0, and the loop goes over the places `i` from 0 to 8. Replace the two `...`. The check calls `eleven_test("111222333")`, then other numbers.
        """),
        L.task("Give back whether the digits times their weights, added up, are divisible by 11."),
    ])
    return


@app.cell(hide_code=True)
def exercise_eleven():
    box_eleven = L.code_box("eleven-test", lines=6)
    box_eleven
    return (box_eleven,)


@app.cell(hide_code=True)
def check_eleven(box_eleven):
    result_eleven = L.check("eleven-test", box_eleven.value)
    result_eleven
    return (result_eleven,)


@app.cell(hide_code=True)
def hints_eleven():
    L.hints(
        "eleven-test",
        [
            "In the worked example, `_total` adds up each digit times its weight, written out one by one. Here the loop does the same, one place `i` at a time.",
            'Each time round the loop, add digit i times its weight to `total`. Digit i is a character, such as `"3"`, and you need the number. After the loop, the number passes when the remainder after dividing by 11 is 0: give back that comparison.',
            "`total = total + WEIGHTS[i] * ___` and `return total % 11 == 0`",
        ],
        why='`total = total + WEIGHTS[i] * int(number[i])` adds each digit times its weight, and `return total % 11 == 0` gives `True` when the remainder is 0. `eleven_test("111222333")` gives `True`: the total is 66, six times 11. `eleven_test("111223333")`, with one digit different, gives `False`.',
    )
    return


@app.cell(hide_code=True)
def text_after_eleven():
    mo.md(r"""
    With a right `eleven_test`, the grid under the check has 91 of its 1,000 squares filled: of the numbers from 111222000 to 111222999, about 1 in 11 passes, and the test cannot tell which of them, if any, belong to a person. The case number 202600038 in the table passes too.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hide():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. hide: one word at a time

        `hide(word)` gets one word of a message, without the punctuation around it. It gives back `"[NAME]"` when the word is one of the names in `NAMES`, `"[PHONE]"` when `is_phone(word)` says it is a mobile number, `"[BSN]"` when the word has nine characters that are all digits and pass the eleven test, and otherwise the word itself. `NAMES` holds the first and last names of the people who wrote the ten messages, as the municipality's own records have them. The box knows `NAMES`, `is_phone`, `is_number` and the course's `eleven_test`, so this exercise works even before your Exercise 1 is right.

        `word in NAMES` is `True` when the word is one of the items of the list. The start already hides names; add the two other rules before `return word`. Call `eleven_test` only on nine digits: on a shorter word, or one with a letter or a dash, it stops with an error, and of a longer number it reads only the first nine digits. An `if` inside an `if` asks two things in a row. The check calls `hide("111222333")`, then other words.
        """),
        L.task("Give back the placeholder for a name, a mobile number or a possible BSN, and every other word as it is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hide():
    box_hide = L.code_box("hide", lines=11)
    box_hide
    return (box_hide,)


@app.cell(hide_code=True)
def check_hide(box_hide):
    result_hide = L.check("hide", box_hide.value)
    result_hide
    return (result_hide,)


@app.cell(hide_code=True)
def hints_hide():
    L.hints(
        "hide",
        [
            "Step 2 of the recipe. The start shows the pattern: an `if` that asks something about `word`, and under it a `return` with the placeholder.",
            "First the phone: `is_phone` asks everything a mobile number needs. Then the BSN: ask whether the word has nine characters, then whether they are all digits, and only then whether they pass the eleven test, each `if` inside the one before.",
            '`if len(word) == 9:`, inside it `if is_number(word):`, inside that `if ___:` with `return "[BSN]"` under it',
        ],
        why='Under the rule for names come `if is_phone(word):` with `return "[PHONE]"`, and `if len(word) == 9:`, `if is_number(word):` and `if eleven_test(word):` with `return "[BSN]"`, each inside the one before. `hide("111222333")` gives `"[BSN]"`; `hide("202600041")` gives the case number back, because it fails the eleven test.',
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your hide on a message of your own

    Type a message in the box: with names from `NAMES`, numbers, or anything else. The picture shows it after your `hide` once Exercise 2 is right, and after the course's until then. Try a phone number with spaces in it, a BSN with one digit changed, or a name that is not in `NAMES`.
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.text_area(value=C.EXAMPLE_MESSAGE[L.lang], label="A message", rows=2, full_width=True)
    try_box
    return (try_box,)


@app.cell(hide_code=True)
def demo_try(box_hide, result_hide, try_box):
    _yours = C.mine(box_hide.value, result_hide.state == "correct")
    _out = mo.Html(C.try_picture(try_box.value, _yours, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_found():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. share_found: what the filter finds

        The course marked by hand which words of the ten messages are personal data: the 32 words in `PII`, such as `"Sanne"`, `"0612345678"` and `"06-23456789"`. The other numbers in the messages are in `OTHER`: `["45", "202600041", "3", "202600038", "2026"]`. `share_found(words)` gets a list `words` and gives back the share of them that `hide` hides: the number of words for which `hide(word)` gives back something else than the word, divided by the number of words. Inside, `found` starts at 0 and counts the words that `hide` changes. On `PII` that is the recall of the filter, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en); on `OTHER`, the share it hides by mistake. The box knows the course's `hide`. Replace the two `...`. The check calls `share_found(PII)`, then other lists.
        """),
        L.task("Give back the share of the words that hide changes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_found():
    box_found = L.code_box("share-found", lines=7)
    box_found
    return (box_found,)


@app.cell(hide_code=True)
def check_found(box_found):
    result_found = L.check("share-found", box_found.value)
    result_found
    return (result_found,)


@app.cell(hide_code=True)
def hints_found():
    L.hints(
        "share-found",
        [
            "Step 4 of the recipe. A word is hidden when what `hide` gives back is not the word itself.",
            "Count a word in `found` when `hide` changes it. After the loop, divide the count by the number of words in the list.",
            "`if ___:` and `return found / len(words)`",
        ],
        why="`if hide(word) != word:` counts the words that `hide` changes, and `return found / len(words)` gives their share. `share_found(PII)` gives `0.71875`: 23 of the 32. `share_found(OTHER)` gives `0.2`: 1 of the 5, the case number 202600038, which passes the eleven test by chance.",
    )
    return


@app.cell(hide_code=True)
def text_after_found():
    mo.md(r"""
    The course's filter finds 23 of the 32 pieces of personal data: a recall of 0.71875. What it misses, you saw under Exercise 2: a phone number with a dash or with +31, a BSN with dots in it, a BSN with a typing error, two names in lower case, the resident next door who is not in `NAMES`, and an e-mail address, for which it has no rule at all. And it hides one case number that is not personal data. A rule you add for one of these may hide more by mistake. Ten made-up messages say little about real ones, and Presidio's 0.707 and DEDUCE's 0.564 were measured on other texts and in other ways, so the three numbers do not compare. But none of them is 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.md(r"""
    ### Exercise 4. A number that passes

    In one of the messages, the nine digits 202600038 pass the eleven test.
    """)
    return


@app.cell(hide_code=True)
def choice_pass_cell():
    choice_pass = L.choice(
        "pass",
        question="What does that tell you about 202600038?",
        options={
            "real": "That it is the BSN of a person who exists",
            "right": "Only that it could be a BSN",
            "sender": "That it is the BSN of whoever wrote it",
        },
        correct="right",
        why={
            "right": "1 in 11 numbers of nine digits pass, and a BSN says nothing about the person; the government's own check also looks in the number register. 202600038 is a case number of the municipality, and it passes by chance.",
            "real": "A pass only says that the number could be a BSN: 1 in 11 numbers of nine digits pass. Whether someone has that number, only the register knows. See *What a pass tells you* in *The idea*.",
            "sender": "The test looks at the digits, not at who wrote them, and 1 in 11 numbers of nine digits pass anyway. See *What a pass tells you* in *The idea*.",
        },
    )
    choice_pass
    return (choice_pass,)


@app.cell(hide_code=True)
def text_exercise_anonymous():
    mo.md(r"""
    ### Exercise 5. Nothing left to find

    A better filter finds every name, phone number and BSN in the ten messages.
    """)
    return


@app.cell(hide_code=True)
def choice_anonymous_cell():
    choice_anonymous = L.choice(
        "anonymous",
        question="Are the ten messages then anonymous?",
        options={
            "yes": "Yes: no names or numbers are left in them",
            "right": "No: what they say can still point to people",
            "list": "Yes, once the municipality deletes NAMES as well",
        },
        correct="right",
        why={
            "right": "A filter hides the pieces it has rules for. The e-mail address is still there, and so is what the messages say: the resident next door who parks a van on the sidewalk, the person who moved last month. Anonymous means that nobody can be found again, by any means reasonably likely to be used, as in [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en).",
            "yes": "The filter hides only what it has rules for: the e-mail address is still there, and so is what the messages say. See the watch-out under *New words and symbols*.",
            "list": "Deleting `NAMES` does not change the messages: the e-mail address, and what the text says, still point to people. See the watch-out under *New words and symbols*.",
        },
    )
    choice_anonymous
    return (choice_anonymous,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en), you replaced names with codes in a table, and a few facts still singled people out. Here you hid names and numbers in free text before it goes into a pipeline: with a check digit that catches typing errors and tells a possible BSN from most other numbers, and with a measurement that shows what slips through. Something always does. In the next lesson, *Privacy by design*, the question becomes how to need fewer such filters: by collecting only what the purpose needs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_anonymous, choice_pass, result_eleven, result_found, result_hide):
    L.wrap_up(result_eleven, result_hide, result_found, choice_pass, choice_anonymous)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
