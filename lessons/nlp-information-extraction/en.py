# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-information-extraction. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-information-extraction", "en")
    C = L.checks   # the paragraphs of 48 Wikipedia pages in English and Dutch, the course's list of their dates, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en), and one about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_groups():
    quick_groups = L.choice(
        "quick-groups",
        question=r'In *Finding patterns: regular expressions*, a pattern with groups takes a clock time apart. What does `re.findall(r"(\d{1,2}):(\d\d)", "at 9:30 and 14:05")` give?',
        options={
            "whole": '`["9:30", "14:05"]`: the whole matches',
            "right": '`[("9", "30"), ("14", "05")]`',
            "numbers": "`[(9, 30), (14, 5)]`: the parts as whole numbers",
            "flat": '`["9", "30", "14", "05"]`: every part in one list of strings',
        },
        correct="right",
        why={
            "right": "With more than one group, `re.findall` gives a tuple for each match, with what each group matched, as strings: the 05 keeps its 0. This lesson makes rows out of such tuples.",
            "whole": "That is what `re.findall` gives for a pattern without groups. What changes when the pattern has two? See [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
            "numbers": "The parts come straight out of the text. What kind of value is a piece of text? See [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
            "flat": "`re.findall` keeps the parts of one match together. In what? See [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
        },
    )
    quick_groups
    return (quick_groups,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="A rule finds 50 dates in a text. 40 of them are dates of the text; the other 10 are not. The text holds 80 dates in all. What are the precision and the recall of the rule?",
        options={
            "swapped": "Precision 0.5, recall 0.8",
            "right": "Precision 0.8, recall 0.5",
            "share": "Precision 0.625, recall 0.8: 50 of the 80 dates, and 40 of the 50",
            "errors": "Precision 0.2, recall 0.5",
        },
        correct="right",
        why={
            "right": "Precision is the share of what the rule found that is right: 40 of 50, 0.8. Recall is the share of the dates in the text that it found: 40 of 80, 0.5. This lesson counts both for rules that take dates out of text.",
            "swapped": "Which of the two is about what the rule found, and which about what the text holds? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "share": "Precision and recall both count the 40 dates that are right. Out of what? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "errors": "0.2 is the share of the dates found that are wrong, 10 of 50. Which share is precision? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A text tells you facts, but a program can only sort, count or look them up once they stand in a table. Jurafsky and Martin open their chapter on information extraction with an analyst who follows airline stocks: for every announcement of a fare increase, the analyst needs the airline, the amount, the date, and the airlines that followed. The facts are in news reports like this one: "Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on flights to some cities also served by lower-cost carriers. American Airlines, a unit of AMR Corp., immediately matched the move …". They describe the task: "This process of information extraction (IE) turns the unstructured information embedded in texts into structured data, for example for populating a relational database to enable further processing."

    This lesson takes the dates out of 2,067 paragraphs from 48 pages of the English Wikipedia: the pages of the development set of SQuAD (Rajpurkar and others, 2016), a collection of questions about Wikipedia, with their Dutch from SQuAD-NL (de Vries and others, 2023), most of it machine translation. The course looked for every date in them that names a day, a month and a year, first with patterns like the ones this lesson writes, and then by hand at every place where a day and a month stand together: 245 dates, the same ones in both languages. With that list, you can count exactly how many dates a rule finds, misses and gets wrong.

    **An everyday example.** A friend mails that dinner is on 12 March at seven, and your mail app offers to put it in your calendar. It took the date and the time out of the text, and turned them into a day and an hour that the calendar understands.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_forms = L.guess(
        "guess",
        question="Of the 245 dates in the English paragraphs, how many are written month first, as in *February 7, 2016*, and not day first, as in *7 February 2016*?",
        options={
            "few": "About 10, from a few American pages",
            "fifty": "About 50",
            "right": "About 110",
            "all": "Nearly all: Wikipedia started in the United States and writes the way Americans do",
        },
        reveal="**108 of the 245: almost half.** Wikipedia's *Manual of Style* asks for one form within a page: \"The dates in the text of any one article should all have the same format (day-first or month-first).\" And it says which: \"Articles on topics with strong ties to a particular English-speaking country should generally use the more common date format for that country.\" So *Super Bowl 50*, *Apollo program* and *American Broadcasting Company* write month first, and *Martin Luther*, *Kenya* and *Doctor Who* day first; 28 of the 30 pages with dates keep to one form. The Dutch translation writes every date day first. A rule for one form finds about half of the English dates.",
    )
    guess_forms
    return (guess_forms,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Information extraction** turns free text into structured data: rows with fixed fields. Often you know beforehand which fields you want. Jurafsky and Martin: "The related task of template filling is to find recurring stereotypical events or situations in documents and fill in the template slots." A **template** holds the fields for one kind of event, and a **slot** is one of those fields. For the airline report, the template has four slots, and Jurafsky and Martin fill them with *United Airlines* as the lead airline, *$6* as the amount, *2006-10-26* as the effective date, and *American Airlines* as the follower.

    Three of those slots hold text taken straight from the report. The date does not: "These slot-fillers may consist of text segments extracted directly from the text, or concepts like times, amounts, or ontology entities that have been inferred through additional processing." The report only says *Friday*, and in a later sentence *Thursday*; which days those are follows from the day the report appeared. Real texts are often like that: "The task is difficult because fully qualified temporal expressions are fairly rare in real texts." This lesson keeps to dates that name a day, a month and a year.

    You can write the form of such a date as a regular expression, with a group for each field. `re.findall` then gives a tuple of strings for every date, and you turn each string into the value of its field: digits with `int`, the name of a month with a dictionary that gives its number. A vertical bar in a pattern means *or*: `(May|June)` matches *May* or *June*. The parentheses keep the choice inside the group; without them, the bar splits the whole pattern in two. That is how most systems for dates still work, Jurafsky and Martin write: "Most current approaches to temporal normalization are rule-based".

    A field of a table is written in one fixed form, so that a program can compare and sort its values. For dates, that form is the international standard **ISO 8601**: the year in four digits, the month in two, the day in two, with hyphens between them. Jurafsky and Martin, for the date July 2, 2007: "The ISO representation for this kind of expression is YYYY-MM-DD, or in this case, 2007-07-02." *23 November 1963* and *November 23, 1963* both become 1963-11-23.

    How good is a rule? You compare its rows with a list of the right rows, a **gold standard**. As in *Beyond accuracy: precision and recall*, a row that the rule finds and the list holds is a true positive, a row it finds that the list does not hold is a false positive, and a row of the list that it does not find is a false negative. Precision is the share of the rows found that are right, and recall the share of the right rows that were found. A row with one wrong field, such as the wrong day, counts twice: as a false positive, and the right row it should have been as a false negative.

    The picture shows four sentences from four pages, and the table the course makes of their dates.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Each date becomes one row, whatever its form: *July 20, 1969* and *9 October 2004* both get a year, a month and a day. The last column writes each row in the form of ISO 8601. The time in the third sentence, 20:17:40, is a fact too, but it has no slot in this table.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | Meaning |
    |---|---|
    | **information extraction** | turning free text into structured data, such as rows with fixed fields |
    | **template** | the fields you want for one kind of event or fact, such as the airline, the amount and the date of a fare increase |
    | **slot** | one field of a template |
    | **ISO 8601** | the international standard for writing dates: YYYY-MM-DD, such as 1963-11-23 |
    | **gold standard** | the list of right answers that the output of a rule is compared with |
    | `"-".join(d)` | joining a dictionary `d` joins its keys, here with `-` between them: `"-".join({"May": 5, "June": 6})` gives `"May-June"` |
    | `for day, month, year in re.findall(pattern, text)` | for a pattern with three groups, `re.findall` gives a tuple of three strings for each match, and the loop takes each tuple apart into three names |
    | `f"{n:02d}"` | an f-string that writes the whole number `n` with at least two digits, with a 0 in front when it has one: `f"{7:02d}"` gives `"07"` |
    | `C.MONTHS_EN`, `C.MONTHS_NL` | the month names of English and of Dutch with their numbers: `{"January": 1, …}` and `{"januari": 1, …}` |
    | `C.text(lang, i)` | paragraph `i` of the 2,067, counting from 0, in English (`"en"`) or Dutch (`"nl"`) |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Choose the fields of a row: here the year, the month and the day.
    2. Write a pattern with a group for each field, and look at what it finds in real text.
    3. Turn each group into the value of its field: digits with `int`, a name with a dictionary.
    4. Write each value in one fixed form, such as ISO 8601 for dates.
    5. Count the right, wrong and missed rows against a gold standard, and improve the pattern where it falls short.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code fills three slots of the airline template from the first two sentences of the report: the airlines with a group for the word before *Airlines*, and the amount with a group for the digits after the dollar sign. In a pattern, `\$` is a dollar sign itself; a `$` alone means the end of the text.
    """)
    return


@app.cell
def example_cell():
    import re

    _report = ("Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on "
               "flights to some cities also served by lower-cost carriers. American Airlines, a unit of AMR Corp., "
               "immediately matched the move")
    _airlines = re.findall(r"([A-Z][a-z]+) Airlines", _report)
    _amounts = re.findall(r"\$(\d+)", _report)
    _row = {"lead": _airlines[0] + " Airlines", "amount": int(_amounts[0]), "follower": _airlines[1] + " Airlines"}
    print(_airlines)
    print(_amounts)
    print(_row)
    return (re,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The pattern for the airlines is a guess that works here: a word with a capital before *Airlines*. *United* gets the slot of the lead airline because it comes first, and *American* that of the follower because it comes second: the rule knows nothing of who follows whom, only of the order in the text. The amount is a whole number now, 6, so a program can add it up. The effective date stays empty: the report says *Friday*, and in a later sentence *Thursday*, and which days those are depends on the date of the report, which the text does not give.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. date_rows: the dates of a text as rows

        Write `date_rows(text, months)`, where `text` is a string and `months` a dictionary of month names with their numbers, such as `{"January": 1, "February": 2, …}`. Give back a list with a row for each date in `text` that is written as a day, a month name and a year, such as *6 December 1989*, in the order of the text. A day is one or two digits and a year four, and neither is part of a longer number or word. A row is a tuple `(year, month, day)` of three whole numbers. The first line of the function joins the month names into `names`, such as `January|February|…`. Replace the two `...`: the pattern, with a group for the day, one for the month and one for the year, and the line that adds the row to `rows`.

        The check calls `date_rows(sentence_en, months_en)` and `date_rows(sentence_nl, months_nl)`, with the sentence *Doctor Who originally ran for 26 seasons on BBC One, from 23 November 1963 until 6 December 1989.*, its Dutch *Doctor Who liep oorspronkelijk 26 seizoenen op BBC One, van 23 november 1963 tot 6 december 1989.*, and the month names of English and of Dutch (in the lesson: `C.SENTENCE_EN`, `C.SENTENCE_NL`, `C.MONTHS_EN` and `C.MONTHS_NL`). Both give `[(1963, 11, 23), (1989, 12, 6)]`. Then cases of its own: a sentence with *1 May 1191*; one with *4 August 1915* and *November 1918*; one with *February 7, 2016*; one with *series 24 in 1991*; a Dutch sentence with *9 oktober 2004*; an empty text; a text with *3 Feb 2020* and *4 March 2020*, with the months `{"Jan": 1, "Feb": 2, "Mar": 3}`; and paragraphs 371 (*Martin Luther*) and 6 (*Super Bowl 50*), counting from 0, in English and in Dutch (in the lesson: `C.text("en", 371)` and so on).
        """),
        L.task("Take the dates of a text apart into rows of a year, a month and a day."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_rows = L.code_box("date_rows", lines=10)
    box_rows
    return (box_rows,)


@app.cell(hide_code=True)
def check_rows(box_rows):
    result_rows = L.check("date_rows", box_rows.value)
    result_rows
    return (result_rows,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "date_rows",
        [
            r"`pattern` is a raw string with three groups: one or two digits with `\d{1,2}`, then a space and a group around `names`, joined in with `+`, then a space and four digits, with `\b` at both ends. In the loop, `rows.append` adds a tuple: `int` turns `year` and `day` into whole numbers, and the dictionary `months` gives the number of the month name in `month`.",
            r'Put `names` between `r"\b(\d{1,2}) ("` and `r") (\d{4})\b"` with `+`: the parentheses around it make the choice of a month one group. Then the row: how do you look up the value of a key in a dictionary?',
            "`        rows.append((int(year), ___, int(day)))`",
        ],
        why=r'`pattern = r"\b(\d{1,2}) (" + names + r") (\d{4})\b"` and, in the loop, `rows.append((int(year), months[month], int(day)))`. The example gives `[(1963, 11, 23), (1989, 12, 6)]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_rows():
    mo.md(r"""
    On the English paragraphs, your rule finds 132 of the 245 dates and makes no wrong rows. It misses every date of *Super Bowl 50*, *Apollo program* and *American Broadcasting Company*: those pages write month first. In the Dutch translation, every date is written day first, and the same rule finds 239 of the 245.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. score: right, wrong and missed

        Write `score(found, gold)`, where `found` is the list of rows that a rule found and `gold` the list of the right rows, the gold standard. A row can be any tuple, and no row occurs twice in a list. Give back a tuple `(right, wrong, missed)`: how many rows of `found` are in `gold`, how many are not, and how many rows of `gold` are not in `found`. The first lines turn both lists into sets, so that `in` is quick on long lists. Replace the two `...`.

        The check calls `score(found_rows, gold_rows)`, with the rows `[(1963, 11, 23), (1989, 12, 7), (2005, 3, 26)]` and the right rows `[(1963, 11, 23), (1989, 12, 6), (2005, 3, 26), (2008, 7, 27)]` (in the lesson: `C.FOUND_ROWS` and `C.GOLD_ROWS`). It gives `(2, 1, 2)`: the row with the 7th of December is wrong, and the 6th of December is missed, as is the row of 2008. Then cases of its own: nothing found; no right rows; both lists empty; every right row found; two rows against three; and, as rows `(paragraph, year, month, day)` against the course's 245 dates, the rows of the course's rule for day first on all English paragraphs and on all Dutch ones, and of a looser rule on the English ones.
        """),
        L.task("Count the right, wrong and missed rows of a rule against the gold standard."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=8)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "In the loop: `if row in gold:` adds 1 to `right`. After the loop, `return` gives three numbers: `right`; the rows of `found` that are not right, `len(found) - right`; and the rows of `gold` that were not found.",
            "`right` counts the rows of `found` that are also in `gold`, and every other row of `found` is wrong. How many rows of `gold` are left that the rule did not find?",
            "`    return right, len(found) - right, ___`",
        ],
        why="In the loop: `if row in gold:` with `right = right + 1` under it, and after the loop `return right, len(found) - right, len(gold) - right`. The example gives `(2, 1, 2)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    The rule for day first has a precision of 1.00 on the English paragraphs: every row it finds is right. Its recall is 0.54: it misses the 108 dates written month first, and 5 written in other ways, such as *28 January to 25 May 1521*, where one year serves two days. A looser rule, with any word between the day and the year, finds the same dates and adds 7 wrong rows, such as *5 in 1970* from *13.5 in 1970*: a precision of 0.95. A second pattern for month first lifts the recall to 0.98. In the Dutch paragraphs, the rule for day first already has a recall of 0.98, and the second pattern changes nothing.

    These numbers say as much about the list as about the rules: the course found 240 of its 245 dates with these same two patterns, and the other 5 by going through every place where a day and a month stand together by hand. A gold standard made with a rule only shows what that rule misses once people check it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_iso():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. iso_dates: one fixed form

        Write `iso_dates(rows)`, where `rows` is a list of rows `(year, month, day)`. Give back a list with each row written as a date of ISO 8601, a text such as `"1969-07-20"`, sorted. Replace the two `...`.

        The check calls `iso_dates(apollo_rows)`, with three dates of the page *Apollo program*: `[(1969, 7, 20), (1961, 5, 25), (1967, 1, 27)]` (in the lesson: `C.APOLLO_ROWS`). It gives `["1961-05-25", "1967-01-27", "1969-07-20"]`. Then cases of its own: no rows; one row; three dates of *Super Bowl 50* in 2012 and 2013; the 15 dates of *Apollo program*; and all 245 dates of the course.
        """),
        L.task("Write rows as dates of ISO 8601, in order."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iso():
    box_iso = L.code_box("iso_dates", lines=6)
    box_iso
    return (box_iso,)


@app.cell(hide_code=True)
def check_iso(box_iso):
    result_iso = L.check("iso_dates", box_iso.value)
    result_iso
    return (result_iso,)


@app.cell(hide_code=True)
def hints_iso():
    L.hints(
        "iso_dates",
        [
            "In the loop: `dates.append` adds an f-string with `year`, `month` and `day`, a hyphen between each two, and `:02d` after each name that should get two digits. After the loop, `return` gives the texts in order with `sorted`.",
            "ISO 8601 writes the month and the day with two digits each: 7 becomes 07. Text sorts by its first character, then by the next one. Why does that put dates of this form in the order of time?",
            '`        dates.append(f"{year}-{month:02d}-{___}")`',
        ],
        why='In the loop: `dates.append(f"{year}-{month:02d}-{day:02d}")`, and after it `return sorted(dates)`. The example gives `["1961-05-25", "1967-01-27", "1969-07-20"]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_iso():
    mo.md(r"""
    Sorted as text, the dates as the page writes them start with *April 12, 1961* and *April 4, 1968* and end with *September 19, 1961*: the order of the alphabet, not of time. Your ISO dates go from 1961-04-12 to 1974-02-08. In the form of ISO 8601, sorting the texts is the same as sorting the days, since the largest unit comes first and every part has a fixed number of digits.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: a pattern of your own

    The box starts with a pattern for a day, any word and a year. The picture compares what it finds in the 2,067 paragraphs of your language with the course's 245 dates: a date counts as found when one match holds it whole, and a match that holds no date is wrong. Jurafsky and Martin warn about rules for times and dates: "A major difficulty for all of these very lexicalized approaches is avoiding expressions that trigger false positives", such as the 1984 of "1984 tells the story of Winston Smith...", which is a book. Can you make the wrong matches go away without losing dates? In English, can you also find the dates written month first? Which dates does no pattern of yours find, and why?
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    pattern_box = mo.ui.text(value=C.PATTERN_START, label="Pattern", full_width=True)
    pattern_box
    return (pattern_box,)


@app.cell(hide_code=True)
def try_look(pattern_box):
    mo.Html(C.pattern_picture(pattern_box.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_names():
    mo.md(r"""
    ### Exercise 4. Dates and names

    Dates have a few fixed forms, so a pattern can describe them. Try the same for the names of people: a pattern for a run of two or more words that each start with a capital and go on in small letters, such as *Nikola Tesla*. In the 11,722 English sentences of Universal NER (Mayhew and others, 2024), people marked every name of a person: 2,028 names. The pattern finds 3,000 runs, and 510 of them are names of people: a precision of 0.17, and a recall of 0.25.
    """)
    return


@app.cell(hide_code=True)
def choice_names_cell():
    choice_names = L.choice(
        "names",
        question="Why does a pattern work so much better for dates than for the names of people?",
        options={
            "case": "`re` cannot tell a capital from a small letter",
            "right": "A name has no fixed form: capitals also start places, firms and sentences, and many names are one word",
            "rare": "Names are rarer in text than dates",
            "marks": "The pattern is right, but the people who marked the sentences missed most names, so its right runs count as wrong",
        },
        correct="right",
        why={
            "right": "A date is a number from 1 to 31, one of twelve month names and four digits; a name has no such form. *Iguazu Falls* and *Buenos Aires* look like names and are places, and *Callum* is a name of one word. Whether a run of words is a name depends on the words around it. The lesson [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en) learns that from examples, with a model.",
            "case": "`[A-Z]` matches only capitals, and `[a-z]` only small letters. Where do the wrong runs come from, then?",
            "rare": "2,028 names in 11,722 sentences is plenty to find. Why are most of the runs not names?",
            "marks": "A mark can be wrong now and then, but *Iguazu Falls* is a place, and the pattern takes it all the same. What does a capital tell you about a word?",
        },
    )
    choice_names
    return (choice_names,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en), a pattern found pieces of text. Here each group of a pattern became a field of a row, and the rows became a table: information extraction with rules. Your `date_rows` took the dates of a text apart into years, months and days, your `score` counted right, wrong and missed rows against a gold standard, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), and your `iso_dates` wrote them in the one form of ISO 8601, in which sorting the texts sorts the days. A rule reaches as far as its form: almost half of the English dates were written in another form, and the names of people have no form that a pattern can catch. The next lesson, *Keywords and terms*, finds the words that say what a document is about.

    **An everyday example.** An app for your expenses reads the receipts you photograph: the date, the shop and the amount go into its columns. A receipt that writes the date in another way lands on the wrong day, or does not show up at all.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_names, result_iso, result_rows, result_score):
    L.wrap_up(result_rows, result_score, result_iso, choice_names)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
