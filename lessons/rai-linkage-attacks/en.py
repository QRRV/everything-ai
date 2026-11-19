# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-linkage-attacks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-linkage-attacks", "en")
    C = L.checks   # a made-up town: its voter list VOTERS, the released visits VISITS, the whole town PEOPLE, and QI


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Hiding in a group: k-anonymity](/course/open/rai-k-anonymity?lang=en) and [Two tables into one: merge](/course/open/data-merge?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_k():
    quick_k = L.choice(
        "quick-k",
        question="Grouped by ZIP code, birth year and sex, a table has three groups: of 4, 9 and 2 rows. What is the largest k for which it is k-anonymous?",
        options={
            "right": "k = 2, the smallest group",
            "largest": "k = 9, the largest group",
            "all": "k = 15, all rows taken together",
        },
        correct="right",
        why={
            "right": "Every combination of values must occur in at least k rows, so the smallest group sets k: 2.",
            "largest": "k-anonymity asks that every group has at least k rows, so the smallest group sets k, not the largest. See *Hiding in a group* in *The idea* of [Hiding in a group: k-anonymity](/course/open/rai-k-anonymity?lang=en).",
            "all": "k is about the groups, not about the whole table: the smallest group has 2 rows. See *Hiding in a group* in *The idea* of [Hiding in a group: k-anonymity](/course/open/rai-k-anonymity?lang=en).",
        },
    )
    quick_k
    return (quick_k,)


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="Tuesday is once in `left` and twice in `right`. How many Tuesday rows does `pd.merge(left, right, on=\"day\")` give?",
        options={
            "first": "One: merge keeps the first match",
            "right": "Two, one for each match",
            "none": "None: the key must be unique",
        },
        correct="right",
        why={
            "right": "Every pair of rows with the same key becomes one row, so Tuesday comes out twice. That is why you count the rows before and after a merge.",
            "first": "merge keeps every match: each pair of rows with the same key becomes one row. See *The idea* of [Two tables into one: merge](/course/open/data-merge?lang=en).",
            "none": "A key may occur more than once; each pair of rows with the same key becomes one row. See *The idea* of [Two tables into one: merge](/course/open/data-merge?lang=en).",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def quick_check_inner():
    quick_inner = L.choice(
        "quick-inner",
        question="Monday is in `left` but not in `right`. Is there a Monday row in `pd.merge(left, right, on=\"day\")`?",
        options={
            "missing": "Yes, with missing values on the right side",
            "right": "No: an inner join keeps only matched keys",
            "longer": "Yes, if `right` has more rows",
        },
        correct="right",
        why={
            "right": "Without `how`, merge makes an inner join: a key that is not in both tables is left out, without a word.",
            "missing": "That is what `how=\"left\"` does. Without `how`, merge makes an inner join, which leaves Monday out. See *New words and symbols* in [Two tables into one: merge](/course/open/data-merge?lang=en).",
            "longer": "The number of rows does not matter: without `how`, merge makes an inner join, which leaves out a key that is not in both. See *New words and symbols* in [Two tables into one: merge](/course/open/data-merge?lang=en).",
        },
    )
    quick_inner
    return (quick_inner,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A table without names can still say who is in it, once someone puts it next to a table with names. Latanya Sweeney showed this in the 1990s by putting the voter list of the city of Cambridge next to medical data that a Massachusetts state agency had released about state employees. Looked at again later, the case is more subtle than its short version: a voter list does not hold everyone, so a name that fits is not always the right name. In this lesson you do both in a made-up town: you link a released table of 1,000 visits to the town's voter list, count the visits that get a name, and then see, with what only the course knows, how many of those names are wrong.

    **An everyday example.** A running club puts the results of a race online without names: only age group, sex and finishing time. A neighbor's running app shows a run that morning, on the same course, with the same time. Put the two side by side, and one result has a name.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="The voter list of Cambridge, Massachusetts, that Sweeney bought held 54,805 voters, with their birth date and one of five ZIP codes. How many of them had a combination of birth date and ZIP code that no other voter on the list had?",
        options={
            "few": "Hardly anyone",
            "half": "Fewer than half",
            "right": "About 7 in 10",
        },
        reveal="**About 7 in 10**: 69%, according to Pierangela Samarati and Latanya Sweeney (1998). With sex too, Sweeney (2000) counts 2 × 5 × 365 × 100 = 365,000 possible combinations over a hundred years of birth dates, for only 54,805 voters. In Exercise 1 you measure the same thing on a made-up voter list.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### Two tables, one key

        Sweeney (2002) describes a common practice: data about people are released "with all explicit identifiers, such as name, address and telephone number, removed on the assumption that anonymity is maintained because the resulting data look anonymous." In Massachusetts, the Group Insurance Commission (GIC) bought health insurance for state employees. Its data held "nearly one hundred attributes per encounter" for "approximately 135,000 state employees and their families", and because the data were believed to be anonymous, "GIC gave a copy of the data to researchers and sold a copy to industry". Then Sweeney writes: "For twenty dollars I purchased the voter registration list for Cambridge Massachusetts and received the information on two diskettes". The list held the name, address, ZIP code, birth date and sex of each voter, and "This information can be linked using ZIP code, birth date and gender to the medical information".

        The governor of Massachusetts at that time, William Weld, lived in Cambridge, and Weld's medical records were in the GIC data. According to the voter list, six people had Weld's birth date, three of them were men, and only one of those lived in Weld's ZIP code.

        That is a **linkage attack**: you join a released table with a second table that has names and shares a quasi-identifier, here ZIP code, birth date and sex. In pandas it is a merge on all three: `pd.merge(released, voters, on=["zip", "birth", "sex"])`. A row matches only when all three columns are the same. A released row that matches exactly one voter gets that voter's name; a row that matches two voters comes out twice, with two candidates; a row that matches none drops out, as in every inner join.
        """),
        mo.Html(C.link_picture("en")),
        mo.md(r"""
        ### Why so many are alone

        Sweeney (2000) explains why the voter list was so revealing: "there were 2 possible values for gender and 5 possible five-digit ZIP codes; birth dates were within a range of 365 days for 100 years. This gives 365,000 unique values, but there were only 54,805 voters." With far more combinations than people, many people have a combination to themselves: they are **unique**, alone on those columns. Samarati and Sweeney (1998), citing Sweeney, give the numbers for the list: of the 54,805 voters, "12% had unique birth dates, 29% were unique with respect to birth date and gender, 69% with respect to birth date and a 5-digit ZIP code", and 97% were identifiable with the full postal code and birth date.

        For the whole United States, Sweeney (2000) estimated that "87% (216 million of 248 million)" of the population "had reported characteristics that likely made them unique based only on {5-digit ZIP, gender, date of birth}". The estimate comes from 1990 census counts, not from names: per ZIP code, the census gives the number of people in each age band, and where a band holds no more people than there are combinations of sex and birth date in it, Sweeney counts all of them as likely alone. Philippe Golle (2006) counted again from the census, with the assumption "that births are uniformly distributed across the days of the year", and found 61% for 1990 and 63% for 2000. Both numbers count people who are likely alone, not people who were named; Sweeney's paper itself warns that knowing that one or a few people share some values is not exactly the same as saying who they are.

        A coarser birth date leaves almost nobody alone. With ZIP code, sex and only the month and year of birth, Sweeney's table gives 3.7% likely unique; with only the year, 0.04%. Golle's table for 2000 gives 4.2% and 0.2%.

        ### A name from an incomplete list

        Daniel Barth-Jones (2012) looked at the case again. The voter list held 54,805 people, while Cambridge had about 99,435 inhabitants in 1996 and 1997, by Barth-Jones's estimate from the censuses of 1990 and 2000: the list held 55% of the city. Of the about 88,000 people old enough to vote, some 31,000 (35%) were not on it. So "many of the individuals who looked unique within the voter list would not actually have been the only person in Cambridge to have their particular set of birth date, gender and ZIP code characteristics." Barth-Jones calls the assumption that a list holds everyone "the myth of the perfect population register": a voter list is no **population register**, a list of everyone who lives somewhere.

        The census of 2000 counted 174 men aged 50 in Weld's ZIP code. Assuming the same number in 1997, and birthdays spread evenly over the year, as Golle assumed, Barth-Jones finds only a 62 percent chance that Weld was the only one of them with that birthday. A short rule gives the same number: each of the other 173 men has a birthday on another day with chance 364/365. One man's birthday does not depend on another's, so the chance that all 173 have another day is the product of their chances: 364/365 multiplied by itself 173 times, (364/365)<sup>173</sup>, about 0.62. With 150 men, the fewest Barth-Jones thinks likely, the rule gives 66%. You write that rule in Exercise 3.

        Barth-Jones argues that Weld was most likely re-identifiable only because Weld was a public figure whose hospital stay was in the news: Weld collapsed at a college commencement in May 1996. And on the voter list: "The whole process of linkage to the voters list was superfluous to Weld's re-identification—it was not necessary and, more importantly, it was not sufficient." That does not make stripped data safe, and Barth-Jones does not say it does: "We must pay attention to the potential for re-identification when data has only been stripped of the directly identifying information such as names and addresses."
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
        | **linkage attack** | putting names on a released table by joining it with a second table that has names and shares a quasi-identifier, such as a voter list |
        | **unique** | the only row with its values in some columns: alone on those columns |
        | **population register** | a list of everyone who lives somewhere; a voter list is not one, because not everyone is on it |
        | **`on=[...]`** | a merge on several columns: a row matches only when all of them are the same, as in `pd.merge(visits, voters, on=["zip", "birth", "sex"])` |
        | **(364/365)<sup>n−1</sup>** | the chance that none of n − 1 other people has your birthday, when birthdays are spread evenly over 365 days |
        """),
        L.watch_out("Unique is not yet identified: a row that is alone in a table gets a name only when a second table with names holds the same values. And a single match in a list that does not hold everyone gives a name that is likely, not certain, to be right. Two details of the Weld story often get mixed up: 54,805 is the number of voters, not of inhabitants; and the story that Sweeney sent the governor's health records to the governor's office is not in Sweeney's paper of 2002: it comes from a retelling by Paul Ohm (2010), who cites another author for it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out how many rows of a released table someone could put a name on:

    1. Find the columns the released table shares with a table that has names, such as ZIP code, birth date and sex.
    2. Measure how many people are alone on those columns: they are the ones a match can single out. That is `share_alone`.
    3. Merge on all of those columns, and count per released row how many named rows match it: a row with exactly one gets a name. That is `linked`.
    4. Before you trust such a name, ask whether the named table holds everyone. If it does not, work out the chance that the match is alone in the whole population. That is `alone`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course made up a town of 30,000 adults, with a voter list and the released table of a health insurer: nobody in it exists. The cell reads both tables, merges them on ZIP code, birth date and sex, and counts what comes out.
    """)
    return


@app.cell
def example_link():
    import pandas as pd

    _visits = pd.read_csv("data/course-town-visits.csv.gz")
    _voters = pd.read_csv("data/course-town-voters.csv.gz")
    print(len(_visits), "visits and", len(_voters), "voters")
    # every visit next to each voter with the same ZIP code, birth date and sex
    _both = pd.merge(_visits, _voters, on=["zip", "birth", "sex"])
    print(_both.head(3))
    print("rows after the merge:", len(_both))
    print("visits with a match:", _both["visit"].nunique())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The released table has no names, yet the first rows of the merge put a diagnosis next to a name. The merge has 805 rows for 711 visits: 289 of the 1,000 visits match no voter and drop out, and some visits match more than one voter, so they come out more than once. Which of the 711 get exactly one name, you count in Exercise 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. share_alone: how many are alone

        `VOTERS` is the voter list of the made-up town: 19,656 rows with the columns `name`, `zip`, `birth` (a date as text, such as `"1961-04-12"`) and `sex`. `PEOPLE` is the whole town as only the course knows it: 30,000 people, with the same columns and two more, `voter` (`True` for the people on the voter list) and `visit` (the id of their visit in the released table, or empty). `QI` is the list `["zip", "birth", "sex"]`.

        `share_alone(table, columns)` gets a table `table` and a list of column names `columns`. It gives back the share of the rows that are the only one with their values in `columns`: a number from 0 to 1. Replace the `...` after `sizes =` with the size of each group of rows with the same values in `columns`, and the `...` after `return` with the share of the rows whose group has one row. The check calls `share_alone(VOTERS, QI)`, then other columns, `PEOPLE`, and the small voter list `FEW_VOTERS` of the picture in *The idea*.
        """),
        L.task("Give back the share of the rows that are the only one with their values in columns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share_alone = L.code_box("share_alone", lines=4)
    box_share_alone
    return (box_share_alone,)


@app.cell(hide_code=True)
def check_share(box_share_alone):
    result_share_alone = L.check("share_alone", box_share_alone.value)
    result_share_alone
    return (result_share_alone,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share_alone",
        [
            "Step 2 of the recipe. The sizes of the groups you know from `k_of`; now you count the groups of size 1, and turn that count into a share of all rows.",
            "`sizes == 1` gives `True` for each group of one row, and `.sum()` counts the `True` values, as in [Asking questions of data](/course/open/data-explore?lang=en). Which function gives the number of rows of `table`?",
            "`sizes = table.groupby(columns).size()` and `return (sizes == 1).sum() / ___`",
        ],
        why="`sizes = table.groupby(columns).size()` gives the size of every group, and `return (sizes == 1).sum() / len(table)` the share of the rows that are alone. `share_alone(VOTERS, QI)` gives 0.86: 86% of the voters are the only one on the list with their ZIP code, birth date and sex.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_linked():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. linked: the rows that get a name

        `VISITS` is the released table of a made-up health insurer in the same town: 1,000 rows with the columns `visit` (an id from `"V0001"` to `"V1000"`), `zip`, `birth`, `sex` and `diagnosis`, and no names.

        `linked(released, voters, columns)` gets a released table `released` with a column `visit`, a voter list `voters`, and a list of column names `columns`. It gives back how many rows of `released` match exactly one row of `voters` on all of `columns`: the rows that get a name. Replace the three `...`: `both` is the merge of `released` and `voters` on `columns`, `per_visit` the number of rows of each visit in `both`, and the last line counts the visits with exactly one. The check calls `linked(VISITS, VOTERS, QI)`, then fewer columns, and the small tables `FEW_VISITS` and `FEW_VOTERS` of the picture in *The idea*.
        """),
        L.task("Give back how many rows of released match exactly one row of voters on columns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_linked():
    box_linked = L.code_box("linked", lines=8)
    box_linked
    return (box_linked,)


@app.cell(hide_code=True)
def check_linked(box_linked):
    result_linked = L.check("linked", box_linked.value)
    result_linked
    return (result_linked,)


@app.cell(hide_code=True)
def hints_linked():
    L.hints(
        "linked",
        [
            "Step 3 of the recipe. A visit that fits two voters is in the merge twice, so count the rows of each visit, and then the visits with exactly one row.",
            "`pd.merge` takes a list of columns with `on=`, as in the worked example. For the count per visit, group `both` by one of its columns and take `.size()`; `.sum()` of a comparison counts its `True` values.",
            "`both = pd.merge(released, voters, on=columns)`, `per_visit = both.groupby(___).size()` and `return (per_visit == 1).sum()`",
        ],
        why="`both = pd.merge(released, voters, on=columns)` puts each visit next to every voter that fits, `per_visit = both.groupby(\"visit\").size()` counts them per visit, and `return (per_visit == 1).sum()` counts the visits with exactly one. `linked(VISITS, VOTERS, QI)` gives 622: 622 of the 1,000 visits get a name.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_alone():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. alone: the chance that a match is the only one

        `alone(n, values)` gets a whole number `n`: a number of people who share everything but one thing, such as the men aged 50 in one ZIP code, who differ only in their birthday. `values` is the number of values that this one thing can take, all equally likely, such as 365 birthdays. It gives back the chance that one of the n people is the only one with their value: that none of the other n − 1 has it. `miss` is the chance that one other person has another value than yours. Replace the first `...` with that chance, and the second with the chance that all n − 1 others miss. The check calls `alone(174, 365)`, then other numbers.
        """),
        L.task("Give back the chance that none of the other n − 1 people has your value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_alone():
    box_alone = L.code_box("alone", lines=4)
    box_alone
    return (box_alone,)


@app.cell(hide_code=True)
def check_alone(box_alone):
    result_alone = L.check("alone", box_alone.value)
    result_alone
    return (result_alone,)


@app.cell(hide_code=True)
def hints_alone():
    L.hints(
        "alone",
        [
            "Step 4 of the recipe, and the rule in *A name from an incomplete list*: with 365 birthdays, one other person misses yours with chance 364/365.",
            "Chances that must all come true are multiplied: `miss` times itself, once for each other person. `**` is to the power of, as in [Variables and numbers](/course/open/py-variables?lang=en).",
            "`miss = (values - 1) / values` and `return miss ** ___`",
        ],
        why="`miss = (values - 1) / values` is the chance that one other person misses your value, and `return miss ** (n - 1)` the chance that all n − 1 do. `alone(174, 365)` gives 0.62: the 62% of Barth-Jones.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_voter_list():
    mo.md(r"""
    ### Exercise 4. What the list showed

    On the Cambridge voter list, only one man had the governor's birth date and ZIP code: Weld.
    """)
    return


@app.cell(hide_code=True)
def choice_voter_list_cell():
    choice_voter_list = L.choice(
        "voter-list",
        question="What did that show?",
        options={
            "sure": "That the matching record was surely the governor's",
            "right": "Only that no other voter matched those values",
            "nothing": "Nothing: voter lists hold no birth dates",
        },
        correct="right",
        why={
            "right": "The list held 54,805 of about 99,435 people in Cambridge, by Barth-Jones's estimate, so a man of 50 with the same birthday who was not on it would not show. With 174 men aged 50 in that ZIP code, `alone(174, 365)` gives 0.62: Barth-Jones's chance that Weld was the only one.",
            "sure": "The list was not the whole city: of the about 88,000 people old enough to vote, some 31,000 were not on it. Someone with the same values could be missing from it. See *A name from an incomplete list* in *The idea*.",
            "nothing": "Sweeney's list held the name, address, ZIP code, birth date and sex of each voter. The question is what a single match on it shows. See *A name from an incomplete list* in *The idea*.",
        },
    )
    choice_voter_list
    return (choice_voter_list,)


@app.cell(hide_code=True)
def text_exercise_estimate():
    mo.md(r"""
    ### Exercise 5. What 87% counts

    Sweeney (2000) estimated that 87% of the people in the US in 1990 were likely the only ones with their ZIP code, sex and birth date. Golle (2006) counted again for 1990 and found 61%.
    """)
    return


@app.cell(hide_code=True)
def choice_estimate_cell():
    choice_estimate = L.choice(
        "estimate",
        question="What do these two numbers count?",
        options={
            "named": "People whose records were linked to a name",
            "right": "People likely unique, from census counts",
            "voters": "Voters whose names were on a public list",
        },
        correct="right",
        why={
            "right": "Both are estimates, from census counts of people per ZIP code and age, of how many are likely alone on those values. Nobody was named. A person who is alone still gets a name only when a second table with names holds the same values.",
            "named": "Nobody was named: both numbers are estimates from census counts of how many people are likely alone. A name needs a second table that holds the same values. See *Why so many are alone* in *The idea*.",
            "voters": "The numbers are about the whole population in the census, not about a voter list, and they count people who are likely alone, not names. See *Why so many are alone* in *The idea*.",
        },
    )
    choice_estimate
    return (choice_estimate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Hiding in a group: k-anonymity](/course/open/rai-k-anonymity?lang=en), a quasi-identifier made the groups; here it is the key of a merge, and every row that is alone on it can get a name from a second table. A coarser birth date makes the groups larger and the single matches fewer, as coarsening did in k-anonymity; a list that does not hold everyone makes some of the remaining names wrong. All of this works on the data after it is released. In the next lesson, *Answering with a coin*, the protection comes earlier: each person adds chance to their own answer before it is collected.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_estimate, choice_voter_list, result_alone, result_linked, result_share_alone):
    L.wrap_up(result_share_alone, result_linked, result_alone, choice_voter_list, choice_estimate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
