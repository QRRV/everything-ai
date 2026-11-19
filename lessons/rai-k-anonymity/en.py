# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-k-anonymity. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-k-anonymity", "en")
    C = L.checks   # UCI Adult as PEOPLE and BANDED, its five quasi-identifiers as QI, and a made-up clinic


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en) and [One number per group: groupby](/course/open/data-groupby?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hash():
    quick_hash = L.choice(
        "quick-hash",
        question="An app replaces every e-mail address in its rows with the SHA-256 hash of that address, and shares the rows. Is the result anonymous?",
        options={
            "yes": "Yes: a hash cannot be turned back into text",
            "right": "No: hash likely addresses and compare",
            "key": "Yes, as long as no key list is kept",
        },
        correct="right",
        why={
            "right": "Whoever can guess the addresses hashes them and compares, as `unmask` did with the café's list. That is pseudonymization: the rows stay personal data.",
            "yes": "There is no need to turn a hash back: whoever can guess the addresses hashes them and compares. See *A hash is not a hiding place* in *The idea* of [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en).",
            "key": "A plain hash needs no key list: whoever can guess the addresses hashes them and compares. See *A hash is not a hiding place* in *The idea* of [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en).",
        },
    )
    quick_hash
    return (quick_hash,)


@app.cell(hide_code=True)
def quick_check_size():
    quick_size = L.choice(
        "quick-size",
        question="`df` is the table of 344 penguins. What does `df.groupby(\"species\").size()` give?",
        options={
            "right": "The number of rows of each species",
            "groups": "How many species there are",
            "mean": "The mean of every column, per species",
        },
        correct="right",
        why={
            "right": "`size` gives, per group, how many rows it has: one number for each species.",
            "groups": "`size` gives one number per group: how many rows that group has. How many groups there are is the length of that result. See *New words and symbols* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "mean": "A mean needs `.mean()`; `size` counts the rows of each group. See *New words and symbols* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
        },
    )
    quick_size
    return (quick_size,)


@app.cell(hide_code=True)
def quick_check_pairs():
    quick_pairs = L.choice(
        "quick-pairs",
        question="What does `df.groupby([\"island\", \"species\"]).size()` give?",
        options={
            "apart": "One count per island, and one per species",
            "right": "One count per island-and-species pair",
            "first": "One count per island; species is left out",
        },
        correct="right",
        why={
            "right": "With a list of columns, a group is a combination of values that occurs, such as Biscoe and Gentoo; `size` counts the rows of each such pair.",
            "apart": "With a list of columns, the groups are the combinations of values, not each column on its own: Biscoe and Gentoo is one group. See step 4 of the *Recipe* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "first": "Both columns make the groups: every combination of island and species that occurs is one group. See step 4 of the *Recipe* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
        },
    )
    quick_pairs
    return (quick_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A common way to share data about people safely is to hide each of them in a crowd: as long as every combination of facts occurs often enough, nobody can be picked out. That idea is called k-anonymity, and it is one of the best-known answers to the problem of the last lesson, where three facts were enough to single out one row. It is easy to measure and to reach: you count, coarsen and leave out. It also has a well-known gap: when everyone in a group shares the same secret, hiding in the group does not protect the secret. In this lesson you measure both on UCI Adult, a table of 48,842 people: k, the size of the smallest group, and l, the smallest number of different secrets in a group.

    **An everyday example.** A company runs a staff survey, and shows the answers only for teams of at least five people, so that no single answer can be traced back. But when all five in a team answer no to "Are you happy with your manager?", the manager knows each of their answers anyway.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="UCI Adult holds 48,842 people from a US survey of 1994. Look only at their age, sex, race, marital status and education. How many of them are the only one in the table with their combination of those five?",
        options={
            "none": "Not a single one",
            "few": "About 400",
            "right": "About 4,000",
        },
        reveal="**About 4,000**: 3,948 people, 8% of the table, are the only one with their age, sex, race, marital status and education. For each of them, those five facts single out one row. You compute the smallest group yourself in Exercise 1.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### Hiding in a group

        In [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en), three facts about Bo left one row: Bo was singled out. Latanya Sweeney (2002) turned that into a demand on a whole table: "A release provides k-anonymity protection if the information for each person contained in the release cannot be distinguished from at least k-1 individuals whose information also appears in the release." What matters are the facts that someone could also know from elsewhere, such as age, ZIP code or sex. Together, such columns are a **quasi-identifier**, a term Sweeney takes from Dalenius. **k-anonymity** asks that every combination of values of the quasi-identifier occurs in at least k rows. In pandas words: group the rows by the quasi-identifier, and the smallest group has at least k rows. Whoever knows only someone's quasi-identifier then finds at least k rows that fit, as `candidates` would, and cannot tell which one is theirs.

        The idea is older than 2002. Sweeney's paper "significantly amends and substantially expands" a paper with Pierangela Samarati, submitted in 1998, and Sweeney credits Samarati with the name. That earlier paper says why it is needed: "Shockingly, there remains a common incorrect belief that if the data looks anonymous, it is anonymous."
        """),
        mo.Html(C.clinic_picture("en")),
        mo.md(r"""
        ### Coarsen or leave out

        A table is rarely k-anonymous by itself. In a second paper of 2002, Sweeney names two ways to get there: "Generalization involves replacing (or recoding) a value with a less specific but semantically consistent value. Suppression involves not releasing a value at all." That paper's example of **generalization**: "the original ZIP codes {02138, 02139} can be generalized to 0213*". In the picture, each age became its ten years and each ZIP code lost its last digit; **suppression** would leave a value out, or a whole row whose group is still too small.

        Both cost something: a coarser value says less, and a row left out is lost to research. For a study of students and taxes in Estonia, Bogdanov et al. (2016) write that "the use of 3-anonymity caused 10%-30% of sample loss, depending on the demographic group"; the study used another way, which comes back in *Computing on secrets*. On UCI Adult, with age, sex, race, marital status and education, 9,810 of the 48,842 people are in a group of fewer than 5: making the table 5-anonymous by only leaving rows out would drop 20% of it.

        ### What k does not promise

        Sweeney's own paper says so: "This property does not guarantee individuals cannot be identified in RT; there may exist other inference attacks that could reveal the identities of the individuals contained in the data", with RT the released table. And k-anonymity rests on an assumption: that the data holder "can accurately identify quasi-identifiers". A column left out of the quasi-identifier, because nobody thought someone could know it, is a way in. The same paper describes three attacks on tables that are k-anonymous: one on the order of the rows (the fix is to shuffle them), one that joins a second release of the same data, coarsened in another way, and one on data that changes over time.

        ### When the whole group shares the secret

        Ashwin Machanavajjhala, Johannes Gehrke, Daniel Kifer and Muthuramakrishnan Venkitasubramaniam (2006) showed two attacks on k-anonymity: "First, we show that an attacker can discover the values of sensitive attributes when there is little diversity in those sensitive attributes. Second, attackers often have background knowledge, and we show that k-anonymity does not guarantee privacy against attackers using background knowledge." A column that should stay secret, such as a condition or an income, is what this lesson calls a **sensitive column**.

        Their example is a fictitious hospital, with a table that is 4-anonymous on ZIP code, age and nationality. Alice knows that Bob, a neighbor, is 31, American, and lives in ZIP code 13053. Four rows fit, and all four say cancer: Alice now knows that Bob has cancer, without knowing which row is Bob's. That is the **homogeneity attack**. In the second attack, what Alice already knows makes the other values of a group very unlikely. On their copy of UCI Adult, without the rows with a missing value (45,222 people), with age, sex, race, marital status and education as the quasi-identifier, they made every minimal 6-anonymous version of the table, coarsened no further than needed: with the salary class (the column `income`) as the secret, 8 of the 9 were open to the homogeneity attack; with the occupation, 1 of the 12.

        ### l-diversity: different secrets in every group

        Their answer: "A q⋆-block is ℓ-diverse if [it] contains at least ℓ 'well-represented' values for the sensitive attribute S. A table is ℓ-diverse if every q⋆-block is ℓ-diverse." A q⋆-block is what this lesson calls a group, and the course writes **l-diversity** with a plain l. They call it a principle, not yet a definition, and make "well-represented" precise in two ways, both stricter than the simplest reading. The simplest reading, at least l different values in every group, got its name, distinct l-diversity, from Ninghui Li, Tiancheng Li and Suresh Venkatasubramanian (2007), who warn that it "does not prevent probabilistic inference attacks". Different values can also mean almost the same: in their example, a group holds gastric ulcer, gastritis and stomach cancer, and an attacker learns that someone in it has a stomach disease. Their verdict on l-diversity: it "is neither necessary nor sufficient to prevent attribute disclosure", that is, a secret value leaking out. They propose t-closeness: in every group, the distribution of the secret values must stay within a distance t of their distribution in the whole table.
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
        | **quasi-identifier** | facts that do not say who someone is on their own, such as age, ZIP code or sex, but can in combination; in a table, a set of columns |
        | **k-anonymity** | every combination of values of the quasi-identifier occurs in at least k rows; the table is then k-anonymous |
        | **k** | the number of rows in the smallest group |
        | **generalization** | replacing a value with a less specific one that still fits, such as 37 with 30-39 |
        | **suppression** | not releasing a value at all, or a whole row |
        | **sensitive column** | a column that should stay secret, such as a condition or an income; the papers call it a sensitive attribute |
        | **homogeneity attack** | learning someone's secret because all rows of their group share it, without knowing which row is theirs |
        | **l-diversity** | every group has at least l well-represented values of the sensitive column; in the simplest reading, at least l different values |
        | **l** | in the simplest reading, the smallest number of different values of the sensitive column in one group |
        | **nunique** | per group, how many different values a column has, leaving missing values out: `df.groupby("sex")["race"].nunique()` |
        """),
        L.watch_out("k-anonymity protects only against someone whose knowledge fits the quasi-identifier you chose, and only against picking out a row. It says nothing about what the rows of a group have in common: count l next to k, and do not call a table anonymous because k is large."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out how well a table hides the people in it:

    1. Choose the quasi-identifier: every column that someone could know from elsewhere, such as age, sex or where someone lives.
    2. Group the rows by those columns, and take the number of rows in the smallest group: that is k. That is `k_of`.
    3. When k is too small, coarsen the columns that make the groups small, such as age into bands of ten years, and leave out the rows whose group still has fewer than k rows. That is `band`.
    4. Count, in every group, the different values of the sensitive column; the smallest count is l. When l is 1, someone's secret is known without finding their row. That is `l_of`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads UCI Adult, groups the people by sex and race, counts the people in each group, and takes the smallest count: the k for these two columns. Then it adds marital status.
    """)
    return


@app.cell
def example_groups():
    import pandas as pd

    _people = pd.read_csv("data/adult.csv.gz")
    # how many people share each combination of sex and race
    _sizes = _people.groupby(["sex", "race"]).size()
    print(_sizes)
    print("smallest group:", _sizes.min())
    print("with marital status too:", _people.groupby(["sex", "race", "marital_status"]).size().min())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With sex and race, the smallest group has 155 people: the women whose race is recorded as Other. On those two columns, the table is 155-anonymous. Add marital status, and two people are the only one with their combination: k drops to 1. Every column you add can split the groups further.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_k():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. k_of: the smallest group

        `PEOPLE` is UCI Adult: 48,842 people, a row each, with columns such as `age`, `sex`, `race`, `marital_status`, `education`, `occupation` and `income` (`"<=50K"` or `">50K"`: at most or more than 50,000 dollars a year). `QI` is the list `["age", "sex", "race", "marital_status", "education"]`: the quasi-identifier of Machanavajjhala and colleagues.

        `k_of(df, columns)` gets a table `df` and a list of column names `columns`. It gives back the number of rows in the smallest group of rows with the same values in `columns`: the k for which `df` is k-anonymous on those columns. Replace the `...` with one expression: group `df` by `columns`, count the rows of each group, and take the smallest count. The check calls `k_of(PEOPLE, QI)`, then fewer columns, and the clinic of the picture: `CLINIC` before and `RELEASED` after coarsening.
        """),
        L.task("Give back the number of rows in the smallest group of rows with the same values in columns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_k():
    box_k_of = L.code_box("k_of", lines=3)
    box_k_of
    return (box_k_of,)


@app.cell(hide_code=True)
def check_k(box_k_of):
    result_k_of = L.check("k_of", box_k_of.value)
    result_k_of
    return (result_k_of,)


@app.cell(hide_code=True)
def hints_k():
    L.hints(
        "k_of",
        [
            "Step 2 of the recipe. One group per combination of values, a count per group, and then one number: the smallest count.",
            "`df.groupby(columns)` makes the groups. Which method counts the rows of each group, and which takes the smallest of those counts?",
            "`return df.groupby(columns).size().___()`",
        ],
        why="`df.groupby(columns).size()` counts the rows of each group, and `.min()` takes the smallest. `k_of(PEOPLE, QI)` gives 1: 3,948 people are the only one with their combination of the five columns.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_band():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. band: ten years instead of one

        `band(age)` gets a whole number `age`, and gives back the ten years it falls in, as a text: `"30-39"` for 37, and also for 30 and for 39. `low` is the lowest age of the band: 30 for 37. Replace the first `...` with an expression for `low`, and the second with the text of the band: from `low` to the highest age of the band, with a hyphen and no spaces. The check calls `band(37)`, then other ages, from 5 to 100.
        """),
        L.task("Give back the ten years the age falls in, as a text such as \"30-39\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_band():
    box_band = L.code_box("band", lines=4)
    box_band
    return (box_band,)


@app.cell(hide_code=True)
def check_band(box_band):
    result_band = L.check("band", box_band.value)
    result_band
    return (result_band,)


@app.cell(hide_code=True)
def hints_band():
    L.hints(
        "band",
        [
            "Step 3 of the recipe. 37 lies between 30 and 39: round it down to a ten. Divide by 10, drop what comes after the point, and multiply by 10 again.",
            "`int(...)` drops what comes after the point, as in [Variables and numbers](/course/open/py-variables?lang=en): `int(3.7)` gives `3`. The band then runs from `low` to the highest age of the band, in an f-string: for 37, from 30 to 39.",
            "`low = int(age / 10) * 10` and `return f\"{low}-{___}\"`",
        ],
        why="`low = int(age / 10) * 10` gives 30 for 37, and `return f\"{low}-{low + 9}\"` gives `\"30-39\"`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_l():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. l_of: different secrets in every group

        `BANDED` is `PEOPLE` with one more column, `band`: the ten years of each person's age, made with the course's `band`. Grouped by `band` and `sex`, it has 18 groups, and the smallest has 17 people: `BANDED` is 17-anonymous on those two columns.

        `l_of(df, columns, secret)` gets a table `df`, a list of column names `columns`, and the name of one more column, `secret`. It gives back the smallest number of different values of `secret` in one group of rows with the same values in `columns`: the l for which `df` is l-diverse, in the simplest reading. Replace the `...` with one expression: group `df` by `columns`, take the column `secret`, count the different values in each group, and take the smallest count. The check calls `l_of(BANDED, ["band", "sex"], "income")`, then other columns and secrets, the clinic, and a small table `GAPS` with a missing value.
        """),
        L.task("Give back the smallest number of different values of secret in one group of rows with the same values in columns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_l():
    box_l_of = L.code_box("l_of", lines=3)
    box_l_of
    return (box_l_of,)


@app.cell(hide_code=True)
def check_l(box_l_of):
    result_l_of = L.check("l_of", box_l_of.value)
    result_l_of
    return (result_l_of,)


@app.cell(hide_code=True)
def hints_l():
    L.hints(
        "l_of",
        [
            "Step 4 of the recipe. As in `k_of`, but in each group you count the different values of one column instead of the rows.",
            "After `df.groupby(columns)`, square brackets take one column, as in [One number per group: groupby](/course/open/data-groupby?lang=en). Which method from *New words and symbols* counts the different values?",
            "`return df.groupby(columns)[secret].___().min()`",
        ],
        why="`df.groupby(columns)[secret].nunique()` counts the different values of `secret` in each group, and `.min()` takes the smallest. `l_of(BANDED, [\"band\", \"sex\"], \"income\")` gives 1: in one group, everyone has the same income class.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bigger():
    mo.md(r"""
    ### Exercise 4. A bigger k

    Machanavajjhala and colleagues also made UCI Adult k-anonymous on age, sex, race, marital status and education, with the salary class as the secret: in every minimal way, once for k = 2, once for k = 5 and once for k = 50.
    """)
    return


@app.cell(hide_code=True)
def choice_bigger_cell():
    choice_bigger = L.choice(
        "bigger-k",
        question="What happened to the homogeneity attack as k grew?",
        options={
            "gone": "It disappeared from k = 5 on",
            "right": "Fewer rows leaked, but it stayed",
            "grew": "It revealed more rows as k grew larger",
        },
        correct="right",
        why={
            "right": "In the journal version of their paper, the salary class of 558 rows was revealed on average at k = 2, of 381.58 at k = 5, and of 202.80 at k = 50, where all 5 minimal tables were open to it. A bigger k asks for bigger groups, not for more varied ones.",
            "gone": "At k = 5, 11 of the 12 minimal tables were still open to it, and at k = 50 all 5 of them. A bigger k asks for bigger groups, not for more varied ones. See *When the whole group shares the secret* in *The idea*.",
            "grew": "Fewer rows were revealed as k grew: 558 on average at k = 2, 202.80 at k = 50. But the attack did not go away. See *When the whole group shares the secret* in *The idea*.",
        },
    )
    choice_bigger
    return (choice_bigger,)


@app.cell(hide_code=True)
def text_exercise_skewness():
    mo.md(r"""
    ### Exercise 5. Two values, one answer

    A test is positive for 1% of the 10,000 people in a table. In the released table, one group has 49 people who tested positive and 1 who tested negative. The group has two different values, so in the simplest reading it is 2-diverse.
    """)
    return


@app.cell(hide_code=True)
def choice_skewness_cell():
    choice_skewness = L.choice(
        "skewness",
        question="What does an attacker learn about someone who is known to be in that group?",
        options={
            "low": "That their chance is 1%, as for everyone",
            "half": "That it is 50-50: there are two values",
            "right": "That they are very likely positive",
        },
        correct="right",
        why={
            "right": "Li, Li and Venkatasubramanian (2007): someone in such a group would be considered 98% positive, rather than 1%. Two different values do not mean two equally likely ones.",
            "low": "The 1% holds for the whole table, not for this group: in it, 49 of the 50 tested positive. See *l-diversity: different secrets in every group* in *The idea*.",
            "half": "Two values can be far from equally likely: 49 of the 50 people in the group tested positive. See *l-diversity: different secrets in every group* in *The idea*.",
        },
    )
    choice_skewness
    return (choice_skewness,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Pseudonymous is not anonymous](/course/open/rai-pseudonymization?lang=en), three facts singled out one row. k-anonymity makes sure that at least k rows fit every combination of values of the quasi-identifier, and l-diversity that the rows of a group do not all share their secret. Both protect only against someone whose knowledge fits the quasi-identifier you chose. In the next lesson, *Linking tables*, you join a table without names with a public one on such columns, and watch the names come back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bigger, choice_skewness, result_band, result_k_of, result_l_of):
    L.wrap_up(result_k_of, result_band, result_l_of, choice_bigger, choice_skewness)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
