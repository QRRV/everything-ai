# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-discrimination-tests. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-discrimination-tests", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_opportunity():
    quick_opportunity = L.choice(
        "quick-opportunity",
        question="A bank's model gives a loan to 0.6 of the women who would pay it back, and to 0.6 of such men. Which criterion does that meet?",
        options={
            "parity": "Demographic parity: the same yes rate",
            "right": "Equal opportunity: the same recall",
            "calibration": "Calibration: a yes means the same",
        },
        correct="right",
        why={
            "parity": "Demographic parity counts the yes among everyone, not only among the people who would pay back. The same share of those people with a yes is the same recall: equal opportunity. See *The idea* in [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).",
            "right": "Of the people who would pay back, the same share gets a loan in both groups: the same true positive rate, equal opportunity.",
            "calibration": "Calibration starts from the people with a yes, and asks how many of them pay back. This question starts from the people who pay back. See *The idea* in [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).",
        },
    )
    quick_opportunity
    return (quick_opportunity,)


@app.cell(hide_code=True)
def quick_check_base():
    quick_base = L.choice(
        "quick-base",
        question="A model has the same precision (below 1) and the same recall in two groups, but more people of group A are positive. What about the false positive rates of the two groups?",
        options={
            "same": "They are the same too",
            "right": "They must differ",
            "zero": "Both must be 0",
        },
        correct="right",
        why={
            "same": "The base rate still changes the false positive rate: in Exercise 2 of [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en), the same precision and recall gave 0.03 and 0.11.",
            "right": "From the base rate, the precision and the recall of a group, its false positive rate follows; different base rates give different rates.",
            "zero": "Only a model without false positives has 0 for both, and then its precision is 1. With the same precision below 1, the rates differ. See *Why not both* in [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).",
        },
    )
    quick_base
    return (quick_base,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lessons you had the model and its data: you could count every yes and every mistake. Often you have neither. A landlord, an employer or a police officer decides, and you only see what comes out. Did they treat groups differently? This lesson shows two tests that work from outside, and where each of them fails. The first of them also works on a model that you can only send inputs to.

    **An everyday example.** Two friends call the same landlord about the same room, an hour apart. They say the same things: the same income, the same date to move in. Only their names differ. One hears that the room is free, the other that it was just rented. One pair of calls proves little; a hundred such pairs prove a lot.
    """)
    return


@app.cell(hide_code=True)
def predict_callback():
    guess_callback = L.guess(
        "callback",
        question="Researchers answered job ads in Boston and Chicago with made-up CVs that were alike except for the name: some had a White-sounding name, some an African-American-sounding name. The White names got a call back 9.65 percent of the time. How often do you think the African-American names did?",
        options={
            "same": "About as often: 9.5 percent",
            "right": "Less often: 6.45 percent",
            "rare": "Hardly ever: 1 percent",
        },
        reveal="**Less often: 6.45 percent.** That is 3.20 percentage points less: the White names got 50 percent more calls, about one per 10 CVs against one per 15. The CVs differed only in the name. *The idea* below shows how such a test is built.",
    )
    guess_callback
    return (guess_callback,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Equal applications.** To test whether a decision depends on a protected attribute, send applications that are the same in everything except that attribute, and count how often each kind gets a yes. Such a test is an **audit study**. Some audit studies send trained people in person, in pairs; Barocas, Hardt and Narayanan write in *Fairness and Machine Learning* that audit studies "were pioneered by the US Department of Housing and Urban Development in the 1970s", to study how minority home buyers and renters were treated. A **correspondence study** is an audit study on paper: the applications are written and made up. Marianne Bertrand and Sendhil Mullainathan (2004) answered more than 1,300 job ads with close to 5,000 CVs, with names "such as Emily Walsh or Greg Baker" or "such as Lakisha Washington or Jamal Jones". Their result: "Resumes with White names have a 9.65 percent chance of receiving a callback. Equivalent resumes with African-American names have a 6.45 percent chance of being called back." In everyday terms: "Applicants with White names need to send about 10 resumes to get one callback whereas applicants with African-American names need to send about 15 resumes." Even the neighborhood was the same in each pair, so, Barocas, Hardt and Narayanan note, "the design likely underestimates the extent to which morally irrelevant characteristics affect callback rates in practice."

    **In the Netherlands.** In 2014, the Sociaal en Cultureel Planbureau (SCP) sent 504 applications that differed only in the name (native Dutch, Hindustani or Moroccan) to 176 real vacancies in the region of The Hague. Applicants with a native Dutch name had a 34 percent chance of success, with a Hindustani name 23 percent, and with a Moroccan name 19 percent. Lex Thijssen, Marcel Coenders and Bram Lancee (2019) sent 4,211 applications to vacancies in ten occupations, and estimated a chance of a positive reply of 46 percent for applicants with a Dutch background against 35 percent for applicants with a migration background. In Dutch, such a test is called a praktijktest. It can also serve as evidence: a 2014 guide calls situation testing "most appropriate for establishing direct discrimination", the kind of [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en) where the protected attribute itself makes the difference.

    **The same test on a model.** Most such studies, Barocas, Hardt and Narayanan write, "are best seen as attempts to test blindness: whether a decision maker directly uses a sensitive attribute." On a model, you change only that attribute. Sainyam Galhotra and colleagues (2017) measure it this way: run the model "on two individuals identical in every way except race", and see whether the output changes. Sahaj Garg and colleagues (2019) swapped words in sentences: a model found "Some people are gay" 98 percent likely to be toxic, and "Some people are straight" 2 percent. But the model without the column `sex` of [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en) passes such a test trivially: it does not see the column, and finds sex through proxies all the same. As the book puts it: "if a system passes a fairness test, we should not interpret it as a certificate that the system is fair."

    **The outcome test.** Often you cannot send test cases: the decisions were made long ago. Then look at what came of them. Camelia Simoiu, Sam Corbett-Davies and Sharad Goel (2017) describe the **outcome test**, which they trace to the economist Gary Becker: it is "based not on the rate at which decisions are made, but on the success rate of those decisions." For loans: minorities "who are granted loans, absent discrimination, should still be found to repay their loans at the same rate as whites who are granted loans." If they repay more often, the lender may be "granting loans only to exceptionally qualified minorities": a higher bar for them, or in the words of the previous lesson, a higher decision threshold. For police searches, the success rate is the **hit rate**, "the proportion of searches that successfully turn up contraband". It is the precision per group of [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en), and it needs only the outcomes of the people who got a yes, which is what an outsider can see.

    **What it found.** John Knowles, Nicola Persico and Petra Todd (1999) looked at 1,590 car searches on a highway in Maryland. Drugs were found in 0.34 of the searches of Black drivers, 0.32 of those of White drivers and 0.11 of those of Hispanic drivers. In their model, the first two numbers were "inconsistent with racial prejudice"; for Hispanic men, the low number was "consistent with a taste for discrimination against this group", on a small sample. Emma Pierson and colleagues (2020) collected data on nearly 100 million traffic stops in the United States. In eight state patrols, searches of White drivers found something illegal 32.0 percent of the time, of Black drivers 29.4 percent and of Hispanic drivers 24.3 percent: "The outcome test thus indicates that search decisions may be biased against Hispanic drivers, but the evidence is more ambiguous for black drivers."

    **Equal bars, unequal success.** Yet the outcome test can mislead. Barocas, Hardt and Narayanan call the step from a different success rate to a different bar "a logical fallacy". The success rate is an average over everyone above the bar, also over the people far above it. If the people above the bar sit further above it in one group, that group's success rate is higher with the very same bar. That is **infra-marginality**: "the measurement is aggregated over samples that are far from the decision threshold (margin)." It may also hide a lower bar, Simoiu and colleagues argue. In Raleigh, North Carolina, Black drivers were searched more often than White drivers, 4 against 2 percent, and searches of Black drivers found something more often, 16 against 13 percent; by the outcome test, the bar was lower for White drivers. Simoiu and colleagues estimated the bars themselves with a threshold test; their estimate put the bar lower for Black drivers, 6 against 9 percent, "suggesting discrimination against blacks". Their test needs a model of how the chances are spread, while, as Barocas, Hardt and Narayanan put it, "the outcome test is model-free".

    **You only see the yes.** The outcome test and the threshold test can only look at the outcomes of the people who got a yes; an audit study does not need them. Whether someone pays a loan back, you only see when they got one. Himabindu Lakkaraju and colleagues (2017) give bail as an example: "we observe the outcome of whether a defendant fails to return for their court appearance only if the human judge decides to release the defendant on bail." This is the **selective labels problem**. From past decisions you can measure the precision per group, but not the recall or the false positive rate: for those you need the outcomes of the people who got a no. Testing equal errors on such data, Barocas, Hardt and Narayanan write, "suffers from selection bias". One way out, a yes for some people whatever the prediction, would "raise ethical concerns", and such experiments "are rarely done in practice".

    **Try it.** The cell below makes up 100 drivers in each of two groups, with the same chances of carrying something illegal: 70 drivers with a chance of 0.01, 20 with 0.2 and 10 with 0.6. An officer searches every driver whose chance is at least the bar of their group. The function `_search` counts per group how many drivers are searched, and how many finds that gives on average: 20 drivers with a chance of 0.2 give 4 finds. First both groups get the bar 0.5, then group B gets a lower bar, 0.1. Run the cell.
    """)
    return


@app.cell
def demo_searches():
    # made-up drivers: per group, a chance of carrying something and how many drivers have it
    _drivers = {
        "group A": {0.01: 70, 0.2: 20, 0.6: 10},
        "group B": {0.01: 70, 0.2: 20, 0.6: 10},
    }


    def _search(bars):
        # per group: the drivers searched, the finds on average, and the share of the searches that finds something
        for group in _drivers:
            searched = 0
            found = 0
            for chance, count in _drivers[group].items():
                if chance >= bars[group]:
                    searched = searched + count
                    found = found + chance * count
            print("  ", group, "| bar", bars[group], "| searched", searched, "of 100", "| found", round(found, 1),
                  "| hit rate", round(found / searched, 2))


    print("the same bar for both groups")
    _search({"group A": 0.5, "group B": 0.5})
    print("a lower bar for group B")
    _search({"group A": 0.5, "group B": 0.1})
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_hits({"title": "The searches of the officer per group",
                                "rows": ["The same bar of 0.5", "A bar of 0.1 for group B"],
                                "xlabels": ["share searched", "hit rate"],
                                "key": "open: group A, filled: group B", "decimal": ".",
                                "alt": "Two panels with two rows each. Left, the share of the drivers searched in group A and group B: the same bar 0.10 and 0.10, a lower bar for group B 0.10 and 0.30. Right, the hit rate: the same bar 0.60 and 0.60, a lower bar for group B 0.60 and 0.33."})),
        mo.md(r"""
        With the same bar, both groups are searched equally often, and the searches find something equally often: 0.6 of the time. With a lower bar for group B, three times as many of its drivers are searched, and the extra searches are of drivers with a chance of only 0.2: the hit rate of group B drops to 0.33. That is the reasoning of the outcome test: a lower hit rate points to a lower bar. Here the groups had the same chances, so the test is right. Exercise 2 shows a case where it is wrong.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **audit study** | a test with pairs of people or applications that are the same except for a protected attribute, to see whether that attribute changes the decision; in Dutch a praktijktest |
        | **correspondence study** | an audit study with written, made-up applications, such as CVs that differ only in the name |
        | **outcome test** | compares the success rate of decisions per group: of the people with a yes, the share for whom it turned out right; a lower rate in one group points to a lower bar there |
        | **hit rate** | in an outcome test of searches: the share of the searches that finds something, the precision of the searches |
        | **infra-marginality** | success rates differ between groups with one bar for everyone, because in one group the people above the bar sit further above it |
        | **selective labels problem** | you only see the outcome of the people who got a yes, so you cannot count the mistakes among the people who got a no |
        """),
        L.watch_out("An equal success rate does not prove that there is no discrimination, and an unequal one does not prove that there is. Pierson and colleagues note that the outcome test \"can similarly fail to detect discrimination when it is present\"."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out from outside whether decisions treat groups differently:

    1. Can you send test cases? Make pairs that are the same except for the protected attribute, send many, and compare per group the share with a yes: an audit study. On a model, change the proxies of the attribute too, not only its column.
    2. Only past decisions? Take per group the success rate of the yes: of the people with a yes, the share for whom it turned out right. A lower rate in one group points to a lower bar there: the outcome test.
    3. Ask whether the groups are spread differently above the bar. Then equal bars can give unequal success rates, and unequal bars equal ones: infra-marginality.
    4. Say what you cannot see: the outcomes of the people with a no. Without them, there is no recall and no false positive rate per group.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The callbacks of Bertrand and Mullainathan: they sent 2,435 CVs with each kind of name. The paper gives percentages; the numbers of calls in the cell, 235 and 157, are worked out from its 9.65 and 6.45 percent of 2,435. The cell works out per kind of name the share that got a call back, and then how many times as often the White names got one: step 1 of the recipe. Run the cell.
    """)
    return


@app.cell
def example_callbacks():
    # Bertrand and Mullainathan (2004): 2,435 CVs per kind of name
    _sent = 2435
    # the calls back, worked out from their percentages
    _called = {"White names": 235, "African-American names": 157}

    _rate = {}
    for _name in _called:
        _rate[_name] = _called[_name] / _sent
    print("called back:", "White names", round(_rate["White names"], 4),
          "| African-American names", round(_rate["African-American names"], 4))
    print("times as often:", round(_rate["White names"] / _rate["African-American names"], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The White names got a call back 0.0965 of the time and the African-American names 0.0645: 9.65 and 6.45 percent, as in the paper. The White names got 1.5 times as many calls, the "50 percent" more of the paper. Exercise 1 does the same for the Dutch study of the SCP.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ratio():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How many times as often

        `success` holds the chance of success per name in the study of the SCP. The loop goes through its keys, and `group` holds the name of one group in each round. Replace the `...` with one line: `ratio[group]`, the chance of native Dutch applicants divided by the chance of `group`, as the worked example divided the two rates. The check also tries other chances.
        """),
        L.task("Work out `ratio[group]` for each group in `success`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ratio():
    box_ratio = L.code_box("ratio", lines=8)
    box_ratio
    return (box_ratio,)


@app.cell(hide_code=True)
def check_ratio(box_ratio):
    result_ratio = L.check("ratio", box_ratio.value)
    result_ratio
    return (result_ratio,)


@app.cell(hide_code=True)
def hints_ratio():
    L.hints(
        "ratio",
        [
            "The worked example divided the rate of the White names by the rate of the African-American names. Here, divide the chance of native Dutch applicants by the chance of each group in turn.",
            "Inside the loop, `success[group]` is the chance of the group of this round, and `success[\"native Dutch\"]` the chance of native Dutch applicants.",
            "`ratio[group] = success[\"native Dutch\"] / ___`.",
        ],
        why="`ratio[group] = success[\"native Dutch\"] / success[group]`. That gives 1.0 for native Dutch, 1.48 for Hindustani and 1.79 for Moroccan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hits():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One bar, two hit rates

        Made-up drivers again, now with the chances of the example of Simoiu, Corbett-Davies and Goel: in group A, 90 drivers with a chance of 0.01 and 10 with 0.75; in group B, 80 with 0.01 and 20 with 0.5. The officer searches every driver whose chance is at least `bar`, 0.1, in both groups. `drivers` holds per group how many drivers have each chance; the inner loop goes through it with `chance` and `count`; `searched` and `found` start at 0 for each group, and the last line divides them into `hit_rate[group]`. Replace the `...` in the inner loop with the `if` of *Try it* and the two lines under it, with `bar` instead of `bars[group]`. The check also tries another bar.
        """),
        L.task("Work out `hit_rate[group]` for each group in `drivers`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hits():
    box_hits = L.code_box("hits", lines=14)
    box_hits
    return (box_hits,)


@app.cell(hide_code=True)
def check_hits(box_hits):
    result_hits = L.check("hits", box_hits.value)
    result_hits
    return (result_hits,)


@app.cell(hide_code=True)
def hints_hits():
    L.hints(
        "hits",
        [
            "Only the drivers with a chance of at least `bar` are searched. For each kind of driver that is searched, add how many there are to `searched`, and how many finds they give on average to `found`.",
            "`if chance >= bar:`, and under it, indented by four more spaces, `searched = searched + count`.",
            "`found = found + ___ * count`.",
        ],
        why="In the inner loop: `if chance >= bar:`, then `searched = searched + count` and `found = found + chance * count`. That gives a hit rate of 0.75 for group A and 0.50 for group B.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_labels():
    mo.md(r"""
    ### Exercise 3. What a bank can measure
    """)
    return


@app.cell(hide_code=True)
def choice_labels_cell():
    choice_labels = L.choice(
        "labels",
        question="A bank only knows whether a loan was paid back for the loans it gave. Which of these can it work out per group from its own past decisions?",
        options={
            "right": "Of those with a loan, the share who paid back",
            "recall": "Of all who would pay back, the share with a loan",
            "fpr": "Of all who would not pay back, the share with a loan",
        },
        correct="right",
        why={
            "right": "Whether a loan was paid back is known for everyone with a loan, so the bank can count per group the share that paid back: the precision, the number of the outcome test.",
            "recall": "For this, the bank needs everyone who would pay back, also the people it turned down. Whether they would have paid back, it never saw: the selective labels problem.",
            "fpr": "For this, the bank needs everyone who would not pay back, also the people it turned down. Whether they would have paid back, it never saw: the selective labels problem.",
        },
    )
    choice_labels
    return (choice_labels,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en), the law asked whether a protected attribute made the difference; an audit study tests exactly that, with pairs that differ only in that attribute. In [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en) you had the model and its data; here you saw decisions from outside. The outcome test compares the precision per group of the people who got a yes; infra-marginality is why equal bars can still give unequal success rates, and selective labels are why equal errors are hard to check afterwards. The next lesson, *Fixing the data: reweighing*, starts on the other side: what you can do when a model treats groups unequally.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_labels, result_hits, result_ratio):
    L.wrap_up(result_ratio, result_hits, choice_labels)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
