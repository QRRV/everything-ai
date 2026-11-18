# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-bias-faces-images. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-faces-images", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Bias at every step](/course/open/rai-bias-lifecycle?lang=en) and [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="A firm trains a voice assistant on recordings from every region, but tests it only on its own staff, who almost all grew up in one region. What kind of bias can that give?",
        options={
            "representation": "Representation bias",
            "right": "Evaluation bias",
            "measurement": "Measurement bias",
        },
        correct="right",
        why={
            "representation": "The training data covers every region, so that is not where the problem is: the test data is. See *Building and using the model* in [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).",
            "right": "Test data that does not look like the people the assistant is used for, so a good score can hide a group: evaluation bias.",
            "measurement": "No label here is a stand-in that measures worse for one group; the problem is who is in the test. See *Building and using the model* in [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="A system gets photos of 200 women. It calls 180 of them women and 20 of them men. What is its recall for the class woman?",
        options={
            "right": "0.9",
            "wrong": "0.1",
            "upside": "1.11",
        },
        correct="right",
        why={
            "right": "180 / (180 + 20) = 0.9: it finds 9 in 10 of the women. In this lesson, 1 minus the recall of a group is its error rate: here 0.1.",
            "wrong": "That is the share it got wrong, 20 of 200. Recall is the share of the women it finds: TP / (TP + FN). See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "upside": "That is 200 / 180, upside down: recall is never above 1. Recall divides the true positives by all women: TP / (TP + FN). See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Five Black women took General Motors to court. The company had not hired Black women before 1964, and all the Black women it hired after 1970 lost their jobs in a layoff by seniority during a later recession. The court ruled that they could not combine the law against race discrimination and the law against sex discrimination into a new "super-remedy". In 1989 the legal scholar Kimberlé Crenshaw used the case to show how a "single-axis framework erases Black women": looking at race and at sex one at a time misses what happens where the two meet. Crenshaw's word for it, intersectionality, comes with a picture of traffic at a crossing: an accident there "can be caused by cars traveling from any number of directions and, sometimes, from all of them." Machine learning has the same blind spot. A system checked per attribute can look fine for women and fine for darker-skinned people, and still fail darker-skinned women.

    **An everyday example.** A school reports that 90 percent of its girls pass, and 90 percent of its pupils from the east of the town. That does not tell you how the girls from the east do: they could be the ones who fail.
    """)
    return


@app.cell(hide_code=True)
def predict_gerrymander():
    guess_gerrymander = L.guess(
        "gerrymander",
        question="A model says yes to half of the men and half of the women, and to half of the Black and half of the White people. The four combinations are equally large. What share of the Black women can it say yes to?",
        options={
            "half": "Exactly half",
            "between": "Between a quarter and three quarters",
            "right": "Anything from none to all",
        },
        reveal="**Anything from none to all.** Michael Kearns and colleagues (2018) give an example: a model that says yes exactly to the Black men and the White women. It says yes to 50 percent of the men and of the women, and to 50 percent of the Black and of the White people, so it looks fair on each attribute. But it says yes to none of the Black women and none of the White men. They call this *fairness gerrymandering*, after the trick of drawing election districts so that one side wins: a model can look fair on a few groups chosen in advance and be unfair to their combinations.",
    )
    guess_gerrymander
    return (guess_gerrymander,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Intersectional fairness.** *Intersectional fairness* is fairness for groups that are combinations of attributes, such as darker-skinned women, and not only for each attribute on its own. You check it in a simple way: count per combination, and compute the metric per combination. Crenshaw's paper is about law, not about statistics; *Gender Shades* used Crenshaw's word for its analysis of four combinations, and the course does the same. How far should you go? Kearns and colleagues warn that you cannot ask for fairness in every subgroup you can think of: any model that makes mistakes could be called unfair to the group "it misclassified", which they call "overfitting" a fairness constraint. So you choose the combinations that matter.

        **Gender Shades.** Joy Buolamwini and Timnit Gebru (2018) built a benchmark of 1,270 members of parliament from three African countries (Rwanda, Senegal, South Africa) and three European ones (Iceland, Finland, Sweden), countries chosen for gender parity in their parliaments: the Pilot Parliaments Benchmark. A dermatologist labeled each person's skin type on the six-point Fitzpatrick scale; types I to III count as lighter, IV to VI as darker. The gender labels are female and male, for people perceived as women or men, a view the paper itself calls reductionist: it "does not adequately capture the complexities of gender or address transgender identities." In April and May 2017 they tested the gender classifiers of Microsoft, Face++ and IBM: systems that label a face as female or male. The error rate of a group is 1 minus its true positive rate: for women, the share of the women the system labeled men, so 1 minus the recall for women. Faces the system did not find at all are left out: only then do the paper's numbers match its counts, and the follow-up paper states that rule.

        | System | women | men | darker | lighter | darker women | lighter men |
        |---|---|---|---|---|---|---|
        | Microsoft | 10.7 | 2.6 | 12.9 | 0.7 | 20.8 | 0.0 |
        | Face++ | 21.3 | 0.7 | 16.5 | 4.7 | 34.5 | 0.8 |
        | IBM | 20.3 | 5.6 | 22.4 | 3.2 | 34.7 | 0.3 |

        *Error rate in percent on the Pilot Parliaments Benchmark, from Table 4 of* Gender Shades.

        All three systems did worst on darker-skinned women, from 20.8 to 34.7 percent, while the highest error rate for lighter-skinned men was 0.8 percent. Darker-skinned women were 21.3 percent of the benchmark, but between 61.0 and 72.4 percent of the errors. Look at the columns: the worst number per attribute, 22.4 for darker-skinned people with IBM, lies far below the 34.7 of the combination. The authors are careful about the cause: darker skin "may be highly correlated with facial geometries or gender display norms that were less represented in the training data". And they chose easy photos on purpose, "an optimistic sample", so on harder photos they expect the error rates to be higher.

        **After the audit.** Inioluwa Deborah Raji and Buolamwini tested again in August 2018. Within seven months, all three companies had released a new version, and the error rate for darker-skinned women had dropped by 17.7 to 30.4 percentage points; for IBM from 34.7 to 16.97 percent. Two companies not named in the first audit, Amazon and Kairos, had 31.4 and 22.5 percent for darker-skinned women.

        **Not face recognition.** *Gender Shades* tested gender classification: guessing a label from one face. Face recognition matches photos of the same person. NIST, the US standards institute, tested 189 mostly commercial face recognition algorithms in 2019, and warns that the gender classification studies "did not evaluate face recognition algorithms, yet the results have been widely cited to indict their accuracy." Its own finding: across demographic groups, the rates of false positives, two different people taken for the same person, "often vary by factors of 10 to beyond 100 times". On application photos they were highest for West and East African and East Asian people; with a number of algorithms developed in China, the effect was reversed for East Asian faces.

        **Image generators.** An image generator makes a picture from a text. Federico Bianchi and colleagues (2023) asked Stable Diffusion for "A photo of the face of" an occupation, 100 images each, and used CLIP, a model that turns images into vectors, to find which group's average face photo each image was closest to. 99 percent of the software developers were represented as White, while 56 percent of the software developers in the US identified as White; housekeepers, nurses and flight attendants were 100 percent represented as female. The images are of nobody: "represented as" is a label from CLIP, not anyone's identity. Alexandra Sasha Luccioni and colleagues (2023) compared three generators with the statistics of the US Bureau of Labor Statistics (BLS) for 146 occupations. They sorted the occupations into five groups by their share of women, from the fifth with the fewest women to the fifth with the most, and measured the share of images in clusters linked to the word woman. Dall·E 2 put far fewer images in clusters linked to women than the statistics had women, in every group; Stable Diffusion 1.4 came closest, and in the group with the fewest women it put slightly more there. In this lesson, "draws fewer women" is short for this measure.

        **The law.** Since 2 February 2025, the European AI Act bans, among other things, two uses that face analysis can serve: inferring the emotions of people at work and in education, except for medical or safety reasons, and biometric categorization, from a face or other biometric data, that infers someone's race, political opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation. Sex and gender are not on that list. Sorting people by sensitive or protected attributes from their biometric data, and emotion recognition elsewhere, count as high-risk instead; after a change to the law in 2026, those obligations apply from 2 December 2027.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **intersectional fairness** | fairness for groups that are combinations of attributes, such as darker-skinned women, checked per combination |
        | **fairness gerrymandering** | a model looks fair on each attribute on its own, and is unfair to their combinations (Kearns and colleagues) |
        | **error rate of a group** | in *Gender Shades*: 1 minus the recall of a group, the share of that group the system labels wrongly |
        """),
        L.watch_out("A table per combination is not all that intersectionality means. Crenshaw's point is that the experience \"is greater than the sum of racism and sexism\"; error rates per cell show where a system fails, not why, and not what it is like to be failed."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a system for intersectional bias:

    1. Choose the attributes that matter for the people the system is used on, and the metric, such as the error rate.
    2. Count per combination: how many people, how many got a result, how many errors. Say what you left out, such as faces the system did not find.
    3. Compute the metric per combination. For one attribute, add up the counts of its cells and divide once; do not average the rates of cells of different sizes.
    4. Compare the worst combination with the worst single attribute, and look at the size of each cell: with few people, a rate jumps around.
    5. Choose the combinations before you look; you cannot check every group you can think of.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Microsoft's numbers from *Gender Shades*, per combination, from the tables in the paper's supplementary material. `classified` is the faces the system found and labeled, `errors` the ones it labeled wrongly. The loop works out the error rate of each cell. Run the cell.
    """)
    return


@app.cell
def example_cells():
    # Microsoft in Gender Shades (2018): per group, the faces it classified and its errors
    _cells = [
        {"skin": "darker", "gender": "female", "classified": 265, "errors": 55},
        {"skin": "darker", "gender": "male", "classified": 300, "errors": 18},
        {"skin": "lighter", "gender": "female", "classified": 295, "errors": 5},
        {"skin": "lighter", "gender": "male", "classified": 385, "errors": 0},
    ]

    for _cell in _cells:
        _rate = _cell["errors"] / _cell["classified"]
        print(_cell["skin"], _cell["gender"], round(_rate, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Four rates, from 0.208 for darker-skinned women to 0.0 for lighter-skinned men, as in the table in *The idea*. A report per attribute would give one number for all women: it mixes the cell of 0.208 with the cell of 0.017. How much does that mixing hide? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One number per attribute

        Work out the error rate of all cells whose `key` has the value `value`, taken together: for `key = "gender"` and `value = "female"`, all women. Add up first, divide last: start with `classified = 0` and `errors = 0`, loop over `cells` with `cell`, and when `cell[key] == value`, add the cell's `"classified"` and `"errors"` to the totals. After the loop, `rate` is `errors` divided by `classified`; it is the last line, so the check sees it. The check also tries the darker cells, and the men.
        """),
        L.task("Work out `rate`: the errors of the cells whose `key` is `value`, divided by their faces classified."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rate():
    box_rate = L.code_box("rate", lines=22)
    box_rate
    return (box_rate,)


@app.cell(hide_code=True)
def check_rate(box_rate):
    result_rate = L.check("rate", box_rate.value)
    result_rate
    return (result_rate,)


@app.cell(hide_code=True)
def hints_rate():
    L.hints(
        "rate",
        [
            "The women's error rate is all the errors on women divided by all the women classified: add up the two cells with `\"female\"`, and divide once, after the loop.",
            "In the loop, an `if` picks the cells: `if cell[key] == value:`. Under it, two lines add the cell's counts to the totals, such as `classified = classified + cell[\"classified\"]`.",
            "After the loop, not inside it: `rate = ___ / classified`.",
        ],
        why="`classified = 0` and `errors = 0`, then `for cell in cells:` with `if cell[key] == value:` under it, and inside that `classified = classified + cell[\"classified\"]` and `errors = errors + cell[\"errors\"]`. After the loop, `rate = errors / classified`. That gives 60 / 560 = `0.107`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_subgroups():
    mo.md(r"""
    ### Exercise 2. Every group?
    """)
    return


@app.cell(hide_code=True)
def choice_subgroups_cell():
    choice_subgroups = L.choice(
        "subgroups",
        question="A team wants to measure its face classifier on every group it can define, down to \"the faces it labels wrongly\". What is the problem with that last group?",
        options={
            "small": "It is too small to measure at all",
            "right": "Every model that makes errors fails it",
            "more": "Nothing: more groups always make a model fairer",
        },
        correct="right",
        why={
            "small": "Its size is not the problem: it can be large. On that group the model is wrong by definition, so every model that makes errors fails it. See *Intersectional fairness* in *The idea*.",
            "right": "Kearns and colleagues: any imperfect classifier could be accused of being unfair to the group of the people it misclassified, which they call \"overfitting\" a fairness constraint. Choose the combinations that matter, such as skin type and gender, before you look.",
            "more": "Measuring more groups does not make a model fairer, and a group made of the model's own errors fails every model that makes errors. See *Intersectional fairness* in *The idea*.",
        },
    )
    choice_subgroups
    return (choice_subgroups,)


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Fewer women than in the statistics

        `quintiles` holds Table 3 of Luccioni and colleagues: per group of occupations, `"bls"` is the percent women in the labor statistics, and `"sd14"`, `"sd2"` and `"dalle2"` are the percent of images that Stable Diffusion 1.4, Stable Diffusion 2 and Dall·E 2 drew in clusters linked to women. Work out `gap`: the mean, over the five groups, of the system's share minus the share in the statistics. Start with `total = 0`, loop over `quintiles` with `quintile`, add `quintile[system] - quintile["bls"]` to `total`, and divide by the number of groups, `len(quintiles)`. A negative gap means the system draws fewer women than there are in those jobs. The check also tries the two other systems.
        """),
        L.task("Work out `gap`: the mean of `quintile[system] - quintile[\"bls\"]` over the groups."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gap():
    box_gap = L.code_box("gap", lines=18)
    box_gap
    return (box_gap,)


@app.cell(hide_code=True)
def check_gap(box_gap):
    result_gap = L.check("gap", box_gap.value)
    result_gap
    return (result_gap,)


@app.cell(hide_code=True)
def hints_gap():
    L.hints(
        "gap",
        [
            "As in Exercise 1: add up in the loop, divide after it. This time you add one difference per group.",
            "The difference of one group is `quintile[system] - quintile[\"bls\"]`: the system's share minus the statistics. After the loop, divide the total by `len(quintiles)`.",
            "Inside the loop: `total = total + ___`.",
        ],
        why="`total = 0`, then `for quintile in quintiles:` with `total = total + quintile[system] - quintile[\"bls\"]` under it. After the loop, `gap = total / len(quintiles)`. That gives `-29.64` for Dall·E 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_law():
    mo.md(r"""
    ### Exercise 4. What does the law ban?
    """)
    return


@app.cell(hide_code=True)
def choice_law_cell():
    choice_law = L.choice(
        "law",
        question="Which of these does the AI Act ban in the EU?",
        options={
            "gender": "A classifier that guesses gender from a face photo",
            "right": "Emotion recognition of pupils in class",
            "nurses": "An image generator that draws every nurse as a woman",
        },
        correct="right",
        why={
            "gender": "The ban on biometric categorization lists race, political opinions, trade union membership, religious or philosophical beliefs, sex life and sexual orientation, but not sex or gender. See *The law* in *The idea*.",
            "right": "Since 2 February 2025 the AI Act bans systems that infer the emotions of people at work and in education, except for medical or safety reasons.",
            "nurses": "None of the AI Act's bans is about who an image generator draws for a job. A skewed picture of who does a job is a harm to measure, as in Exercise 3. See *The law* in *The idea*.",
        },
    )
    choice_law
    return (choice_law,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Bias at every step](/course/open/rai-bias-lifecycle?lang=en), one overall score hid a group: evaluation bias. Here a score per attribute hid a combination: count per cell, and compute the rate of a cell, and of an attribute, from the counts. The error rate of *Gender Shades* is 1 minus the recall of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), per group. For an image generator the question was not how often it is wrong, but whether who it draws matches the world, or a stereotype of it. The next lesson, *Bias in speech and in ads*, asks the same of voices, and of who gets to see an ad.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_law, choice_subgroups, result_gap, result_rate):
    L.wrap_up(result_rate, choice_subgroups, result_gap, choice_law)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
