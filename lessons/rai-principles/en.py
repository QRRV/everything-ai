# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-principles. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-principles", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_duty():
    quick_duty = L.choice(
        "quick-duty",
        question="Sharing patient records without asking would let a hospital save more lives. Why could deontology still say no?",
        options={
            "outcome": "Because in the end it would save fewer lives than it promises",
            "character": "Because a doctor of good character would feel uneasy sharing them",
            "right": "Because people have a right to be asked, whatever the outcome",
        },
        correct="right",
        why={
            "outcome": "That is still an argument about outcomes, the question of consequentialism. Which theory holds on to duties and rights, whatever the outcome? See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "character": "That is the question of virtue ethics. Which theory holds on to duties and rights? See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "right": "Deontology holds on to duties and rights whatever the outcome. Kant held that you may never treat a person merely as a means.",
        },
    )
    quick_duty
    return (quick_duty,)


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="In the hospital of that lesson, nearby first prevents 32 readmissions and need first 28, but need first helps a larger share of the worst-off group. Which theory points to nearby first?",
        options={
            "right": "Consequentialism, which adds up the good over everyone",
            "contract": "A fair agreement, which looks after the worst off",
            "care": "Care ethics, which starts from who depends on whom in the region",
        },
        correct="right",
        why={
            "right": "Consequentialism adds up what each option brings about, and 32 is more than 28.",
            "contract": "Rawls argued that behind the veil of ignorance you would look after the worst off, and that points to need first. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "care": "Care ethics looks at relationships and at what people need, not at a total. Which theory adds up? See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Since 2016, governments, companies, universities and others have published principles for AI by the dozen. In 2019, Anna Jobin, Marcello Ienca and Effy Vayena collected 84 of these documents and counted which principles each one names. At first sight the world agrees: a handful of words comes back almost everywhere. This lesson counts those words, and then looks at what the documents mean by them.

    **An everyday example.** Almost every restaurant calls its food fresh. For one, that means picked this morning; for another, frozen last month and thawed today. The word is the same; what you get is not.
    """)
    return


@app.cell(hide_code=True)
def predict_transparency():
    guess_transparency = L.guess(
        "transparency",
        question="Of the 84 documents with principles for AI, how many do you think name transparency?",
        options={
            "quarter": "About 20",
            "half": "About 40",
            "right": "About 70",
        },
        reveal="**73 of the 84**, more than any other principle. Justice and fairness came second, with 68 (Jobin, Ienca and Vayena, 2019). *The idea* shows the rest.",
    )
    guess_transparency
    return (guess_transparency,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Governments, companies and others write down what they think good AI is, as short principles. Three sets are well known:

        - The **OECD AI Principles** (2019, updated in 2024) were the first that governments agreed on together. They name five: inclusive growth, sustainable development and well-being; respect for the rule of law, human rights and democratic values, including fairness and privacy; transparency and explainability; robustness, security and safety; and accountability.
        - UNESCO's *Recommendation on the Ethics of Artificial Intelligence* was adopted by all 193 of its member states in November 2021.
        - In 2019, the EU's High-Level Expert Group on AI described **trustworthy AI**: lawful, ethical and robust, with seven requirements, among them human oversight, transparency and fairness. In 2020 it turned them into **ALTAI**, a list of questions an organization answers about its own system.

        Companies tend to speak of **responsible AI**: building and using AI in line with such principles, with the ways of working that help.

        The picture shows what Jobin, Ienca and Vayena found in their 84 documents. Five principles come back in more than half, and four of the words, beneficence, non-maleficence, autonomy and justice, were already the core of medical ethics (Beauchamp and Childress, 1979). That reads as agreement. But the same study found that the documents mean quite different things by the same word, give different reasons for it, and say different things about how to put it into practice. **Transparency**, the most common, can mean saying that AI is used, publishing the data or the code, or explaining a decision.

        Who wrote them is uneven too. The USA and the United Kingdom together wrote more than a third of the documents; none came from a country in Africa or South America on its own: those regions appear only through international organizations.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **OECD AI Principles** | the first principles for AI that governments agreed on together (2019, updated in 2024): five, such as human rights, transparency and accountability |
        | **trustworthy AI** | the EU expert group's name for AI that is lawful, ethical and robust, with seven requirements such as human oversight, transparency and fairness |
        | **ALTAI** | the Assessment List for Trustworthy AI (2020): the expert group's questions to test a system against those seven requirements |
        | **responsible AI** | building and using AI in line with such principles, and the ways of working and rules that help |
        | **transparency** | being open about an AI system: that it is used, what it does, how it was made and on what data |
        """),
        L.watch_out("How often a principle is named says how popular the word is, not what it means or whether anyone acts on it. Two documents can both name fairness and ask for things that clash."),
        mo.Html(C.picture({"title": "In how many of the 84 documents each principle is named", "x": "documents", "half": "half",
                           "names": ["transparency", "justice and fairness", "non-maleficence", "responsibility", "privacy", "beneficence",
                                     "freedom and autonomy", "trust", "sustainability", "dignity", "solidarity"],
                           "alt": "Of 84 documents with principles for AI, 73 name transparency, 68 justice and fairness, 60 non-maleficence, 60 responsibility, 47 privacy, 41 beneficence, 34 freedom and autonomy, 28 trust, 14 sustainability, 13 dignity and 6 solidarity. A dashed line marks half, at 42."})),
        mo.md(r"""
        Counts from Jobin, Ienca and Vayena (2019), Table 2. The dashed line is at half of the 84 documents.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a set of principles for AI:

    1. List the principles it names.
    2. For each one, find what this document means by it: a definition, an example, a rule.
    3. Look for how it is put into practice: a list of questions such as ALTAI, a person in charge.
    4. Compare it with another set: the same word, the same meaning?
    5. Note what is missing, and who wrote it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The six principles at the bottom of the picture. For each, the share of the 84 documents that name it: its count divided by the number of documents. Run the cell.
    """)
    return


@app.cell
def example_shares():
    _named = {"beneficence": 41, "freedom and autonomy": 34, "trust": 28, "sustainability": 14, "dignity": 13, "solidarity": 6}
    _documents = 84

    for _principle in _named:
        print(_principle, round(_named[_principle] / _documents, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Even beneficence, doing good, is named in fewer than half of the documents, and sustainability in only 0.17 of them. Which principles do more than half name? That is Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_common():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. More than half

        `named` holds the counts of all 11 principles and `documents` the number of documents; the loop goes through the principles with `principle`. Replace the `...` with two lines: an `if` that compares the share, worked out as in the worked example, with half, and under it the line that keeps the principle. The last line is `common`, so the check sees it. The check also tries another number of documents.
        """),
        L.task("Put in `common` the names of the principles that more than half of the documents name."),
    ])
    return


@app.cell(hide_code=True)
def exercise_common():
    box_common = L.code_box("common", lines=9)
    box_common
    return (box_common,)


@app.cell(hide_code=True)
def check_common(box_common):
    result_common = L.check("common", box_common.value)
    result_common
    return (result_common,)


@app.cell(hide_code=True)
def hints_common():
    L.hints(
        "common",
        [
            "The share as in the worked example, now inside an `if`: two lines inside the loop.",
            "Which number stands for half, when you compare a share? And what does the loop do with a principle that passes?",
            "`if named[principle] / documents > ___:` and under it `common.___(principle)`.",
        ],
        why="`if named[principle] / documents > 0.5:` with `common.append(principle)` under it keeps transparency, justice and fairness, non-maleficence, responsibility and privacy. Beneficence just misses, with 41 of 84.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_meanings():
    mo.md(r"""
    ### Exercise 2. One word, three meanings

    Each of these six cases is called transparency in some guideline. Put each one with what it makes transparent.
    """)
    return


@app.cell(hide_code=True)
def sort_meanings_cell():
    sort_meanings = L.sort("meanings")
    sort_meanings
    return (sort_meanings,)


@app.cell(hide_code=True)
def check_meanings(sort_meanings):
    result_meanings = L.check("meanings", sort_meanings)
    result_meanings
    return (result_meanings,)


@app.cell(hide_code=True)
def hints_meanings():
    L.hints(
        "meanings",
        [
            "Ask for each case: is it about one person's decision, about telling people that AI is there, or about how the whole system was made?",
            "Two cases are about a single case of a single person. Two others only tell you that AI is involved, and explain nothing.",
            "Explain one decision: the refused loan and \\_\\_\\_. Say that AI is used: the chat window and \\_\\_\\_. Open up the system: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Explain one decision:** the refused loan and the flagged message. **Say that AI is used:** the chat window and the label. **Open up the system itself:** the data and the limits.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fairness():
    mo.md(r"""
    ### Exercise 3. The same word
    """)
    return


@app.cell(hide_code=True)
def choice_fairness_cell():
    choice_fairness = L.choice(
        "fairness",
        question="Two companies both name fairness as a principle. For company A, it means that its system makes as many mistakes for one group of customers as for another; for company B, that the same rule holds for every customer. What does the shared word tell you?",
        options={
            "same": "That their systems will treat the same people in the same way",
            "right": "Little: one word can ask for quite different things",
            "stronger": "That company A takes fairness more seriously than company B",
        },
        correct="right",
        why={
            "same": "The two meanings can even pull apart: one rule for every customer can still make more mistakes for one group. What did Jobin, Ienca and Vayena find about the meanings? See *The idea*.",
            "right": "The words agree, the meanings do not, as Jobin, Ienca and Vayena found. What a principle asks for, you only learn from what it means in that document.",
            "stronger": "Both companies mean something by fairness; they mean different things. Nothing here says which one does more for it. See *The idea*.",
        },
    )
    choice_fairness
    return (choice_fairness,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) you met theories that each ask their own question. A principle is shorter: a word such as fairness or no harm, which each theory can fill in its own way, and that is part of why so many documents agree on it. In the next lesson, *When principles are only words*, you test whether a principle changes anything at all.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fairness, result_common, result_meanings):
    L.wrap_up(result_common, result_meanings, choice_fairness)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
