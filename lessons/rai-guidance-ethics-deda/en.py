# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-guidance-ethics-deda. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-guidance-ethics-deda", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Designing with values](/course/open/rai-design-with-values?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hierarchy():
    quick_hierarchy = L.choice(
        "quick-hierarchy",
        question="In a values hierarchy, what comes directly under a value?",
        options={
            "requirement": "A requirement with a number to test",
            "right": "Norms that say what it asks here",
            "code": "The code that the team writes",
        },
        correct="right",
        why={
            "requirement": "Requirements come at the bottom, under the norms. See *The idea* in [Designing with values](/course/open/rai-design-with-values?lang=en).",
            "right": "Norms say what the value asks of this system; requirements under them make the norms checkable.",
            "code": "Code is not part of the hierarchy: the hierarchy says what the code must do. See *The idea* in [Designing with values](/course/open/rai-design-with-values?lang=en).",
        },
    )
    quick_hierarchy
    return (quick_hierarchy,)


@app.cell(hide_code=True)
def quick_check_requirement():
    quick_requirement = L.choice(
        "quick-requirement",
        question="Which of these is a requirement that an engineer can check?",
        options={
            "right": "A CV is deleted after at most 30 days",
            "respect": "The tool treats every user with respect and care",
            "privacy": "Applicants' privacy is protected",
        },
        correct="right",
        why={
            "respect": "Respect is a value, and treating people with care is a norm at most: how would you test it? See *The idea* in [Designing with values](/course/open/rai-design-with-values?lang=en).",
            "right": "It has a number, so a test can pass or fail: a requirement.",
            "privacy": "Privacy is the value itself. What would you measure? See *The idea* in [Designing with values](/course/open/rai-design-with-values?lang=en).",
        },
    )
    quick_requirement
    return (quick_requirement,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Many towns, schools and government bodies want to use data and AI, and few of them have an ethicist on staff. Two Dutch methods give a team a way to think it through together, in an afternoon, with the people involved. Both are used by Dutch municipalities and government bodies. This lesson walks through both.

    **An everyday example.** Before a family gets a dog, they sit down and talk it through: who walks it, what if someone is allergic, what does it cost? The question is not whether dogs are good, but how a dog would fit into this family.
    """)
    return


@app.cell(hide_code=True)
def predict_table():
    guess_table = L.guess(
        "table",
        question="A Dutch town wants a camera on a square that warns the police when a fight starts. It holds a guidance ethics workshop about it. Who do you think sits at the table?",
        options={
            "ethicists": "Mostly ethicists from the nearest university",
            "lawyers": "The town's own lawyers and its officials",
            "right": "Everyone involved, residents included",
        },
        reveal="**Everyone involved**: residents, police officers, shop owners, the makers of the camera and the town. The method leans on the experience of the people who will live with the technology more than on outside experts.",
    )
    guess_table
    return (guess_table,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Both methods are for a team that starts on a concrete project, not for philosophers.

        **Guidance ethics** (in Dutch *begeleidingsethiek*), from the philosopher Peter-Paul Verbeek and ECP, a Dutch platform for the information society, asks not *whether* a technology should be allowed, but *how* it can be used well. A workshop goes in three steps, shown in the picture. First the case: what the technology is and where it will be used. Then the dialogue: who the actors are, what good and bad effects it will have for each of them, and which values are at stake. Last the options for action, in three places: in the technology itself, in its surroundings such as rules and the organization, and in the people who use it.

        **DEDA**, the Data Ethics Decision Aid (in Dutch *De Ethische Data Assistent*), comes from the Utrecht Data School and was made with the city of Utrecht. It is a set of questions in themes, such as where the data come from, the algorithm, responsibility and communication. A team goes through them together and writes its answers down, so that later it can show why it decided what it decided.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **guidance ethics** | a Dutch approach in which users, makers and others involved discuss a concrete technology together, to steer how it is used rather than to say yes or no to it |
        | **DEDA** | the Data Ethics Decision Aid of the Utrecht Data School: questions in themes that a team, often at a government body, goes through together for a data or algorithm project |
        """),
        L.watch_out("A workshop is not a check. Both methods help a team ask good questions and write down its choices; whether the choices are carried out still needs requirements that someone tests, as in [Designing with values](/course/open/rai-design-with-values?lang=en)."),
        mo.Html(C.picture({"title": "The three steps of a guidance ethics workshop",
                           "steps": ["The case", "The dialogue", "Options for action"],
                           "texts": ["What is the technology, and where will it be used?",
                                     "Who are the actors? What good and bad effects will it have for each? Which values are at stake?"],
                           "places": ["in the technology", "in the surroundings: rules, organization", "in the people who use it"],
                           "alt": "Three steps side by side. 1, the case: what is the technology, and where will it be used? 2, the dialogue: who are the actors, what good and bad effects will it have for each, which values are at stake? 3, options for action: in the technology, in the surroundings, and in the people who use it."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To think a data or AI project through as a team:

    1. Describe the case: the technology and the place where it will be used.
    2. Invite the actors: the people who make it, use it and live with it.
    3. Go through the effects and the values, or through DEDA's themes, one at a time.
    4. Write down each answer, and mark the questions that are still open.
    5. Choose options for action in the technology, the surroundings and the people, and decide who carries out each one.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small team, made up by the course, starts on two of DEDA's themes. `questions` says how many questions each theme has for this project, `answered` how many the team has answered. The share answered is the one divided by the other. Run the cell.
    """)
    return


@app.cell
def example_two_themes():
    _questions = {"source": 10, "algorithm": 4}
    _answered = {"source": 7, "algorithm": 1}

    for _theme in _questions:
        print(_theme, _answered[_theme] / _questions[_theme])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The team has answered 0.7 of its questions about where its data come from, and only 0.25 of those about the algorithm: step 4 of the recipe shows where to go on. Exercise 1 does the same for a team with four themes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_done():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Where is the team?

        Another team, also made up, is further along. `questions` holds how many questions each theme has, `answered` how many the team has answered, and the loop goes through the themes with `theme`. Replace the `...` with one line, as in the worked example, but put each share in `done` instead of printing it. The last line is `done`, so the check sees it. The check also tries other answers.
        """),
        L.task("Put in `done`, under each theme, the share of its questions the team has answered."),
    ])
    return


@app.cell(hide_code=True)
def exercise_done():
    box_done = L.code_box("done", lines=7)
    box_done
    return (box_done,)


@app.cell(hide_code=True)
def check_done(box_done):
    result_done = L.check("done", box_done.value)
    result_done
    return (result_done,)


@app.cell(hide_code=True)
def hints_done():
    L.hints(
        "done",
        [
            "The division of the worked example, with its answer stored under the theme.",
            "Which of the two numbers goes on top of the division? And how do you put a value under a key in a dictionary?",
            "`done[theme] = ___[theme] / ___[theme]`",
        ],
        why="`done[theme] = answered[theme] / questions[theme]` gives `{'source': 1.0, 'algorithm': 0.5, 'responsibility': 0.25, 'communication': 1.0}`: only a quarter of the questions about responsibility are answered.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_places():
    mo.md(r"""
    ### Exercise 2. Where does it act?

    The workshop about the camera on the square came up with six options for action. Put each one where it acts.
    """)
    return


@app.cell(hide_code=True)
def sort_places_cell():
    sort_places = L.sort("places")
    sort_places
    return (sort_places,)


@app.cell(hide_code=True)
def check_places(sort_places):
    result_places = L.check("places", sort_places)
    result_places
    return (result_places,)


@app.cell(hide_code=True)
def hints_places():
    L.hints(
        "places",
        [
            "Ask for each option: what changes, the camera, the rules and reports around it, or what a person knows?",
            "Two options change the camera itself. Two others change nothing about the camera or the rules, only what people know.",
            "In the technology: blurred and \\_\\_\\_. In the surroundings: a rule and \\_\\_\\_. In the people: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**In the technology:** blurred and the square only. **In the surroundings:** a rule and a report. **In the people who use it:** training and feedback.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_how():
    mo.md(r"""
    ### Exercise 3. Not whether, but how
    """)
    return


@app.cell(hide_code=True)
def choice_how_cell():
    choice_how = L.choice(
        "how",
        question="An ethics committee is asked to say yes or no to the camera on the square. Which question does guidance ethics ask instead?",
        options={
            "allowed": "Whether this technology should be allowed at all, yes or no",
            "law": "Which law the technology must obey",
            "right": "How this technology can be used well, here and now",
        },
        correct="right",
        why={
            "allowed": "That is the yes-or-no question of the committee. Guidance ethics starts from the technology as it is coming, and asks something else. See *The idea*.",
            "law": "The law matters, but guidance ethics asks about values and use, not only about what is legal. See *The idea*.",
            "right": "Guidance ethics asks how the technology can be used well: it looks for options in the technology, the surroundings and the people, instead of a single verdict.",
        },
    )
    choice_how
    return (choice_how,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Designing with values](/course/open/rai-design-with-values?lang=en) you went from a value to requirements for one system. Guidance ethics and DEDA are ways to make those choices together, with the people involved, and to write them down. In the next lesson, *Who is responsible?*, you ask who answers for a system when it goes wrong anyway.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_how, result_done, result_places):
    L.wrap_up(result_done, result_places, choice_how)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
