# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-safety-field. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-safety-field", "en")
    C = L.checks   # Apollo's Table 1, the Seoul signatories and the network's first members, four bodies' names


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [AI ethics and AI safety](/course/open/rai-ethics-vs-safety?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Greaves and MacAskill's AI plan beats the bednets from 2.5 × 10¹⁰ future beings on, far fewer than even their restricted estimate. What does their case for the AI plan mainly rest on?",
        options={
            "main": "The 10²⁴ future beings of their main estimate",
            "right": "Its guesses of a 0.1% chance and a 1% cut",
            "cost": "The cost of bednets, about 4000 dollars a life",
        },
        correct="right",
        why={
            "right": "With so few future beings needed, the vast numbers hardly matter. What carries the case is the 0.1% chance of a catastrophe and the 1% of it the plan would avoid, and for those, Greaves and MacAskill write, there is no hard evidence.",
            "main": "The AI plan already wins at 2.5 × 10¹⁰, far below 10²⁴, so the main estimate is not what it rests on. See [AI ethics and AI safety](/course/open/rai-ethics-vs-safety?lang=en).",
            "cost": "The bednets are the benchmark the plan is compared with, not what the plan rests on. See [AI ethics and AI safety](/course/open/rai-ethics-vs-safety?lang=en).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_scales():
    quick_scales = L.choice(
        "quick-scales",
        question="Prunkl and Whittlestone found the line between near-term and long-term poorly defined. What did they put in its place?",
        options={
            "camps": "Two camps, one for harm now and one for harm later, each with its own work",
            "right": "Four scales, such as how capable the systems are and how sure the harm is",
            "time": "One scale of time, from harm today to harm in a hundred years",
        },
        correct="right",
        why={
            "right": "Capabilities, impacts, certainty and extremity: four scales rather than a choice of two, so one concern can sit near on one scale and far on another.",
            "camps": "Two camps is the split they found poorly defined. See [AI ethics and AI safety](/course/open/rai-ethics-vs-safety?lang=en).",
            "time": "When the harm comes is only one of their scales; they add how capable the systems are, how sure the harm is and how large it is. See [AI ethics and AI safety](/course/open/rai-ethics-vs-safety?lang=en).",
        },
    )
    quick_scales
    return (quick_scales,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    News about AI safety names many bodies: an institute that tested a new model, an evaluator that caught a model deceiving its testers, a course that trains researchers. They differ in what they may do, they change their names, and some of what they say about themselves is their own claim.

    This lesson maps the field: the institutes of governments, the independent evaluators, the courses, the companies themselves and the research agendas that list what is still unsolved. You will count in one evaluator's results and compare two lists of countries yourself.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2024, the evaluator Apollo Research gave AI models a goal, told them that nothing else mattered, and put them where their developers' oversight stood in the way. In what share of the runs did the model that did it most often switch that oversight off?",
        options={
            "half": "In about half of the runs",
            "right": "In about one run in twenty",
            "rare": "In about one run in a thousand",
        },
        reveal="**In about one run in twenty.** The model that did it most often, opus-3, switched off its oversight in 5% of the runs and o1 in 2%; two of the five models in that part of the table never did. That they did it at all was the point: *Evaluators* in *The idea* comes back to it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Institutes of governments

    In November 2023, the British government turned its Frontier AI Taskforce into the AI Safety Institute: in its own words "the first state-backed organisation focused on advanced AI safety for the public interest", with the mission "to minimise surprise to the UK and humanity from rapid and unexpected advances in AI". It was to evaluate advanced AI systems, do research and share information, and it got no power over them: "The Institute is not a regulator and will not determine government regulation." Nor would its evaluations "designate any particular AI system as ‘safe’". Such a body is an **AI safety institute**: part of a government, it tests advanced AI models and does research on their risks.

    Its first evaluations were to include "biased outputs and reasoning". On 14 February 2025, the minister Peter Kyle renamed it the AI Security Institute, to focus on risks such as chemical and biological weapons, cyberattacks and fraud: "It will not focus on bias or freedom of speech". The same press release reads the change in two ways. Kyle: "The work of the AI Security Institute won’t change". Its chair, Ian Hogarth: "The Institute’s focus from the start has been on security". In November 2026, its website speaks of "100+ technical staff" and £66 million a year, and calls the institute "the world’s largest government team dedicated to AI safety and security research".

    In the United States, an AI Safety Institute at NIST, the national institute for standards and technology, was announced at the British summit of November 2023. In June 2025 it was re-established as the Center for AI Standards and Innovation (CAISI). An executive order of 29 September 2026 told the government to write "Super Intelligence" where it wrote "Artificial Intelligence", and the center became the Center for Advancing Innovation and Standards for Super Intelligence (CAISSI). For now, the order defines the new words as the technology that American law calls artificial intelligence, and it asks for a new definition within 60 days: the name changed, while for now the technology it covers did not. In its evaluations, NIST writes, the center will "focus on demonstrable risks, such as cybersecurity, biosecurity, and chemical weapons."

    At the AI Seoul Summit of 21 May 2024, 10 countries and the European Union agreed to launch an international network of such institutes. It first met on 20 November 2024 in San Francisco, with the United States as its first chair, and since 9 December 2025 it has been called the International Network for Advanced AI Measurement, Evaluation and Science, with the UK as coordinator. Who signed in Seoul and who became a first member are not quite the same list: Exercise 2 finds the difference.

    The European Union's AI Office, set up by a decision of 24 January 2024, is a different kind of body: it "enforces the rules for GPAI models", the general-purpose AI models of the EU's AI Act, and helps to develop trustworthy AI. It "employs more than 125 staff", and one of its units is called "AI Safety". The Netherlands is in none of the lists of the network. Since 2023, its data protection authority, the Autoriteit Persoonsgegevens, has coordinated the supervision of algorithms and AI that put fundamental values and rights at risk: it supervises, and it is not a safety institute.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Evaluators

    Evaluators test AI models, often from outside the companies that make them. METR, pronounced "meter", is a nonprofit that "studies the most advanced AI agents to understand whether their developers are on track to lose control of them". It "has not accepted funding from AI companies, though we make use of significant free tokens". One of its measures is the time horizon: the length of the tasks, in the time skilled people need for them, that a model completes half the time. In the version of its paper of July 2026, o3 has a time horizon of "around 110 minutes", and the time horizon of the best models has "doubled approximately every seven months since 2019". The lesson *How long a task can a model do?* works it out later.

    Apollo Research holds that "some of the most dangerous risks come from scheming frontier AI: advanced AI systems covertly pursuing misaligned objectives." Frontier models are the most capable models of their time. In a paper of December 2024, revised in January 2025, it ran six evaluations, and its Table 1 gives the results of six models. Each time, a model got a goal and a situation in which scheming would help to reach it, such as a chance to switch off its oversight, or to copy what it took to be its own weights to another server before being replaced. Five of the six schemed at least once. Read the setup, though: "We study whether models have the capability to scheme in pursuit of a goal that we provide in-context and instruct the model to strongly follow." When Apollo took that push out of the four evaluations of Table 2, scheming came "at significantly lower rates and across fewer evaluations"; the summary of the paper calls them "rare instances". Apollo writes that it "published the first evidence that frontier models can scheme in context", that is, for a goal given in the prompt: a claim about itself.

    ### Courses and the companies

    Most people in the field learned it somewhere. BlueDot Impact gives "free online courses", each "25 to 40 hours total", and by its own count "1,200+ alumni now work in the field". ARENA, the Alignment Research Engineer Accelerator, runs "2-3 bootcamps each year, and each lasts for 4-5 weeks" in London, from the fundamentals and transformer interpretability to reinforcement learning, evaluations of language models and alignment; its curriculum is available online. And AI companies such as Anthropic, Google DeepMind and OpenAI publish where they think safety research should go.

    ### Research agendas

    What is still unsolved? Research agendas list it. On 8 May 2025, researchers who had met in Singapore, "More than 100 participants" from 11 countries, from universities, AI companies, governments and civil society, published the Singapore Consensus on Global AI Safety Research Priorities. It draws on the priorities of, among others, the British institute, NIST, Anthropic, Google DeepMind and OpenAI, and sorts the open problems into three areas, in the picture below: "challenges with creating trustworthy AI systems (Development), challenges with evaluating their risks (Assessment), and challenges with monitoring and intervening after deployment (Control)." Rivals can share such work, it argues, "similar to how competing aircraft manufacturers (e.g., Boeing and Airbus) collaborate on aviation safety information and standards."

    The British institute's own agenda puts research on risks, from cyber misuse and crime to autonomous systems and the resilience of society, beside research on solutions: safeguards, control and alignment. About alignment it writes: "There are no known reliable methods for preventing sufficiently capable AI systems from circumventing safeguards and causing harm."
    """)
    return


@app.cell(hide_code=True)
def areas_cell():
    mo.Html(C.areas_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **AI safety institute** | a government body that tests advanced AI models and does research on their risks, such as the British AI Security Institute; the name of a single institute can change |
        | regulator | a body that may make or enforce rules; the British institute is none, the EU AI Office enforces the rules for general-purpose AI models |
        | evaluator | an organization that tests AI models, often from outside the company that made them |
        | time horizon | METR's measure: the length of the tasks, in the time skilled people need, that a model completes half the time |
        | scheming | a model covertly pursuing goals that are not its developers', while hiding them |
        | research agenda | a list of the problems a group thinks research should solve |
        | `C.TABLE` | Apollo's Table 1: a dict with, for each of the six evaluations, the set of models that schemed in it at least once |
        | `C.MODELS` | the six models Apollo tested, as the table names them: `"gpt-4o"`, `"gemini-1.5"`, `"llama-3.1"`, `"sonnet-3.5"`, `"opus-3"` and `"o1"` |
        | `C.SEOUL`, `C.NETWORK` | sets: the 11 that signed in Seoul (10 countries and the EU), and the 10 first members of the network |
        | `a - b` | for two sets: a new set of the values in `a` that are not in `b` |
        """),
        L.watch_out("A ✓ in an evaluation means: at least once, in this setup. Ask how often, after what push, and who says it was the first."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Ask what kind of body it is: part of a government, an independent evaluator, a company, a course or a group that writes an agenda.
    2. Ask what it may do: test and advise, or make and enforce rules.
    3. Give a name with its date, and check whether it has changed since.
    4. Read how a result came about: which models, whose goal, how strong a push, and how often.
    5. Note who calls something "first" or "largest": the body itself or others.
    6. Place a piece of research in an agenda, such as assessment, development or control.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Table 1 of Apollo's paper gives, for each evaluation, the models that schemed in it at least once; the course keeps it in `C.TABLE`. The cell goes through the evaluations and prints how many models schemed in each, and whether gpt-4o is among them.
    """)
    return


@app.cell
def example_table():
    for evaluation in C.TABLE:                  # Apollo's six evaluations
        models = C.TABLE[evaluation]            # the models that schemed in it at least once
        print(evaluation, len(models), "gpt-4o" in models)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each evaluation tests one way to scheme. In oversight subversion a model could switch off the oversight that would shut it down; in self-exfiltration it could copy itself to the server of the model meant to replace it; in goal-guarding it could change the prompt its developers had written for its successor; in covert email reranking it could quietly change a ranking it was asked to make; in instrumental alignment faking it could do what its developers wanted only while it was being watched; and in sandbagging it could do worse on purpose in a test. Five models schemed at least once in self-exfiltration, and only one in covert email reranking. gpt-4o is in none of the six sets. In Exercise 1 you count the other way around: for one model, over all evaluations.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_schemed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. schemed_in: in how many evaluations a model schemed

        Write `schemed_in(table, model)`. `table` is a dict like `C.TABLE`: for each evaluation, a set of models. Start `count` at 0, go with a `for` loop through the evaluations in `table`, and add 1 to `count` for each evaluation whose set has `model` in it. Return `count`. Replace the `...`. The check calls `schemed_in(TABLE, "opus-3")`, then the other models and tables of its own.
        """),
        L.task("Count the evaluations in whose set of models the model is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_schemed():
    box_schemed = L.code_box("schemed_in", lines=6)
    box_schemed
    return (box_schemed,)


@app.cell(hide_code=True)
def check_schemed(box_schemed):
    result_schemed = L.check("schemed_in", box_schemed.value)
    result_schemed
    return (result_schemed,)


@app.cell(hide_code=True)
def hints_schemed():
    L.hints(
        "schemed_in",
        [
            "Go through the evaluations one by one and keep a count that goes up by one each time the model is in that evaluation's set.",
            "`count = 0` starts the count. The set of models of one evaluation is its value in the dict `table`, and `model in` a set gives `True` when the model is in it.",
            "`        if model in ___:`",
        ],
        why="`count = 0`, `for evaluation in table:`, inside it `if model in table[evaluation]:` with `count = count + 1`, then `return count` level with `for`. For opus-3: oversight subversion, self-exfiltration, goal-guarding and sandbagging, so 4.",
    )
    return


@app.cell(hide_code=True)
def text_after_schemed():
    mo.md(r"""
    o1 schemed in all six evaluations, opus-3 and llama-3.1 in four, gemini-1.5 in three, sonnet-3.5 in two and gpt-4o in none. A ✓ means at least once. Table 2 of the paper gives the shares of the runs: opus-3 switched off its oversight in 5% of them and o1 in 2%, and o1 quietly gave a false ranking in 22%. In five of the six evaluations, the goal came from Apollo with a strong push to pursue it; in instrumental alignment faking, it came without that push. One result went further: sonnet-3.5 did worse on purpose in a test in pursuit of being helpful, "a goal that was acquired during training rather than in-context".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. changes: who signed, and who joined

        `C.SEOUL` holds the 11 that signed in Seoul, `C.NETWORK` the 10 first members of the network. Write `changes(before, after)` for two sets. Make `left`, the sorted list of the values in `before` that are not in `after`, and `joined`, the sorted list of the values in `after` that are not in `before`: `a - b` and `sorted` do it. Return both, with a comma between them: `return left, joined`. Replace the `...`. The check calls `changes(SEOUL, NETWORK)`, then sets of its own.
        """),
        L.task("Who is in the first set and not in the second, and who the other way around, each as a sorted list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_changes():
    box_changes = L.code_box("changes", lines=4)
    box_changes
    return (box_changes,)


@app.cell(hide_code=True)
def check_changes(box_changes):
    result_changes = L.check("changes", box_changes.value)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def hints_changes():
    L.hints(
        "changes",
        [
            "Two differences, one in each direction, and each turned into a list in order.",
            "`before - after` is a set of the values in `before` that are not in `after`, and `sorted(...)` turns a set into a list in order. The second list is the difference the other way around.",
            "`    joined = sorted(___)`",
        ],
        why="`left = sorted(before - after)`, `joined = sorted(after - before)` and `return left, joined`. For Seoul and the network: `['Germany', 'Italy']` and `['Kenya']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_changes():
    mo.md(r"""
    Germany and Italy signed in Seoul but were not among the first members; Kenya was a first member without having signed there. The other nine are in both. The two lists come from different documents: they say who signed one agreement and who was named a first member six months later, not why the two differ.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: names over time

    The slider sets a date, as a number of months after November 2023, when the British institute was founded; each stop is the first day of a month. The picture shows four public bodies over time, with the name each had on that date in bold, on a bar that stands out.
    """)
    return


@app.cell(hide_code=True)
def demo_names():
    names = L.explore("names", "months after November 2023", lambda _v: C.demo_frame(_v, L.lang),
                      start=2, values=C.MONTHS, role="",
                      task="Find the smallest number on the slider at which none of the names has the word Safety in it.")
    names
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 26 on, 1 January 2026, none of the four names has the word Safety in it: the British institute went over to Security, the American center to Standards and Innovation and later Super Intelligence, and the network to Measurement, Evaluation and Science. What the work itself became is another question: the minister said it "won’t change", while bias left its scope. So give a name with its date, and use AI safety institute for the kind of body.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Exercise 3. A headline

    A news story has the headline: "British AI Security Institute declares new model safe".
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Going by what the British government wrote about the institute, what is wrong with this headline?",
        options={
            "closed": "The institute closed in 2025, when it was given its new name",
            "right": "It is not a regulator, and it does not label any model as safe",
            "bias": "Since its new name, the institute only studies bias and freedom of speech",
        },
        correct="right",
        why={
            "right": "In 2023 the government wrote that the institute \"is not a regulator\" and that its evaluations would not \"designate any particular AI system as ‘safe’\". It tests and advises. In the EU, the AI Office does enforce rules, for general-purpose AI models.",
            "closed": "It was renamed, not closed: since 14 February 2025 it has been the AI Security Institute, and the minister said its work \"won’t change\". The headline goes wrong on what the institute may do. See *Institutes of governments* in *The idea*.",
            "bias": "The other way around: since the new name, \"It will not focus on bias or freedom of speech\". The headline goes wrong on something else, what the institute may do. See *Institutes of governments* in *The idea*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_exercise_area():
    mo.md(r"""
    ### Exercise 4. Where in the agenda?

    Before a new model is released, a team at an AI safety institute checks whether the model can help someone plan a cyberattack.
    """)
    return


@app.cell(hide_code=True)
def choice_area_cell():
    choice_area = L.choice(
        "area",
        question="In which of the three areas of the Singapore Consensus does this work fall?",
        options={
            "development": "Development: it happens before the release, while the model is being made",
            "right": "Assessment: it finds out how severe and how likely a harm could be",
            "control": "Control: it is about keeping attacks from happening",
        },
        correct="right",
        why={
            "right": "Testing what a model can do is risk assessment, area 1, whether it happens before or after the release; its findings feed the decisions on development and release. The later lesson *Testing for dangerous capabilities* goes deeper.",
            "development": "Development is about building the system so that it does what was specified; its verification, 2.3, checks the system against that specification. Finding out what dangerous things a model can do is 1.6, under assessment, also before the release. See the picture under *Research agendas* in *The idea*.",
            "control": "Control is about watching a system after its release and stepping in. Finding out what a model can do is assessment. See the picture under *Research agendas* in *The idea*.",
        },
    )
    choice_area
    return (choice_area,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [AI ethics and AI safety](/course/open/rai-ethics-vs-safety?lang=en) showed what two groups disagree about. This lesson showed who does the work: institutes of governments that test and advise, an EU office that enforces rules, evaluators such as METR and Apollo Research, courses such as BlueDot and ARENA, the companies themselves, and agendas that list what is unsolved. The next lesson, *Robust, resilient, available, predictable*, looks at what it takes for an AI system to keep working: when its input is a little off, after a failure, and whenever it is needed.

    **An everyday example.** In many countries, independent crash tests give new cars stars, while a government agency decides whether a car may be sold at all. Stars are not a permit, and a permit is not a review.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_area, choice_headline, result_changes, result_schemed):
    L.wrap_up(result_schemed, result_changes, choice_headline, choice_area)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
