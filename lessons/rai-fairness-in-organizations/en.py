# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-fairness-in-organizations. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-fairness-in-organizations", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ratio():
    quick_ratio = L.choice(
        "quick-ratio",
        question="A model's selection rates are 0.30 for women, 0.50 for men and 0.40 for a third group. Fairlearn's demographic parity ratio divides the smallest selection rate by the largest. What is it?",
        options={
            "right": "0.6",
            "third": "0.75",
            "top": "0.8",
        },
        correct="right",
        why={
            "right": "0.30 / 0.50 = 0.6: the smallest selection rate divided by the largest, over all groups.",
            "third": "That is 0.30 / 0.40, the women against the third group. The ratio takes the smallest and the largest rate of all groups: 0.30 / 0.50. See [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
            "top": "That is 0.40 / 0.50, the third group against the men. The ratio takes the smallest and the largest rate of all groups: 0.30 / 0.50. See [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
        },
    )
    quick_ratio
    return (quick_ratio,)


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="Over ten samples of 1,000 people from UCI Adult, reweighing raised the women's yes rate divided by the men's by anything from 0.07 to 0.51. A paper reports one such sample, with a rise of 0.45. What does that one number tell you?",
        options={
            "exact": "Exactly how much the method helps everyone",
            "right": "Little, another split differs a lot",
            "fails": "That the method does not work",
        },
        correct="right",
        why={
            "exact": "One split of a small table says little: over ten splits, the rise ran from 0.07 to 0.51. See [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
            "right": "Another sample could have given 0.07. Report how far the results lie apart over several splits, and use more than one table.",
            "fails": "A rise of 0.45 is no sign that it fails. But one split of a small table says little either way. See [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    This module was about measuring and fixing models. In practice, fairness work often stalls before anyone opens a notebook. Kenneth Holstein and colleagues (2019) interviewed 35 people who build machine learning systems in industry, and surveyed 267 practitioners. Only 21 percent said their team prioritizes fairness "A lot" or "A great deal"; 36 percent said "Not at all". Many teams checked for fairness only after a customer complained, while they looked for security risks before anything went wrong. One practitioner told Michael Madaio and colleagues (2020) what happens when there is no set moment for it: "There's no checkpoint where someone's supposed to say something, and so you can only do it by being this annoying squeaky wheel".

    **An everyday example.** In a restaurant kitchen, someone has to check the temperature of the fridges. If nobody owns that check and there is no set time for it, it happens the day after a guest falls ill. With a list on the door that says who checks, when, and what happens when a fridge is too warm, it happens every morning.
    """)
    return


@app.cell(hide_code=True)
def predict_suspect():
    guess_suspect = L.guess(
        "suspect",
        question="Holstein and colleagues asked practitioners whether their team had found fairness issues in its products. Of those who said no, how many suspected there might be issues they had not yet detected?",
        options={
            "few": "Fewer than one in five",
            "half": "About half of them",
            "right": "Four in five",
        },
        reveal="**Four in five.** Of the respondents who got the question, almost half, 49 percent, said their team had found possible fairness issues. Of the 51 percent whose team had not, most, 80 percent, suspected issues they had not yet detected. Mind the denominators: only 63 percent of the respondents got this question, and each percentage counts only those who did.",
    )
    guess_suspect
    return (guess_suspect,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Where it stalls.** Most of the people Holstein and colleagues interviewed saw data collection, not model development, as the most important place to fix things; one engineer called their team's data collection "almost like the wild west". Many teams could collect data on groups "only ... at coarser levels, if at all", and 70 percent of the 69 percent who got that question said tools to audit fairness without individual demographic data would be at least "Very" useful. Madaio and colleagues (2022) worked with 33 practitioners in ten teams at three technology companies. They found it hard to choose the metrics, to know which people and groups to focus on, and to collect data to evaluate per group, and their choices followed the business: "If the organization is talking only about business performance, then the project is going to prioritize the groups that are the most important for the business". McKane Andrus and colleagues (2021) interviewed 38 practitioners from 26 organizations. "Almost every participant described access to demographic data as a significant barrier". Companies that collected their own data almost all had gender and age; outside employment, healthcare and some financial services, few had data on race, so most did not check their systems for racial bias. Some had to justify a bias analysis with expected gains in key performance indicators, and if it failed, the work could just be abandoned. Andrus and colleagues do not conclude that teams should simply collect more data on people: that raises its own questions, such as privacy, consent, which categories to use, and whether a company should guess them at all.

        **Reactive, proactive, anticipatory.** Bogdana Rakova and colleagues (2021) interviewed 26 people from 19 organizations on four continents, in late 2019: "many organizations behave reactively, fewer are now proactive, and respondents aspire for their organizations to become anticipatory in the future." In reactive organizations the most common push to act was "catastrophic media attention", and one person asked in meetings: "What if ProPublica found out about ...?" About 40 percent, 11 of the 26, volunteered time for responsible AI work outside their official job. A few organizations are proactive: they have set processes to evaluate and review their systems, with the work and the accountability often spread over several teams, and some have internal review boards. Every organization, they write, needs a way to answer four questions: "When and how do we act?", "How do we measure success?", "What are the internal structures we rely on?" and "How do we resolve tensions?" It is crucial, they write, to ask whether a system is appropriate in the first place, and "the decision to cease further development is respected and cannot be overruled by other considerations."

        **Pause points.** Madaio and colleagues (2020) designed a fairness checklist together with 48 practitioners from 12 technology companies. It follows the stages of a product: envision, define, prototype, build, launch and evolve. From aviation and medicine they took the lesson that a checklist is used when it is tied to the right moments, "pause points". So each stage names moments that teams already have: envisioning meetings and pre-mortem screenings, where a team imagines in advance how the product could go wrong, spec and design reviews, go/no-go discussions and code reviews, the ship review before launch, and regular product reviews after it. One item is "consider aborting development". The checklist warns: "Undertaking the items in this checklist will not guarantee fairness. The items are intended to prompt discussion and reflection." The authors add that a checklist can give the impression that a simple compliance process guarantees fairness.

        **A fairness review.** None of these papers defines the term; the course uses it for what they describe together. A *fairness review* is a planned moment in the work on a product where a team with mixed roles measures the system per group, writes down who could be harmed and what will be done about it, and someone with the authority to do so decides to go on, to change it or to stop. Planned: it is in the schedule, like a code review, and does not wait for a complaint. Mixed roles: the people who build the system, someone who knows the law, someone who knows the people it affects. A decision: and a decision to stop holds.

        **In the Netherlands and the EU.** The Dutch government has its own way of doing such a review: the IAMA, the *Impact Assessment Mensenrechten en Algoritmes*, made by Utrecht University for the Ministry of the Interior. Version 2 appeared on 16 February 2026; version 1 is from 2021. It calls itself "een instrument voor discussie en besluitvorming", an instrument for discussion and decision-making, and "geen afvinklijstje", not a list to tick off. It advises a team of 4 to 7 people, with at least a project leader, a data scientist and a lawyer. Its five parts ask why, what and how, then go through human rights, and close; along the way it asks which assumptions and bias lie in the input data, and whether the algorithm makes a direct or indirect distinction that equal treatment law covers. The team writes an advice on whether to use the algorithm and on what conditions; the person responsible for it decides. If the human rights part does not hold up, the advice is to go back to the drawing board. Version 2 includes what Article 27 of the AI Act asks: the fundamental rights impact assessment (FRIA). It applies to public bodies, private entities that provide public services, and deployers of systems that score people's creditworthiness or assess risk and set prices for life and health insurance. Before first using such a high-risk AI system, they must describe how and how often they will use it, which groups it may affect, the risks of harm, the human oversight, and what they will do if the risks occur, including complaint mechanisms. They report the results to the market surveillance authority. Critical infrastructure is left out. For these systems the rules apply from 2 December 2027. And since December 2022, Dutch government bodies can describe their algorithms in the public Algorithm Register, algoritmes.overheid.nl; publishing there is not yet compulsory.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **fairness review** | a planned moment in the work on a product where a team with mixed roles measures the system per group, writes down who could be harmed and what will be done, and someone with authority decides to go on, change it or stop; the course's working definition |
        | **pause point** | a moment a team already has, such as a design review or the ship review, at which a checklist is used |
        | **reactive** and **proactive** | acting after complaints or bad press, or with set processes to evaluate and review a system in advance (Rakova and colleagues) |
        | **IAMA** | the Dutch *Impact Assessment Mensenrechten en Algoritmes*: a structured discussion that ends in an advice on whether to use an algorithm |
        | **FRIA** | the fundamental rights impact assessment that Article 27 of the AI Act asks of some deployers of high-risk AI systems |
        """),
        L.watch_out("A review is not a box to tick. The checklist of Madaio and colleagues says that undertaking its items will not guarantee fairness, and the IAMA calls itself no list to tick off: both are there to help a team talk and decide. A review that never changes or stops anything is a ritual."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To set up fairness reviews in a team:

    1. Put them in the schedule: pick moments the team already has, such as a design review, a go/no-go discussion and the ship review, and one after launch.
    2. Name for each moment who leads it and who takes part: the people who build the system, someone who knows the law, someone who knows the people it affects.
    3. Agree before you look on what you measure, for which groups, and how large a gap may be; say where the data on groups comes from, or why you do not have it.
    4. Write down who could be harmed and what will be done: change the system, add human oversight, or stop.
    5. Let someone with the authority decide, and let a decision to stop hold.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A team's plan, made up by the course: the moments at which it looks at fairness, each with a stage of the checklist and who leads it. The loop counts, for each stage, how many moments fall in it. Run the cell.
    """)
    return


@app.cell
def example_plan():
    # the stages of the checklist, in order, and a team's plan
    _stages = ["envision", "define", "prototype", "build", "launch", "evolve"]
    _moments = [
        {"moment": "design review", "stage": "define", "lead": "designer"},
        {"moment": "code review", "stage": "build", "lead": "engineer"},
        {"moment": "go/no-go discussion", "stage": "build", "lead": "product manager"},
        {"moment": "ship review", "stage": "launch", "lead": "director"},
    ]

    for _stage in _stages:
        _count = 0
        for _moment in _moments:
            if _moment["stage"] == _stage:
                _count = _count + 1
        print(_stage, _count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Three stages get 0: envision, prototype and evolve. Nobody asks whether the system should be built, and nobody looks again after launch, so a problem there comes up only when someone complains. Which stages does another plan leave out? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Stages without a moment

        Work out `missing`: the stages, in the order of `stages`, in which no moment of `moments` falls. Start with `missing = []` and loop over `stages` with `stage`. For each stage, count the moments in it as in the worked example: `count = 0`, a loop over `moments` with `moment`, and `count = count + 1` when `moment["stage"] == stage`. After that inner loop, if `count == 0`, `missing.append(stage)`. The check also tries another plan.
        """),
        L.task("Work out `missing`: the stages in which no moment of the plan falls, in the order of `stages`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing", lines=20)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "As in the worked example: a loop over the stages, and inside it a loop over the moments that counts. This time you do not print the count; you keep the stages where it stays 0.",
            "Before the inner loop, `count = 0`; inside it, `if moment[\"stage\"] == stage:` with `count = count + 1`. After the inner loop, at the level of `for moment`, check `if count == 0:`.",
            "Under `if count == 0:`: `missing.append(___)`.",
        ],
        why="`missing = []`, then `for stage in stages:` with `count = 0` under it, and `for moment in moments:` with `if moment[\"stage\"] == stage:` and `count = count + 1`. After the inner loop, `if count == 0:` with `missing.append(stage)`. That gives `['define', 'prototype', 'evolve']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proactive():
    mo.md(r"""
    ### Exercise 2. Before or after?
    """)
    return


@app.cell(hide_code=True)
def choice_proactive_cell():
    choice_proactive = L.choice(
        "proactive",
        question="In the words of Rakova and colleagues, which of these makes a team's fairness work proactive rather than reactive?",
        options={
            "complaint": "Fixing each problem a customer reports",
            "right": "A review at set moments before launch",
            "values": "A statement of values on the company website",
        },
        correct="right",
        why={
            "complaint": "Acting on complaints is the reactive way: the problem has already reached people. See *Reactive, proactive, anticipatory* in *The idea*.",
            "right": "Proactive organizations have set processes to evaluate and review their systems in advance, with the work and the accountability spread over several teams. Reactive ones act after complaints or bad press.",
            "values": "A statement says what a company wants, not when anyone checks. Proactive work has set moments and people who own them. See *Reactive, proactive, anticipatory* in *The idea*.",
        },
    )
    choice_proactive
    return (choice_proactive,)


@app.cell(hide_code=True)
def text_exercise_behind():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Which groups need work?

        In a review before launch, a team looks at its speech recognizer per group. Suppose it had the word error rates in `wer`, which Allison Koenecke and colleagues (2020) measured for five speech recognition systems on average: the errors in a transcript per word said, so lower is better. Before looking, the team agreed on `limit`: how far above the best group's WER a group may lie. Work out `behind`: the groups, in the order of `wer`, whose WER is above `best + limit`, where `best` is the lowest WER. First collect the values in a list with a loop over `wer` and take `best = min(values)`; then start `behind = []` and loop over `wer` again. The check also tries a smaller limit.
        """),
        L.task("Work out `behind`: the groups whose WER is more than `limit` above the lowest WER."),
    ])
    return


@app.cell(hide_code=True)
def exercise_behind():
    box_behind = L.code_box("behind", lines=16)
    box_behind
    return (box_behind,)


@app.cell(hide_code=True)
def check_behind(box_behind):
    result_behind = L.check("behind", box_behind.value)
    result_behind
    return (result_behind,)


@app.cell(hide_code=True)
def hints_behind():
    L.hints(
        "behind",
        [
            "Two steps: first the best group's WER, the lowest one; then every group that lies more than `limit` above it.",
            "A loop `for group in wer:` gives the names; `wer[group]` gives the WER. Collect the WERs with `values.append(wer[group])`, and take `best = min(values)`.",
            "In the second loop: `if wer[group] > ___:` with `behind.append(group)` under it.",
        ],
        why="`values = []`, `for group in wer:` with `values.append(wer[group])`, and `best = min(values)`. Then `behind = []` and `for group in wer:` with `if wer[group] > best + limit:` and `behind.append(group)`. That gives `['black men', 'black women']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fria():
    mo.md(r"""
    ### Exercise 4. Who must do a FRIA?
    """)
    return


@app.cell(hide_code=True)
def choice_fria_cell():
    choice_fria = L.choice(
        "fria",
        question="Under the AI Act, which of these deployers must assess the impact on fundamental rights before it first uses its AI system?",
        options={
            "shop": "A shop that uses AI to sort its incoming mail",
            "right": "A bank that scores loan applicants with AI",
            "grid": "A grid operator that uses AI to keep its grid stable",
        },
        correct="right",
        why={
            "shop": "Sorting mail is not a high-risk use. Article 27 names public bodies, private entities that provide public services, and deployers of credit scoring and of pricing for life and health insurance. See *In the Netherlands and the EU* in *The idea*.",
            "right": "Scoring the creditworthiness of people is a high-risk use for which Article 27 asks every deployer, also a private bank, to assess the impact on fundamental rights before first use. For these systems it applies from 2 December 2027.",
            "grid": "Critical infrastructure counts as high-risk, but Article 27 leaves it out: no fundamental rights impact assessment is asked there. See *In the Netherlands and the EU* in *The idea*.",
        },
    )
    choice_fria
    return (choice_fria,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en) the question was what a fairness number can show. Here it is who computes it, when, and what happens next. The measurements of this module only help when someone has the moment, the data and the authority to act on them: a fairness review puts those in the schedule. The IAMA and the FRIA are such reviews, written down for governments and some companies; they ask for a discussion and a decision, not for ticked boxes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fria, choice_proactive, result_behind, result_missing):
    L.wrap_up(result_missing, choice_proactive, result_behind, choice_fria)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
