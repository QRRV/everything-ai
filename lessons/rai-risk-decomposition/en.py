# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-risk-decomposition. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-risk-decomposition", "en")
    C = L.checks   # a made-up AI assistant on a pharmacy's website: its hazards, an answer and a day at a time


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Robust, resilient, available, predictable](/course/open/rai-reliability?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_robust():
    quick_robust = L.choice(
        "quick-robust",
        question="The course's digit model reads 890 of the 1,000 test digits right, and 452 when the digits move three pixels to the right. Which of the four words is that difference about?",
        options={
            "right": "Robustness: keeping its level on input that is a little different",
            "available": "Availability: the share of requests that do not fail with an error",
            "predictable": "Predictability: giving the same answer each time it sees the same digit",
        },
        correct="right",
        why={
            "right": "The digits are the same digits, a little to one side, and the model loses its level: that is robustness, measured as in Hendrycks and Dietterich's tests.",
            "available": "The model answered for every digit; it was wrong more often. Availability counts answers that do not arrive. See [Robust, resilient, available, predictable](/course/open/rai-reliability?lang=en).",
            "predictable": "The model gives the same answer for the same image every time; the shifted digits are other images. See [Robust, resilient, available, predictable](/course/open/rai-reliability?lang=en).",
        },
    )
    quick_robust
    return (quick_robust,)


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="A dice game: you win €6 if you throw a six, and nothing otherwise. What is the expected value of what you win?",
        options={
            "half": "€3: exactly halfway between nothing and €6",
            "right": "€1: €6 times 1/6, plus €0 times 5/6",
            "prize": "€6: what you win when you throw a six",
        },
        correct="right",
        why={
            "right": "Each value times its probability, added up: 6 × 1/6 + 0 × 5/6 = 1. On average, a throw wins €1.",
            "half": "That is the plain mean of €0 and €6, as if both were equally likely; nothing comes five times as often. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "prize": "That is the best case, which comes one throw in six. The expected value weighs each value by its probability. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A pharmacy puts an AI assistant on its website that answers questions about medicines. Three things can go wrong: it can answer rudely, it can give a wrong dose, and it can miss that someone describes an emergency. Rude answers happen most often; a missed emergency does the most harm. Which should the team work on first?

    How often something goes wrong does not answer that, and neither does how bad it is. This lesson combines the two into risk, adds risk up over everything that can go wrong, and then splits it into four factors, each a place where risk can be lowered.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A made-up chatbot answers 1,000,000 questions a day. 1 question in 100 asks for something dangerous, and the chatbot refuses 99.9% of those. How many dangerous answers does it give a day?",
        options={
            "one": "About 1 a day",
            "right": "About 10 a day",
            "thousand": "About 1,000 a day",
        },
        reveal="**About 10 a day.** 1 in 100 of 1,000,000 questions is 10,000 dangerous questions, and 0.1% of those is 10: 3,650 a year. A chance that looks tiny for one question becomes a steady number when many questions come: that is exposure, which *Exposure and vulnerability* in *The idea* puts in the equation.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Hazard and risk

    In its chapter on safety engineering, Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (2024) defines the hazard: "A hazard is a source of danger that could cause harm." A flammable liquid is one, and so is broken glass. A **hazard** is not yet a risk: a shark is a hazard, but it brings no risk as long as nobody goes into the water. "A threat is a hazard with malicious or adversarial intent", such as a hacker.

    ### The classic risk equation

    "Risk can be broken down into probability and severity", the textbook goes on: the chance that an accident happens, and how much harm it does if it does. As a formula, risk(hazard) = P(hazard) × severity(hazard), "the classic formulation of risk". A volcano that will probably erupt brings little risk when nobody lives near it, as the harm is small; a volcano that will probably stay dormant brings little risk too, even with many people around it, as the chance is small. NIST's AI Risk Management Framework of 2023 defines risk along the same lines, as a combination of both, without a formula, in words adapted from the standard ISO 31000:2018: "the composite measure of an event’s probability of occurring and the magnitude or degree of the consequences of the corresponding event". NIST adds that consequences "can be positive, negative, or both"; this lesson, like the textbook, counts only harm.

    To judge the risk of an AI system, the textbook says, you need to know two things: the chance that it does something unintended, and how bad the consequences are. "A demonstration of a potential catastrophic hazard does not necessarily imply the risk is high": that a model can do something harmful in a test does not yet say how likely it is to do so in use. The textbook asks for both: what a system can do, and what it tends to do.

    ### Adding up the hazards

    "The total risk of a system is the sum of the risks of all associated hazards": risk = Σ P(hazard) × severity(hazard), where Σ adds one term for each hazard. That is an expected value, as in [Expected value and variance](/course/open/prob-expectation?lang=en): each harm times its probability, added up. Risk is the expected harm. The textbook adds up "the risk estimates for each independent hazard": the sum treats the hazards as separate events.

    Zero risk is not the goal, as "reducing risk to zero is likely impossible". A better goal, the textbook suggests, is to make catastrophic risks negligible, for instance "less than 0.01% of an existential catastrophe per century".
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Exposure and vulnerability

    People who work on disasters split risk further. On 1 December 2016, an expert group set up by the United Nations General Assembly set out its terms for disaster risk reduction. Disaster risk is the loss of life, the injuries and the damage that could happen in a given period, "determined probabilistically as a function of hazard, exposure, vulnerability and capacity". Exposure is people, buildings and other goods "located in hazard-prone areas"; vulnerability is the conditions "which increase the susceptibility of an individual, a community, assets or systems to the impacts of hazards"; capacity is the strengths and means to manage the risk. The report gives no formula, only "a function of". It also names residual risk: the disaster risk "that remains even when effective disaster risk reduction measures are in place".

    The textbook uses a similar split, with its own meanings and without capacity, and multiplies the factors: the **disaster risk equation**. "Risk can be broken down into exposure, probability, severity, and vulnerability." On a slippery floor, exposure is how many people walk across it and how long it stays wet; probability, how likely someone who walks on it is to fall; severity, how hard the floor is, marble or carpet; vulnerability, how easily the one who falls breaks a bone, which depends on age, for example. Its exposure is close to the UN's, but not the same: not where people are, but how much they are exposed. Probability and severity together are the "intrinsic hazard level". For each hazardous event h, risk = P(h) × severity(h) × exposure(h) × vulnerability(h), and the total adds up the events.

    The textbook warns: "Rather than being mathematically rigorous, this equation is intended to convey that increasing any of the terms being multiplied will increase the risk, and reducing any of these terms will reduce it." One factor at zero makes the risk of that hazard zero, "regardless of how large the other factors are"; the other hazards keep their risk. A common extension divides by the ability to cope: how well people recover. For harm that nobody can recover from, such as an existential catastrophe, the equation would give a risk that is "astronomically large or infinite".

    ### Three places to lower AI risk

    Each factor is a place to lower risk, and the textbook links three research areas to them. Robustness, against attacks for example, lowers the vulnerability of AI systems themselves: how easily they can be pushed into doing harm; monitoring lowers exposure, by spotting hazards and avoiding them; control lowers the hazard level itself, by keeping AI systems controllable and less hazardous. "To reduce AI risk, it is crucial to pursue and develop all three, rather than relying on just one." For proxy gaming, where an AI system games an imperfect measure of what is meant (see [When a measure becomes a target](/course/open/rai-goodhart?lang=en)), the textbook gives an example of each: to lower the exposure, monitoring that flags odd behavior early; to lower the hazard level, a goal closer to what is meant and less power for the system; to lower the vulnerability, better proxies, more robust AI systems or less dependence on them. In a paper of 2021, Hendrycks, Nicholas Carlini, John Schulman and Jacob Steinhardt gave that role, lowering the hazard itself, to another area: "alignment makes systems inherently less hazardous".
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **hazard** | a source of danger that could cause harm; it brings risk only when someone or something is exposed to it |
        | threat | a hazard with someone behind it who means harm |
        | risk | the chance of a harmful event combined with how much harm it does; in the classic risk equation, probability times severity |
        | severity | how much harm an event does when it happens; in this lesson in made-up harm points |
        | **disaster risk equation** | risk as probability × severity × exposure × vulnerability, added up over the hazardous events: a rule of thumb, not exact mathematics |
        | exposure | how much someone is exposed to a hazard: how many people, how often, for how long |
        | vulnerability | how susceptible those who are exposed are to harm from the hazard |
        | residual risk | the risk that remains after the measures against it |
        | Σ | the sum: Σ over the hazards adds one term for each hazard |
        | `C.HAZARDS`, `C.EVENTS` | the made-up pharmacy assistant: its three hazards with a chance per answer and a severity, and the same three for a whole day, with four factors |
        """),
        L.watch_out("Risk is neither the chance nor the harm alone: a rare hazard with great harm can carry more risk than a common one with little. And a factor at zero removes the risk of one hazard, not of the whole system."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. List the hazards: what can go wrong, and for whom.
    2. For each hazard, estimate the chance and the severity, with one unit of harm for all of them.
    3. Multiply them, and add up over the hazards: the expected harm.
    4. If you know more, split further: how much exposure there is, and how vulnerable those exposed are.
    5. In the largest risks, look for the factor that is easiest to lower.
    6. Treat the numbers as estimates: a factor is seldom exactly zero, and what remains after the measures is residual risk.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell computes the risk of one made-up hazard, a harmful answer: first by the classic risk equation, for one answer, then by the disaster risk equation, for a day, and then for a day with half the exposure.
    """)
    return


@app.cell
def example_harm():
    p, severity = 0.002, 3                                # made up: a harmful answer, 3 harm points
    print(p * severity)                                   # the classic risk equation: per answer
    exposure, vulnerability = 5000, 0.5                   # 5,000 answers a day; half the harm lands
    print(p * severity * exposure * vulnerability)        # harm points a day
    print(p * severity * (exposure / 2) * vulnerability)  # half the exposure
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    For one answer, the risk is 0.006 harm points; for a day of 5,000 answers, where half the harm reaches the person who asks, it is 15. Half the exposure gives 7.5, half the risk, and so does half of any other factor. In Exercise 1 you add up the risks of several hazards.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. total_risk: all hazards together

        `C.HAZARDS` is a list of three dicts, one for each hazard of the pharmacy's assistant, each with a `"name"`, a chance per answer `"p"` and a severity in harm points `"severity"`. Write `total_risk(hazards)`. Start `total` at 0; for each dict `h` in `hazards`, add its chance times its severity to `total`; return `total`. Replace the `...`. The check calls `total_risk(HAZARDS)`, then other lists of hazards, an empty one too.
        """),
        L.task("The risk of all hazards together, per answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_total():
    box_total = L.code_box("total_risk", lines=5)
    box_total
    return (box_total,)


@app.cell(hide_code=True)
def check_total(box_total):
    result_total = L.check("total_risk", box_total.value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total_risk",
        [
            "The classic risk equation for each hazard, then the sum over the hazards, as with an expected value.",
            "A `for` loop over `hazards` gives one dict at a time, and `h[\"p\"]` is its chance. Add each hazard's risk to `total`, and return `total` after the loop.",
            "`        total = total + ___`",
        ],
        why="`total = 0`, `for h in hazards:` with `total = total + h[\"p\"] * h[\"severity\"]` inside it, then `return total`. For HAZARDS: 0.65 harm points per answer.",
    )
    return


@app.cell(hide_code=True)
def text_after_total():
    mo.md(r"""
    For one answer, the three hazards bring 0.65 harm points: 0.05 from rude answers, 0.1 from wrong doses and 0.5 from missed emergencies. The rarest hazard, one missed emergency in 10,000 answers, carries most of the risk, as it does 5,000 times the harm of a rude answer, which comes 500 times as often.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_disaster():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. disaster_risk: four factors

        `C.EVENTS` holds the same three hazards for a whole day, now with four factors: `"p"`, the chance per question of the kind it is about; `"severity"`; `"exposure"`, how many such questions come in a day; and `"vulnerability"`, the share of the harm that reaches the people who ask. All of them are made up. Of the 2,000 questions a day, 500 are about a dose and 200 describe an emergency; a rude answer can come with any question, so its exposure is all 2,000, and it reaches everyone it is given to, vulnerability 1.0. The chances are those of `C.HAZARDS`, per question of its kind: 10 missed emergencies in 10,000 emergency questions are 1 in 10,000 answers, as 1 question in 10 describes an emergency. A wrong dose has a vulnerability of 0.25, as three in four check the leaflet in the box; a missed emergency 0.5, as about half would call for help anyway. Write `disaster_risk(events)`: make an empty dict `risks`; for each dict `e` in `events`, put the product of its four factors in `risks` under `e["name"]`; return `risks`. Replace the `...`. The check calls `disaster_risk(EVENTS)`, then other lists.
        """),
        L.task("The harm points a day of each hazardous event, in a dict."),
    ])
    return


@app.cell(hide_code=True)
def exercise_disaster():
    box_disaster = L.code_box("disaster_risk", lines=5)
    box_disaster
    return (box_disaster,)


@app.cell(hide_code=True)
def check_disaster(box_disaster):
    result_disaster = L.check("disaster_risk", box_disaster.value)
    result_disaster
    return (result_disaster,)


@app.cell(hide_code=True)
def hints_disaster():
    L.hints(
        "disaster_risk",
        [
            "The same loop as in Exercise 1, but each event gets its own place in a dict instead of a place in one sum.",
            "Put a value in a dict with `risks[key] = value`; the key here is the event's name. The four factors are under the keys `\"p\"`, `\"severity\"`, `\"exposure\"` and `\"vulnerability\"`.",
            "`        risks[e[\"name\"]] = ___`",
        ],
        why="`risks = {}`, `for e in events:` with `risks[e[\"name\"]] = e[\"p\"] * e[\"severity\"] * e[\"exposure\"] * e[\"vulnerability\"]` inside it, then `return risks`. For EVENTS: 100 harm points a day from rude answers, 50 from wrong doses and 500 from missed emergencies.",
    )
    return


@app.cell(hide_code=True)
def text_after_disaster():
    mo.md(r"""
    A day brings 650 harm points: 100 from rude answers, 50 from wrong doses and 500 from missed emergencies. Without the vulnerability it would be 1,300: the 0.65 per answer of Exercise 1 times 2,000 answers. Halving any factor of the missed emergency saves 250 harm points, more than all rude answers together. So the team can pick the factor that is easiest to change: show the emergency number above the chat box, so that fewer people type an emergency into it, which lowers the exposure; or put a line about what to do in an emergency under every answer, which lowers the vulnerability.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: more missed emergencies

    The slider sets how many of 10,000 questions that describe an emergency the assistant misses, from 1 to 20; `C.EVENTS` has 10. The picture shows the harm points a day of each hazardous event, and the total.
    """)
    return


@app.cell(hide_code=True)
def demo_missed():
    missed = L.explore("missed", "missed per 10,000 emergency questions", lambda _v: C.demo_frame(_v, L.lang),
                       start=1, values=C.MISSED, role="",
                       task="Find the smallest number on the slider at which the missed emergency carries more than half of the harm points a day.")
    missed
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 4 on, the missed emergency carries more than half: 200 of the 350 harm points a day. At 1 it carries 50 of the 200. Every extra miss in 10,000 adds 50 harm points a day, while the other two stay at 150 together. A rare event with great harm can go from a small part of the risk to most of it through a small change in its chance.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_demonstration():
    mo.md(r"""
    ### Exercise 3. One demonstration

    A team writes 500 prompts to trick a customer-service model. One of them gets the model to promise a customer a refund of €10,000.
    """)
    return


@app.cell(hide_code=True)
def choice_demonstration_cell():
    choice_demonstration = L.choice(
        "demonstration",
        question="What does this demonstration say about the risk of the model in use?",
        options={
            "high": "The risk is high: what a model has done once in a test, it can do again in use",
            "right": "Little on its own: it shows the hazard, not how often it occurs in use",
            "euro": "€20 per customer: the 0.2% of the test times the €10,000 refund",
        },
        correct="right",
        why={
            "right": "The test shows that the hazard exists, and what it costs when it happens. How often it happens with real customers, who do not write prompts to trick the model, is still unknown, and the classic risk equation needs both.",
            "high": "\"A demonstration of a potential catastrophic hazard does not necessarily imply the risk is high\", the textbook says of catastrophic hazards, and it holds here too: the chance in use is still unknown. See *The classic risk equation* in *The idea*.",
            "euro": "0.2% is the share of prompts written to trick the model that worked, not the chance with real customers; that chance is still unknown. See *The classic risk equation* in *The idea*.",
        },
    )
    choice_demonstration
    return (choice_demonstration,)


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Exercise 4. One factor at zero

    A bank's assistant has three hazards: wrong advice about loans, a rude answer and a leaked account number. The bank no longer lets it answer questions about loans, so the exposure of the first hazard becomes 0.
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "zero",
        question="By the disaster risk equation, what happens to the assistant's total risk?",
        options={
            "zero": "It becomes 0, as one factor of the product is now 0",
            "right": "It falls by that hazard's risk; the other two hazards still count",
            "same": "It stays the same, as the chance of wrong loan advice has not changed",
        },
        correct="right",
        why={
            "right": "One factor at zero makes the risk of that hazard zero, \"regardless of how large the other factors are\". The total adds up the hazards, and the other two keep their risk.",
            "zero": "The product is per hazard; the total is a sum over the hazards, and the rude answer and the leaked account number still count. See *Exposure and vulnerability* in *The idea*.",
            "same": "The chance per question did not change, but no loan question comes in any more: with exposure 0, the product for that hazard is 0. See *Exposure and vulnerability* in *The idea*.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Robust, resilient, available, predictable](/course/open/rai-reliability?lang=en) measured how often a model fails. This lesson weighed failures by their harm: risk is chance times severity, added up over the hazards, the expected value of [Expected value and variance](/course/open/prob-expectation?lang=en) with harm as the value; the disaster risk equation adds exposure and vulnerability. The next lesson, *Nines of reliability*, turns the chance of failure around into reliability, and counts it in nines.

    **An everyday example.** Cycling home in the dark: lights lower the probability of a crash, riding slower lowers its severity, cycling less often lowers the exposure, and a helmet lowers the vulnerability of your head.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_demonstration, choice_zero, result_disaster, result_total):
    L.wrap_up(result_total, result_disaster, choice_demonstration, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
