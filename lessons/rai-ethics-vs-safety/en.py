# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ethics-vs-safety. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ethics-vs-safety", "en")
    C = L.checks   # Greaves and MacAskill's numbers: three plans, three estimates of the future, the bednets


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en) and [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_risk():
    quick_risk = L.choice(
        "quick-risk",
        question="Many news sites start writing their stories with the same AI model. No model makes a mistake and nobody means harm, yet the news grows ever more alike. In Zwetsloot and Dafoe's terms, what kind of risk is this?",
        options={
            "misuse": "Misuse: the news sites use the model to harm their readers",
            "right": "Structural: it comes from how AI changes the environment and its incentives",
            "accident": "An accident: the model does something its makers did not intend, such as writing errors",
        },
        correct="right",
        why={
            "right": "Nobody misuses the model and it does not fail; the harm comes from many sites making the same choice. That is the structural perspective Zwetsloot and Dafoe add to misuse and accidents.",
            "misuse": "Nobody here means to cause harm, and misuse needs someone who does. See [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en).",
            "accident": "The model does what it was built to do: write stories, without mistakes. The harm comes from many sites making the same choice. See [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en).",
        },
    )
    quick_risk
    return (quick_risk,)


@app.cell(hide_code=True)
def quick_check_theory():
    quick_theory = L.choice(
        "quick-theory",
        question="A hospital can fund a program that saves 10 lives this year, or one that saves 12 lives but breaks a promise it made to its patients. Which theory says the second is better because more lives are saved in total?",
        options={
            "duty": "Deontology: a promise counts whatever the outcome",
            "right": "Consequentialism: it adds up what a choice brings about",
            "virtue": "Virtue ethics: it asks what a person of good character would do in the hospital's place",
        },
        correct="right",
        why={
            "right": "Consequentialism judges a choice by what it brings about, added up over everyone: 12 lives are more than 10. This lesson meets the same kind of sum, with far larger numbers.",
            "duty": "Deontology is the theory that would object: for it, the broken promise counts whatever the outcome. See [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "virtue": "Virtue ethics asks about character, not about totals. See [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_theory
    return (quick_theory,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the spring of 2023, two statements about AI came out two months apart. One said that the harms of AI are "real and present"; the other, that the risk of extinction from AI should be a global priority. Both came from people who study what can go wrong with AI, and they disagree about where to look first.

    This lesson looks at the two groups, AI ethics and AI safety, at the argument with which some of the second put the far future first, at the critique of that argument, and at what the two share. You will redo the sums of that argument yourself.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A safe option saves 1,000 lives for sure. A risky option has a chance of one in a trillion (10⁻¹²) to save a quintillion (10¹⁸) lives, and otherwise saves none. Weighing each outcome by its chance, how many lives does the risky option save on average?",
        options={
            "none": "Almost none: it nearly always saves nobody",
            "right": "A million: a thousand times as many",
            "same": "About 1,000: the chance makes up for the size",
        },
        reveal="**A million: a thousand times as many.** 10¹⁸ × 10⁻¹² = 10⁶. Yet it nearly always saves nobody. The example is Greaves and MacAskill's own, for the worry they call fanaticism: *Their own doubts* in *The idea* comes back to it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Two groups, one quarrel

    The earlier modules of this domain, from ethics and fairness to privacy, belong to what is usually called AI ethics: harm that AI systems do now, to people alive now, such as unfair decisions, exploited workers and power in few hands. This module is about AI safety, which also takes in catastrophes that far more capable systems might cause later. In 2023 the two clashed in public.

    In March 2023, the Future of Life Institute published a letter that asked for a pause of at least six months on training AI systems more powerful than GPT-4; by the count of its critics, more than 2,000 people had signed it by 31 March. That day, in a statement on the website of the DAIR Institute, Timnit Gebru, Emily M. Bender and Angelina McMillan-Major, three of the authors of the paper known as "Stochastic Parrots", wrote: "The harms from so-called AI are real and present and follow from the acts of people and corporations deploying automated systems." The letter's hypothetical risks, they wrote, "are the focus of a dangerous ideology called longtermism that ignores the actual harms resulting from the deployment of AI systems today."

    On 30 May 2023, the Center for AI Safety put one sentence online: "Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war." On its page, in November 2026, the first names under it are Geoffrey Hinton, Yoshua Bengio, Demis Hassabis, Sam Altman and Dario Amodei. It gives no probability and no date.

    ### Longtermism

    **Longtermism**, in the words of the philosopher William MacAskill, "is the view that we should be doing much more to protect future generations." It rests on three ideas: "future people have moral worth, there could be very large numbers of future people, and that what we do today can affect how well or poorly their lives go." That is why longtermists focus on existential risks, "risks that threaten the destruction of humanity's long-term potential"; the ones they highlight include advanced artificial intelligence, engineered pathogens and nuclear war.

    In a working paper of June 2021, *The case for strong longtermism*, Hilary Greaves and William MacAskill argue for more: "strong longtermism: the view that impact on the far future is the most important feature of our actions today." The far future starts, in their example, 100 years after a decision.

    ### The sum behind it

    Greaves and MacAskill make the case with numbers. As the benchmark for good done now they take bednets against malaria, which save a life for about 4000 dollars: 0.025 lives per 100 dollars. In their words: "We use the cost-effectiveness of antimalarial bednet distribution as an approximate upper bound on attainable near-future benefits per unit of spending." Against it they set plans that lower the risk of a catastrophe in the next 100 years: the extinction of humanity for asteroids and pandemics, and for AI a catastrophe they count as at least as bad. Such a catastrophe would cost all future lives, so the lives a plan saves on average are the drop in that risk times the expected number of future beings: the number you get when every possible future is weighed by its chance, an expected value, which [Expected value and variance](/course/open/prob-expectation?lang=en) treats in full.

    How many future beings? Their table gives six scenarios, in the picture below, from humanity on Earth for a million years, 10¹⁴ lives, to digital life across the Milky Way, 10⁴⁵. They work with three estimates: main, 10²⁴, which they call the least that is reasonable; low, 10¹⁸; and restricted, 10¹⁴, as if humanity stays on Earth and nothing larger has any chance.

    Their three plans, which the course keeps in `C.PLANS`, each with a risk, the share of it the plan removes (its cut) and a cost:

    - asteroids: the remaining chance that an asteroid of 10 km or more hits Earth in the next 100 years is estimated at 1 in 150 million, and they assume that 1% of such impacts would end humanity. Finding every remaining one costs at most 1.2 billion dollars, and once one is found on a collision course, deflecting it and preparing remove 5% of the risk.
    - pandemics: the chance of an extinction-level pandemic in the next 100 years is estimated at about 1 in 22,000, an average they take of two of three published estimates, which range from 1 in 600,000 to 1 in 50; and 250 billion dollars for stronger health care would remove at least 1% of it.
    - AI: "even a highly conservative assessment would assign at least a 0.1% chance to an AI-driven catastrophe (as bad as or worse than human extinction) over the coming century", and 1 billion dollars of carefully targeted spending would avoid it in at least 1% of the cases.

    For these sums they count value as total welfare, the consequentialist kind of sum of the quick check. They also argue that a version of their thesis holds without it: "We suggested that deontic strong longtermism might well be true even if consequentialism is false."
    """)
    return


@app.cell(hide_code=True)
def size_cell():
    mo.Html(C.size_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Their own doubts

    Greaves and MacAskill name the weak points themselves: "In our own view, the weakest points in the case for axiological strong longtermism are the assessment of numbers for the cost-effectiveness of particular attempts to benefit the far future, the appropriate treatment of cluelessness, and the question of whether an expected value approach to uncertainty is too ‘fanatical’ in this context." The guess is their example of fanaticism: a tiny chance of an enormous number wins the sum, though it almost always comes to nothing. They regard it as "one of the most plausible ways in which the argument for strong longtermism might fail." Even so, they "tentatively conclude that considerations of fanaticism do not undermine the argument for strong longtermism", in part because the chances may not be small enough for that at the level of society. Their thesis also compares, and no more: "We do not claim, and nor do we believe, that options cannot deliver large benefits without being near-best for the far future, or that available near-future benefits are small in any absolute sense."

    ### TESCREAL

    The critics see more than a sum. In 2024, Timnit Gebru and Émile P. Torres published *The TESCREAL bundle: Eugenics and the promise of utopia through artificial general intelligence*. **TESCREAL** is an acronym they coined "while writing an early draft of this paper", for seven ideologies: "transhumanism, Extropianism, singularitarianism, (modern) cosmism, Rationalism, Effective Altruism, and longtermism". They argue that the push for AGI, artificial general intelligence, comes from this bundle, and that "the normative framework that motivates much of this goal is rooted in the Anglo-American eugenics tradition of the twentieth century." About AI safety they write: "Framing the AGI agenda as a safety issue allows companies working toward it to describe themselves as ‘AI safety’ organizations safeguarding humanity's future, while simultaneously creating unsafe products, centralizing power, and evading accountability".

    They add that "not everyone associated with ideologies in this bundle believes in the totality of the dominant views in this bundle, and some people may even object to being bundled in this manner." Gebru founded DAIR; by the paper's own account, its other author was an advocate of the movement for nearly a decade before leaving it. They also quote a version of Greaves and MacAskill's paper from 2019: "Short-run effects act as little more than tie-breakers." That sentence is not in the version of 2021 quoted here.

    ### More in common than it seems

    In 2020, Carina Prunkl and Jess Whittlestone wrote that the line between "near-term" and "long-term" "is poorly defined, and is used differently across people and contexts". They split it into four questions, each a scale rather than a choice of two: capabilities (current systems or much more advanced ones), impacts (soon or much later), certainty (well understood or speculative) and extremity (harm at every scale or mainly the largest). In 2022, Benjamin Bucknall and Shiri Dori-Hacohen proposed that "certain already-documented effects of AI can act as existential risk factors": harms of today that make a catastrophe from another source more likely, or harder to respond to. Dan Hendrycks, the director of the Center for AI Safety, told BBC News that future risks and present concerns "shouldn't be viewed antagonistically": "Addressing some of the issues today can be useful for addressing many of the later risks tomorrow." And the DAIR statement agreed with some proposals of the letter it criticized, such as "‘provenance and watermarking systems to help distinguish real from synthetic’ media".
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **longtermism** | the view that we should do much more to protect future generations; strong longtermism holds that impact on the far future is the most important feature of what we do today |
        | **TESCREAL** | Gebru and Torres's name for seven related ideologies, from transhumanism to longtermism, which they argue drive the push for AGI |
        | existential risk | a risk that threatens the destruction of humanity's long-term potential, such as its extinction |
        | expected number of future beings | how many beings will live, with every possible future weighed by its chance |
        | benchmark | what a plan is compared with: here the bednets, 0.025 lives per 100 dollars |
        | `C.PLANS` | the three plans by name, `"asteroids"`, `"pandemics"` and `"AI"`: dicts with the keys `"risk"` (the chance in the next 100 years of extinction from that cause, or for AI of a catastrophe at least as bad), `"cut"` (the share of that risk the plan removes) and `"cost"` (in dollars) |
        | `C.ESTIMATES` | the expected number of future beings: `"main"` 10²⁴, `"low"` 10¹⁸ and `"restricted"` 10¹⁴ |
        | `C.BEDNETS` | the benchmark: 100 / 4000 = 0.025 lives per 100 dollars |
        | `1e24` | how Python writes 10²⁴, a 1 with 24 zeros; `5e-16` is 5 × 10⁻¹⁶ |
        """),
        L.watch_out("A sum with a huge number in it is only as good as its weakest guess: here, the chance that a plan works."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Ask what harm each side means, to whom, and when.
    2. Find the numbers behind a priority, and mark which of them are measured and which are guesses.
    3. Work out from which value of a guess the conclusion changes.
    4. Read the critics and the defenders in their own words, and say who claims what.
    5. Describe a concern on several scales, such as how capable the systems are and how certain the harm is, rather than with one label.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    From 1996, NASA's Spaceguard Survey tracked asteroids near Earth, at a cost of 71 million dollars by 2012. Greaves and MacAskill estimate that it lowered the chance of extinction by at least 5 × 10⁻¹⁶ per 100 dollars. The cell works out what that is worth in lives on each of their three estimates, beside the bednets.
    """)
    return


@app.cell
def example_spaceguard():
    print("bednets:", 100 / 4000, "lives per 100 dollars")
    drop = 5e-16                            # Spaceguard: the drop in the chance of extinction per 100 dollars
    for name in C.ESTIMATES:                # main, low and restricted
        lives = drop * C.ESTIMATES[name]    # lives on average per 100 dollars
        print(name, round(lives, 2), "lives per 100 dollars")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    These are their figures: 500 million, 500 and 0.05 lives per 100 dollars. On the main and the low estimate the survey beats the bednets by far; on the restricted one, 0.05 against 0.025, by a factor of two. The drop per 100 dollars here was given; in the exercise you work it out from a risk, a cut and a cost.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lives():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. lives_per_100: what 100 dollars of a plan is worth

        Write `lives_per_100(plan, estimates)`. `plan` is a dict with the keys `"risk"`, `"cut"` and `"cost"`, like the plans in `C.PLANS`; `estimates` is a dict with a number of future beings for each name, like `C.ESTIMATES`. Compute `drop`, the risk times the cut: how much the whole cost lowers that risk. Make an empty dict `result` and, with a `for` loop over `estimates` that takes each `name`, store under that name the drop times its number of future beings, divided by the cost, times 100. Return `result`. Replace the `...`. The check calls `lives_per_100(PLANS["AI"], ESTIMATES)`, then the two other plans, and plans and estimates of its own.
        """),
        L.task("For each estimate: the risk times the cut, times the number of future beings, per 100 dollars of the cost."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lives():
    box_lives = L.code_box("lives_per_100", lines=6)
    box_lives
    return (box_lives,)


@app.cell(hide_code=True)
def check_lives(box_lives):
    result_lives = L.check("lives_per_100", box_lives.value)
    result_lives
    return (result_lives,)


@app.cell(hide_code=True)
def hints_lives():
    L.hints(
        "lives_per_100",
        [
            "First the drop the whole plan buys, then, for every estimate, what that drop is worth in lives, and last per 100 dollars instead of for the whole cost.",
            "`drop = plan[\"risk\"] * plan[\"cut\"]` gives the drop. `result = {}` starts an empty dict, and `result[name] = ...` stores a value under a name. The number of future beings of an estimate is its value in the dict `estimates`.",
            "`        result[name] = drop * ___ / plan[\"cost\"] * 100`",
        ],
        why="`drop = plan[\"risk\"] * plan[\"cut\"]`, `result = {}`, then `for name in estimates:` with `result[name] = drop * estimates[name] / plan[\"cost\"] * 100`, and `return result`. For the AI plan: 0.001 × 0.01 × 10¹⁴ / 10⁹ × 100 = 100 lives on the restricted estimate.",
    )
    return


@app.cell(hide_code=True)
def text_after_lives():
    mo.md(r"""
    For the AI plan, 100 dollars is worth a trillion, a million and 100 lives on the main, the low and the restricted estimate: Greaves and MacAskill's own figures. Asteroids give about 280,000, 0.28 and 0.000028, and pandemics about 180 million, 180 and 0.018; their figures, 300,000, 0.3 and 0.00003 and 200 million, 200 and 0.02, are these rounded to one digit. Against 0.025 for the bednets, all three plans win on the main and the low estimate. On the restricted estimate only the AI plan does; for asteroids, in their words, the example "supports strong longtermism on our main and low estimates, though not on the restricted estimate."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_break_even():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. break_even: from how many future beings a plan wins

        The estimates of the future differ by ten powers of ten, so it helps to turn the question around. Write `break_even(plan, benchmark)`: the number of future beings at which 100 dollars of the plan is worth exactly `benchmark` lives. Compute `drop` as in Exercise 1. The lives per 100 dollars are drop × beings / cost × 100; work out which number of beings makes that equal to `benchmark`, and return it. Replace the `...`. The check calls `break_even(PLANS["AI"], BEDNETS)`, the AI plan against the bednets, then the two other plans and cases of its own.
        """),
        L.task("Turn the sum of Exercise 1 around: which number of future beings makes 100 dollars of the plan worth exactly the benchmark?"),
    ])
    return


@app.cell(hide_code=True)
def exercise_break_even():
    box_break_even = L.code_box("break_even", lines=3)
    box_break_even
    return (box_break_even,)


@app.cell(hide_code=True)
def check_break_even(box_break_even):
    result_break_even = L.check("break_even", box_break_even.value)
    result_break_even
    return (result_break_even,)


@app.cell(hide_code=True)
def hints_break_even():
    L.hints(
        "break_even",
        [
            "Write benchmark = drop × beings / cost × 100 and get beings alone on one side.",
            "Multiply both sides by the cost, then divide both by the drop and by 100. In Python, a product you divide by goes between parentheses.",
            "`    return benchmark * plan[\"cost\"] / ___`",
        ],
        why="`drop = plan[\"risk\"] * plan[\"cut\"]` and `return benchmark * plan[\"cost\"] / (drop * 100)`. For the AI plan: 0.025 × 10⁹ / (10⁻⁵ × 100) = 2.5 × 10¹⁰.",
    )
    return


@app.cell(hide_code=True)
def text_after_break_even():
    mo.md(r"""
    The asteroid plan needs 9 × 10¹⁶ future beings, the pandemic plan 1.4 × 10¹⁴ and the AI plan 2.5 × 10¹⁰: 25 billion, which humanity on Earth would reach in two and a half centuries at the 10¹⁰ lives per century of their table. So the AI case hardly needs the vast numbers; it rests on the 0.1% and the 1%. For those, they write: "There is no hard quantitative evidence to guide cost-effectiveness estimates for AI safety work." Expert judgment, they add, tends to put the chance of an existential catastrophe from artificial superintelligence at 1 to 10%. Estimating such numbers is the first of the weak points they name themselves.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: how many future beings?

    The slider sets the expected number of future beings, as a power of ten, from 10¹⁰ to 10²⁴. The bars show what 100 dollars of each plan is worth in lives, on a scale where each tick is ten times as much; the dashed line is the bednets.
    """)
    return


@app.cell(hide_code=True)
def demo_future():
    future = L.explore("future", "future beings, as a power of ten", lambda _v: C.demo_frame(_v, L.lang),
                       start=14, values=C.POWERS, role="",
                       task="Find the smallest number on the slider at which all three plans beat the bednets.")
    future
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 10¹⁷ on, all three plans beat the bednets, the asteroid plan only just: 0.0278 against 0.025. At 10¹⁰, none does. In between, the answer turns on a number that nobody can measure, the expected size of the future; the scenarios in their own table span 31 powers of ten. That is where the critics push, and where Greaves and MacAskill place their own doubts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.md(r"""
    ### Exercise 3. A label on a blog

    A researcher audits hiring models for unfair decisions, and gives money to an Effective Altruism charity that buys bednets. A blog post calls the researcher "a TESCREAList, so a eugenicist".
    """)
    return


@app.cell(hide_code=True)
def choice_label_cell():
    choice_label = L.choice(
        "label",
        question="What does Gebru and Torres's paper itself say that bears on this?",
        options={
            "self": "That TESCREAL is the name the members of these movements chose for themselves",
            "right": "That not everyone tied to these ideologies holds all of their dominant views",
            "outside": "That Effective Altruism is not one of the seven ideologies, so the label does not fit",
        },
        correct="right",
        why={
            "right": "The paper says \"not everyone associated with ideologies in this bundle believes in the totality of the dominant views in this bundle\". The eugenics link is the authors' thesis about the ideologies, not a finding about each person who gives to a charity.",
            "self": "Gebru and Torres coined the acronym themselves, \"while writing an early draft of this paper\", and say some people may object to being bundled. See *TESCREAL* in *The idea*.",
            "outside": "Effective Altruism is one of the seven, the EA in TESCREAL. What bears on the blog is what the paper says about people: not everyone associated with these ideologies holds all of their dominant views. See *TESCREAL* in *The idea*.",
        },
    )
    choice_label
    return (choice_label,)


@app.cell(hide_code=True)
def text_exercise_scales():
    mo.md(r"""
    ### Exercise 4. Near or far?

    A research group studies how the algorithms that choose which posts people see today could, over decades, wear down trust in elections, until a country can no longer agree on how to answer a crisis.
    """)
    return


@app.cell(hide_code=True)
def choice_scales_cell():
    choice_scales = L.choice(
        "scales",
        question="Is this near-term or long-term work?",
        options={
            "near": "Near-term: it is about systems that exist today, so it belongs to AI ethics",
            "long": "Long-term: its harm comes decades from now and could reach a whole country, so it belongs to AI safety alone",
            "right": "One label does not fit: it is about current systems, but its impacts come late and are large",
        },
        correct="right",
        why={
            "right": "On Prunkl and Whittlestone's scales it sits near on capabilities, far on impacts, and high on extremity, with certainty somewhere in between. Bucknall and Dori-Hacohen would call such a harm a possible risk factor: a harm of today that makes a later catastrophe harder to respond to.",
            "near": "It is about today's systems, but the harm it studies comes late and could be large: one label hides that. See *More in common than it seems* in *The idea*.",
            "long": "Its harm comes late, but the systems are today's, and the harm builds up from effects that are already documented. See *More in common than it seems* in *The idea*.",
        },
    )
    choice_scales
    return (choice_scales,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en) sorted risks by where they come from, and [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) showed theories that weigh the same choice in different ways. Here the two meet: the priority some in AI safety give the far future is argued with a consequentialist sum with very large numbers, though Greaves and MacAskill hold that it may also stand without consequentialism, and the critics object to the sum, to what it leaves out and to the ideas behind it. The next lesson, *Who works on AI safety?*, looks at the institutes, labs and courses that make up the field.

    **An everyday example.** A family that weighs repairing a leaking roof now against saving for a child's studies in fifteen years faces the same question on a small scale: how much should what comes later count against what hurts now?
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_label, choice_scales, result_break_even, result_lives):
    L.wrap_up(result_lives, result_break_even, choice_label, choice_scales)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
