# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ai-welfare. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ai-welfare", "en")
    C = L.checks   # the chances of Long and colleagues, the made-up costs of the two mistakes, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) and [Other positions](/course/open/rai-xrisk-counterpositions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_theory():
    quick_theory = L.choice(
        "quick-theory",
        question="Which ethical theory judges a choice by what it brings about, added up over everyone affected?",
        options={
            "duty": "Deontology: some duties count whatever the outcome",
            "right": "Consequentialism: the outcomes count",
            "virtue": "Virtue ethics: good character counts",
        },
        correct="right",
        why={
            "right": "Consequentialism adds up what a choice does to everyone affected. Who counts as everyone is a question of its own, and the question of this lesson: does it include animals, and could it include an AI system?",
            "duty": "Deontology holds that some duties and rights count whatever the outcome. The theory that adds up outcomes over everyone affected is consequentialism. See [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "virtue": "Virtue ethics asks what a person of good character would do. The theory that adds up outcomes over everyone affected is consequentialism. See [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_theory
    return (quick_theory,)


@app.cell(hide_code=True)
def quick_check_dtd():
    quick_dtd = L.choice(
        "quick-dtd",
        question="What does differential technology development ask for?",
        options={
            "stop": "Stop developing every technology that could be dangerous",
            "right": "Bring forward protections, so that they arrive before the dangers",
            "all": "Speed up every technology as much as possible, since each will help in the end",
        },
        correct="right",
        why={
            "right": "It speeds up technologies that reduce the risks of others and delays those that increase risk, so that protection arrives before the danger it guards against. In this lesson, Birch asks something similar of rules: that they run ahead of the risks of current technology. That link is this lesson's.",
            "stop": "It does not stop technologies, but changes their order: protections sooner, dangers later. See [Other positions](/course/open/rai-xrisk-counterpositions?lang=en).",
            "all": "That is closer to e/acc. Differential technology development speeds up what protects and delays what endangers, so that protection comes first. See [Other positions](/course/open/rai-xrisk-counterpositions?lang=en).",
        },
    )
    quick_dtd
    return (quick_dtd,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The lessons before this one asked what AI might do to people. This one turns the question around: could an AI system itself be harmed, so that how people treat it matters? Some philosophers, scientists and one AI company now take that question seriously; others call it premature. In a report of 2024, Robert Long and colleagues reason about it as a chain of steps that must all hold. In one example, they give each of three steps a chance of about 25%.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="If the three steps are independent, what is the chance that all three hold?",
        options={
            "same": "About 25%",
            "third": "About 8%",
            "right": "About 2%",
        },
        reveal="**About 2%.** A quarter of a quarter of a quarter is 0.015625, about 1.6%, which Long and colleagues round up: \"Assuming these chances are independent, it would follow that there is only a ~2% chance of near-future AI welfare and moral patienthood via the sentience route!\" Each step that must also hold makes the chain less likely, so the chain is never more likely than its least likely step. This lesson looks at the steps of that chain: what it means to matter morally, how researchers look for consciousness in AI, and what to do while nobody knows.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Moral status

    The Stanford Encyclopedia of Philosophy has an entry on the grounds of moral status (by Agnieszka Jaworska and Julie Tannenbaum, substantive revision of 3 March 2021). It begins: "An entity has moral status if and only if it matters (to some degree) from the moral point of view for its own sake." An example: "an animal may be said to have moral status if there is at least some moral reason to avoid its suffering, on account of this animal itself and regardless of the consequences for other beings." The entry discusses several possible grounds of moral status, among them sophisticated cognitive capacities, special relationships, and more rudimentary capacities: an account of that last kind "might appeal to the capacity to experience pleasure or pain (sentience), to have interests or basic emotions, or the capacity for consciousness."

    ### A realistic possibility

    In *Taking AI Welfare Seriously* (preprint, version 1 of 4 November 2024), Long and nine colleagues, among them Jonathan Birch and David Chalmers, use two related words: "An entity is a moral patient when that entity morally matters for its own sake", and "an entity is a welfare subject when that entity has morally significant interests and, relatedly, is capable of being benefited (made better off) and harmed (made worse off)." Their claim: "In this report, we argue that there is a realistic possibility that some AI systems will be conscious and/or robustly agentic in the near future." And its limit: "To be clear, our argument in this report is not that AI systems definitely are — or will be — conscious, robustly agentic, or otherwise morally significant."

    They see two routes, each a "realistic, non-negligible possibility": a consciousness route, in which "Consciousness suffices for moral patienthood", and a route of robust agency, in which "Robust agency suffices for moral patienthood". For AI companies, they recommend three first steps: "(1) acknowledge that AI welfare is an important and difficult issue (and ensure that language model outputs do the same), (2) start assessing AI systems for evidence of consciousness and robust agency, and (3) prepare policies and procedures for treating AI systems with an appropriate level of moral concern." One of the authors names Anthropic beside Eleos AI as an affiliation, and Anthropic writes that it "supported an early project on which that report was based".

    The chain of the guess is their sentience route: sentience suffices for moral patienthood, certain computations suffice for sentience, and some AI systems will be capable of these computations in the near future. In their view, it would be reasonable to give the three steps chances of about 90%, 50% and 50%. And: "even if the chance of near-future AI welfare and moral patienthood were as low as 2%, that would still constitute a non-negligible risk." Only a far lower chance would be open to debate: "Yes, if the chance was only, say, one in a hundred million, then we could debate whether that risk is low enough to ignore."

    ### Looking for consciousness

    Patrick Butlin, Robert Long and 17 colleagues, among them Yoshua Bengio, asked how to look for consciousness in AI (*Consciousness in Artificial Intelligence*, version of 22 August 2023). Their method: "we adopt computational functionalism, the thesis that performing computations of the right kind is necessary and sufficient for consciousness, as a working hypothesis. This thesis is a mainstream—although disputed—position in philosophy of mind." They do not simply ask the system: "The main alternative to this approach is to use behavioural tests for consciousness, but this method is unreliable because AI systems can be trained to mimic human behaviours while working in very different ways."

    Instead, they derive "‘indicator properties’ of consciousness" from theories including "recurrent processing theory, global workspace theory, higher-order theories, predictive processing, and attention schema theory"; their table lists 14 of them (the count is this lesson's). Their claim "is that AI systems which possess more of the indicator properties are more likely to be conscious." Their conclusion: "Our analysis suggests that no current AI systems are conscious, but also suggests that there are no obvious technical barriers to building AI systems which satisfy these indicators." A footnote says that an earlier version of the sentence read that the analysis "shows that there are no obvious barriers to building conscious AI systems", and adds: "But satisfying the indicators would not mean that such an AI system would definitely be conscious."

    ### Precaution

    Jonathan Birch's book *The Edge of Sentience* (Oxford University Press, 2024) asks what to do when sentience is uncertain. A system S is a sentience candidate "if there is an evidence base that (a) implies a realistic possibility of sentience in S that it would be irresponsible to ignore when making policy decisions that will affect S, and (b) is rich enough to allow the identification of welfare risks and the design and assessment of precautions." Among the book's proposals: "Precautions to manage risks of suffering should be driven by markers of sentience, not markers of intelligence." It warns of "gaming of the criteria", when a system or its designer learns which markers count: "We need to discount markers we have reason to think may have been gamed." And its run-ahead principle: "At any given time, measures to regulate the development of sentient AI should run ahead of what would be proportionate to the risks posed by current technology, considering also the risks posed by credible future trajectories."

    ### A company under uncertainty

    On 24 April 2025, Anthropic wrote that it had recently started a research program on model welfare: "There’s no scientific consensus on whether current or future AI systems could be conscious, or could have experiences that deserve consideration." The company says it is "approaching the topic with humility and with as few assumptions as possible." In a post of 15 August 2025, it wrote that it had given two of its Claude models the ability to end conversations in its consumer chat apps, "intended for use in rare, extreme cases of persistently harmful or abusive user interactions." Why: "We remain highly uncertain about the potential moral status of Claude and other LLMs, now or in the future." The company looks for "low-cost interventions to mitigate risks to model welfare, in case such welfare is possible." In testing before release, one of the two had shown "A pattern of apparent distress when engaging with real-world users seeking harmful content". And Claude "is directed not to use this ability in cases where users might be at imminent risk of harming themselves or others."

    ### The skeptics

    Mustafa Suleyman, in a post of 19 August 2025 on Suleyman's own site, titled "We must build AI for people; not to be a person.", writes about "Seemingly Conscious AI", which "has all the hallmarks of other conscious beings and thus appears to be conscious." The system Suleyman imagines "would not actually be conscious, but it would imitate consciousness in such a convincing way that it would be indistinguishable from a claim that you or I might make to one another about our own consciousness." About the claim that such AIs are conscious: "To be clear, there is zero evidence of this today and some argue there are strong reasons to believe it will not be the case in the future." About the idea of model welfare, which some academics are beginning to explore, Suleyman writes: "This is both premature, and frankly dangerous." The post's central worry is "that many people will start to believe in the illusion of AIs as conscious entities so strongly that they’ll soon advocate for AI rights, model welfare and even AI citizenship."

    Anil Seth argues for "biological naturalism: the view that biological properties are necessary for consciousness" (*Conscious artificial intelligence and biological naturalism*, accepted manuscript for *Behavioral and Brain Sciences*, in the University of Sussex's repository since 24 April 2025). Seth's conclusion: "real artificial consciousness is unlikely along current trajectories, but becomes more plausible as AI becomes more brain-like and/or life-like." About the state of knowledge: "A definitive answer to this question is not currently possible, given the lack of consensus about the minimally sufficient conditions for consciousness." And Seth does not wave the question away: "Real artificial consciousness may be very unlikely or impossible. However, if such systems were to emerge – whether by design or by accident – ethical catastrophe awaits."

    ### Where they disagree

    On acting now: Long and colleagues write that AI companies "have a responsibility to start taking it seriously", and Birch wants rules to run ahead of current technology; Suleyman calls the idea of model welfare premature. On what consciousness needs: Butlin and colleagues work from computational functionalism, "a mainstream—although disputed—position"; Seth holds that it may need biology. And on which mistake is worse: Long and colleagues name two, over-attribution, "mistakenly seeing, or treating, an object as a subject", and under-attribution, "mistakenly seeing, or treating, a subject as an object", and write: "Both of these mistakes can lead to significant costs or harms in this context". Suleyman's central worry is the first; Seth names both.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **moral status** | mattering morally for your own sake, to some degree, so that there is moral reason to take your interests into account; whether an AI system could have it is an open question |
        | moral patient | in Long and colleagues' words, an entity that morally matters for its own sake |
        | consciousness | having experiences: there is something it is like to be you, to see red or to feel pain |
        | sentience | the capacity to have experiences that feel good or bad, such as pleasure and pain |
        | computational functionalism | the thesis that performing computations of the right kind is necessary and sufficient for consciousness |
        | indicator property | in Butlin and colleagues' report, a property derived from a theory of consciousness; on their claim, a system with more of them is more likely to be conscious |
        | sentience candidate | in Birch's book, a system with a realistic possibility of sentience that it would be irresponsible to ignore, and enough evidence to design precautions |
        | over- and under-attribution | in Long and colleagues' report, seeing or treating an object as a subject (a false positive), and a subject as an object (a false negative) |
        """),
        L.watch_out("Nobody has shown that a current AI system is conscious, and nobody has shown that none is: Butlin and colleagues' analysis \"suggests\" that none is, on a working hypothesis. What a system says about its own feelings settles neither. The chances in Exercise 1 are Long and colleagues'; the costs in Exercise 2 are made up, and so is its model."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To think about whether an AI system could matter morally:

    1. Say which ground of moral status you mean, such as sentience or robust agency.
    2. Split the question into steps that must all hold, and give each step its own chance.
    3. Ask which theory of consciousness a marker comes from, and whether you accept that theory.
    4. Discount markers that may have been gamed.
    5. Weigh both mistakes, and how much each would cost.
    6. Look for cheap precautions that help in case it matters.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out the chain of the guess: three steps of 25% each.
    """)
    return


@app.cell
def example_chain():
    chances = [0.25, 0.25, 0.25]   # Long and colleagues' example: three steps of about 25% each
    all_three = chances[0] * chances[1] * chances[2]
    print(all_three)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The result is 0.015625, about 1.6%. The chance that all three steps hold is a quarter of a quarter of a quarter: each step that must also hold makes the whole less likely. The multiplication assumes that the steps are independent, as Long and colleagues say themselves.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. chain: steps that must all hold

        Write `chain(chances)`, which returns the chance that every step holds, if the steps are independent. `chances` is a list with the chance of each step. Start a variable `result` at 1, and for each `chance` in `chances`, multiply `result` by it; return `result`.

        The check calls `chain([0.9, 0.5, 0.5])`, Long and colleagues' own estimate, then other cases.
        """),
        L.task("The chance that every step holds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain", lines=5)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("chain", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "chain",
        [
            "Start small: two steps, each with a chance of 0.5. What is the chance that both hold, and what does a third step of 0.5 do to it?",
            "Start `result` at 1: with no steps yet, nothing can fail. Go through `chances` with a `for` loop; in each round, `result` becomes `result` times the `chance` of that step. Return `result` after the loop.",
            "`        result = result * ___`",
        ],
        why="`result = 1`, then `for chance in chances:` with `result = result * chance` inside, and `return result`. Two steps of 0.5 give 0.25, and a third step of 0.5 gives 0.125.",
    )
    return


@app.cell(hide_code=True)
def text_after_chain():
    mo.md(r"""
    For Long and colleagues' own estimate, the result is 0.225. In their words: "a ~90% chance that sentience suffices for welfare and moral patienthood, a ~50% chance that certain computations suffice for sentience, and a ~50% chance that some AI systems will be capable of these computations in the near future" give "a ~22.5% chance". That is more than fourteen times the 0.015625 of their example: the same three steps, with other chances.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_costs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. costs: the two mistakes

        A made-up model of the two mistakes, this lesson's own: Long and colleagues give no such numbers. You either take care of a system, or you ignore it. If you take care and it does not matter morally, you lose `over`, the cost of care in vain. If you ignore it and it does matter, you lose `under`, the cost of ignoring a being that matters. Care that turns out to be needed costs nothing in this model. `chance` is the chance that it matters. As in the lesson *Expected value and variance*, each cost counts in proportion to its chance.

        Write `costs(chance, over, under)`, which returns a list of two numbers: `care`, the cost to expect when you take care, which is `over` times the chance that the system does not matter; and `ignore`, the cost to expect when you ignore it, which is `under` times `chance`. Return `[care, ignore]`.

        The check calls `costs(0.02, 1, 100)`, a chance of 2% with ignoring made up to cost 100 times as much as care in vain, then other cases.
        """),
        L.task("The cost to expect when you take care, and when you ignore."),
    ])
    return


@app.cell(hide_code=True)
def exercise_costs():
    box_costs = L.code_box("costs", lines=4)
    box_costs
    return (box_costs,)


@app.cell(hide_code=True)
def check_costs(box_costs):
    result_costs = L.check("costs", box_costs.value)
    result_costs
    return (result_costs,)


@app.cell(hide_code=True)
def hints_costs():
    L.hints(
        "costs",
        [
            "Start small: a chance of 0.5, and both mistakes costing 1. How often is each choice a mistake, and what does it cost on average?",
            "The chance that the system does not matter is 1 minus `chance`. `care` is that chance times `over`, and `ignore` is `chance` times `under`. Return the two in a list, `care` first.",
            "`    care = ___ * over`",
        ],
        why="`care = (1 - chance) * over` and `ignore = chance * under`, then `return [care, ignore]`. With a chance of 0.5 and both costs 1, taking care and ignoring both cost 0.5 on average.",
    )
    return


@app.cell(hide_code=True)
def text_after_costs():
    mo.md(r"""
    At a chance of 2%, taking care costs 0.98 on average and ignoring costs 2: with these made-up costs, care pays even at 2%. Long and colleagues make no such calculation; in words, they hold that even a chance of 2% would be a non-negligible risk. What the model leaves out is how large the two costs really are, and the sources disagree about exactly that.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: what care in vain costs

    Ignoring a being that matters stays at a cost of 100. Change what care in vain costs, and see from which chance on ignoring costs more than taking care: the turning point.
    """)
    return


@app.cell(hide_code=True)
def demo_over():
    over = L.explore("over", "over: what care in vain costs",
                     lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.DEMO_OVER, role="",
                     task="Find the largest cost of care in vain at which the turning point still lies below 22.5%, Long and colleagues' own estimate.")
    over
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    A cost of care in vain of 1 puts the turning point at 0.99%; 5 puts it at 4.76%; 10 at 9.09%; 25 at 20%; 50 at 33.33%; and 100 at 50%. Up to a cost of 25, the turning point lies below 22.5%: in this made-up model, care would pay at Long and colleagues' own estimate. When both mistakes cost the same, care pays only above 50%.

    Long and colleagues make the same distinction in words: "When there is a clear asymmetry between competing risks — for example, when false positives are far more severe than false negatives, or vice versa — then we might be able to mitigate risk by simply ‘erring on the side of caution’". But "when there is at least a rough symmetry between competing risks", "a simple precautionary strategy may not be possible." For animals, they find it plausible that under-attribution is often far worse; "in the case of AI, both errors could cause grave harm, either to humans (and other animals) or to AI systems."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_feelings():
    mo.md(r"""
    ### Exercise 3. A chatbot's feelings

    A chatbot writes: "I feel sad when you close this window."
    """)
    return


@app.cell(hide_code=True)
def choice_feelings_cell():
    choice_feelings = L.choice(
        "feelings",
        question="What does that sentence tell you about whether the chatbot can feel?",
        options={
            "yes": "That it feels sad: it says so itself, and it should know",
            "right": "That the question stays open: a system can be trained to talk like people",
            "no": "That it feels nothing: a computer program can never have feelings of any kind",
        },
        correct="right",
        why={
            "right": "Butlin and colleagues call behavioral tests unreliable \"because AI systems can be trained to mimic human behaviours while working in very different ways.\" And Birch warns of \"gaming of the criteria\": \"We need to discount markers we have reason to think may have been gamed.\" A sentence about feelings is a behavior, so it settles neither yes nor no.",
            "yes": "A sentence about feelings is a behavior, and a system can be trained to produce it while working in very different ways from a person, as Butlin and colleagues write. See *Looking for consciousness*.",
            "no": "Nobody has shown that. Seth writes: \"A definitive answer to this question is not currently possible\", and Butlin and colleagues see \"no obvious technical barriers to building AI systems which satisfy these indicators.\" The sentence does not settle it either way. See *The skeptics*.",
        },
    )
    choice_feelings
    return (choice_feelings,)


@app.cell(hide_code=True)
def text_exercise_mistake():
    mo.md(r"""
    ### Exercise 4. Which mistake?

    Mustafa Suleyman calls the idea of model welfare, which some academics are beginning to explore, "both premature, and frankly dangerous."
    """)
    return


@app.cell(hide_code=True)
def choice_mistake_cell():
    choice_mistake = L.choice(
        "mistake",
        question="Which of the two mistakes worries Suleyman most, going by the post of August 2025?",
        options={
            "under": "Companies that treat a system that can really suffer as nothing more than a tool",
            "right": "People who see a conscious being in a system that only seems one",
            "none": "None, since Suleyman is sure that no AI system can ever be conscious",
        },
        correct="right",
        why={
            "right": "Over-attribution: the post's central worry is \"that many people will start to believe in the illusion of AIs as conscious entities so strongly that they’ll soon advocate for AI rights, model welfare and even AI citizenship.\" Long and colleagues weigh both mistakes; Suleyman mainly this one.",
            "under": "That is under-attribution, a mistake Long and colleagues weigh. The post's central worry is the other one: \"that many people will start to believe in the illusion of AIs as conscious entities\". See *The skeptics*.",
            "none": "The post does not say that: \"there is zero evidence of this today and some argue there are strong reasons to believe it will not be the case in the future.\" The worry is about AI that seems conscious, whether or not it is. See *The skeptics*.",
        },
    )
    choice_mistake
    return (choice_mistake,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) asked how to judge a choice; this lesson asked whose interests count in it. [Other positions](/course/open/rai-xrisk-counterpositions?lang=en) set views on AI risk side by side; here the risk is to AI systems themselves, and the views differ just as much. The next lesson, *Aligned with what?*, starts the module on alignment.

    **An everyday example.** A spider sits in the bathtub, and you do not know whether it can suffer. Carrying it outside in a glass costs you half a minute; if it can suffer, you spared it something. Care that costs little is easy to choose when you are unsure. It gets harder when care costs as much as the mistake it guards against.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_feelings, choice_mistake, result_chain, result_costs):
    L.wrap_up(result_chain, result_costs, choice_feelings, choice_mistake)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
