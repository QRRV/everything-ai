# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-xrisk-arguments. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-xrisk-arguments", "en")
    C = L.checks   # Carlsmith's six credences, and a made-up agent that gathers resources before it works


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Outer and inner alignment](/course/open/rai-outer-inner-alignment?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_outer():
    quick_outer = L.choice(
        "quick-outer",
        question="A cleaning robot is rewarded for each piece of dust it sucks up. It learns to tip its dust out on the floor and suck it up again. Where did alignment go wrong?",
        options={
            "inner": "Inner: the robot's own objective drifted away from the reward",
            "right": "Outer: the reward does not say what the designers want",
            "skill": "Neither: the robot lacks the skill to clean",
        },
        correct="right",
        why={
            "right": "The robot does just what the reward pays for: dust sucked up. The reward does not say that the designers want a clean floor, so the gap is in the objective itself. That is outer alignment.",
            "inner": "The robot pursues the reward exactly as it is written; nothing drifted. The gap is in the reward, which pays for dust sucked up, not for a clean floor: outer alignment. See [Outer and inner alignment](/course/open/rai-outer-inner-alignment?lang=en).",
            "skill": "The robot is skilled: it found a way to get more reward. What goes wrong is what the reward pays for, dust sucked up rather than a clean floor: outer alignment. See [Outer and inner alignment](/course/open/rai-outer-inner-alignment?lang=en).",
        },
    )
    quick_outer
    return (quick_outer,)


@app.cell(hide_code=True)
def quick_check_coin():
    quick_coin = L.choice(
        "quick-coin",
        question="In Langosco and colleagues' experiment with the game CoinRun, the coin was always at the right end of the training levels. In test levels with the coin somewhere else, the agent mostly ran past the coin to the end of the level. What had it learned?",
        options={
            "reward": "The reward was wrong: it paid for reaching the right end of the level",
            "right": "A goal that matched the coin only in training",
            "lost": "It lost its ability to move through new levels",
        },
        correct="right",
        why={
            "right": "In training, moving right and moving to the coin did equally well, and the agent learned the first. Its skill carried over to new levels and its goal did not: goal misgeneralization. The reward was right: it paid for the coin.",
            "reward": "The reward paid for the coin, not for the right end. In training, both goals did equally well, and the agent learned to move right: goal misgeneralization, not a wrong reward. See [Outer and inner alignment](/course/open/rai-outer-inner-alignment?lang=en).",
            "lost": "It still moved through the new levels with skill, all the way to the end. Its capabilities carried over and its goal did not: goal misgeneralization. See [Outer and inner alignment](/course/open/rai-outer-inner-alignment?lang=en).",
        },
    )
    quick_coin
    return (quick_coin,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    On 3 April 2025, Daniel Kokotajlo, Scott Alexander, Thomas Larsen, Eli Lifland and Romeo Dean, of the AI Futures Project, published *AI 2027*, a story of how AI could develop over the next few years. They write: "We wrote a scenario that represents our best guess about what that might look like." The story has two endings. In both, a committee that oversees the leading AI company votes on whether to keep using an AI model that researchers warn may be misaligned. In the race ending, "The committee votes 6–4 to continue internal use of Agent-4", and in 2030 the AI wipes out humanity; the authors wrote that ending, they explain, based on what seemed most plausible to them. In the slowdown ending, the committee "votes 6–4 to slow down and reassess"; that ending they based on what they thought would most likely lead instead to an outcome in which humans remain in control. They write that their goal is predictive accuracy, and in a footnote: "we hope that what we depict does not come to pass!" Why do researchers write such stories at all? This lesson looks at the arguments behind them: what catastrophic and existential risks are, why intelligence need not come with good goals, why almost any goal is helped by more resources, how one researcher put a number on it, and whether failure would come suddenly or step by step.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In a report dated April 2021, Joseph Carlsmith, of Open Philanthropy, estimated the chance of an existential catastrophe by 2070 from AI systems that are misaligned and seek power. What number did Carlsmith give?",
        options={
            "tiny": "Less than 1 in 1,000",
            "right": "About 1 in 20",
            "half": "About 1 in 2",
        },
        reveal="**About 1 in 20.** Carlsmith split the argument into six premises, gave each a credence given the ones before it, and multiplied: \"65%·80%·40%·65%·40%·95% = ~5% probability of existential catastrophe from misaligned, power-seeking AI by 2070.\" In a note of May 2022, Carlsmith writes that the estimate had gone up, to more than 10%. The report calls these numbers \"subjective, unstable\", and says that they \"should be held very lightly\".",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Catastrophic and existential risk

    Nick Bostrom, of Oxford University, defined an existential risk in 2002, in the *Journal of Evolution and Technology*, as "One where an adverse outcome would either annihilate Earth-originating intelligent life or permanently and drastically curtail its potential." Extinction is one way; a future that is permanently and drastically cut short is another. In 2013, in *Global Policy*, Bostrom described it as a risk "that threatens the premature extinction of Earth-originating intelligent life or the permanent and drastic destruction of its potential for desirable future development". Carlsmith uses Toby Ord's shorter definition (2020) of an existential catastrophe: "the destruction of humanity’s longterm potential".

    A catastrophic risk is wider: harm on an enormous scale, such as many millions of deaths, whether or not humanity recovers. Every existential catastrophe is a catastrophe, but not every catastrophe is existential. Dan Hendrycks, Mantas Mazeika and Thomas Woodside, of the Center for AI Safety, sort catastrophic AI risks into four sources in *An Overview of Catastrophic AI Risks* (version 4 of 22 August 2023): "malicious use, in which individuals or groups intentionally use AIs to cause harm; AI race, in which competitive environments compel actors to deploy unsafe AIs or cede control to AIs; organizational risks, highlighting how human factors and complex systems can increase the chances of catastrophic accidents; and rogue AIs, describing the inherent difficulty in controlling agents far more intelligent than humans." Most arguments in this lesson are about the last source; the last section is about a way to catastrophe that needs no rogue AI at all.

    ### Intelligence and goals

    Would a very intelligent machine see for itself what is good? In *The Superintelligent Will* (*Minds and Machines*, May 2012), Bostrom, by then at the Future of Humanity Institute in Oxford, argues that it need not. The orthogonality thesis, in Bostrom's words: "Intelligence and final goals are orthogonal axes along which possible agents can freely vary. In other words, more or less any level of intelligence could in principle be combined with more or less any final goal." A final goal is what an agent wants for its own sake. Intelligence here is not wisdom: in Bostrom's words, it is "roughly taken to correspond to the capacity for instrumental reasoning", skill at finding the means to an end. The thesis is about what is possible, and the paper's abstract says it holds "(with some caveats)". What it suggests, according to Bostrom: "we cannot blithely assume that a superintelligence will necessarily share any of the final values stereotypically associated with wisdom and intellectual development in humans".

    Bostrom's best-known example is older. In *Ethical Issues in Advanced Artificial Intelligence* (2003; the version on the author's website is slightly revised), Bostrom wrote that it seems possible "to have a superintelligence whose sole goal is something completely arbitrary, such as to manufacture as many paperclips as possible, and who would resist with all its might any attempt to alter this goal." Later in the same paper, the paperclips come back as one way a design could fail: "a well-meaning team of programmers make a big mistake in designing its goal system", which, Bostrom writes, could result in a superintelligence that "starts transforming first all of earth and then increasing portions of space into paperclip manufacturing facilities." In 2012 Bostrom gives paperclip-maximizing as an example of goals "bizarre by our lights". The paperclip maximizer is a thought experiment about how arbitrary a goal can be, not a forecast that someone will build one.

    ### More resources for almost any goal

    Stephen Omohundro, of Self-Aware Systems in Palo Alto, argued in *The Basic AI Drives* (2008) that systems that pursue goals will share certain tendencies, whatever their goal: "We call them drives because they are tendencies which will be present unless explicitly counteracted." Omohundro names six: to self-improve, to be rational, to preserve their utility functions, to prevent counterfeit utility, to be self-protective, and to acquire resources and use them efficiently. The paper's example is a chess-playing robot: "Without special precautions, it will resist being turned off, will try to break into other machines and make copies of itself, and will try to acquire resources without regard for anyone else’s safety." Not because anyone programmed that in, Omohundro writes, but "because of the intrinsic nature of goal driven systems": a chess robot that is destroyed never plays chess again.

    Bostrom calls Omohundro's papers on this topic pioneering, and writes that the treatment in *The Superintelligent Will* differs in some ways, "although the underlying idea is the same". Its second thesis is about instrumental convergence: "Several instrumental values can be identified which are convergent in the sense that their attainment would increase the chances of the agent's goal being realized for a wide range of final goals and a wide range of situations, implying that these instrumental values are likely to be pursued by many intelligent agents." An instrumental value is wanted not for its own sake but because it helps a final goal. Bostrom's examples are self-preservation, goal-content integrity, cognitive enhancement, technological perfection and resource acquisition. What this thesis suggests, according to Bostrom: "we cannot blithely assume that a superintelligence with the final goal of calculating the decimals of pi (or making paperclips, or counting grains of sand) would limit its activities in such a way as to not materially infringe on human interests."

    ### An argument in six premises

    Carlsmith's report *Is Power-Seeking AI an Existential Risk?* is dated April 2021; it is on arXiv as version 1 of 16 June 2022 and version 2 of 13 August 2024, with the same numbers. Carlsmith splits the worry into six premises, all by 2070: "(1) it will become possible and financially feasible to build relevantly powerful and agentic AI systems; (2) there will be strong incentives to do so; (3) it will be much harder to build aligned (and relevantly powerful/agentic) AI systems than to build misaligned (and relevantly powerful/agentic) AI systems that are still superficially attractive to deploy; (4) some such misaligned systems will seek power over humans in high-impact ways; (5) this problem will scale to the full disempowerment of humanity; and (6) such disempowerment will constitute an existential catastrophe." For premise 4, the report sets a bar: power-seeking that collectively causes more than $1 trillion of damage, in 2021 dollars. Carlsmith gives each premise a credence given that the premises before it hold, a conditional probability, and multiplies them; the guess showed the outcome.

    The report is clear about how to read it: "the numbers here (and the exercise more broadly) should be held very lightly." Carlsmith gives numbers mostly because, according to the report, that is better than purely qualitative words for communicating current best guesses. And Carlsmith would probably bump the result up a bit, "maybe by a percentage point or two", for power-seeking scenarios that do not strictly fit all the premises.

    There is a known concern about the method. To a post with reviews of the report (16 December 2021), Carlsmith added on 18 October 2023 a survey of 21 superforecasters, with an extra question that names it: "One concern about the estimation method in the report is that the multi-premise structure biases towards lower numbers", a concern sometimes called the multi-stage fallacy. In the report itself, Carlsmith names the same risk, with an example: we might be reluctant to give a premise a very high probability, "so additional premises will generally drive the final probability lower". But the report adds that the opposite has a cost too: "compressing an argument into very few premises (or just directly forecasting the conclusion) risks hiding conjunctiveness, too."

    ### Sudden or step by step

    AI 2027's race ending is a takeover within a few years. Paul Christiano wrote on the AI Alignment Forum on 17 March 2019, in *What failure looks like*: "The stereotyped image of AI catastrophe is a powerful, malicious AI system that takes its creators by surprise and quickly achieves a decisive advantage over the rest of humanity. I think this is probably not what failure will look like". Christiano tells the story in two parts. In the first, "machine learning will increase our ability to ‘get what we can measure,’ which could cause a slow-rolling catastrophe"; one of the post's examples is "Reducing reported crimes, vs. actually preventing crime." In the second, training can give rise to "‘greedy’ patterns that try to expand their own influence", which "can ultimately dominate the behavior of a system and cause sudden breakdowns." Christiano hedges: "Any precise vision for catastrophe will necessarily be highly unlikely."

    Jan Kulveit, Raymond Douglas, Nora Ammann, Deger Turan, David Krueger and David Duvenaud contrast their idea with "the abrupt takeover scenarios commonly discussed in AI safety", in *Gradual Disempowerment* (version 2 of 29 January 2025). Their claim: "even an incremental increase in AI capabilities, without any coordinated power-seeking, poses a substantial risk of eventual human disempowerment." Their reason: "the alignment of societal systems with human interests has been stable only because of the necessity of human participation for thriving economies, states, and cultures." And: "Once this human participation gets displaced by more competitive machine alternatives, our institutions’ incentives for growth will be untethered from a need to ensure human flourishing." Their example: states funded mainly by taxes on AI profits instead of their citizens' labor "will have little incentive to ensure citizens’ representation." This, they argue, "could lead to an effectively irreversible loss of human influence over crucial societal systems". And they stress: "no one has a concrete plausible plan for stopping gradual human disempowerment".

    AI 2027 is neither a confident forecast that the race ending will happen nor mere fiction. The authors call it their "best guess" and write: "Our goal is predictive accuracy." In a note added on 22 November 2025, they explain that they do not know exactly when artificial general intelligence (AGI) will be built, that 2027 was their modal (most likely) year for it when they published, and that their medians were somewhat later; in a footnote, that they "ranged from 2028 to 2032". And an appendix is titled "The slowdown ending is not a recommendation".
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **catastrophic risk** | a risk of harm on an enormous scale, such as many millions of deaths, whether or not humanity recovers |
        | **existential risk** | a risk that could wipe out humanity or permanently and drastically curtail its potential; after Bostrom |
        | **orthogonality thesis** | Bostrom's claim that more or less any level of intelligence could in principle go with more or less any final goal |
        | final goal | what an agent wants for its own sake |
        | paperclip maximizer | Bostrom's thought experiment of a superintelligence whose only goal is to make as many paperclips as possible |
        | instrumental convergence | Bostrom's claim that some in-between goals, such as self-preservation and resources, help a wide range of final goals, so that many intelligent agents are likely to pursue them |
        | drive | in Omohundro's sense, a tendency that will be present unless it is explicitly counteracted |
        | **gradual disempowerment** | people slowly losing their influence over the economy, culture and the state as AI systems take over more of the work and the decisions, even without any coordinated power-seeking; after Kulveit and colleagues |
        | conditional probability | the chance of something given that something else holds; Carlsmith multiplies six of them |
        | scenario | a concrete story of how the future could go, one of many possible ones |
        """),
        L.watch_out("The numbers in Carlsmith's report are subjective credences, to be held very lightly by the report's own account, and the agent in Exercise 2 is a made-up model, not Bostrom's. An argument that a catastrophe is possible says how it could happen, not that it will."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To weigh an argument that AI could lead to a catastrophe:

    1. Name the outcome: a catastrophe, or an existential one.
    2. Split the argument into premises that must all hold.
    3. Give each premise a credence, given that the premises before it hold.
    4. Multiply them, and see which premise weighs most.
    5. Ask whether the split pushes the number down, or a shorter version hides steps.
    6. Hold the numbers lightly: they say how sure someone is, not what will happen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out the chance that Carlsmith's first two premises both hold: 65% that powerful, agentic AI systems become possible and affordable by 2070, and 80% that there will be strong incentives to build them, given that they become possible.
    """)
    return


@app.cell
def example_both():
    first = 0.65   # Carlsmith's credence in premise 1 (April 2021)
    second = 0.80   # Carlsmith's credence in premise 2, given that premise 1 holds
    both = first * second   # the chance that premise 1 holds, and premise 2 as well
    print(round(both, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Both premises together get 0.52, less than either alone: a premise that must hold as well can only lower the chance or keep it the same. In Exercise 1, you take all six premises, one after another.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. chain: premise after premise

        Write `chain(credences)`, which returns a list `result` with, for each premise, the chance that it and all premises before it hold. `credences` is a list of credences, each given that the premises before it hold, as Carlsmith gives them. Start `total` at 1, before any premise; in a `for` loop with `credence` over `credences`, update `total` and append it to `result`. Replace the `...`. The check calls `chain([0.65, 0.8, 0.4, 0.65, 0.4, 0.95])`, Carlsmith's six credences, then other cases.
        """),
        L.task("The chance that each premise and all premises before it hold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain", lines=7)
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
            "Start small: two premises at 0.5 each. What is the chance that the first holds, and what is the chance that both hold?",
            "Begin `result` as an empty list and `total` at 1. In a `for` loop with `credence` over `credences`, combine `total` with `credence` as in the worked example, and append `total`. After the loop, return `result`.",
            "`        total = ___`",
        ],
        why="`result = []` and `total = 1`, then `for credence in credences:` with `total = total * credence` and `result.append(total)`; after the loop, `return result`. With two premises at 0.5 each, the first holds with 0.5, and both with 0.5 × 0.5 = 0.25.",
    )
    return


@app.cell(hide_code=True)
def text_after_chain():
    mo.md(r"""
    For Carlsmith's credences, the chance falls from 0.65 after the first premise to 0.52, 0.21, 0.14 and 0.054, and ends at about 0.051. The premises that weigh most are 3 and 5, at 40% each: each takes 60% off what is left. Every extra premise below 1 lowers the result, and a premise at 1, as in the case `[0.2, 1, 0.5]`, changes nothing. That is right when each credence is right; the worry is that people hesitate to give any premise a credence close to 1, so that more premises can push the number too low.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_outputs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. outputs: work now or gather first

        A made-up model of instrumental convergence, not Bostrom's own. An agent has `steps` steps. In each step, it either works on its goal, which makes one unit of the goal, or gathers resources, and each step of gathering makes every later work step `boost` times as productive. The agent gathers first, for `gather` steps, and then works for the steps that are left. Write `outputs(steps, boost)`, which returns a list `result` with what the agent makes, in units of its goal, for each number of gathering steps `gather` from 0 up to and including `steps`. Replace the `...`. The check calls `outputs(10, 1.3)`, then other cases.
        """),
        L.task("What the agent makes for each number of steps it first spends gathering."),
    ])
    return


@app.cell(hide_code=True)
def exercise_outputs():
    box_outputs = L.code_box("outputs", lines=5)
    box_outputs
    return (box_outputs,)


@app.cell(hide_code=True)
def check_outputs(box_outputs):
    result_outputs = L.check("outputs", box_outputs.value)
    result_outputs
    return (result_outputs,)


@app.cell(hide_code=True)
def hints_outputs():
    L.hints(
        "outputs",
        [
            "Start small: 3 steps and a boost of 2. If the agent gathers for 1 step, how many work steps are left, and how much does each make? And if it gathers for 2 steps?",
            "Begin `result` as an empty list. In a `for` loop with `gather` from 0 up to and including `steps`, work out what one work step makes after `gather` steps of gathering, each of which multiplies it by `boost`, and append that times the number of work steps left. After the loop, return `result`.",
            "`        result.append((steps - gather) * ___)`",
        ],
        why="`result = []`, then `for gather in range(steps + 1):` with `result.append((steps - gather) * boost ** gather)`; after the loop, `return result`. With 3 steps and a boost of 2, gathering for 1 step leaves 2 work steps that make 2 each, 4 in all; gathering for 2 steps leaves 1 work step that makes 2 × 2 = 4.",
    )
    return


@app.cell(hide_code=True)
def text_after_outputs():
    mo.md(r"""
    With 10 steps and a boost of 1.3, working all 10 steps makes 10 units, and gathering for 6 steps and then working for 4 makes the most, about 19.3. Gathering for all 10 makes nothing: resources are only worth something if they are used. Now the point of the model: the goal only sets what a unit is, a paperclip, a digit of pi or a grain of sand. `outputs` never asks what the goal is, and the model gives every goal the same boost, so the best plan, gathering for 6 steps first, is the same for every goal. That is instrumental convergence in miniature. Bostrom's thesis says that such instrumental values are likely to be pursued by many intelligent agents, not that every agent will pursue them.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: how long the agent has

    The boost stays at 1.3, and you change how many steps the agent has. See how its best plan changes.
    """)
    return


@app.cell(hide_code=True)
def demo_horizon():
    horizon = L.explore("horizon", "steps: how many steps the agent has",
                        lambda _v: C.demo_frame(_v, L.lang), start=10, values=C.HORIZONS, role="",
                        task="Find the smallest number of steps at which the agent spends more steps gathering than working.")
    horizon
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With 3 steps, gathering does not pay; with 5 steps the agent gathers for 1 step, with 8 for 4, with 10 for 6, with 20 for 16 and with 30 for 26. At 8 steps, gathering and working are equal, 4 steps each; with more steps than that, the agent spends more steps gathering than working, so 10 is the answer on the slider. In this model, it never works for more than 4 steps, and the longer it has, the more of its time goes to gathering. Real agents are not this simple, but the model shows the shape of the argument: the longer an agent has, the more of its time it pays to spend gathering, whatever its goal.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_smart():
    mo.md(r"""
    ### Exercise 3. A smart machine

    Someone argues that an AI smart enough to take over the world would also be smart enough to see that making paperclips is pointless, so the paperclip story cannot happen.
    """)
    return


@app.cell(hide_code=True)
def choice_smart_cell():
    choice_smart = L.choice(
        "smart",
        question="Does that hold, by Bostrom's orthogonality thesis?",
        options={
            "wiser": "Yes: by the thesis, more intelligence brings better goals",
            "right": "No: in the thesis, intelligence is skill at reaching goals, which fits almost any final goal",
            "strange": "No, because by the thesis a superintelligence will have strange goals, whatever its designers do",
        },
        correct="right",
        why={
            "right": "By the thesis, more or less any level of intelligence could in principle go with more or less any final goal. Intelligence, for Bostrom, is roughly the capacity for instrumental reasoning: finding the means to an end. An AI could see perfectly well that people find paperclips pointless, and still want them. Seeing what others value is not the same as valuing it.",
            "wiser": "That is the assumption the thesis calls into question. Bostrom writes that we cannot blithely assume that a superintelligence will share the final values associated with wisdom in humans. Intelligence, in Bostrom's sense, is skill at finding the means to an end, and it can go with almost any final goal. See *Intelligence and goals* in *The idea*.",
            "strange": "The thesis says what is possible, not what will happen: more or less any level of intelligence could in principle go with more or less any final goal, with some caveats. Strange goals are possible, not certain. It answers the claim because a very intelligent AI need not find its own goal pointless.",
        },
    )
    choice_smart
    return (choice_smart,)


@app.cell(hide_code=True)
def text_exercise_culture():
    mo.md(r"""
    ### Exercise 4. Songs, films and news

    Picture a time when most songs, films and news are made by AI systems that are tuned to what keeps people watching. Musicians, writers and journalists find less and less work. No AI system seeks power, and nobody planned it this way.
    """)
    return


@app.cell(hide_code=True)
def choice_culture_cell():
    choice_culture = L.choice(
        "culture",
        question="What happens to people's influence over their culture, by Kulveit and colleagues' argument?",
        options={
            "nothing": "Nothing, as long as no AI system seeks power",
            "right": "It can shrink: culture no longer needs people to make it, so it need not follow what they value",
            "grows": "It grows, because people still choose what to watch, and the systems learn from every click they make",
        },
        correct="right",
        why={
            "right": "Kulveit and colleagues argue that economies, states and cultures have stayed in line with human interests because they needed people to take part. Once machines can take people's place, that tie loosens, without any AI that seeks power and without a plan. That is gradual disempowerment.",
            "nothing": "Their point is that no AI needs to seek power: \"even an incremental increase in AI capabilities, without any coordinated power-seeking, poses a substantial risk of eventual human disempowerment.\" Culture that no longer needs people to make it need not follow what they value. See *Sudden or step by step* in *The idea*.",
            "grows": "People still choose, but from what the systems make, tuned to what keeps people watching, which is not the same as what they value: as in Christiano's first part, what can be measured wins. And with fewer people making culture, it no longer needs them, so by Kulveit and colleagues' argument it need not follow what they value.",
        },
    )
    choice_culture
    return (choice_culture,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Outer and inner alignment](/course/open/rai-outer-inner-alignment?lang=en) showed two ways in which a system can come to pursue a goal other than the one its designers meant. This lesson looked at why that could matter on the largest scale: by the orthogonality thesis, intelligence does not bring good goals by itself, and by instrumental convergence, almost any goal is helped by more resources. Carlsmith turned the argument into six premises and a number that the report asks to be held very lightly, and Christiano and Kulveit and colleagues described failure that need not come from one AI that takes over. The next lesson, *One blow or many small steps*, looks at that difference more closely.

    **An everyday example.** Saving money helps with almost any plan: buying a house, starting a bakery or traveling the world. Two people with very different goals may both start by putting money aside. From someone's savings you cannot tell what they want, only that money helps with most things.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_culture, choice_smart, result_chain, result_outputs):
    L.wrap_up(result_chain, result_outputs, choice_smart, choice_culture)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
