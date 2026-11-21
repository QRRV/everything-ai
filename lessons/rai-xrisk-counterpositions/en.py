# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-xrisk-counterpositions. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow true risk

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-xrisk-counterpositions", "en")
    C = L.checks   # made-up dangers with the protection against each, and a made-up soft pause


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [One blow or many small steps](/course/open/rai-accumulative-risk?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="In the previous lesson's made-up model, a society's resilience is 80. Each year takes off a harm of 5, and the society wins back 2, but never more than up to 100. What is its resilience after two years?",
        options={
            "again": "77: each new year starts again from 80",
            "right": "74: the second year starts from 77",
            "gain": "84: the society wins back 2 in each year",
        },
        correct="right",
        why={
            "right": "Each year starts where the year before ended: 80 − 5 + 2 = 77 after the first year, and 77 − 5 + 2 = 74 after the second. Neither is above 100, so the top of 100 changes nothing here.",
            "again": "Then the harms would never build up. Each year starts where the year before ended: 77 after the first year, then 77 − 5 + 2 = 74. See [One blow or many small steps](/course/open/rai-accumulative-risk?lang=en).",
            "gain": "The harm comes off as well: each year takes off 5 and wins back 2, so the resilience falls by 3 a year, from 80 to 77 and then to 74. See [One blow or many small steps](/course/open/rai-accumulative-risk?lang=en).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def quick_check_lockin():
    quick_lockin = L.choice(
        "quick-lockin",
        question="By Finnveden, Riedel and Shulman's definition, when is a feature of the world locked in?",
        options={
            "bad": "Once bad values have won out over all the good ones",
            "right": "Once how it turns out in the long run is far less uncertain",
            "extinct": "Only once humanity has gone extinct and nothing can change any more",
        },
        correct="right",
        why={
            "right": "In their definition, a feature is locked in at some time if, before it, there is notable uncertainty about how the feature will turn out in the long run, and after it, that uncertainty has been significantly reduced. Good values can be locked in as well as bad ones.",
            "bad": "Lock-in is about how much the long-run future can still change, not about whether the values are good: good values can be locked in as well. See [One blow or many small steps](/course/open/rai-accumulative-risk?lang=en).",
            "extinct": "Extinction is one example of a lock-in event, not the only one: in global value lock-in, all powerful actors come to hold values from a much smaller set than before. See [One blow or many small steps](/course/open/rai-accumulative-risk?lang=en).",
        },
    )
    quick_lockin
    return (quick_lockin,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The previous lessons set out why some researchers fear a catastrophe from AI, in one blow or in many small steps. Not everyone sees it that way. Some see AI as a normal technology, however transformative; some want the attention on the harms AI does today; some want AI built faster; and some want it built faster, but with protection first. Behind that last view is an idea: speed up the technologies that protect against others, and hold back the dangerous ones, so that each protection is there before the danger it guards against. Nick Bostrom gave the idea a name: differential technological development.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In which text did Bostrom give the idea this name?",
        options={
            "book": "In the book Superintelligence, from 2014",
            "right": "In a paper on existential risks, from 2002",
            "preprint": "In a joint preprint on the concept, from 2022",
        },
        reveal="**In a paper on existential risks, from 2002**, of which a first version is from 2001. In it, Bostrom writes: \"Our focus should be on what I want to call differential technological development: trying to retard the implementation of dangerous technologies and accelerate implementation of beneficial technologies, especially those that ameliorate the hazards posed by other technologies.\" Later texts point elsewhere: a chapter by Anders Sandberg and a post of 2022 credit the idea to Bostrom's book *Superintelligence* (2014). Both come from the authors of a preprint by Jonas Sandbrink and colleagues, and the post says of that preprint: \"To our knowledge, this is the first academic article seeking to comprehensively articulate a principle of differential technology development.\" This lesson looks at that idea, and at the other positions beside it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### AI as normal technology

    Arvind Narayanan and Sayash Kapoor published *AI as Normal Technology* with the Knight First Amendment Institute in April 2025. The name does not mean that AI matters little: "To view AI as normal is not to understate its impact—even transformative, general-purpose technologies such as electricity and the internet are ‘normal’ in our conception." They call the statement that AI is normal technology three things: "a description of current AI, a prediction about the foreseeable future of AI, and a prescription about how we should treat it." They expect the transformative economic and societal impacts of AI to be slow, "on the timescale of decades". And: "We view AI as a tool that we can and should remain in control of, and we argue that this goal does not require drastic policy interventions or technical breakthroughs." The essay, they write, "has the unusual goal of stating a worldview rather than defending a proposition." Nor do they expect progress to stall. Of the world they describe, in which AI is far more advanced than it is today, they write: "We are not claiming that AI progress—or human progress—will stop at that point."

    In their view, catastrophic misalignment is "(by far) the most speculative of the risks that we discuss". They explain the word: "By speculative risks, we mean those for which there is epistemic uncertainty about whether or not the true risk is zero—uncertainty that can potentially be resolved through further observations or research." Their main defense against it lies not in the models themselves, but further on, where AI is put to use: "In our view, the primary defense against misalignment, again, lies downstream. The defenses needed against misuse that we discussed earlier—from hardening critical infrastructure to improving cybersecurity—will also serve as protection against potential misalignment risks."

    They put more weight on systemic risks, among them "concentration of power, erosion of social trust, pollution of the information ecosystem" and "mass surveillance": "If AI is normal technology, these risks become far more important than the catastrophic ones discussed above." They advocate "reducing uncertainty as a first-rate policy goal and resilience as the overarching approach to catastrophic risks." And they see a trade-off: "Defense against superintelligence requires humanity to unite against a common enemy, so to speak, concentrating power and exercising central control over AI technology." They are more concerned, they write, about risks from people using AI for their own ends.

    ### Harms first

    Others say that talk of extinction draws attention away from the harms that are already here. An unsigned editorial in *Nature* of 27 June 2023 writes about the idea that AI could lead to human extinction: "But, like a magician’s sleight of hand, it draws attention away from the real issue: the societal harms that AI systems and tools are causing now, or risk causing in future." The editorial says that many AI researchers and ethicists it spoke to are frustrated by the doomsday talk, and that this talk is problematic in at least two ways. The first: "the spectre of AI as an all-powerful machine fuels competition between nations to develop AI so that they can benefit from and control it. This works to the advantage of tech firms: it encourages investment and weakens arguments for regulating the industry." Further on: "Fearmongering narratives about existential risks are not constructive. Serious discussion about actual risks, and action to contain them, are."

    Alex Hanna, director of research at the Distributed AI Research Institute, and Emily M. Bender, professor of linguistics at the University of Washington, wrote in *Scientific American* on 12 August 2023, after naming wrongful arrests, surveillance, defamation and deepfake pornography: "These issues, and not the imagined potential to wipe out humanity, are the real threat of artificial intelligence." The word "imagined" is theirs; the article is an opinion piece, about where attention should go.

    ### Faster

    A post of 10 July 2022, *Notes on e/acc principles and tenets*, with the byline "Beff Jezos and bayes", is one of the early texts of effective accelerationism, e/acc for short; it points to an older "original post by Swarthy". It says that "higher variance marketplaces and competition are better at identifying and capitalizing on utility from our environment over other methods such as top-down optimal control", and that "over-regulating technologies suppresses variance and hence slows down progress". It also says: "Effective accelerationism aims to follow the ‘will of the universe’". Its summary "in a nutshell" includes "Stop fighting the thermodynamic will of the universe" and "You cannot stop the acceleration".

    Marc Andreessen's *The Techno-Optimist Manifesto* (16 October 2023) calls its own view techno-optimism: "We believe in accelerationism – the conscious and deliberate propulsion of technological development – to ensure the fulfillment of the Law of Accelerating Returns." On AI: "We believe any deceleration of AI will cost lives. Deaths that were preventable by the AI that was prevented from existing is a form of murder." Under the heading "The Enemy", it counts “existential risk”, “trust and safety” and “tech ethics” among the names of what it calls "a mass demoralization campaign". Its list of "Patron Saints of Techno-Optimism" starts with the accounts @BasedBeffJezos and @bayeslord. The manifesto came more than a year after the e/acc post.

    ### d/acc

    In *My techno-optimism* (27 November 2023), Vitalik Buterin writes of being "sympathetic to the e/acc perspective in a lot of contexts", but adds that "it is possible to take this perspective too far", pointing to concerns about AI and to the e/acc enthusiasm for military technology. Buterin also distrusts a common answer to such risks: "Across the board, I see far too many plans to save the world that involve giving a small group of people extreme and opaque power and hoping that they use it wisely." That leads Buterin to a different philosophy: "I will refer to this philosophy by the name of d/acc." And: "The ‘d’ here can stand for many things; particularly, defense, decentralization, democracy and differential." Buterin ties it to an older idea: "This core idea, that some technologies are defense-favoring and are worth promoting, while other technologies are offense-favoring and should be discouraged, has roots in effective altruist literature under a different name: differential technology development."

    A year later, in *d/acc: one year later* (5 January 2025): "The core idea of d/acc is simple: decentralized and democratic differential defensive acceleration. Build technologies that shift the offense/defense balance toward defense, and do so in a way that does not rely on handing over more power to centralized authorities." Buterin's first strategy for the risks of AI is liability: whoever causes legally actionable damage could be sued. Only as a second strategy, under the heading "Global ‘soft pause’ button on industrial-scale hardware", does Buterin write: "If I was convinced that we need something more ‘muscular’ than liability rules, this is what I would go for. The goal would be to have the capability to reduce worldwide available compute by ~90-99% for 1-2 years at a critical period, to buy more time for humanity to prepare."

    ### Differential technology development

    Bostrom's section on the idea starts from a limit: "The limited enforceability of technological bans restricts the set of feasible policies from which we can choose." What can be affected, by Bostrom's account, is "the rate of development of various technologies and potentially the sequence in which feasible technologies are developed and implemented." In biotechnology, for example, Bostrom would promote vaccines, drugs, protective gear, sensors and diagnostics, and delay biological warfare agents as much as possible. For nanotechnology: "the desirable sequence would be that defense systems are deployed before offensive capabilities become available to many independent powers". For AI, Bostrom did not argue for slowing down. Superintelligence has its own dangers, the paper says, but "these are dangers that we will have to face at some point no matter what. But getting superintelligence early is desirable because it would help diminish other risks." Superintelligence, the paper says, would speed up nanotechnology, "thus shortening the period of vulnerability between the development of dangerous nanoreplicators and the deployment of adequate defenses."

    In 2022, Jonas B. Sandbrink, Hamish Hobbs, Jacob L. Swett, Allan Dafoe and Anders Sandberg put the idea forward as a principle, in a preprint, with a slightly different name. Sandberg quotes it in a book chapter that is partly based on that preprint; the PDF is from March 2024 and gives Reuben College and the Future of Humanity Institute at the University of Oxford as Sandberg's affiliations. The principle reads: "Differential technology development (DTD): Leverage risk-reducing interactions between technologies by affecting their relative timing." In the chapter, Sandberg names three kinds of help: "This can happen by safety technologies, by defensive technologies, or by substitute technologies arriving early or instead of risk-increasing technologies." The chapter also points to the Mohists of the Warring States era in China, who spread knowledge of defensive warfare to make wars of aggression impractical: "This may also be the earliest attempt to speed up certain technologies to reduce the harm done by other technologies."

    ### An objection from a researcher

    Yann LeCun, in episode #416 of the Lex Fridman Podcast (transcript posted on 7 March 2024), calls most of the assumptions behind catastrophe scenarios false. The first: "So the first assumption is that the emergence of super intelligence is going to be an event, that at some point we’re going to figure out the secret and we’ll turn on a machine that is super intelligent, and because we’d never done it before, it’s going to take over the world and kill us all. That is false. It’s not going to be an event." Many people would build such systems, in LeCun's view: "If some other goes rogue, then we can use the good ones to go against the rogue ones. So it’s going to be my smart AI police against your rogue AI." And: "They’re not going to have the desire to dominate because the desire to dominate is something that has to be hardwired into an intelligent system." LeCun's objection is aimed at the picture of one decisive event; whether it also answers the accumulative picture of the previous lesson, LeCun does not say there. That reading is this lesson's.

    ### Where they disagree

    On control: Kasirzadeh, in the previous lesson, calls for centralized control measures against decisive risk. Buterin writes: "Embracing centralized control for the sake of safety has permanent appeal to a subset of people", and adds that readers know many examples, and their downsides. Narayanan and Kapoor head a section "Nonproliferation is infeasible to enforce and leads to single points of failure". And the e/acc post prefers competition to "top-down optimal control". On speed: for Andreessen, any deceleration of AI costs lives; Buterin would want a soft pause only as a fallback; and Bostrom points not to speed alone, but to the rate and the sequence.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **differential technology development** | speeding up technologies that reduce the risks of others and delaying those that increase risk, so that protection arrives before the danger it guards against; Bostrom (2002) called it differential technological development |
        | effective accelerationism (e/acc) | a movement that wants to speed up technology as much as possible; one of its early texts is a post of July 2022 |
        | defensive accelerationism (d/acc) | Buterin's d/acc (2023), in Buterin's words defensive (or decentralization, or differential) acceleration: speeding up technologies that favor defense, in a decentralized and democratic way |
        | AI as normal technology | Narayanan and Kapoor's worldview (2025): AI as a tool people can and should stay in control of, transformative like electricity and the internet |
        | speculative risk | after Narayanan and Kapoor, a risk for which it is uncertain whether it is zero at all, uncertainty that further observations or research could resolve |
        | soft pause | in Buterin's proposal, the capability to sharply reduce worldwide available compute for a short time at a critical period |
        """),
        L.watch_out("These are positions, not findings: Narayanan and Kapoor state a worldview, the Nature editorial and Hanna and Bender's article are opinion pieces, and the manifesto and Buterin's posts are essays. None of them gives its own estimate of how likely a catastrophe is; Buterin cites the numbers of others, which the next lesson looks at. The models in Exercises 1 and 2 are made up, not Bostrom's, Sandbrink and colleagues' or Buterin's."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To place a position on AI risk:

    1. Find what it worries about most: a catastrophe, today's harms, or holding progress back.
    2. Ask what it wants done about speed: faster, slower, or a different order.
    3. Ask who would hold control: a few central actors, or many.
    4. Look for what it says about the order in which technologies arrive.
    5. Check what kind of text it is: a worldview, an opinion piece, a manifesto, an essay or a paper.
    6. Quote it in its own words, hedges included.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out one made-up pair: a dangerous technology that arrives in year 4, and the protection against it, which arrives in year 7, or, if it is brought forward, in year 3.
    """)
    return


@app.cell
def example_pair():
    risk = 4   # made-up: the year the dangerous technology arrives
    guard = 7   # made-up: the year the protection against it arrives
    early_guard = 3   # made-up: the same protection, brought forward
    print(max(0, guard - risk), max(0, early_guard - risk))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With the protection in year 7, the danger is unguarded for 3 years. With the protection in year 3, it is there first: 3 − 4 is −1, and `max` turns that into 0, because a protection that comes early leaves no unguarded years, not a negative number of them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_exposure():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. exposure: years without protection

        A made-up model of differential technology development, not Bostrom's or Sandbrink and colleagues': five dangerous technologies, each with the year it arrives, and a protection against each, with the year the protection arrives. Bostrom writes of "the period of vulnerability between the development of dangerous nanoreplicators and the deployment of adequate defenses"; this model counts such periods in years.

        Write `exposure(risks, guards)`, which returns a list `result` with, for each pair, the years the dangerous technology is out before its protection arrives, and 0 if the protection is there first. `risks` and `guards` are lists of the same length: `risks[k]` is the year the danger of pair `k` arrives, and `guards[k]` the year its protection arrives. In a `for` loop with `k` over `range(len(risks))`, append the number of each pair to `result`.

        The check calls `exposure([3, 6, 7, 9, 11], [6, 5, 12, 10, 11])`, five made-up pairs, then other cases.
        """),
        L.task("The unguarded years of each pair."),
    ])
    return


@app.cell(hide_code=True)
def exercise_exposure():
    box_exposure = L.code_box("exposure", lines=5)
    box_exposure
    return (box_exposure,)


@app.cell(hide_code=True)
def check_exposure(box_exposure):
    result_exposure = L.check("exposure", box_exposure.value)
    result_exposure
    return (result_exposure,)


@app.cell(hide_code=True)
def hints_exposure():
    L.hints(
        "exposure",
        [
            "Start small: one pair with the danger in year 4 and the protection in year 7, and one with the danger in year 6 and the protection in year 5. How many years is each danger unguarded?",
            "Begin `result` as an empty list. In a `for` loop with `k` over `range(len(risks))`, work out the unguarded years of pair `k` as in the worked example, never below 0, and append them. After the loop, return `result`.",
            "`        result.append(max(0, ___))`",
        ],
        why="`result = []`, then `for k in range(len(risks)):` with `result.append(max(0, guards[k] - risks[k]))`; after the loop, `return result`. With the danger in year 4 and the protection in year 7, the danger is unguarded for 3 years; with the danger in year 6 and the protection in year 5, the protection is there first, so 0.",
    )
    return


@app.cell(hide_code=True)
def text_after_exposure():
    mo.md(r"""
    For the five made-up pairs, the answer is `[3, 0, 5, 1, 0]`: 9 unguarded years in all. In pair 2, the protection comes a year before its danger, and in pair 5 in the same year, so neither leaves an unguarded year. A protection that comes early does not make up for one that comes late elsewhere: pair 2's spare year does nothing for pair 3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_progress():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. progress: a soft pause

        A made-up model of Buterin's soft pause, not Buterin's own: each year adds as much progress as the share of the world's compute that is available in it, 1 in a normal year. Write `progress(shares)`, which returns a list `result` with the progress made by the end of each year. `shares` is a list with the share of each year. Start `total` at 0; in a `for` loop with `share` over `shares`, work out the new `total` and append it to `result`.

        The check calls `progress([1, 1, 0.05, 0.05, 1, 1, 1, 1])`, eight made-up years with a pause of 2 years at 95% less compute, within the "~90-99%" and "1-2 years" of Buterin's proposal, then other cases.
        """),
        L.task("The progress made by the end of each year."),
    ])
    return


@app.cell(hide_code=True)
def exercise_progress():
    box_progress = L.code_box("progress", lines=7)
    box_progress
    return (box_progress,)


@app.cell(hide_code=True)
def check_progress(box_progress):
    result_progress = L.check("progress", box_progress.value)
    result_progress
    return (result_progress,)


@app.cell(hide_code=True)
def hints_progress():
    L.hints(
        "progress",
        [
            "Start small: three years with shares of 1, 0.5 and 1. How much progress has been made by the end of each of the three years?",
            "Begin `result` as an empty list and `total` at 0. In a `for` loop with `share` over `shares`, work out the new `total` from the old one and this year's share, and append it. After the loop, return `result`.",
            "`        total = ___`",
        ],
        why="`result = []` and `total = 0`, then `for share in shares:` with `total = total + share` and `result.append(total)`; after the loop, `return result`. With shares of 1, 0.5 and 1, the progress is 1 by the end of the first year, 1.5 by the end of the second and 2.5 by the end of the third.",
    )
    return


@app.cell(hide_code=True)
def text_after_progress():
    mo.md(r"""
    For the eight made-up years, the progress by the end of year 8 is 6.1, against 8 without a pause. Without a pause, a progress of 6.1 would be reached after 6.1 years; with this pause, only at the end of year 8, 1.9 years later. That is the time this pause buys: 2 × 0.95 years. Buterin's aim is "to buy more time for humanity to prepare". The model leaves out a lot: progress need not follow compute this closely, the pause in Buterin's proposal covers only industrial-scale hardware, and nothing else changes in the model during the pause. Buterin also calls both strategies, liability and the pause button, "only temporary stopgaps".
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: protection sooner

    The five made-up pairs of Exercise 1 stay the same. Bring every protection the same number of years sooner, and see how many unguarded years are left.
    """)
    return


@app.cell(hide_code=True)
def demo_sooner():
    sooner = L.explore("sooner", "sooner: how many years earlier every protection comes",
                       lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.DEMO_SOONER, role="",
                       task="Find the smallest number of years sooner at which no dangerous technology is left unguarded.")
    sooner
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With every protection on its own schedule, the five dangers are unguarded for 9 years in all; with every protection 1 year sooner, for 6; 2 years sooner, for 4; 3 years sooner, for 2; 4 years sooner, for 1; and from 5 years sooner, for none. The last unguarded year belongs to pair 3, whose protection comes 5 years after its danger. Bringing everything the same number of years sooner, dangers and protections alike, would change none of the gaps: in this model, what counts is when each protection comes compared with its danger. A faster pace that shrinks every gap, as Bostrom expected superintelligence to do for nanotechnology, is not in this model. Bostrom calls developments that advance offense and defense equally "neutral from a security perspective", unless they are done by countries "we identify as responsible"; then, in Bostrom's words, they are advantageous "to the extent that they increase our technological superiority over our potential enemies".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_normal():
    mo.md(r"""
    ### Exercise 3. A newspaper column

    A column reads: Narayanan and Kapoor call AI a normal technology, so in their view AI will not change much, and there are no serious risks to worry about.
    """)
    return


@app.cell(hide_code=True)
def choice_normal_cell():
    choice_normal = L.choice(
        "normal",
        question="Does the column get Narayanan and Kapoor right?",
        options={
            "speculative": "Yes: they call catastrophic misalignment a speculative risk",
            "right": "No: normal does not mean small, and they name serious risks",
            "half": "Partly: AI will change a lot, but they see no serious risks in it",
        },
        correct="right",
        why={
            "right": "Narayanan and Kapoor write that to view AI as normal \"is not to understate its impact—even transformative, general-purpose technologies such as electricity and the internet are ‘normal’ in our conception.\" They also name systemic risks, such as concentration of power and mass surveillance, which in their view become far more important if AI is normal technology.",
            "speculative": "By speculative they mean that it is uncertain whether the risk is zero, uncertainty that further observations or research could resolve; that is not the same as no risk. And they name systemic risks, such as concentration of power and mass surveillance. See *AI as normal technology*.",
            "half": "They do see serious risks: systemic ones, such as concentration of power, erosion of social trust and mass surveillance, which in their view become far more important than the catastrophic ones if AI is normal technology. See *AI as normal technology*.",
        },
    )
    choice_normal
    return (choice_normal,)


@app.cell(hide_code=True)
def text_exercise_dacc():
    mo.md(r"""
    ### Exercise 4. Fast and spread out

    A group wants all AI built as fast as possible, by many small companies rather than a few big ones. Which technologies come first does not matter to the group. It calls its view d/acc.
    """)
    return


@app.cell(hide_code=True)
def choice_dacc_cell():
    choice_dacc = L.choice(
        "dacc",
        question="What does the group's view leave out of d/acc, as Buterin describes it?",
        options={
            "nothing": "Nothing, because d/acc is all about spreading power over many hands",
            "right": "Favoring technologies that shift the balance toward defense",
            "slow": "Slowing all AI down until it is safe, as d/acc requires",
        },
        correct="right",
        why={
            "right": "Buterin's core idea is \"decentralized and democratic differential defensive acceleration\": \"Build technologies that shift the offense/defense balance toward defense\". Buterin sets it apart from \"Decentralized acceleration, but don't care about the ‘differential defensive’ part. Basically, be an e/acc, but decentralized.\" The group is fast and spread out, but not differential.",
            "nothing": "Spreading power is one half. The other half is the differential, defensive part: \"Build technologies that shift the offense/defense balance toward defense\". Without it, Buterin calls a view \"be an e/acc, but decentralized\". See *d/acc*.",
            "slow": "d/acc speeds technology up; it favors some technologies over others. Buterin would want a soft pause only as a fallback, if Buterin were convinced that something stronger than liability rules is needed. What the group leaves out is the differential, defensive part. See *d/acc*.",
        },
    )
    choice_dacc
    return (choice_dacc,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [One blow or many small steps](/course/open/rai-accumulative-risk?lang=en) set two kinds of catastrophic risk side by side, decisive and accumulative. This lesson looked at positions beside that view: Narayanan and Kapoor, for whom AI is a normal technology, however transformative; writers who want the attention on today's harms; e/acc and Andreessen, who want technology faster; Buterin's d/acc, faster but with defense first; and differential technology development, the idea that the order in which technologies arrive matters. The positions disagree most about speed and about who should hold control. The next lesson, *What experts expect*, looks at what researchers expect, in numbers.

    **An everyday example.** A town opens a new swimming lake. It can open sooner or later, but what matters more is whether the lifeguards start before the first swimmers arrive, or a month after. Opening sooner or later does nothing about that month without lifeguards; only starting the lifeguards first does.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dacc, choice_normal, result_exposure, result_progress):
    L.wrap_up(result_exposure, result_progress, choice_normal, choice_dacc)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
