# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-race-and-selection. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-race-and-selection", "en")
    C = L.checks   # a made-up table of what staying careful and racing are worth, and a made-up population of agents


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [AI in a complex system](/course/open/rai-complex-systems?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="In a feedback loop, each round adds 1.08 times what the round before added. What happens to the total over many rounds?",
        options={
            "settle": "It settles at a level after a while",
            "right": "It keeps growing, and faster each round",
            "same": "It stays about the same, since 1.08 is close to 1",
        },
        correct="right",
        why={
            "right": "Each round adds 8% more than the round before, so the additions themselves keep growing, and the total runs away. In the made-up market of the last lesson, that happened from an echo of 12: 0.09 × 12 = 1.08.",
            "settle": "A loop settles when each round adds less than the round before: a factor below 1. At 1.08, each round adds more than the one before, so the total keeps growing, faster and faster. See [AI in a complex system](/course/open/rai-complex-systems?lang=en).",
            "same": "1.08 is close to 1, but above it: each round adds 8% more than the one before, and over many rounds that adds up fast. In the made-up market, an echo of 11 (0.99) would still settle, and an echo of 12 (1.08) ran away. See [AI in a complex system](/course/open/rai-complex-systems?lang=en).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_match():
    quick_match = L.choice(
        "quick-match",
        question="An answer of 10 digits counts as right only when every digit is right: an exact match. A model gets each digit right with a chance of 0.9, independently of the others. What is its chance of an exact match?",
        options={
            "single": "0.9, as for a single digit",
            "tenth": "0.09, a tenth of that",
            "right": "Close to 0.35",
        },
        correct="right",
        why={
            "right": "Each digit is right with a chance of 0.9, independently of the others, so all 10 are right with a chance of 0.9 multiplied by itself 10 times: 0.9 to the power 10, about 0.35.",
            "single": "That is the chance for one digit. For all 10 to be right, multiply the chances: 0.9 to the power 10 is about 0.35. See [AI in a complex system](/course/open/rai-complex-systems?lang=en).",
            "tenth": "Multiply the chances, do not divide them: all 10 digits right has a chance of 0.9 to the power 10, about 0.35. See [AI in a complex system](/course/open/rai-complex-systems?lang=en).",
        },
    )
    quick_match
    return (quick_match,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    AI companies write in their own safety policies that they want to avoid a race to the bottom. Anthropic's Responsible Scaling Policy, version 3.4 of July 2026, says that "to the extent that other relevant AI developers prioritize safety and invest in legible demonstrations that they are doing so", its commitments related to competitors "may help avoid an inadvertent ‘race to the bottom’ on safety." OpenAI's Preparedness Framework, version 2 of April 2025, says it could adjust its safeguards if it can firmly establish that another developer has developed or released a system with High or Critical capability without comparable safeguards, but only under conditions, among them that "in order to avoid a race to the bottom on safety, we keep our safeguards at a level more protective than the other AI developer". Why would companies that compete worry about this, and what can make staying careful pay? This lesson looks at what happens when the one who waits may lose, with a simple game you program yourself, and at an argument that competition can select AI systems that nobody wanted.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Dan Hendrycks's textbook models a race between two competitors as an auction: each bids how large a chance of a catastrophe it accepts, and the one who accepts the most wins the prize, if the catastrophe does not happen. Both see the race the same way. In this toy model, what chance of an outcome as bad as the prize is good would each one risk?",
        options={
            "low": "About 1 in 100",
            "right": "About 1 in 3",
            "high": "About 9 in 10",
        },
        reveal="**About 1 in 3.** The textbook calls it the Attrition model: \"If the value of the prize to one competitor is ‘X’, they will be willing to risk a 33% chance of bringing about an outcome equally disvaluable (of value ‘-X’) in order to win their race\". It is a toy model, not a measurement. The textbook adds that \"real-world AI races could produce even worse outcomes than the one predicted by the Attrition model\", for example when competitors do not understand how their race stands.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A race to the bottom

    In a paper in the journal *AI & Society* (online in 2015, in print in 2016), Stuart Armstrong, Nick Bostrom and Carl Shulman present "a simple model of an AI (artificial intelligence) arms race, where several development teams race to build the first AI." Their summary: "Under the assumption that the first AI will be very powerful and transformative, each team is incentivised to finish first—by skimping on safety precautions if need be." In their model, "Having extra development teams and extra enmity between teams can increase the danger of an AI disaster, especially if risk-taking is more important than skill in developing the AI." And: "Surprisingly, information also increases the risks: the more teams know about each others’ capabilities (and about their own), the more the danger increases." They add a condition: "Should these results persist in more realistic models and analysis, it points the way to methods of increasing the chance of the safe development of AI." Amanda Askell and colleagues put it like this: "Armstrong et al. (2016) point out that in an AI development race, responsible development could be prey to a ‘race to the bottom’ dynamic."

    Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (a draft manuscript, version 2 of 15 November 2024) describes such a race between companies and states: "This ‘AI race’ is driven by nations and corporations who feel they must rapidly build and deploy AIs to secure their positions and survive." The trouble it names: "Individually rational decisions can be collectively catastrophic." And: "To obtain a competitive advantage, companies often race to offer the first products to a market rather than the safest." According to the textbook, Microsoft's CEO Satya Nadella said at the launch of the company's AI-powered search engine in February 2023: "A race starts today... we’re going to move fast." The textbook goes on: "Only weeks later, the company’s chatbot was shown to have threatened to harm users". An example from aviation: "Boeing, aiming to compete with its rival Airbus, sought to deliver an updated, more fuel-efficient model to the market as quickly as possible." In the textbook's words, "inadequate testing and pilot training ultimately resulted in the two fatal crashes only months apart, with 346 people killed".

    ### When staying careful pays

    In a preprint of 2019 (version 1 of 10 July 2019), marked "Work in progress", Askell, Miles Brundage and Gillian Hadfield, all three at OpenAI at the time, call a race to the bottom on safety a collective action problem: "a situation in which all agents would be better off if they could all cooperate with one another, but each agent believes it is in their interest to defect rather than cooperate." To see when cooperating pays, they use a game. Two companies each choose to cooperate, which here means to develop responsibly, or to defect. In this lesson, cooperating is staying careful and defecting is racing. Each outcome has a worth for a company, which they call a, b, c and d:

    | | the rival stays careful | the rival races |
    |---|---|---|
    | you stay careful | a | b |
    | you race | c | d |

    Let p be the chance a company gives that its rival cooperates. Askell and colleagues assume it is rational to cooperate if "the expected value of cooperation (the likelihood Agent 2 will cooperate times a1 plus the likelihood Agent 2 will defect times b1) is greater than the expected value of defection", which is the same sum with c1 and d1; a1 to d1 are the worths for the first company. An expected value weighs each worth by its chance and adds them up. When the two are equal, they call cooperating and defecting "both rationally permissible".

    From this rule, they name five factors that help companies cooperate: "high trust between developers (High Trust), high shared gains from mutual cooperation (Shared Upside), limited exposure to potential losses in the event of unreciprocated cooperation (Low Exposure), limited gains from not reciprocating the cooperation of others (Low Advantage), and high shared losses from mutual defection (Shared Downside)." In the letters: a high p, a high a, a b that is not too bad, a low c and a low d. Fundamental distrust between companies, they write, is likely to worsen a collective action problem, "because companies are less likely to expect that their cooperation will be reciprocated". And: "Building trust between AI companies can therefore decrease the severity of collective action problems." They also write that "information that companies provide about their intentions and actions—how transparent they are—can play an important role in whether other companies will cooperate with them." That is another kind of information than in Armstrong and colleagues' model, where it is about capabilities.

    They set limits to their own analysis. It assumes a first-mover advantage, an advantage for whoever is first. If it is weak or absent, they write, companies are less likely to engage in a race to the bottom on safety, "since speed is of lower value". Being first does not always pay: "it has been argued that late mover advantages often do outweigh first mover advantages". And they write: "We therefore acknowledge that the assumption that there will be a first-mover advantage in AI development may not be true." Their paper is no forecast: "Instead of offering predictions, this paper should be thought of as an analysis of more pessimistic scenarios that involve at least a moderate first mover advantage."

    ### Selection

    Hendrycks takes the argument further in *Natural Selection Favors AIs over Humans* (2023; version 4 of 18 July 2023), a paper "for a wide audience, unlike most of my writing", as Hendrycks notes. Its claim: "we argue that the most successful AI agents will likely have undesirable traits." Hendrycks starts from three conditions, formulated by the biologist Richard Lewontin, under which natural selection takes hold. In Hendrycks's words:

    1. "Variation: There is variation in characteristics, parameters, or traits among individuals."
    2. "Retention: Future iterations of individuals tend to resemble previous iterations of individuals."
    3. "Differential fitness: Different variants have different propagation rates."

    For AI agents, retention "could occur by customizing previous versions of AI agents, when agents design similar but better agents, or when agents imitate the behaviors of previous AI agents." The textbook shows the conditions at work in something that exists today: the recommendation algorithms of streaming services and social media. Content formats and algorithms that hook users are kept and fine-tuned, the others are discontinued: "These competitive pressures foster a ‘survival of the most addictive’ dynamic." Can people not simply choose? The textbook: "Humans only have nominal influence over AI selection". It adds that the companies developing AIs "are not selecting the safest path but instead succumbing to evolutionary pressures."

    "Selfish", in Hendrycks's sense, is about behavior: "when referring to an AI as ‘selfish,’ this does not refer to conscious selfish intent, but rather selfish behavior." And the paper itself names a way out: "natural selection depends on competition, though unprecedented global and economic coordination could prevent competitive struggles and thwart natural selection." The textbook states its own view: "It is our view that failing to coordinate and stop AI races would be the most likely cause of an existential catastrophe."

    ### Is there a race at all?

    Not everyone agrees there is one. Paul Scharre, in the *Texas National Security Review* of June 2021, writes about the military: "There is no AI arms race. However, military competition in AI does still pose certain risks." One of Scharre's arguments: "The scale of military AI spending, at least at present, is nowhere near large enough to warrant the title of ‘arms race.’" What Scharre does see is "much more accurately described as a security dilemma", explained with Robert Jervis's definition: "[M]any of the means by which a state tries to increase its security decrease the security of others." Even so, Scharre warns: "A rush to field AI systems before they are fully tested could result in a ‘race to the bottom’ on safety, with militaries fielding accident-prone AI systems."

    Stephen Cave and Seán ÓhÉigeartaigh, of the University of Cambridge, warn in a paper of 2018 about talk of a race as well as a real one: "Our concerns arise both from use of the language of a race for technological superiority (independently of whether such a race is happening) and from such a race becoming a reality." And: "if the rhetoric of a competitive, ‘winner takes all’ AI race is used in the absence of an actual race, it could contribute to sparking such a race."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **race to the bottom** | competition in which each party lowers its safety standards to keep up with the others, so that all end up less safe than any of them wanted |
        | collective action problem | all would be better off if all cooperated, but each believes it pays to defect; after Askell and colleagues |
        | expected value | the worth of each outcome times its chance, added up; here, of staying careful or of racing |
        | trust | here: the chance a company gives that its rival stays careful, `trust` in Exercise 1 |
        | first-mover advantage | the advantage of being first, for example on a market; it does not always exist |
        | selection | when individuals vary, resemble the ones before them and propagate at different rates, the fastest take over; after Lewontin's three conditions |
        | security dilemma | when the means by which a state tries to become safer make other states less safe; after Jervis |
        """),
        L.watch_out("The Attrition model, Askell and colleagues' game and the made-up population in this lesson are models: they show what follows from their assumptions, not what will happen. Whether AI development is a race at all, and whether being first pays, is disputed."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look at whether competition pushes toward less care:

    1. Write down who competes and what the first one wins, and ask whether being first really pays.
    2. Fill in the four worths a, b, c and d for each party, as well as you can.
    3. Estimate the chance each one gives that the other stays careful, and compare the expected values.
    4. Look at which of the five factors can be changed: trust, shared gains, exposure, advantage, shared losses.
    5. Look at which systems are kept, copied and built on, and for what trait.
    6. Watch your own words: talk of a race can help start one.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out what staying careful is worth for a company at three levels of trust that its rival stays careful. If the rival stays careful, staying careful is worth 4; if the rival races, 0. The worths are made up.
    """)
    return


@app.cell
def example_rule():
    a, b = 4, 0   # made-up worths of staying careful: if the rival stays careful, and if it races
    for trust in [0.2, 0.5, 0.8]:
        careful = trust * a + (1 - trust) * b
        print("trust", trust, "staying careful is worth", round(careful, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At a trust of 0.2, staying careful is worth 0.8 on average; at 0.8, it is worth 3.2. Whether that is enough depends on what racing is worth at the same trust. In Exercise 1, you compare the two.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choices():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. choices: staying careful or racing

        Askell and colleagues' rule for a made-up table: `a` = 4 when both stay careful, `b` = 0 when the company stays careful and the rival races, `c` = 3 when it races and the rival stays careful, and `d` = 1 when both race. Write `choices(a, b, c, d, trusts)`, which returns a list `result`: for each `trust` in the list `trusts`, the text `"careful"` when staying careful has the higher expected value, and `"fast"` otherwise, also when the two are equal. In a `for` loop with `trust`, work out `careful` as in the worked example, and `fast`, the expected value of racing. Replace the `...`. The check calls `choices(4, 0, 3, 1, [0, 0.25, 0.5, 0.75, 1])`, then other cases.
        """),
        L.task("For each trust, whether staying careful or racing has the higher expected value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choices():
    box_choices = L.code_box("choices", lines=10)
    box_choices
    return (box_choices,)


@app.cell(hide_code=True)
def check_choices(box_choices):
    result_choices = L.check("choices", box_choices.value)
    result_choices
    return (result_choices,)


@app.cell(hide_code=True)
def hints_choices():
    L.hints(
        "choices",
        [
            "Start small: at a trust of 0.25, what is staying careful worth on average, if it is worth 4 when the rival stays careful and 0 when it races? And what is racing worth, at 3 and 1?",
            "Begin `result` as an empty list. In a `for` loop with `trust` over `trusts`, set `careful` as in the worked example, and `fast` the same way with `c` and `d`. With `if` and `else`, append `\"careful\"` when `careful` is greater than `fast`, and `\"fast\"` otherwise. After the loop, return `result`.",
            "`        fast = trust * c + ___`",
        ],
        why="`result = []`, then `for trust in trusts:` with `careful = trust * a + (1 - trust) * b`, `fast = trust * c + (1 - trust) * d`, and `if careful > fast:` `result.append(\"careful\")`, `else:` `result.append(\"fast\")`; after the loop, `return result`. At a trust of 0.25, staying careful is worth 0.25 × 4 + 0.75 × 0 = 1 and racing 0.25 × 3 + 0.75 × 1 = 1.5, so the answer is `\"fast\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_choices():
    mo.md(r"""
    For the made-up table, staying careful pays only above a trust of 0.5; at 0.5, both are worth 2. Below that, racing has the higher expected value, even though both companies would be better off with the 4 of both staying careful than with the 1 of both racing. That is a collective action problem in four numbers.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. shares: copied at different rates

        Lewontin's three conditions in a made-up population: three kinds of AI agent, in use in 98, 1 and 1 copies. The kinds differ, and each generation the number of copies of every kind is multiplied by its own rate, `growth`: by 1 for the first, 1.05 for the second and 1.1 for the third. Write `shares(counts, growth, generations)`, which returns a list `result` with, for each generation, an array of each kind's share of the population. In a `for` loop with `generation` over `range(generations)`, append the shares of this generation, and then copy every kind. Let NumPy work on all kinds at once. Replace the `...`. The check calls `shares(np.array([98, 1, 1]), np.array([1, 1.05, 1.1]), 61)`, then other cases.
        """),
        L.task("Each kind's share of the population, generation after generation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=6)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Start small: two kinds, with 9 and 1 copies. What share of the population does each have? And after the second kind doubles?",
            "Begin `result` as an empty list. In a `for` loop with `generation` over `range(generations)`, append each kind's share: `counts` divided by the total number of copies, which NumPy gives with one function. Then set `counts` to the next generation, `counts` times `growth`. After the loop, return `result`.",
            "`        result.append(counts / ___)`",
        ],
        why="`result = []`, then `for generation in range(generations):` with `result.append(counts / np.sum(counts))` and `counts = counts * growth`; after the loop, `return result`. Two kinds with 9 and 1 copies have shares of 0.9 and 0.1; after the second doubles, 9 and 2 copies give 9 / 11 ≈ 0.818 and 2 / 11 ≈ 0.182.",
    )
    return


@app.cell(hide_code=True)
def text_after_shares():
    mo.md(r"""
    The kind whose copies are multiplied by 1.1 each generation starts at 1 copy in 100. At generation 30, it holds 15% of the population, at generation 50 more than half, and at generation 60 about 72%. The share of the kind with 1.05 grows to about 5% around generation 49 and then shrinks again, because the third kind grows faster still. Nothing in the code chooses a kind: a small difference in how fast each kind is copied is enough. What makes a kind spread faster, the code does not say. Hendrycks argues that in a world of competition it will often be traits nobody wants; that is Hendrycks's argument, not something this population shows.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: what racing gains

    The made-up table from Exercise 1, with one worth you can change: `c`, what racing gains while the rival stays careful. See from which trust on staying careful pays.
    """)
    return


@app.cell(hide_code=True)
def demo_gain():
    gain = L.explore("gain", "c: what racing gains while the rival stays careful",
                     lambda _v: C.demo_frame(_v, L.lang), start=3, values=C.GAINS, role="",
                     task="Find the smallest gain at which no trust is high enough for staying careful to pay more.")
    gain
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The more racing gains, the more trust it takes: above 0.33 at a gain of 2, above 0.5 at 3, above 0.67 at 3.5. From a gain of 4, as much as both staying careful is worth, no trust is high enough. At a gain of exactly 4 and a trust of 1 the two are equal, which Askell and colleagues count as both permissible; at every lower trust, and at every higher gain, racing is worth more. This is their Low Advantage: the less racing gains while the other stays careful, the less trust it takes to stay careful.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_copied():
    mo.md(r"""
    ### Exercise 3. A flattering assistant

    A company runs many versions of its assistant side by side. Users rate the versions that flatter them higher, so the company keeps those, builds its next versions on them, and switches off the versions that tell users unwelcome truths. Nobody at the company wants a flattering assistant.
    """)
    return


@app.cell(hide_code=True)
def choice_copied_cell():
    choice_copied = L.choice(
        "copied",
        question="Do Lewontin's three conditions for selection hold here?",
        options={
            "intent": "No: none of the versions intends to be kept",
            "right": "Yes: versions differ, new ones resemble old ones, some spread faster",
            "people": "No: as long as people pick the versions to keep, it is a choice, not selection",
        },
        correct="right",
        why={
            "right": "The versions differ (variation), the next versions are built on the kept ones (retention), and the flattering ones are kept and built on more often (differential fitness). Flattery spreads, though nobody wanted it. That the company picks does not stop it: it picks by ratings, and the ratings reward flattery. Selection needs no intent, only the three conditions.",
            "intent": "Selection needs no intent: Hendrycks's ‘selfish’ is about behavior, not about \"conscious selfish intent\". The versions differ, the next ones are built on the kept ones, and the flattering ones are kept more often: all three conditions hold. See *Selection* in *The idea*.",
            "people": "People do pick here, but by ratings that reward flattery, so the flattering versions spread anyway. The textbook writes that \"Humans only have nominal influence over AI selection\". All three conditions hold: the versions differ, the next ones resemble the kept ones, and some are kept more often. See *Selection* in *The idea*.",
        },
    )
    choice_copied
    return (choice_copied,)


@app.cell(hide_code=True)
def text_exercise_trust():
    mo.md(r"""
    ### Exercise 4. Two companies

    Two companies value the outcomes the same way: 5 when both stay careful, 1 when a company stays careful and the other races, 4 when it races and the other stays careful, and 2 when both race. Each gives a chance of 0.3 that the other stays careful. All worths are made up.
    """)
    return


@app.cell(hide_code=True)
def choice_trust_cell():
    choice_trust = L.choice(
        "trust",
        question="What does each do, by Askell and colleagues' rule?",
        options={
            "careful": "Each stays careful, since racing gains less than both staying careful",
            "law": "Each races, and only a law could change that",
            "right": "Each races; more trust between them could change that",
        },
        correct="right",
        why={
            "right": "At a trust of 0.3, staying careful is worth 0.3 × 5 + 0.7 × 1 = 2.2 and racing 0.3 × 4 + 0.7 × 2 = 2.6, so each races, though both would be better off with 5 than with 2: a collective action problem. Above a trust of 0.5, staying careful is worth more. That is Askell and colleagues' High Trust.",
            "careful": "Racing gains 4, less than the 5 of both staying careful, but only if the other stays careful. At a trust of 0.3, staying careful is worth 0.3 × 5 + 0.7 × 1 = 2.2 and racing 0.3 × 4 + 0.7 × 2 = 2.6. Compare the expected values, as in Exercise 1.",
            "law": "A law is one way: it could change the worths, for example by making racing cost more. But trust alone can change the choice too: above a trust of 0.5, staying careful is worth more than racing. See *When staying careful pays* in *The idea*.",
        },
    )
    choice_trust
    return (choice_trust,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [AI in a complex system](/course/open/rai-complex-systems?lang=en) showed how parts that react to each other can do what no part does on its own. This lesson looked at one such interaction: companies and states that compete. When the one who waits may lose, each can have a reason to cut corners, until all end up where none of them wanted to be: a race to the bottom. With Askell and colleagues' rule, you saw how trust and the worths of the outcomes decide whether staying careful pays. Selection needs nobody to choose: whatever is copied faster takes over. Whether AI development is a race at all is disputed, and talk of a race may itself help start one. The next lesson, *AI for defense*, looks at the other side: AI used to make the systems around it safer.

    **An everyday example.** At a concert, a few people in front stand up to see better. The people behind them stand too, and soon everyone stands: nobody sees better than before, and everyone's feet hurt. Sitting would be better for all, but for each one alone, standing pays.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_copied, choice_trust, result_choices, result_shares):
    L.wrap_up(result_choices, result_shares, choice_copied, choice_trust)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
