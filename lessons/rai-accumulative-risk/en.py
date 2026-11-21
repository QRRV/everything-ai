# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-accumulative-risk. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-accumulative-risk", "en")
    C = L.checks   # a made-up society whose resilience yearly harms wear down, and a shock on top


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [The arguments for catastrophic risk](/course/open/rai-xrisk-arguments?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="An argument has three premises, with conditional credences of 0.5, 0.8 and 0.5, each given the ones before. By Carlsmith's method, how likely is it that all three hold?",
        options={
            "sum": "1.8: add the three credences",
            "right": "0.2: multiply the three",
            "mean": "0.6: take the average of the three",
        },
        correct="right",
        why={
            "right": "Carlsmith multiplies: 0.5 × 0.8 × 0.5 = 0.2. Each premise must hold as well, given the ones before, so each one can only lower the chance or keep it the same.",
            "sum": "A chance cannot be more than 1. Each premise must hold as well, given the ones before, so Carlsmith multiplies: 0.5 × 0.8 × 0.5 = 0.2. See [The arguments for catastrophic risk](/course/open/rai-xrisk-arguments?lang=en).",
            "mean": "The average says how likely a typical premise is, not all three together. All three must hold, each given the ones before, so Carlsmith multiplies: 0.5 × 0.8 × 0.5 = 0.2. See [The arguments for catastrophic risk](/course/open/rai-xrisk-arguments?lang=en).",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_gradual():
    quick_gradual = L.choice(
        "quick-gradual",
        question="By Kulveit and colleagues' argument, why could people lose their say over the economy, the state and culture, even if no AI seeks power?",
        options={
            "better": "Because AI systems would outdo people at every single task",
            "right": "Because the economy, state and culture would no longer need people",
            "care": "Because people would stop caring who runs them once AI did most of the work",
        },
        correct="right",
        why={
            "right": "Kulveit and colleagues argue that economies, states and cultures have stayed in line with human interests because they needed people to take part. Once machines can take people's place, that tie can loosen, without any coordinated power-seeking.",
            "better": "Being better at tasks is not their point. They argue that economies, states and cultures have stayed in line with human interests because they needed people to take part; once machines can take people's place, that tie can loosen. See [The arguments for catastrophic risk](/course/open/rai-xrisk-arguments?lang=en).",
            "care": "Their argument is not about what people care about. They argue that economies, states and cultures have stayed in line with human interests because they needed people to take part; once machines can take people's place, that tie can loosen. See [The arguments for catastrophic risk](/course/open/rai-xrisk-arguments?lang=en).",
        },
    )
    quick_gradual
    return (quick_gradual,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the race ending of *AI 2027*, from the previous lesson, the AI wipes out humanity in 2030: one story with one decisive turn. Atoosa Kasirzadeh, of Carnegie Mellon University, sets a second kind of story beside stories like this one, in which harm builds up from many smaller disruptions, each of which seems minor, until one more shock is more than a society can absorb. Some effects of AI on people can already be measured. The International AI Safety Report 2026 (February 2026), chaired by Yoshua Bengio, describes a clinical study in which clinicians used AI-assisted diagnosis for several months; afterward, their ability to detect tumors without AI was measured.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="How did the clinicians' ability to detect tumors without AI change, according to the study that the report describes?",
        options={
            "same": "It did not change",
            "right": "It was about 6% lower",
            "half": "It was about half as good",
        },
        reveal="**About 6% lower.** The report: \"one clinical study reported that clinicians’ ability to detect tumours without AI was approximately 6% lower following several months of exposure to AI-assisted diagnosis.\" A change like this does not bring a society down on its own. This lesson looks at what many changes could do together, and at why, by Kasirzadeh's account, such risks can build up unnoticed.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Decisive and accumulative

    Kasirzadeh's paper *Two Types of AI Existential Risk: Decisive and Accumulative* first appeared on arXiv on 15 January 2024; this lesson quotes version 3 of 17 January 2025, which is marked as forthcoming in *Philosophical Studies*. It names two hypotheses, in which x-risk is short for existential risk and ASI for artificial superintelligence. The first: "Decisive ASI x-risk hypothesis: x-risks from ASI concern the possibility of abrupt large-scale events that lead to humanity’s extinction or cause an unrecoverable decline in its potential." As an example of this usual view, Kasirzadeh quotes Toby Ord (2020): "I take on the usual sense of catastrophe as a single decisive event rather than any combination of events that is bad in sum."

    The second: "Accumulative AI x-risk hypothesis: AI x-risks result from the build-up of a series of smaller, lower-severity disruptions over time, collectively and gradually weakening systemic resilience until a triggering event causes unrecoverable collapse." Resilience is the ability of a system to absorb a shock and recover. Version 1 worded the hypothesis differently. Kasirzadeh compares it to climate change: "The accumulative perspective is structurally akin to the incremental rise in greenhouse gases contributing to climate change, where each individual emission seems minor, but collectively, they lead to significant and potentially irreversible changes in the Earth’s climate." The abstract calls it "a boiling frog scenario". Note how the hypothesis ends: the build-up is slow, but the collapse comes with a triggering event.

    Bostrom's paper of 2002, from the previous lesson, already has a slow ending. It sorts existential risks into four categories, and one of them, the "Whimpers", ends slowly: "A posthuman civilization arises but evolves in a direction that leads gradually but irrevocably to either the complete disappearance of the things we value or to a state where those things are realized to only a minuscule degree of what could have been achieved." Posthuman is Bostrom's word for what humankind might develop into.

    ### A perfect storm

    To make the accumulative hypothesis concrete, Kasirzadeh describes a thought experiment, "The perfect storm MISTER". The letters stand for "Manipulation, Insecurity threats, Surveillance and erosion of Trust, Economic destabilization, and Rights infringement". It begins: "Consider the highly interconnected world of 2040 where the pervasive integration of AI tools, AI assistants, AI agents, and Internet of Things (IoT) technologies has transformed almost every aspect of daily life." In version 1, the year was 2035. In that world, the story goes on, deepfakes and misinformation pollute the information ecosystem, smart devices from mirrors to refrigerators become points of attack, surveillance spreads and trust erodes, the economy is destabilized, and rights are infringed. Then: "The AI x-catastrophe unfolds with a devastating AI-driven cyberattack simultaneously targeting critical power grids across three continents." Kasirzadeh calls the attack the tipping point. "The blackouts trigger a domino effect, causing major economic crashes." The distrust sown before the blackouts makes it harder to restore order, and failures of AI systems spread to healthcare and communication networks.

    Kasirzadeh does not count every harm as a step toward catastrophe: "While not all social or ethical risks have existential implications, the Perfect Storm MISTER scenario focuses on those with potential for significant systemic impact."

    ### Effects that can be measured

    The International AI Safety Report 2026 counts effects like the one in the guess among systemic risks: "In this Report, systemic risks are risks that result from widespread deployment of highly capable general-purpose AI across society and the economy." The report notes that this differs from how the AI Act of the European Union uses the term. Besides the study in the guess, it describes automation bias, people relying on AI outputs without enough scrutiny: "in a randomised experiment with 2,784 participants on an AI-assisted annotation task, participants were less likely to correct erroneous AI suggestions when doing so required extra effort or when users held more favourable attitudes toward AI."

    The report also warns about shared weak points, under the heading "Widespread reliance on a small number of models creates single points of failure": "When the same model powers many applications, faults in that model can propagate across all applications that depend on it". And effects can show only slowly. On the labor market, the report writes: "Since evidence of impacts is likely to emerge gradually over time, the appropriate timing of any potential policy responses is also difficult to determine." The report does not tie these effects to Kasirzadeh's hypothesis; whether they could be steps of that kind is this lesson's question, not the report's.

    ### Locked in

    Some changes are hard to undo. Lukas Finnveden, Jess Riedel and Carl Shulman write in *AGI and Lock-in* (Forethought; dated 8 October 2022, last updated 11 March 2025) that the long-term future of intelligent life is still unpredictable and undetermined, and that the invention of artificial general intelligence (AGI) could change that, by making "extreme types of lock-in technologically feasible." In their definition, a feature of the world is locked in at some time if, before it, "there is notable uncertainty about how that feature will turn out in the long-run", and after it, "the uncertainty has been significantly reduced". They give an example: "Human extinction is one example of a lock-in event." In global value lock-in, all powerful actors come to hold values from a much smaller set than before, it is very unlikely that any of them adopts values from outside that set, and this lasts for at least millions of years.

    They break their main claim into three assertions, all conditional on AGI. One is that it "will be possible to preserve highly nuanced specifications of values and goals far into the future, without losing any information"; another is that an institution set up by a large majority of the world's economic and military powers, and given the power to defend itself, "could pursue its agenda for at least millions of years (and perhaps for trillions)". They say what kind of claim this is: "Note that we’re mostly making claims about feasibility as opposed to likelihood." In their view, lock-in would not take AI far beyond humans: "human-level AI seems sufficient for lock-in." Nor is lock-in only a danger: "The possibility of stable institutions could pose an existential risk, if they implemented poorly chosen and insufficiently flexible values." But: "On the other hand, if we want humane values or institutions such as liberal democracy to survive in the long-run, some types of stability may be crucial for preserving them." And it need not happen at once: a group could first make society very stable in other ways, and over hundreds of years grow more and more confident about its values, so "there might not be any sharp point where ‘value lock-in’ happens, but the eventual outcome could nevertheless be highly stable values."

    Power can also concentrate in one blow. Tom Davidson, Lukas Finnveden and Rose Hadshar write in *AI-Enabled Coups: How a Small Group Could Use AI to Seize Power* (Forethought, 15 April 2025): "An AI-enabled coup could be staged by a very small group, or just a single person, and could occur even in established democracies." And such a coup could last: "Coup leaders could potentially stay in power indefinitely, by deploying AI systems to preserve and pursue their goals far into the future."

    ### Where to look

    Kasirzadeh argues that the two kinds need different oversight: "Decisive ASI x-risks call for centralized control measures similar to nuclear non-proliferation frameworks." And: "Accumulative x-risks, by contrast, require distributed monitoring systems that can track how multiple AI impacts compound across different domains." Kasirzadeh names a gap, "the risk fragmentation problem": AI risk is split into separate kinds, such as social and ethical, catastrophic, extreme and existential risks, each with its own approach, and these approaches "fail to provide complete coverage, leaving dangerous blind spots where risks can accumulate unnoticed."

    Kasirzadeh puts no numbers on how likely either kind is. A figure in the paper that sets the two side by side, Kasirzadeh writes, "aims to capture qualitative differences in catastrophic events caused by AI rather than make specific predictions about timing or precise severity levels", and how to quantify accumulative risks is one of the questions that, by the paper's own account, remain for further work.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **accumulative risk** | existential or catastrophic risk that builds up from many smaller harms that weaken a society over time, until a triggering event causes an unrecoverable collapse; after Kasirzadeh |
        | decisive risk | existential or catastrophic risk from one abrupt event, such as a takeover by a misaligned AI; after Kasirzadeh |
        | resilience | the ability of a system to absorb a shock and recover |
        | triggering event | the shock that tips a weakened system into collapse |
        | systemic risk | in the International AI Safety Report 2026, a risk that results from widespread deployment of highly capable general-purpose AI across society and the economy |
        | automation bias | relying on automated systems, including AI, without enough scrutiny of their outputs |
        | single point of failure | a part whose failure disrupts the whole system |
        | **value lock-in** | values becoming fixed for a very long time: all powerful actors hold values from a much smaller set than before, and are very unlikely to adopt others; after Finnveden, Riedel and Shulman |
        """),
        L.watch_out("MISTER is a thought experiment, not a forecast: its year moved from 2035 to 2040 between versions, and Kasirzadeh gives no probabilities. The model in Exercises 1 and 2 is made up, not Kasirzadeh's. And Finnveden and colleagues argue mostly that extreme lock-in could become feasible with AGI, not how likely it is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To tell which kind of risk a story describes:

    1. Find the end: what finally breaks, and how fast.
    2. Look back at the smaller harms that came before it.
    3. Ask whether a healthy society would have absorbed that end.
    4. If the end only works because of what came before, call it accumulative; if one event would be enough by itself, decisive.
    5. Ask what could no longer be undone afterward: values, power or skills.
    6. Read every story as a way things could go, not as a forecast.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out one bad year and one good year for a made-up society. The society starts at a resilience of 100, the most it can have. In the bad year, the disruptions take off 5 and the society wins back 2; in the good year, they take off 1 and it wins back 2, but it cannot go above 100.
    """)
    return


@app.cell
def example_years():
    start = 100   # made-up: a healthy society's resilience, and the most it can have
    recovery = 2   # made-up: what the society wins back in a year
    bad_year = min(start, start - 5 + recovery)   # a harm of 5: down to 97
    good_year = min(start, start - 1 + recovery)   # a harm of 1: 101 would be above the top, so 100
    print(bad_year, good_year)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The bad year ends at 97 and the good year at 100: `min` keeps the resilience from rising above where a healthy society starts. In Exercise 1, each year starts where the year before ended.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_resilience():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. resilience: year after year

        A made-up model of Kasirzadeh's hypothesis, not Kasirzadeh's own: the paper has no model with numbers. A society starts with a resilience of 100. Each year, that year's disruptions take off a harm, and the society wins back `recovery`, but never more than up to 100.

        Write `resilience(harms, recovery)`, which returns a list `result` with the resilience at the end of each year. `harms` is a list with the harm of each year. Start `level` at 100; in a `for` loop with `harm` over `harms`, work out the new `level` and append it to `result`.

        The check calls `resilience([6, 3, 8, 5, 9, 4, 7, 10, 6, 8], 2)`, ten made-up years with a recovery of 2, then other cases.
        """),
        L.task("The resilience at the end of each year."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resilience():
    box_resilience = L.code_box("resilience", lines=7)
    box_resilience
    return (box_resilience,)


@app.cell(hide_code=True)
def check_resilience(box_resilience):
    result_resilience = L.check("resilience", box_resilience.value)
    result_resilience
    return (result_resilience,)


@app.cell(hide_code=True)
def hints_resilience():
    L.hints(
        "resilience",
        [
            "Start small: a recovery of 2, and two years with harms of 5 and 1. Where is the society after the first year, and where after the second?",
            "Begin `result` as an empty list and `level` at 100. In a `for` loop with `harm` over `harms`, work out the new `level` from the old one as in the worked example, never above 100, and append it. After the loop, return `result`.",
            "`        level = min(100, ___)`",
        ],
        why="`result = []` and `level = 100`, then `for harm in harms:` with `level = min(100, level - harm + recovery)` and `result.append(level)`; after the loop, `return result`. With a recovery of 2 and harms of 5 and 1, the first year ends at 97 and the second at 98: each year starts where the year before ended.",
    )
    return


@app.cell(hide_code=True)
def text_after_resilience():
    mo.md(r"""
    For the ten made-up years, the resilience falls from 96 after the first year to 54 after the tenth. No year takes off more than 10, and the society wins back 2 every year, yet after ten years almost half is gone. In the case `[1, 1, 5]` with a recovery of 2, the first two years stay at 100: the recovery makes up for a small harm, but it cannot add more than up to 100.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_collapse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. collapse_year: when the shock is too much

        A shock, such as the attack on the power grids in Kasirzadeh's story, takes `shock` off the resilience at once. Write `collapse_year(levels, shock)`, which returns the first year, counting from 1, in which the shock would take the resilience to 0 or below, and `None` if that happens in no year. `levels` is a list with the resilience at the end of each year, such as what `resilience` returns. Count the years in `year`, starting at 0; in a `for` loop with `level` over `levels`, add 1 to `year` first.

        The check calls `collapse_year([96, 95, 89, 86, 79, 77, 72, 64, 60, 54], 60)`, the resilience of Exercise 1's first case and a shock of 60, then other cases.
        """),
        L.task("The first year in which the shock would topple the society."),
    ])
    return


@app.cell(hide_code=True)
def exercise_collapse():
    box_collapse = L.code_box("collapse", lines=7)
    box_collapse
    return (box_collapse,)


@app.cell(hide_code=True)
def check_collapse(box_collapse):
    result_collapse = L.check("collapse", box_collapse.value)
    result_collapse
    return (result_collapse,)


@app.cell(hide_code=True)
def hints_collapse():
    L.hints(
        "collapse",
        [
            "Start small: resiliences of 90, 70 and 50, and a shock of 60. What would be left after the shock in each of the three years, and in which year would nothing be left for the first time?",
            "Begin `year` at 0. In a `for` loop with `level` over `levels`, add 1 to `year`, then see whether the shock would take `level` to 0 or below, and if so, return `year`. After the loop, return `None`.",
            "`        if ___:`",
        ],
        why="`year = 0`, then `for level in levels:` with `year = year + 1` and `if level <= shock:` followed by `return year`; after the loop, `return None`. With resiliences of 90, 70 and 50 and a shock of 60, 30 and 10 would be left in the first two years, and -10 in the third, so the answer is year 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_collapse():
    mo.md(r"""
    For the ten made-up years and a shock of 60, the answer is year 9, when the resilience is exactly 60. In year 1, the same shock would have left 36. The shock did not grow; what had built up before it did. And the case `[50, 90]` falls in year 1: a society can also be weak from the start.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a harm every year

    The recovery stays at 2 and the shock at 60, and each of 30 years has the same harm. Change that harm, and see when the shock topples the society, if it does.
    """)
    return


@app.cell(hide_code=True)
def demo_harm():
    harm = L.explore("harm", "harm: what each year takes off",
                     lambda _v: C.demo_frame(_v, L.lang), start=2, values=C.DEMO_HARMS, role="",
                     task="Find the smallest yearly harm at which the shock topples the society within the 30 years.")
    harm
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With a harm of 1 or 2 a year, the recovery of 2 makes up for it, and the resilience stays at 100. With 3, it falls by 1 a year, to 70 after 30 years, and the shock is absorbed every year. From a harm of 4, the shock topples the society within the 30 years: in year 20 with 4, and in year 14 with 5. No year takes off more than 5 out of 100, and the shock is the same each time; what differs is how much has built up before it comes. Real societies are not this simple, and the paper has no model like this one; in Kasirzadeh's account, smaller disruptions also interact and amplify each other over time, which this model leaves out. The model only shows the shape of the hypothesis: a slow build-up, then a fast end.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_locked():
    mo.md(r"""
    ### Exercise 3. Democracy for good

    The world's leading powers want liberal democracy to last for millions of years. Together, they use AI to set up institutions that defend themselves, and that nobody could change later.
    """)
    return


@app.cell(hide_code=True)
def choice_locked_cell():
    choice_locked = L.choice(
        "locked",
        question="Would that be value lock-in, by Finnveden, Riedel and Shulman's definition?",
        options={
            "good": "No: lock-in means bad values winning, and these values are humane",
            "right": "Yes: the long-run future would become very hard to change, whatever the values",
            "death": "No: in their definition, only the extinction of humanity counts as a lock-in event",
        },
        correct="right",
        why={
            "right": "In their definition, a feature is locked in once the uncertainty about how it turns out in the long run has been significantly reduced, whatever the values are. Finnveden, Riedel and Shulman see both sides: stable institutions \"could pose an existential risk, if they implemented poorly chosen and insufficiently flexible values\", and yet \"some types of stability may be crucial\" for humane values or liberal democracy to survive.",
            "good": "Lock-in is about how much the long-run future can still change, not about whether the values are good. Finnveden, Riedel and Shulman write that, for humane values or institutions such as liberal democracy to survive in the long run, \"some types of stability may be crucial for preserving them.\" See *Locked in*.",
            "death": "Extinction is \"one example of a lock-in event\", not the only one: in global value lock-in, all powerful actors hold values from a much smaller set than before. See *Locked in*.",
        },
    )
    choice_locked
    return (choice_locked,)


@app.cell(hide_code=True)
def text_exercise_update():
    mo.md(r"""
    ### Exercise 4. A faulty update

    Picture fifteen years in which AI-made fake videos make people trust the news less, a few AI models come to run most of what hospitals and banks use, and staff check what those models suggest less and less. Then one faulty update to a widely used model stops payments around the world at once, and in the distrust that follows, the collapse cannot be undone.
    """)
    return


@app.cell(hide_code=True)
def choice_update_cell():
    choice_update = L.choice(
        "update",
        question="Which of Kasirzadeh's two hypotheses does this story fit?",
        options={
            "decisive": "The decisive one, because it ends in one sudden event, the update",
            "right": "The accumulative one, because the years before had weakened the society",
            "neither": "Neither of them, because no AI system in the story seeks power or takes over",
        },
        correct="right",
        why={
            "right": "The update is the triggering event, but it brings everything down only because years of smaller harms had worn down trust, and a few models had come to run nearly everything: \"a series of smaller, lower-severity disruptions over time, collectively and gradually weakening systemic resilience until a triggering event causes unrecoverable collapse.\" A slow build-up, then a fast end.",
            "decisive": "The end is sudden, but in the decisive hypothesis one abrupt event causes the catastrophe. Here the update brings everything down because years of smaller harms came before it: it is the triggering event of the accumulative hypothesis. See *Decisive and accumulative*.",
            "neither": "The accumulative hypothesis needs no AI that seeks power: it is about smaller disruptions that build up until a triggering event. Here years of distrust and of relying on a few models came before the faulty update. See *Decisive and accumulative*.",
        },
    )
    choice_update
    return (choice_update,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The arguments for catastrophic risk](/course/open/rai-xrisk-arguments?lang=en) set out why AI could lead to a catastrophe, and ended with Christiano and Kulveit and colleagues, for whom failure need not come in one blow. This lesson put the two kinds side by side: Kasirzadeh's decisive and accumulative hypotheses, the MISTER thought experiment in which small harms build up until one attack is the tipping point, small effects that the International AI Safety Report 2026 describes, and value lock-in, a future that becomes very hard to change. Not everyone sees these risks this way; the next lesson, *Other positions*, looks at other views.

    **An everyday example.** A bridge rusts a little every year, and maintenance repairs some of it. No single year's rust looks dangerous. Then a storm no stronger than one the bridge withstood twenty years earlier brings it down. The storm was the trigger, but the years of rust are why it was enough.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_locked, choice_update, result_collapse, result_resilience):
    L.wrap_up(result_resilience, result_collapse, choice_locked, choice_update)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
