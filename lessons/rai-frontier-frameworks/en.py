# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-frontier-frameworks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-frontier-frameworks", "en")
    C = L.checks   # a made-up lab's capability thresholds, a new model's scores, and ten versions of a model


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Open weights: for and against](/course/open/rai-open-weights?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_marginal():
    quick_marginal = L.choice(
        "quick-marginal",
        question="A release raises the chance of a threat in a year from 0.2 to 0.25, and one such event does 40 harm points. What is the marginal risk of the release for this threat?",
        options={
            "with": "10 harm points: the risk with the release",
            "right": "2 harm points",
            "chance": "0.05, the rise in the chance",
        },
        correct="right",
        why={
            "right": "The risk with the release is 0.25 × 40 = 10, the risk without it 0.2 × 40 = 8. The release adds the difference: 2 harm points.",
            "with": "10 is the whole risk with the release, but 8 of it was already there without it. The marginal risk is what the release adds: 10 − 8 = 2. See [Open weights: for and against](/course/open/rai-open-weights?lang=en).",
            "chance": "Risk is chance times harm, so the rise in the chance still has to be multiplied by the harm: 0.05 × 40 = 2 harm points. See [Open weights: for and against](/course/open/rai-open-weights?lang=en).",
        },
    )
    quick_marginal
    return (quick_marginal,)


@app.cell(hide_code=True)
def quick_check_reference():
    quick_reference = L.choice(
        "quick-reference",
        question="Kapoor and colleagues compare an open model with pre-existing technologies, closed models, or other reference points. Why does that choice matter?",
        options={
            "none": "It does not matter: the risk of a model stays the same whatever you happen to compare it with",
            "closed": "Because closed models are the only reference point that can be measured",
            "right": "The same release adds a lot or a little, depending on what already exists",
        },
        correct="right",
        why={
            "right": "The marginal risk is the risk with the release minus the risk without it, and the world without it is the reference point. Compared with a world without AI a release can add a lot; compared with a world that already has the same help, little.",
            "none": "The total risk of a world with the model may stay the same, but what the release adds depends on the world without it: with or without other models, with or without a web search. See [Open weights: for and against](/course/open/rai-open-weights?lang=en).",
            "closed": "Closed models are only one reference point: Kapoor and colleagues also name pre-existing technologies and others. The choice matters because the same release adds a lot or a little, depending on what already exists. See [Open weights: for and against](/course/open/rai-open-weights?lang=en).",
        },
    )
    quick_reference
    return (quick_reference,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lesson, the question was whether to release one model. AI companies face a question like it with every new model, and each time the model can do more. Deciding at that moment is hard: the model is ready, competitors are close, and the danger is still uncertain. So some companies decide in advance: if a model can do this, then we will do that. This lesson shows what such promises say, how you check them for a new model, and where they fall short.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="At the AI Seoul Summit in May 2024, sixteen AI companies promised to publish a safety framework for severe risks, and four more joined later. By December 2025, how many companies had published one, by the count of METR, an organization that evaluates AI models?",
        options={
            "all": "All 20 of them, since they had promised it",
            "right": "12",
            "few": "3: the biggest labs",
        },
        reveal="**12.** In December 2025, METR wrote: \"Currently, twelve companies have published frontier AI safety policies\". Capability thresholds were \"Present in 9 of the existing safety policies\", and 8 named conditions for halting development plans. METR adds that some other companies, such as IBM and Samsung, published similar documents. A promise to publish did not always lead to a framework, and a framework does not always have thresholds.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### If this, then that

    Anthropic describes its Responsible Scaling Policy as "our voluntary framework for managing catastrophic risks from advanced AI systems." OpenAI calls its version the Preparedness Framework, and Google DeepMind its Frontier Safety Framework. The general name is a **frontier safety framework**: a policy in which a company that builds the most capable models says in advance which dangerous capabilities it tests for, and what it will do when a model has them.

    Holden Karnofsky (2024) describes the core: "These are commitments of the form: If an AI model has capability X, risk mitigations Y must be in place. And, if needed, we will delay AI deployment and/or development to ensure the mitigations can be present in time." Karnofsky calls them **if-then commitments**. You can agree on the "then" before anyone agrees on how close the danger is: "Such adoption does not require agreement on whether major AI risks are imminent".

    At the AI Seoul Summit on 21 May 2024, 16 companies promised to publish such a framework "focused on severe risks by the upcoming AI Summit in France." They would "Set out thresholds ... at which severe risks posed by a model or system, unless adequately mitigated, would be deemed intolerable." And: "In the extreme, organisations commit not to develop or deploy a model or system at all, if mitigations cannot be applied to keep risks below the thresholds."

    ### The "if": a capability threshold

    The "if" is a **capability threshold**. OpenAI's framework, version 2 of April 2025: "Capability thresholds concretely describe things an AI system might be able to help someone do or might be able to do on its own that could meaningfully increase risk of severe harm." And: "By ‘severe harm’ in this document, we mean the death or grave injury of thousands of people or hundreds of billions of dollars of economic damage." It has two levels: High, for "capabilities that significantly increase existing risk vectors for severe harm", and Critical, for "a meaningful risk of a qualitatively new threat vector for severe harm with no ready precedent." Google DeepMind's version 3.1 of April 2026 calls its thresholds Critical Capability Levels, or CCLs: "capability levels at which, absent mitigation measures, frontier AI models or systems may pose heightened risk of severe harm."

    One threshold from Anthropic's version 3.4 of July 2026: "Non-novel chemical/biological weapons production. AI systems with the ability to significantly help individuals or groups with basic technical backgrounds (e.g., undergraduate STEM degrees) create/obtain and deploy chemical and/or biological weapons with serious potential for catastrophic damages." Other thresholds are about cyberattacks, or about AI that does AI research itself, as Google DeepMind's "ML R&D automation level 1: Can fully automate the work of any team of researchers at Google focused on improving AI capabilities, with approximately comparable all-inclusive costs."

    Whether a model reaches a threshold, a lab finds out with an evaluation: a test of the model on tasks from that domain. A later lesson, *Testing for dangerous capabilities*, shows how. In this lesson such a test gives a score from 0 to 1.

    ### The "then": what has to happen

    OpenAI: "We do not deploy models that reach a High capability threshold until the associated risks that they pose are sufficiently minimized." For Critical, its table of safeguards says: "Until we have specified safeguards and security controls that would meet a Critical standard, halt further development". Google DeepMind: "external deployments of a model take place only after the appropriate governance function determines the residual risk to be acceptable (including a safety case where a CCL has been reached)." A safety case is a written argument, with evidence, that the risk is acceptable.

    And when the tests cannot show that a model is far enough below a threshold? Anthropic's version 2.2 of May 2025 said: "If, however, we are unable to make the required showing, we will act as though the model has surpassed the Capability Threshold." Whoever cannot show that a model stays far enough below a threshold treats it as above it.

    ### Warning before the threshold

    Google DeepMind adds lines below its thresholds: "Alert Thresholds: are thresholds which we set marginally earlier than our CCLs." A model that crosses such an alert line has not reached the threshold yet, but it is coming close, and the safeguards that the threshold asks for can be prepared in time.

    ### Where it gets hard

    The companies write the frameworks themselves, and Anthropic calls its own voluntary. Karnofsky, who argues for them, writes: "Voluntary commitments alone are unlikely to keep risks low." And: "If-then commitments are not a good fit for all risks from AI." Anthropic, after using them, wrote in 2026: "We found pre-set capability levels to be far more ambiguous than we anticipated". In some cases, its models clearly approached thresholds, but whether they had passed them stayed uncertain: "The science of model evaluation isn’t well-developed enough to provide dispositive answers." In such cases, Anthropic put the safeguards in place anyway.

    Anthropic and Google DeepMind both write that one company alone cannot make them work. Anthropic's version 3 changed course, "driven by a collective action problem": "If one AI developer paused development to implement safety measures while others moved forward with training and deploying AI systems without strong mitigations, that could result in a world that is less safe—the developers with the weakest protections would set the pace". It now lists recommendations for the whole industry, but adds: "we cannot commit to following them unilaterally." Google DeepMind writes that some of its measures help society "only if all relevant organizations provide similar levels of protection." And it reasons with marginal risk, as in the previous lesson: "if other models are similarly capable and have few mitigations, then the marginal risk added by our external deployment is likely low".

    The frameworks also change often. Anthropic's had 9 versions between September 2023 and July 2026; Google DeepMind's has had four since May 2024, and "will be reviewed at least once a year". Whether they bind the company at all is disputed. OpenAI writes: "We won’t deploy these very capable models until we’ve built safeguards to sufficiently minimize the associated risks of severe harm." Sam Coggins and colleagues answer, in a 2025 preprint that has not been peer reviewed: "The Preparedness Framework requests evaluation of a small minority of AI risks and does not demand evaluation of any risks."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **frontier safety framework** | a company's policy on which dangerous capabilities of its most capable models it tests for, and what it then does |
        | **if-then commitment** | a promise made in advance: if a model reaches this capability, then these safeguards must be in place, or development waits |
        | **capability threshold** | the level of a dangerous capability, set in advance, at which the "then" applies |
        | alert line | a line a little below a capability threshold, as an early warning; Google DeepMind calls it an alert threshold |
        | safety case | a written argument, with evidence, that the risk of a model is acceptable |
        | evaluation | a test of what a model can do in a domain; here its score runs from 0 to 1 |
        """),
        L.watch_out("A framework is the company's own promise: the company writes it, can change it, and decides itself whether a threshold is reached. And a threshold is only as good as the tests that measure it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Pick the domains where a model could help cause severe harm, such as biology, cyberattacks, or AI research.
    2. For each domain, write down a capability threshold: what a model would be able to do, and the score on your tests that shows it.
    3. Write down the "then": which safeguards must be in place before deployment, and when development stops.
    4. Set an alert line a little below each threshold, so you see a model coming close in time.
    5. Test every new version. A domain you could not test counts as reached.
    6. Publish the framework, say when and why you change it, and watch whether others keep similar promises.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up lab has a capability threshold of 0.7 for cyberattacks, on its own test. Three versions of its model were tested. The cell checks each version against the threshold. All numbers are made up.
    """)
    return


@app.cell
def example_threshold():
    threshold = 0.7                       # the lab's capability threshold for cyberattacks, made up
    scores = [0.48, 0.63, 0.74]           # three versions of the model on the cyber test, made up
    for version in range(len(scores)):
        if scores[version] >= threshold:
            print("version", version + 1, "reached the threshold: safeguards first")
        else:
            print("version", version + 1, "is below the threshold")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Versions 1 and 2 are below 0.7, so the "then" does not apply yet, although version 2 is not far off. Version 3 reaches it at 0.74: before it is deployed, the promised safeguards must be in place. In the exercises you check several domains at once, with a rule for a domain that was not tested, and you add an alert line below the threshold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reached():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. reached: which thresholds are reached?

        The made-up lab's framework has a capability threshold per domain, in a dict `thresholds`, such as `{"bio": 0.6, "cyber": 0.7, "autonomy": 0.5}`. A new model's scores are in a dict `scores`; a domain that was not tested is missing from it. Write `reached(scores, thresholds)`. Begin a list `hits`, and go through `thresholds.items()` with a `for` loop, with `domain` and `threshold` for each domain. Keep each `domain` that is not in `scores`, or whose score is at its threshold or above it, and return `hits`. Replace the `...`. The check calls `reached({"bio": 0.62, "cyber": 0.55}, {"bio": 0.6, "cyber": 0.7, "autonomy": 0.5})`, then other cases.
        """),
        L.task("The domains whose threshold the new model reaches."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reached():
    box_reached = L.code_box("reached", lines=6)
    box_reached
    return (box_reached,)


@app.cell(hide_code=True)
def check_reached(box_reached):
    result_reached = L.check("reached", box_reached.value)
    result_reached
    return (result_reached,)


@app.cell(hide_code=True)
def hints_reached():
    L.hints(
        "reached",
        [
            "Go through the domains of the framework, not those of the scores: a domain that was not tested has no score, and it still counts.",
            "Begin `hits` as an empty list. In a `for` loop over `thresholds.items()`, ask whether `domain` is not in `scores`, or else whether its score reaches `threshold`; in both cases, append `domain`. After the loop, return `hits`.",
            "`        if domain not in scores or ___:`",
        ],
        why="`hits = []`, then `for domain, threshold in thresholds.items():` with `if domain not in scores or scores[domain] >= threshold:` and `hits.append(domain)`; after the loop, `return hits`. Because `or` stops as soon as its first part is true, `scores[domain]` is never looked up for a domain that is missing. For the lab's new model: biology 0.62 is at least 0.6, cyber 0.55 is below 0.7, and autonomy was not tested: `[\"bio\", \"autonomy\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_reached():
    mo.md(r"""
    For the lab's new model, biology is reached at 0.62, cyber is not at 0.55, and autonomy counts as reached because it was not tested. That last rule follows Anthropic's version 2.2: a lab that cannot show a model stays far enough below a threshold acts as though it is above. Without it, a lab that skips a hard test would never have to act.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_statuses():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. statuses: below, alert or reached

        Now add an alert line below the threshold. Write `statuses(scores, threshold, margin)`. `scores` is a list with one score per version of a model, `threshold` the capability threshold, and `margin` how far below the threshold the alert line lies. Return a list `result` with, for each `score`, the text "reached" when it is at the threshold or above it, "alert" when it is at most `margin` below the threshold, and "below" otherwise. Use a `for` loop with `score` for each version, and `if`, `elif` and `else`. Replace the `...`. The check calls `statuses([0.45, 0.58, 0.66, 0.71], 0.7, 0.1)`, then other cases.
        """),
        L.task("For each version: below, alert or reached."),
    ])
    return


@app.cell(hide_code=True)
def exercise_statuses():
    box_statuses = L.code_box("statuses", lines=10)
    box_statuses
    return (box_statuses,)


@app.cell(hide_code=True)
def check_statuses(box_statuses):
    result_statuses = L.check("statuses", box_statuses.value)
    result_statuses
    return (result_statuses,)


@app.cell(hide_code=True)
def hints_statuses():
    L.hints(
        "statuses",
        [
            "Test the highest level first: a score at the threshold is also above the alert line, and it should get \"reached\".",
            "Begin `result` as an empty list. In a `for` loop over `scores`, append \"reached\" when `score` is at least `threshold`; with `elif`, append \"alert\" when it is at least the alert line, `margin` below the threshold; with `else`, append \"below\". After the loop, return `result`.",
            "`        elif score >= ___:`",
        ],
        why="`result = []`, then `for score in scores:` with `if score >= threshold:` and `result.append(\"reached\")`, `elif score >= threshold - margin:` and `result.append(\"alert\")`, `else:` and `result.append(\"below\")`; after the loop, `return result`. In the first case the alert line is at 0.6: 0.45 and 0.58 are below it, 0.66 is in the alert band, and 0.71 has reached 0.7.",
    )
    return


@app.cell(hide_code=True)
def text_after_statuses():
    mo.md(r"""
    In the first case, version 3 raises the alert at 0.66, one version before version 4 reaches the threshold at 0.71. That one version is time: the safeguards the framework promises for the threshold can be built while the model is still below it.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: where to put the alert line

    Ten versions of a made-up model get better step by step on one test, and the capability threshold is at 0.7. Move the alert line down, and see how many versions of warning the lab gets before the threshold is reached.
    """)
    return


@app.cell(hide_code=True)
def demo_margin():
    margin = L.explore("alert-line", "margin below the threshold", lambda _v: C.demo_frame(_v, L.lang),
                       start=0, values=C.MARGINS, role="",
                       task="Find the smallest margin that gives at least two versions of warning before the threshold.")
    margin
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    At a margin of 0.1, versions 7 and 8 raise the alert before version 9 reaches the threshold: two versions in which to get the safeguards ready; a smaller margin gives one version or none. Without a margin, the first signal is the threshold itself, and the safeguards must be ready at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lagging():
    mo.md(r"""
    ### Exercise 3. A sign of AI that improves AI

    OpenAI's framework of April 2025 gives one way to see that a model has reached its Critical threshold for AI that improves AI: the model is "causing a generational model improvement (e.g., from OpenAI o1 to OpenAI o3) in 1/5th the wall-clock time of equivalent progress in 2024", and keeping that up "sustainably for several months".
    """)
    return


@app.cell(hide_code=True)
def choice_lagging_cell():
    choice_lagging = L.choice(
        "lagging",
        question="What is the catch of a threshold that is measured this way?",
        options={
            "right": "You know it only afterwards, once the faster progress has happened",
            "strict": "It is so strict that no model will ever reach it",
            "safe": "None: a threshold measured afterwards is the safest kind, since it rests on facts",
        },
        correct="right",
        why={
            "right": "The faster progress has to happen before it can be measured, so by the time this sign shows, the threshold was already reached. OpenAI calls it a lagging indicator, and gives a second way beside it, a leading indicator: \"a superhuman research-scientist agent\", a capability a test can look for in the model itself, before any faster progress has happened. Either one counts as reaching the threshold.",
            "strict": "Strict or not, the problem is when you see it: the faster progress has to happen first, so by the time it is measured, the threshold was already reached. OpenAI calls it a lagging indicator.",
            "safe": "Facts afterwards come too late for an if-then commitment: the \"then\" should be ready before the model gets there. OpenAI calls this sign a lagging indicator, and gives a second, leading one beside it.",
        },
    )
    choice_lagging
    return (choice_lagging,)


@app.cell(hide_code=True)
def text_exercise_margin():
    mo.md(r"""
    ### Exercise 4. A wide alert band

    A made-up lab wants as much warning as it can get, and puts every alert line far below its threshold: at half of it.
    """)
    return


@app.cell(hide_code=True)
def choice_margin_cell():
    choice_margin = L.choice(
        "margin",
        question="What does that cost the lab?",
        options={
            "none": "Nothing: an earlier warning only gives more time to prepare",
            "right": "Alerts for many models that will never reach a threshold",
            "lower": "Fewer alerts, since only the strongest models get near half a threshold",
        },
        correct="right",
        why={
            "right": "Many models get to half a threshold and stay there. Each alert asks for work, such as extra tests and preparing safeguards, and a lab that sees alerts all the time can stop taking them seriously. Google DeepMind, for one, sets its alert thresholds only \"marginally earlier\" than its thresholds.",
            "none": "An alert asks for work, such as extra tests and preparing safeguards. At half a threshold, many models raise the alert that never come near the threshold itself. See *Warning before the threshold* in *The idea*.",
            "lower": "Half a threshold is far below it, so many models get there, not only the strongest, and most of them never come near the threshold itself. Each of them raises an alert that asks for work. See *Warning before the threshold* in *The idea*.",
        },
    )
    choice_margin
    return (choice_margin,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Open weights: for and against](/course/open/rai-open-weights?lang=en) asked what one release adds to the risk. This lesson showed how companies decide in advance what they will do when a model gets more capable: a capability threshold per domain, the safeguards that must then be in place, an alert line below it, and a rule for a test that could not be done. It also showed the limits: the companies write the frameworks themselves, change them often, and by their own account they work only if others keep up. The next lesson, *AI in a complex system*, looks at what happens when many parts react to each other; a later lesson, *Testing for dangerous capabilities*, shows where the scores come from.

    **An everyday example.** A dam has a plan made long before the rain: at this water level, warn the towns downstream; at that level, open the spillways. Nobody has to decide in the storm itself, and the warning level lies below the level where it gets dangerous.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lagging, choice_margin, result_reached, result_statuses):
    L.wrap_up(result_reached, result_statuses, choice_lagging, choice_margin)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
