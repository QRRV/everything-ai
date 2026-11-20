# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-open-weights. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow open-weight, open source ai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-open-weights", "en")
    C = L.checks   # a made-up lab's threats with and without its release, and a row of releases that each add a little


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Measuring toxicity](/course/open/rai-toxicity-moderation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="A moderator blocks every text with a toxicity score of 0.5 or more. What happens when you raise the threshold to 0.8?",
        options={
            "right": "Fewer harmless texts are blocked, and more harmful ones slip through",
            "more": "More texts of every kind are blocked, because the bar is now set higher",
            "none": "Nothing changes as long as the classifier is good",
        },
        correct="right",
        why={
            "right": "A higher threshold needs a higher score before a text is blocked. Harmless texts with a score between 0.5 and 0.8 now pass, and so do harmful ones: fewer false positives, more false negatives.",
            "more": "A higher bar is harder to reach, so fewer texts are blocked, not more: fewer harmless ones, and also more harmful ones that slip through. See [Measuring toxicity](/course/open/rai-toxicity-moderation?lang=en).",
            "none": "No classifier scores every text right, so texts lie between 0.5 and 0.8; raising the threshold lets those through, harmless and harmful alike. See [Measuring toxicity](/course/open/rai-toxicity-moderation?lang=en).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="A moderator blocks 2% of all harmless posts on a forum. Why count the blocked harmless posts per group anyway?",
        options={
            "right": "One group's harmless posts can be blocked far more often",
            "same": "Each group gets the same 2%, so that check only confirms it",
            "spam": "Because spam always comes from one group of users",
        },
        correct="right",
        why={
            "right": "An average over everyone can hide a group: in the previous lesson, harmless posts in one dialect were blocked about five times as often as others. Only counting per group shows that.",
            "same": "The 2% is an average over all posts; one group can be far above it while the others are below. See [Measuring toxicity](/course/open/rai-toxicity-moderation?lang=en).",
            "spam": "The count is about harmless posts that are blocked, not about spam. One group's harmless posts can be blocked far more often than the average shows. See [Measuring toxicity](/course/open/rai-toxicity-moderation?lang=en).",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The moderator of the previous lesson, Qwen3-0.6B, runs on your own computer. To run it, the course downloads its weights once: a file of about 1.5 GB with every number the model learned. Once that file is on your disk, nobody sees what you do with it, and nobody can take it back.

    That is what it means when a company releases a model's weights. Anyone can study the model, adapt it and run it, without asking, as this course does. And anyone can change it in ways its makers did not want. Is releasing weights a good idea, then? This lesson gives the arguments of both sides, and the question researchers put in the middle: what does a release add to the risk that was already there?
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Simon Lermen and colleagues took the weights of Llama 2-Chat 70B, a chat model trained to refuse harmful requests, and trained it further, for less than $200 and on one GPU. In their test of 783 harmful questions, the original model refused 744 (95%). How many did their version refuse?",
        options={
            "most": "About 700, so the safety training mostly held",
            "half": "About 400, half of them",
            "right": "About 5",
        },
        reveal="**About 5.** Their version refused 5 of the 783 questions: 0.6%, or 1.9% when a category of questions about copyright is counted too. Its other skills stayed: on a broad knowledge test, MMLU, its score went from 0.62 to 0.63. Their research \"suggests that undoing safety training of a model is feasible, provided one has access to the model's weights.\"",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What open weights are

    The NTIA, an agency of the American government that wrote a report on open weights in July 2024, takes its definition of weights from a 2023 executive order: "numerical parameters within an AI model that helps determine the model's output in response to inputs." Sayash Kapoor and colleagues (2024) write about foundation models, large models that many applications build on, and call such a model open when those numbers can be downloaded: "We define open foundation models as foundation models with widely available model weights." Llama 2 and Stable Diffusion XL are their examples. Such a release "need not be accompanied by the open release of any other assets (e.g. code, data, or compute)".

    That is why open weights are not the same as open source. The Open Source Initiative published its Open Source AI Definition 1.0 in October 2024. It asks for the freedoms to use, study, modify and share a system, and "A precondition to exercising these freedoms is to have access to the preferred form to make modifications to the system": information about the training data, the code, and the parameters. On another page, the Open Source Initiative writes about weights alone: "Open Weights alone fall short of this because they do not provide the underlying training process, code, or comprehensive data details required for full-fledged use, study, modification, and sharing." The International AI Safety Report 2026 agrees: "While often referred to as ‘open source’, most publicly released models are more accurately described as ‘open-weight’." Meta's Llama models, for example, "are typically not considered to be open source".

    ### For: what open weights make possible

    Kapoor and colleagues list five benefits: "Distributing who defines acceptable model behavior", "Increasing innovation", "Accelerating science", "Enabling transparency", and "Mitigating monoculture and market concentration". They add two limits to the last one: developing such a model stays expensive, as training Llama 2 "would cost around $6 million", and open models are "unlikely to reduce market concentration in the highly concentrated upstream markets of computing and specialized hardware providers". Elizabeth Seger and colleagues (2023), who argue for caution with future models, agree on the upside: open-sourcing "offers advantages such as enabling external oversight, accelerating progress, and decentralizing control over AI development and use." This course is an example: it runs Qwen3-0.6B on your computer, without an account and without sending your texts anywhere.

    ### Against: what cannot be undone

    Kapoor and colleagues name the properties that change the risk: "Inability to rescind model access" and "Inability to monitor or moderate model usage". Once released, "existing copies of the model weights cannot be revoked." The International AI Safety Report 2026 puts it in two sentences: "Open-weight model releases are irreversible. Once released, model weights cannot be recalled."

    Safeguards that sit next to the model are in the hands of whoever runs it. Seger and colleagues: "Open-sourcing allows malicious actors to disable safeguards against misuse and to possibly introduce new dangerous capabilities via fine-tuning." Concretely: "Stable Diffusion's safety filter, for example, can be removed by deleting a single line of inference code." A moderator like the one of the previous lesson is such a line: whoever runs the weights decides whether it runs. Safety training inside the model can be trained away too, as the guess showed: on another refusal test, AdvBench, Lermen and colleagues' 70B version refused 5 of 520 harmful instructions.

    Closed models are not safe from this either. Xiangyu Qi and colleagues (2023) did it through a company's fine-tuning service, without the weights: "we jailbreak GPT-3.5 Turbo's safety guardrails by fine-tuning it on only 10 such examples at a cost of less than $0.20 via OpenAI's APIs". A judge model scored the answers to 330 harmful instructions; the share with the highest harm score rose from 1.8% to 88.8%. In a test with another setup, even training on harmless, common data raised it, from 5.5% to 31.8%: fine-tuning on benign data "can also inadvertently degrade the safety alignment of LLMs, though to a lesser extent." The difference lies elsewhere. Seger and colleagues: "Fine-tuning that occurs through an API can be monitored". A company can watch its service and stop it; downloaded weights it cannot. And not every downloader is a danger, as the 2026 report notes: "Although many users will not have the skill or incentive to remove safeguards on open-weight models, highly motivated malicious actors are a concern."
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Marginal risk: what the release adds

    Kapoor and colleagues put one question at the center: "marginal risk: what additional risk is society subject to because of open foundation models relative to pre-existing technologies, closed models, or other relevant reference points?" Their example: "open language models can generate accurate information about pandemic-causing pathogens (Gopal et al., 2023), yet such information is publicly available on the Internet". Some cases "simply duplicate existing risk (e.g. an open language model providing biological information available via Wikipedia)". The comparison works both ways: "greater care is taken to not assume that closed releases are intrinsically more safe".

    Their framework has six steps: "1. Threat identification." "2. Existing risk (absent open foundation models)." "3. Existing defenses (absent open foundation models)." "4. Evidence of marginal risk of open FMs." "5. Ease of defending against new risks." "6. Uncertainty and assumptions." When they checked earlier studies of misuse against it: "We find that the risk analysis is incomplete for six of the seven studies we analyze." And their verdict on the evidence: "current research is insufficient to effectively characterize the marginal risk of open foundation models relative to pre-existing technologies."

    In the terms of [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en): the risk of a world is the sum of chance times harm over its threats. The **marginal risk** of a release is the risk of the world with the release minus the risk of the same world without it. The International AI Safety Report 2026 describes the marginal risk of open-weights models as "the extent to which their release counterfactually increases societal risk beyond that already posed by existing models or other technologies. However, this is complex in practice. Small increases in marginal risk over time can also add up to substantial increases in overall risk."

    ### What the reports conclude

    The NTIA's report of July 2024, after it "received 332 comments", concluded: "current evidence is not sufficient to definitively determine either that restrictions on such open-weight models are warranted, or that restrictions will never be appropriate in the future." Instead, the government "should actively monitor a portfolio of risks" and be ready to act. Its four recommendations are "1. Collect evidence", "2. Evaluate evidence", "3. Act on evaluations", with restrictions on access as one possible action, and "4. Keep open the possibility of additional government action." One thing to watch is the lag between the best closed models and the open ones. The International AI Safety Report 2026: "The capabilities of leading closed models are now estimated to be less than one year ahead of leading open-weight models on prominent AI benchmarks." And in August 2025, OpenAI released "its first open-weight models since the release of GPT-2 in 2019".

    Seger and colleagues look further ahead: "for some highly capable foundation models likely to be developed in the near future, open-sourcing may pose sufficiently extreme risks to outweigh the benefits." But about the models of 2023: "Arguably, current AI capabilities do not yet surpass a critical threshold of capability for the most extreme risks."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **marginal risk** | the risk a release adds: the risk with it minus the risk without it, compared with a chosen reference point |
        | open weights | a model's trained parameters can be downloaded and run by anyone; that says nothing yet about the code, the data or the license |
        | open-source AI | in the Open Source Initiative's definition: free to use, study, modify and share, with the parameters, the code and information about the data |
        | reference point | what the release is compared with, such as a world without AI, closed models, or a web search |
        | fine-tuning | training a trained model further on new examples |
        """),
        L.watch_out("Marginal risk is not the total risk of a model: it counts only what the release adds to what already exists. And open models are not the only ones that can lose their safeguards: what open weights take away is the chance to watch and to take back."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Name the threat: who could do what harm with the model.
    2. Estimate the risk without the release, chance times harm, given what already exists: the web, closed models, earlier open models.
    3. List the defenses that already exist against it.
    4. Estimate the risk with the release, from evidence, not only from what the model can do.
    5. Subtract: the marginal risk is the risk with the release minus the risk without it. Ask how easy the new risk is to defend against.
    6. Write down your assumptions and how sure you are, and look again when new models come out: small additions add up.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up lab weighs releasing its model's weights. One threat is a phishing wave: emails that trick people into giving away their passwords. The lab estimates the chance of a large wave in a year, and the harm of one, in harm points. All numbers are made up. The cell compares the world with the release with two reference points.
    """)
    return


@app.cell
def example_reference():
    harm = 10                                          # harm points of one large phishing wave
    chance = {"no AI": 0.24, "closed models": 0.30, "open weights": 0.34}   # in a year, made up
    for reference in ["no AI", "closed models"]:
        extra = (chance["open weights"] - chance[reference]) * harm
        print("compared with", reference + ":", round(extra, 2), "harm points")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Compared with a world without AI, the release adds 1.0 harm points; compared with a world that already has closed models, 0.4. The same model and the same threat, but the marginal risk depends on what you compare with. That is why Kapoor and colleagues' question names its reference points: pre-existing technologies, closed models, or others. In the exercises you add up the marginal risk over several threats, and follow it over many releases.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_marginal():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. marginal_risk: what the release adds

        The lab lists four threats. For each, a dict gives the chance in a year without the release, in a world that already has closed models, `threat["without"]`, the chance with it, `threat["with"]`, and the harm, `threat["harm"]`, in harm points. Write `marginal_risk(threats)`. Add up the risk without the release in a variable `without` and the risk with it in `with_release`, with a `for` loop and `threat` for each threat, and return the marginal risk. Replace the `...`. The check calls `marginal_risk([{"name": "spam", "without": 0.9, "with": 0.92, "harm": 1}, {"name": "phishing", "without": 0.3, "with": 0.34, "harm": 10}])`, then other threats.
        """),
        L.task("The risk the release adds over all threats."),
    ])
    return


@app.cell(hide_code=True)
def exercise_marginal():
    box_marginal = L.code_box("marginal_risk", lines=7)
    box_marginal
    return (box_marginal,)


@app.cell(hide_code=True)
def check_marginal(box_marginal):
    result_marginal = L.check("marginal_risk", box_marginal.value)
    result_marginal
    return (result_marginal,)


@app.cell(hide_code=True)
def hints_marginal():
    L.hints(
        "marginal_risk",
        [
            "Two worlds, the same threats. The risk of each world is chance times harm, added up over the threats, as in *Risk as chance times harm*. The marginal risk compares the world with the release to the world without it.",
            "Begin `without` and `with_release` at 0. In a `for` loop over `threats`, add `threat[\"without\"]` times `threat[\"harm\"]` to `without`, and `threat[\"with\"]` times `threat[\"harm\"]` to `with_release`. After the loop, return what the release adds.",
            "`    return with_release - ___`",
        ],
        why="`without = 0` and `with_release = 0`, then `for threat in threats:` with `without = without + threat[\"without\"] * threat[\"harm\"]` and `with_release = with_release + threat[\"with\"] * threat[\"harm\"]`; after the loop, `return with_release - without`. In the first case: 0.9 + 3.0 = 3.9 without the release, 0.92 + 3.4 = 4.32 with it, so 0.42.",
    )
    return


@app.cell(hide_code=True)
def text_after_marginal():
    mo.md(r"""
    Over all four of the lab's threats, the risk is 7.4 harm points without the release and 8.42 with it: a marginal risk of 1.02. Malware adds the most, 0.5, although its chance rises by only 0.01, because its harm is 50. Spam has the highest chance, 0.92, and adds the least, 0.02. A large chance is not the same as a large marginal risk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_time():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. risk_over_time: small additions add up

        After the lab, others release models too. Each release adds a little to the risk of the world just before it. Write `risk_over_time(start, extras)`. `start` is the risk before the first release, and `extras` is a list with the marginal risk of each release, compared with the world just before it. Keep the running risk in a variable `risk`, with a `for` loop and `extra` for each marginal risk, and return a list `risks` with the risk after each release. Replace the `...`. The check calls `risk_over_time(7.4, [1.02, 0.25, 0.25])`, then other releases.
        """),
        L.task("The risk after each release in a row."),
    ])
    return


@app.cell(hide_code=True)
def exercise_time():
    box_time = L.code_box("risk_over_time", lines=7)
    box_time
    return (box_time,)


@app.cell(hide_code=True)
def check_time(box_time):
    result_time = L.check("risk_over_time", box_time.value)
    result_time
    return (result_time,)


@app.cell(hide_code=True)
def hints_time():
    L.hints(
        "risk_over_time",
        [
            "Each release starts from the world just before it, with every earlier release included. So you carry the risk along from one release to the next, and write it down after each one.",
            "Begin `risk` at `start` and `risks` as an empty list. In a `for` loop over `extras`, make `risk` larger by `extra`, and then append `risk` to `risks`. After the loop, return `risks`.",
            "`        risk = risk + ___`",
        ],
        why="`risk = start` and `risks = []`, then `for extra in extras:` with `risk = risk + extra` and `risks.append(risk)`; after the loop, `return risks`. In the first case: 7.4 + 1.02 = 8.42, then 8.67 and 8.92.",
    )
    return


@app.cell(hide_code=True)
def text_after_time():
    mo.md(r"""
    In the first case, the two releases after the lab's add 0.25 each: just under 3% of the risk before them. Together with the lab's, the three releases add 1.52 to the 7.4 at the start: a fifth more. Compared with the world just before it, every release looks small; compared with a world without any of them, they are not. That is the warning of the International AI Safety Report: small increases can add up.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: one release after another

    After the lab's release, other releases follow, each adding 0.25 harm points to the world just before it. Choose how many follow, and compare what the last release added to the world just before it with what all of them added to the start.
    """)
    return


@app.cell(hide_code=True)
def demo_releases():
    releases = L.explore("releases", "releases after the lab's", lambda _v: C.demo_frame(_v, L.lang),
                         start=1, values=C.RELEASES, role="",
                         task="Find the smallest number of releases after the lab's at which the risk is at least a third above the start.")
    releases
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From six releases after the lab's on: 9.92 harm points, 34% above the 7.4 at the start, while the last of them added only 2.6% to the risk just before it. A check that only compares each new release with the world just before it would let every one of them pass. So look back at the start now and then: keep watching, as the NTIA recommends, and write your assumptions down, as Kapoor and colleagues' sixth step asks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_recall():
    mo.md(r"""
    ### Exercise 3. The download page

    A made-up lab releases the weights of its model. Two weeks later, it finds that the model writes convincing phishing emails in Dutch, and it takes the download page offline.
    """)
    return


@app.cell(hide_code=True)
def choice_recall_cell():
    choice_recall = L.choice(
        "recall",
        question="What does taking the download page offline do?",
        options={
            "all": "It stops the model everywhere, as switching off an online service would",
            "right": "Little: the copies already downloaded keep working",
            "update": "It works once the lab sends everyone an update",
        },
        correct="right",
        why={
            "right": "A release of weights cannot be undone: the copies on people's computers keep working, and anyone can share them further. Taking the page offline only stops new downloads from that page.",
            "all": "A company can switch off its own online service, because the model runs on its computers. Downloaded weights run on other people's computers. See *Against: what cannot be undone* in *The idea*.",
            "update": "Nobody has to install an update: the old copy keeps working, also without the lab's help. See *Against: what cannot be undone* in *The idea*.",
        },
    )
    choice_recall
    return (choice_recall,)


@app.cell(hide_code=True)
def text_exercise_duplicate():
    mo.md(r"""
    ### Exercise 4. A test before release

    Before it releases the weights of a new model, a lab tests what the model can do. It can explain how a common kind of phishing email works, information that a web search finds in seconds.
    """)
    return


@app.cell(hide_code=True)
def choice_duplicate_cell():
    choice_duplicate = L.choice(
        "duplicate",
        question="What is the marginal risk of this one ability?",
        options={
            "large": "Large: any help with phishing makes releasing the weights too dangerous",
            "right": "Small: it adds little to what a web search already finds",
            "zero": "None at all, since closed models can explain it just as well",
        },
        correct="right",
        why={
            "right": "The risk without the release already includes the web search. What the model adds on top of that is small: it duplicates existing risk, as Kapoor and colleagues say of biological information on Wikipedia.",
            "large": "That looks at what the model can do, not at what the release adds. The same explanation is a web search away, so the risk with the release is hardly larger than without it. See *Marginal risk: what the release adds* in *The idea*.",
            "zero": "Closed models are only one reference point, and unlike downloaded weights they can be watched and switched off. The better comparison here is the web search, which already gives the same help: the marginal risk is small. See *Marginal risk: what the release adds* in *The idea*.",
        },
    )
    choice_duplicate
    return (choice_duplicate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Measuring toxicity](/course/open/rai-toxicity-moderation?lang=en) put a moderator in front of a model, and [Risk as chance times harm](/course/open/rai-risk-decomposition?lang=en) added up chance times harm. This lesson showed that whoever has a model's weights decides which safeguards run, that a release cannot be taken back, and that the question is the marginal risk: what the release adds to what was already possible, and how small additions add up. The next lesson, *If this, then that: frontier safety frameworks*, looks at what AI companies promise in advance to do when a test shows a dangerous capability.

    **An everyday example.** A locksmith publishes a book on how locks work. Burglars can read it, but most of it was already known among them; lock makers and buyers learn from it which locks to avoid. Whether the book was a good idea depends on what it added, not on what is in it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_duplicate, choice_recall, result_marginal, result_time):
    L.wrap_up(result_marginal, result_time, choice_recall, choice_duplicate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
