# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-expert-forecasts. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-expert-forecasts", "en")
    C = L.checks   # twenty made-up answers, Cotra's weights with made-up chances, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Other positions](/course/open/rai-xrisk-counterpositions?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gap():
    quick_gap = L.choice(
        "quick-gap",
        question="In the previous lesson's made-up model of differential technology development, a dangerous technology arrives in year 6, and the protection against it in year 4. For how many years is the danger unguarded?",
        options={
            "minus": "−2: the protection comes two years early, so minus two",
            "right": "0: the protection is there before the danger",
            "gap": "2: the years between the two arrivals",
        },
        correct="right",
        why={
            "right": "The unguarded years are the years the danger is out before its protection arrives, and never fewer than 0: `max(0, 4 - 6)` is 0. A protection that comes early leaves no unguarded years, not a negative number of them.",
            "minus": "A number of unguarded years cannot be negative: a protection that comes early leaves none, so `max(0, 4 - 6)` is 0. See [Other positions](/course/open/rai-xrisk-counterpositions?lang=en).",
            "gap": "The protection arrives in year 4, before the danger in year 6, so the danger is never unguarded. Those 2 years would count only if the protection came after the danger. See [Other positions](/course/open/rai-xrisk-counterpositions?lang=en).",
        },
    )
    quick_gap
    return (quick_gap,)


@app.cell(hide_code=True)
def quick_check_median():
    quick_median = L.choice(
        "quick-median",
        question="Five students travel to school in 10, 12, 13, 15 and 60 minutes. Which is larger, the mean or the median of their trips?",
        options={
            "right": "The mean: the one long trip pulls it up",
            "median": "The median: it is the middle trip, 13 minutes",
            "same": "Neither: both describe the middle of the same five trips",
        },
        correct="right",
        why={
            "right": "The mean is 110 / 5 = 22 minutes, and the median 13, the middle trip once the trips are sorted. One extreme value pulls the mean, but hardly moves the median.",
            "median": "The median is indeed 13 minutes, the middle trip. But the mean is 110 / 5 = 22 minutes: the trip of 60 minutes pulls it up. See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "same": "Both describe a middle, but not the same one: the mean is 110 / 5 = 22 minutes, the median 13. One long trip pulls the mean, but hardly moves the median. See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
        },
    )
    quick_median
    return (quick_median,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Headlines about AI risk often carry one number, such as a 10% chance that AI wipes out humanity. Behind such a number there is usually a survey of AI researchers, and a survey gives many answers, not one. In a survey by AI Impacts in December 2024, 1,489 researchers each answered a question about the chance that AI leads to human extinction or a similarly permanent and severe outcome. The median of their answers was 10%.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="What was the mean of those 1,489 answers?",
        options={
            "below": "About 5%: below the median",
            "same": "About 10%: the same as the median",
            "right": "About 18%: above the median",
        },
        reveal="**About 18%: above the median.** The report writes: \"The group’s average probability is at least around 18%.\" The \"at least\" is there because two of the three questions behind it are narrower than the first, so the report treats the three together as an approximate lower bound. 12% of the participants gave a zero chance, and a third gave at least 20%: the high answers pull the mean up, as the one long trip did in the quick check. This lesson looks at numbers like these: where they come from, how far they spread, and what they are worth.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Expert surveys

    AI Impacts has run the Expert Survey on Progress in AI, ESPAI for short, "periodically since 2016 with minimal modifications". Each time, it asks researchers who published at top AI venues the same kinds of questions: when machines will be able to do what, and how likely very good or very bad outcomes are. In the preprint of their paper on the survey of 2023 (first version January 2024), Katja Grace and colleagues write: "In the largest survey of its kind, we surveyed 2,778 researchers who had published in top-tier artificial intelligence (AI) venues". That was a response rate of 15%, and not everyone saw every question: "Because participants received randomized subsets of the questions, the number of responses is far less than 2,778 for most individual questions." The three questions about extinction got 1,321, 661 and 655 answers.

    The next survey ran in December 2024, among people who had published in 2023 at six venues: NeurIPS, ICML, ICLR, AAAI, JMLR and IJCAI. Of 19,874 addresses, 2,052 responded, 10%; 1,580 of them stayed in the main analysis. The report, by Grace and colleagues, came out in September 2026. The survey asks about extinction in three ways. The first question reads: "What probability do you put on future AI advances causing human extinction or similarly permanent and severe disempowerment of the human species?" The second asks the same about "human inability to control future advanced AI systems"; the third adds "within the next 100 years". Each participant got one of the three, the first twice as often as each of the others. In percent:

    | Question | Answers | Mean | Median |
    |---|---|---|---|
    | AI advances | 744 | 18.3 | 10 |
    | inability to control | 392 | 18.5 | 9 |
    | within the next 100 years | 353 | 17.5 | 5 |
    | the three together | 1,489 | 18.2 | 10 |

    The report also gives the spread: "The middle 50% of estimates ranged from 1% to 25%." In the survey of 2023, the median of the three together was 5%. Of 2024, the report writes: "For the first time in these surveys, the median AI researcher says human extinction or permanent and severe disempowerment from future AI has a 10% chance, given no further conditions."

    The same survey asks when. It describes high-level machine intelligence, HLMI for short, as occurring "when unaided machines can accomplish every task better and more cheaply than human workers", and adds: "Think feasibility, not adoption." How the question is put matters: half of the participants got a year and gave a probability, the other half got a probability and gave a year, and the report notes: "We have observed consistently different results from these framings across all editions of this survey, with the fixed-years framing producing generally later predictions." In the report's aggregate forecast, HLMI "reaches a 10% chance in 2027 and a 50% chance in 2042 (18 years from the survey)". And: "Over the course of eight years, the forecast horizon shrank from 45 years out to 18, getting about 3.4 years closer for each year that passed."

    In the journal version of their paper on the survey of 2023 (*Journal of Artificial Intelligence Research*, October 2025), the authors say what such numbers are worth: "However, they are experts in AI research, not AI forecasting, and might thus lack generic forecasting skills, or expertise in non-technical factors that influence the trajectory of AI. While AI experts’ predictions should not be seen as a reliable guide to objective truth, they can provide one important piece of the puzzle."

    ### Superforecasters

    In the *Existential Risk Persuasion Tournament*, XPT for short, the Forecasting Research Institute had experts and superforecasters forecast the same questions, from June to October 2022. Its working paper (version of August 2023) describes the main group as "89 superforecasters who scored highly in previous tournaments and 80 experts in different areas of existential risk"; 32 of the experts were AI experts. It defines extinction as "reduction of the global population to less than 5,000", and a catastrophe as 10% or more of humans dying within 5 years. For AI by 2100, the median superforecaster put a catastrophe at 2.13% and extinction at 0.38%; the median AI expert put them at 12% and 3%. For extinction, the mean of the AI experts was 13.61%, more than four times their median. Of its three questions about extinction (from AI, from nuclear war, and in total), the paper writes: "Across these three questions and five aggregation methods, the mean was always the highest, often two or more times as large as the median."

    Some questions of the tournament have since been settled. On 2 September 2025, the institute wrote: "Respondents underestimated AI progress, especially superforecasters". AI systems reached gold-level performance at the International Mathematical Olympiad in July 2025, "an outcome to which superforecasters gave only 2.3% probability and domain experts 8.6%."

    ### p(doom)

    Wikipedia (page last edited on 15 November 2026): "In the AI safety field, P(doom) is the probability of existentially catastrophic outcomes (so-called ‘doomsday scenarios’) as a result of artificial intelligence." The term was "presumably first used by programmer Tim Tyler in a 2009 comment on LessWrong"; the "presumably" is Wikipedia's. Isaac King warned in a LessWrong post of 1 January 2024 that the term "can refer to many different things", such as the chance that AI kills us within 5 to 10 years, or within 80 to 200 years.

    Arvind Narayanan and Sayash Kapoor, in a post of 26 July 2024: "Our central claim is that AI x-risk forecasts are far too unreliable to be useful for policy, and in fact highly misleading." One reason: "For existential risk from AI, there is no reference class, as it is an event like no other. To be clear, this is a matter of degree, not kind." One of their sections is titled "Subjective probabilities are feelings dressed up as numbers". And: "We have no objection to AI x-risk forecasting as an academic activity, and forecasts may be helpful to companies and other private decision makers."

    Numbers change on the way. In the previous lesson's essay of November 2023, Buterin writes that, according to a survey from 2022, "on average, researchers think that there is a 5-10% chance that AI will literally kill us all". Grace and colleagues' paper on the survey of 2023 also gives the results of 2022 in a table: two of the questions about extinction had medians of 5% and 10%, and means of 15.7% and 20.5%; so Buterin's "on average" matches the medians, not the means. That reading is this lesson's. Melanie Mitchell, in a post of 23 April 2023, looked into a widely shared claim that half of AI researchers give AI at least a 10% chance of killing us all: the question behind it "received only 162 replies".

    ### Biological anchors

    Ajeya Cotra's report *Forecasting TAI with biological anchors*, dated July 2020, asks when a transformative model could be trained. Cotra defines transformative AI, TAI for short, by its effect: "‘software’ (i.e. a computer program or collection of computer programs) that has at least as profound an impact on the world’s trajectory as the Industrial Revolution did." The method compares the compute that training would take with biology. The lifetime anchor is the computation of a human brain in its first billion seconds, up to about age 32: "(1e15 FLOP/s) * (1e9 seconds) = 1e24 FLOP". From each anchor, Cotra adjusts to a hypothesis about training: the Lifetime Anchor hypothesis estimates "a median of ~1e29 FLOP". The evolution anchor is the computation done over the course of evolution: "~1e41 FLOP".

    Cotra does not pick one hypothesis, but weighs them all into one forecast. For three hypotheses about neural networks: "I assigned 20% to Short Horizon, 30% to Medium Horizon, and 15% to Long Horizon." And for the rest: "I assign 5% probability to the Lifetime Anchor hypothesis, 10% probability to the Genome Anchor hypothesis, 10% probability to the Evolution Anchor hypothesis, and 10% probability to the possibility that the amount of computation that would be required to train a transformative model with 2020 architectures and algorithms is higher (perhaps astronomically higher) than any of the hypotheses predict."

    The result: "My best guess assumptions result in a median year of ~2052 (32 years from the time of writing) for when the amount of computation required to train a transformative model may become affordable. My aggressive assumptions place this median at ~2036, and my conservative assumptions place it at ~2100." And: "I am tentatively adopting ~2050 as my median forecast for TAI." The report says of itself: "This is a work in progress and does not represent Open Philanthropy’s institutional view." Cotra shared it on LessWrong in September 2020 as a "draft report on timelines to transformative AI". Two years later, in August 2022, Cotra moved the median to "~2040".

    ### Where they disagree

    On whether to use such numbers at all: for the authors of the paper on the survey of 2023, they are "one important piece of the puzzle"; for Narayanan and Kapoor, they are "far too unreliable to be useful for policy". On whom to ask: for AI extinction by 2100, the median AI expert in the XPT gave 3%, the median superforecaster 0.38%, and the paper calls the gaps between experts and superforecasters "robust across aggregation methods". And on how to ask: in every edition of the survey, asking for the chance by a given year gave later forecasts than asking for the year of a given chance. The three questions about extinction, by contrast, "received similar answers", in the report's words; their means lie between 17.5% and 18.5%.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **expert survey** | a survey that asks many researchers in a field the same questions, such as how likely an outcome is or when a capability arrives; the answers are stated beliefs, not measurements |
        | **p(doom)** | someone's probability that AI leads to an existential catastrophe; what exactly it is a probability of, and by when, differs from person to person |
        | **biological anchors** | Cotra's method (2020) for forecasting transformative AI: estimate the compute that training would take by comparing with biology, such as a human brain over a lifetime or all of evolution, and ask when that compute becomes affordable |
        | superforecaster | in the XPT, someone who scored highly in earlier forecasting tournaments |
        | HLMI | high-level machine intelligence, in the survey: when unaided machines can do every task better and more cheaply than human workers |
        | TAI | transformative AI, in Cotra's report: software with at least as profound an impact on the world as the Industrial Revolution |
        """),
        L.watch_out("An answer in a survey is a stated probability, not a measured risk. Say who was asked, which question, and which number of all the answers: a median, a mean or a share. The answers in Exercise 1 and the chances in Exercise 2 are made up; only the weights in Exercise 2 are Cotra's."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a number about what experts expect:

    1. Find who was asked, how they were chosen, and how many answered.
    2. Read the exact question: which outcome, and by when.
    3. See which single number stands for all the answers, and what kind of number it is.
    4. Look at how far the answers spread, and how many gave zero.
    5. If several answers or hypotheses are combined into one, find out how each is weighted.
    6. Read what the authors themselves say the number is worth.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out the mean and the median of five made-up answers, in percent, to a question like the survey's.
    """)
    return


@app.cell
def example_answers():
    import numpy as np

    answers = np.array([1, 2, 5, 10, 90])   # made-up: five answers, in percent
    print(answers.mean(), np.median(answers))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The mean is 21.6 and the median 5. Four of the five answers are 10 or less; the one answer of 90 pulls the mean up to more than four times the median. The median stays 5 if that answer becomes 50 or 100: it is still the middle answer that counts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_describe():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. describe: four numbers for many answers

        Twenty made-up answers, in percent, to a question like the survey's. Write `describe(answers)`, which returns a list of four numbers: `mean`, the mean of the answers; `median`, their median; `low`, their 0.25 quantile; and `high`, their 0.75 quantile. Between `low` and `high` lies the middle half of the answers, like the survey's "middle 50%". Make an array `a` of `answers` first, and return `[mean, median, low, high]`.

        The check calls `describe([10, 0, 5, 50, 1, 20, 10, 2, 100, 5, 1, 10, 30, 0, 15, 5, 80, 1, 20, 10])`, the twenty made-up answers, then other cases.
        """),
        L.task("The mean, the median and the middle half of the answers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_describe():
    box_describe = L.code_box("describe", lines=10)
    box_describe
    return (box_describe,)


@app.cell(hide_code=True)
def check_describe(box_describe):
    result_describe = L.check("describe", box_describe.value)
    result_describe
    return (result_describe,)


@app.cell(hide_code=True)
def hints_describe():
    L.hints(
        "describe",
        [
            "Start small: the five answers 1, 2, 5, 10 and 90 of the worked example. What are their mean and their median, and between which two answers lies the middle half of them?",
            "Make an array `a` of `answers`. Work out `mean` with `.mean()` and `median` with `np.median`; `low` and `high` come from `np.quantile`, with the fraction of the answers that lies below each. Return the four in a list.",
            "`    low = np.quantile(a, ___)`",
        ],
        why="`a = np.array(answers)`, then `mean = a.mean()`, `median = np.median(a)`, `low = np.quantile(a, 0.25)` and `high = np.quantile(a, 0.75)`, and `return [mean, median, low, high]`. For the answers 1, 2, 5, 10 and 90, the mean is 21.6, the median 5, and the middle half lies between 2 and 10.",
    )
    return


@app.cell(hide_code=True)
def text_after_describe():
    mo.md(r"""
    For the twenty made-up answers, the mean is 18.75 and the median 10, and the middle half runs from 1.75 to 20. Thirteen of the twenty answers are 10 or less, yet the mean is almost twice the median: the answers of 50, 80 and 100 pull it up. The made-up answers look like the survey's on purpose: in December 2024, the mean was 18.2, the median 10, and the middle half ran from 1 to 25.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mixture():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. mixture: one chance from several hypotheses

        A made-up model of the step in which Cotra weighs the hypotheses, not Cotra's own model. The seven weights are Cotra's, in the order of the quotes above: short, medium and long horizon, lifetime, genome, evolution, and more than any of them. The chances are made up: for each hypothesis, the chance that the compute for a transformative model is affordable by some made-up year. The overall chance is each chance times its weight, added up.

        Write `mixture(weights, chances)`, which returns that overall chance. `weights` and `chances` are lists of the same length. Make arrays `w` of `weights` and `c` of `chances`, and return the sum of each weight times its chance.

        The check calls `mixture([0.2, 0.3, 0.15, 0.05, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.9, 0.4, 0.0, 0.0])`, then other cases.
        """),
        L.task("The overall chance, from the weights and the chances."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mixture():
    box_mixture = L.code_box("mixture", lines=7)
    box_mixture
    return (box_mixture,)


@app.cell(hide_code=True)
def check_mixture(box_mixture):
    result_mixture = L.check("mixture", box_mixture.value)
    result_mixture
    return (result_mixture,)


@app.cell(hide_code=True)
def hints_mixture():
    L.hints(
        "mixture",
        [
            "Start small: two hypotheses with the weights 0.75 and 0.25, and the chances 0.4 and 0.8. How much does each hypothesis contribute, and what is the overall chance?",
            "Make arrays `w` of `weights` and `c` of `chances`. Multiplying the two arrays multiplies each weight by its own chance; add up those products with NumPy and return the total.",
            "`    return np.sum(___)`",
        ],
        why="`w = np.array(weights)` and `c = np.array(chances)`, then `return np.sum(w * c)`. With the weights 0.75 and 0.25 and the chances 0.4 and 0.8, the first hypothesis adds 0.3 and the second 0.2: 0.5 overall.",
    )
    return


@app.cell(hide_code=True)
def text_after_mixture():
    mo.md(r"""
    For the made-up chances, the result is 0.29, or 29%. The two possibilities with no chance at all still count: their 20% of the weight pulls the result down. Cotra's own numbers for 2036 come from the whole model: "there is a ~23% probability that the computation required to train a transformative model would be available by 2036, and I would adjust that downward to ~18% for my all-things-considered probability of TAI by 2036". And: "I think a very broad range, from ~2% to ~45%, could potentially be defensible."
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a few high answers

    The twenty made-up answers of Exercise 1 stay the same. Add answers of 100, one at a time, and see what the mean and the median do.
    """)
    return


@app.cell(hide_code=True)
def demo_extra():
    extra = L.explore("extra", "extra: how many answers of 100 are added",
                      lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.DEMO_EXTRA, role="",
                      task="Find the smallest number of extra answers of 100 at which the mean is more than three times the median.")
    extra
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Without extra answers, the mean is 18.75; with 1 extra answer of 100, 22.62; with 2, 26.14; with 3, 29.35; with 4, 32.29; and with 5, 35. The median stays 10 all the way. From 4 extra answers on, the mean is more than three times the median, while 13 of the 24 answers are still 10 or less. The XPT paper explains its means the same way: "forecasts on these questions tended to be in the 0-5% range, with a long tail of higher forecasts pulling up the mean."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Exercise 3. A headline

    A news site writes: AI researchers put the chance that AI wipes out humanity at 18%, according to the survey of December 2024.
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="What should the news site have added?",
        options={
            "nothing": "Nothing, since 1,489 answers make 18% a measured risk",
            "right": "That 18% is the mean, and the median answer was 10%",
            "few": "That only a few dozen researchers took part in the survey",
        },
        correct="right",
        why={
            "right": "The report gives 18% as the group's average, \"at least around 18%\", and 10% as the median; and \"The middle 50% of estimates ranged from 1% to 25%.\" The question was also about extinction \"or similarly permanent and severe disempowerment\", not only about wiping out humanity.",
            "nothing": "1,489 answers make a large survey, but each answer is a stated probability, not a measurement; the authors of the paper on the survey of 2023 write that the predictions of AI experts \"should not be seen as a reliable guide to objective truth\". See *Expert surveys*.",
            "few": "1,580 researchers stayed in the main analysis, and 1,489 of them answered one of the three questions about extinction. What the headline leaves out is which number 18% is: the mean, with a median of 10%. See *Expert surveys*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_exercise_pdoom():
    mo.md(r"""
    ### Exercise 4. A budget by p(doom)

    A government wants to set its budget for AI safety by the p(doom) of experts: the higher the median in the latest survey, the more money.
    """)
    return


@app.cell(hide_code=True)
def choice_pdoom_cell():
    choice_pdoom = L.choice(
        "pdoom",
        question="What objection would Narayanan and Kapoor raise, going by their post of July 2024?",
        options={
            "low": "Experts always give numbers far too low",
            "right": "Such forecasts are too unreliable to guide policy",
            "pointless": "Forecasting AI risk is pointless, even as academic research",
        },
        correct="right",
        why={
            "right": "Their central claim: \"AI x-risk forecasts are far too unreliable to be useful for policy, and in fact highly misleading.\" For existential risk from AI, they write, \"there is no reference class, as it is an event like no other.\"",
            "low": "They do not say that the numbers are too low, or too high: they say that such numbers are too unreliable to be useful for policy. See *p(doom)*.",
            "pointless": "They write: \"We have no objection to AI x-risk forecasting as an academic activity, and forecasts may be helpful to companies and other private decision makers.\" Their objection is to policy built on such numbers. See *p(doom)*.",
        },
    )
    choice_pdoom
    return (choice_pdoom,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Other positions](/course/open/rai-xrisk-counterpositions?lang=en) set positions on AI risk side by side, and Buterin cited the numbers of others there. This lesson looked at where such numbers come from: expert surveys such as AI Impacts' ESPAI, a tournament of superforecasters and experts, the personal p(doom), and Cotra's biological anchors. With [Describing data: mean, median and spread](/course/open/stats-describe?lang=en), you can read them: which question, which number of all the answers, and how far they spread. The next lesson, *Can an AI system matter morally?*, turns from what AI might do to people to how people should treat AI.

    **An everyday example.** Five friends guess how long a drive will take. Four say about 2 hours, one says 6. The mean is 2.8 hours and the median 2. "Our friends expect 2.8 hours" is true, and still misleading. And if none of them has driven that road before, neither number says much about the drive.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_headline, choice_pdoom, result_describe, result_mixture):
    L.wrap_up(result_describe, result_mixture, choice_headline, choice_pdoom)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
