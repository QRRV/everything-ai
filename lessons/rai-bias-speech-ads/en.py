# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-bias-speech-ads. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-speech-ads", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Bias in face analysis and image generators](/course/open/rai-bias-faces-images?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pool():
    quick_pool = L.choice(
        "quick-pool",
        question="A system makes 30 errors on 200 faces of group A, and 0 errors on 100 faces of group B. What is the error rate of A and B together?",
        options={
            "right": "0.1",
            "mean": "0.075",
            "only-a": "0.15",
        },
        correct="right",
        why={
            "right": "30 errors on 300 faces: 0.1. Add up the counts of the cells, and divide once.",
            "mean": "That is the mean of 0.15 and 0.0, as if both groups were the same size. Add up the errors and the faces first: 30 / 300. See *Recipe* in [Bias in face analysis and image generators](/course/open/rai-bias-faces-images?lang=en).",
            "only-a": "That is the rate of group A alone, 30 of 200. Together it is 30 errors on 300 faces. See *Recipe* in [Bias in face analysis and image generators](/course/open/rai-bias-faces-images?lang=en).",
        },
    )
    quick_pool
    return (quick_pool,)


@app.cell(hide_code=True)
def quick_check_hide():
    quick_hide = L.choice(
        "quick-hide",
        question="A report says a speech recognizer works as well for women as for men, and as well for older as for younger speakers. What can it still hide?",
        options={
            "nothing": "Nothing, if both numbers are equal",
            "right": "A worse result for older women",
            "one": "Only a gap per attribute",
        },
        correct="right",
        why={
            "nothing": "Equal numbers per attribute can hide a combination: that is fairness gerrymandering. See *Intersectional fairness* in [Bias in face analysis and image generators](/course/open/rai-bias-faces-images?lang=en).",
            "right": "A number per attribute mixes the combinations, as Microsoft's 0.107 for women hid the 0.208 of darker-skinned women. You see a combination only when you count per combination.",
            "one": "The report already shows that there is no gap per attribute. What it can hide is a combination, such as older women. See *Intersectional fairness* in [Bias in face analysis and image generators](/course/open/rai-bias-faces-images?lang=en).",
        },
    )
    quick_hide
    return (quick_hide,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You talk to a phone, a car or a help line, and a system writes down what you say. Allison Koenecke and colleagues (2020) gave five of these systems interviews with Black and with White speakers in the US. For every system, "the error rates for black speakers are nearly twice as large". A system that hears you worse makes you repeat yourself, puts the wrong words in your subtitles, or does not let you through. Ads have a quieter version of the same problem. An advertiser can choose to show a job ad to everyone, and the platform still decides which of them actually see it. In a test by Muhammad Ali and colleagues (2019), five ads for jobs in the lumber industry reached over 90 percent men, while five ads for janitors, with the same audience and the same bidding strategy, reached over 65 percent women.

    **An everyday example.** A shop pays someone to hand out flyers for a job in the whole town, and to reach as many people as possible for the money. The flats on the hill take longer to walk to, so the flyers end up in the streets below. Nobody chose to leave the hill out, yet the people there never hear about the job.
    """)
    return


@app.cell(hide_code=True)
def predict_clicks():
    guess_clicks = L.guess(
        "clicks",
        question="A job ad for science and technology was meant to reach women and men alike, but fewer women saw it. When women did see it, did they click on it less or more often than men?",
        options={
            "less": "Less often",
            "same": "Just as often",
            "right": "More often",
        },
        reveal="**More often.** In the data of Anja Lambrecht and Catherine Tucker (2019), women clicked 1.77 times per 1,000 times they saw the ad, men 1.41 times. So the platform did not show the ad less to women because they were less interested in it. *The idea* explains what did happen.",
    )
    guess_clicks
    return (guess_clicks,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Word error rate.** Automatic speech recognition, ASR for short, turns speech into written text. To score it, you line up what the system wrote with a transcript that people made, and count the words the system substituted (S), deleted (D) and inserted (I). The word error rate is their sum divided by the number of words in the correct transcript (N): WER = (S + D + I) / N. Lower is better. Because inserted words count too, the WER can be above 1.

        **Speech recognition.** Koenecke and colleagues tested the systems of Amazon, Apple, Google, IBM and Microsoft on interviews with 42 White and 73 Black speakers from five US cities: 19.8 hours of audio, matched on the age and gender of the speakers, in 2,141 snippets per group, each about 17 seconds long. The average WER was 0.35 for Black speakers and 0.19 for White speakers. Microsoft did best, with 0.27 against 0.15; Apple did worst, with 0.45 against 0.23. Per combination, as in *Gender Shades*: 0.41 for Black men, 0.30 for Black women, 0.21 for White men and 0.17 for White women. If a WER above 0.5 makes a transcript unusable, 23 percent of the snippets of Black speakers were unusable, against 1.6 percent of those of White speakers.

        Where does the gap come from? The interviews also held 206 short phrases, of 5 to 8 words, that Black and White speakers both said. On those identical phrases the gap stayed:

        | System | Black speakers | White speakers |
        |---|---|---|
        | Apple | 0.28 | 0.12 |
        | IBM | 0.21 | 0.10 |
        | Google | 0.17 | 0.11 |
        | Amazon | 0.18 | 0.08 |
        | Microsoft | 0.13 | 0.07 |

        *Average WER on 206 identical phrases, from Table 2 of Koenecke and colleagues.*

        The words were the same, so the difference points to how they sound. The authors trace it to the acoustic models, the part of a system that links short pieces of audio to speech sounds: their findings suggest that the systems are confused by how African American Vernacular English sounds, "rather than the grammatical or lexical characteristics". They write that the WERs on these phrases are about twice as large for each system. Read the table as well: for Google it is 0.17 against 0.11, 1.55 times.

        **Not one direction.** Rachael Tatman (2017) tested YouTube's automatic captions on 80 speakers, eight women and eight men from each of five regions: California, Georgia, New England, New Zealand and Scotland. The WER was higher for women, and higher for speakers from Scotland. Tatman calls the result for women "somewhat surprising given earlier studies which found the opposite result"; Koenecke and colleagues found it higher for men. There is no rule of thumb for which group a system hears worse: you measure it.

        **Targeting and delivery.** On an ad platform, the advertiser chooses who may see an ad, such as all adults in a city: the targeting. Then, auction by auction, the platform decides which of them actually see it: the delivery. Facebook explained its auction as "The ad that wins an auction and gets shown is the one with the highest total value", made of the bid, the estimated action rates, and the ad's quality and relevance. Ali and colleagues ran their own ads, each with the same audience. The budget alone moved the share of men from over 55 percent, for ads with very low budgets, to under 45 percent, for ads with high budgets. Their job ads ran for 24 hours, all with the same audience and the same bidding strategy: the lumber jobs reached over 90 percent men and over 70 percent White users, the janitor jobs over 65 percent women and over 75 percent Black users. Housing ads reached an estimated 27 percent White users for luxury rentals, and 49 percent for cheap houses for sale.

        **A cheaper audience.** Anja Lambrecht and Catherine Tucker (2019) looked at a job ad for science, technology, engineering and math, shown on Facebook in 191 countries. It was meant to be gender neutral, yet it was shown to more than 20 percent more men than women. Women who saw it clicked more often than men, which rules out a lack of interest. Their explanation: younger women are a prized audience, so showing them an ad costs more. An algorithm that only optimizes cost-effectiveness shows the ad to the cheaper audience: other advertisers crowd the women out.

        **Not the qualifications.** Is the skew perhaps fair because the jobs suit one group more? Basileal Imana, Aleksandra Korolova and John Heidemann (2021) ran pairs of ads for the same kind of job at two companies with a different share of women, such as delivery drivers for Domino's, where 98 percent are men, and for Instacart, where more than 50 percent are women. On Facebook the Domino's ad reached a larger share of men than the Instacart ad. Both ads ran at the same time, for similar jobs, so they ask for similar qualifications: those cannot explain the difference. On LinkedIn they found no such skew; one of the nine runs there was just above the threshold for statistical significance, in the unexpected direction.

        **What was done about it.** In June 2022, the US Department of Justice settled a case with Meta under the Fair Housing Act, the US law against discrimination in housing: its "first case challenging algorithmic bias" under that law. It alleged that Meta's ad delivery used machine learning that relied in part on race, national origin and sex to help decide who of an advertiser's audience saw a housing ad. In a settlement that a court approved, Meta agreed to stop using a tool for housing ads called Special Ad Audience, to pay the maximum penalty of $115,054, and to build the Variance Reduction System (VRS): for housing ads, it reduces the difference between the eligible audience, the users who fit the advertiser's targeting, and the actual audience, those who saw the ad, by sex and by estimated race or ethnicity. Race is estimated from surnames and places, not asked. In January 2023 the two sides agreed on targets for the vast majority of housing ads: by the end of that year, a difference of at most 10 percent for 91.7 percent of those ads by sex, and for 81.0 percent by race or ethnicity. The settlement put this under court oversight until 27 June 2026.

        **In the EU.** In the EU, since 17 February 2024, the Digital Services Act forbids online platforms to show ads based on profiling with special categories of personal data, such as racial or ethnic origin, and ads based on profiling of minors; every ad must say the main parameters that decided who sees it. A skew that comes from prices and optimization, as above, is not what those rules are about.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **speech recognition** (ASR) | turning speech into written text, such as subtitles |
        | **word error rate** (WER) | (S + D + I) / N: the words substituted, deleted and inserted, divided by the words in the correct transcript; lower is better, and it can be above 1 |
        | **targeting** and **delivery** | the advertiser chooses who may see an ad: the targeting; the platform decides which of them do: the delivery |
        | **ad delivery bias** | a skew in who sees an ad that comes from the platform's delivery, its predictions and the prices in its auctions, and not from the advertiser's targeting |
        """),
        L.watch_out("23 percent of the snippets, not 23 percent of the speakers. The share counts snippets of about 17 seconds, each with its WER averaged over the five systems. Say what you count: snippets, speakers or people who saw an ad."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a speech system or an ad for bias:

    1. Choose the groups and the measure: the WER per group of speakers, or the share of each group among the people who saw an ad.
    2. Compare like with like: speakers matched on age and gender, or the same phrases; ads with the same audience, budget and bidding.
    3. Count first, then divide once per group: errors over words, or the group's impressions over all impressions.
    4. Look at a limit that matters to people, such as a WER above 0.5 that makes a transcript unusable, and say whether you count snippets or people.
    5. Look for the cause with a second test: identical phrases rule out the words, and a pair of similar job ads rules out the qualifications.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two snippets, lined up by hand. In the first, someone said "we are going to the market on friday" and the system wrote "we going to a market on friday now": *are* is deleted, *the* is substituted by *a*, and *now* is inserted. In the second, someone said "thank you" and the system wrote "think you very much indeed": one word substituted and three inserted. The loop works out the WER of each. Run the cell.
    """)
    return


@app.cell
def example_snippets():
    # two snippets: the words in the correct transcript, and the words substituted, deleted and inserted
    _snippets = [
        {"words": 8, "substituted": 1, "deleted": 1, "inserted": 1},
        {"words": 2, "substituted": 1, "deleted": 0, "inserted": 3},
    ]

    for _snippet in _snippets:
        _errors = _snippet["substituted"] + _snippet["deleted"] + _snippet["inserted"]
        print(_snippet["words"], "words, WER", round(_errors / _snippet["words"], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    0.375 for the first snippet: three errors on eight words. 2.0 for the second: four errors on two words, so the WER is above 1. Koenecke and colleagues counted a transcript as unusable when its WER is above 0.5: the first is still usable, the second is not. What share of a set of snippets is unusable? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_wer():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Unusable transcripts

        `snippets` holds ten snippets of one speaker, made up by the course. Work out `unusable`: the share of the snippets whose WER is above `limit`. Start with `count = 0`, loop over `snippets` with `snippet`, work out its `errors` and its `wer` as in the worked example, and when `wer > limit`, add 1 to `count`. After the loop, divide `count` by the number of snippets, `len(snippets)`; `unusable` is the last line, so the check sees it. The check also tries a lower limit.
        """),
        L.task("Work out `unusable`: the share of the snippets whose WER is above `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wer():
    box_wer = L.code_box("wer", lines=25)
    box_wer
    return (box_wer,)


@app.cell(hide_code=True)
def check_wer(box_wer):
    result_wer = L.check("wer", box_wer.value)
    result_wer
    return (result_wer,)


@app.cell(hide_code=True)
def hints_wer():
    L.hints(
        "wer",
        [
            "First the WER of one snippet, as in the worked example; then count how many snippets lie above the limit, and divide by how many there are.",
            "In the loop: `errors` is the sum of the three counts, `wer = errors / snippet[\"words\"]`, and under `if wer > limit:` you add 1 to `count`.",
            "After the loop, not inside it: `unusable = count / ___`.",
        ],
        why="`count = 0`, then `for snippet in snippets:` with `errors = snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]` and `wer = errors / snippet[\"words\"]` under it, and `if wer > limit:` with `count = count + 1`. After the loop, `unusable = count / len(snippets)`. That gives 4 / 10 = `0.4`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cause():
    mo.md(r"""
    ### Exercise 2. Same words, more errors
    """)
    return


@app.cell(hide_code=True)
def choice_cause_cell():
    choice_cause = L.choice(
        "cause",
        question="On 206 identical phrases, said by Black and by White speakers, the five systems still made more errors for the Black speakers. What does that point to?",
        options={
            "words": "Rarer words in what the speakers said",
            "right": "How the words sound",
            "grammar": "The grammar of the speakers' sentences",
        },
        correct="right",
        why={
            "words": "The phrases were identical, so both groups said the same words. See *Speech recognition* in *The idea*.",
            "right": "The words and the grammar were the same, so what differs is how they sound. Koenecke and colleagues trace the gap to the acoustic models, the part of a system that links short pieces of audio to speech sounds.",
            "grammar": "The phrases were identical, so the grammar was the same for both groups. See *Speech recognition* in *The idea*.",
        },
    )
    choice_cause
    return (choice_cause,)


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Who saw the ad?

        `cells` holds Table 2 of Lambrecht and Tucker: per age group and gender, the `"impressions"`, the times someone saw the ad, and the `"clicks"`. Work out `share`: the share of `gender` in everything `count` counts, for `count = "impressions"` and `gender = "female"` the women's share of the impressions. Start with `total = 0` and `mine = 0`, loop over `cells` with `cell`, add every cell's `cell[count]` to `total`, and when `cell["gender"] == gender`, add it to `mine` as well. After the loop, `share` is `mine` divided by `total`. The check also tries the clicks, and the men.
        """),
        L.task("Work out `share`: the cells of `gender` added up for `count`, divided by all cells added up for `count`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=27)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "A share is a part divided by the whole: the women's impressions divided by all impressions. Add up both in one loop, and divide once, after it.",
            "In the loop, every cell goes into `total`: `total = total + cell[count]`. Under `if cell[\"gender\"] == gender:`, the cell goes into `mine` too.",
            "After the loop: `share = ___ / total`.",
        ],
        why="`total = 0` and `mine = 0`, then `for cell in cells:` with `total = total + cell[count]` under it, and `if cell[\"gender\"] == gender:` with `mine = mine + cell[count]`. After the loop, `share = mine / total`. That gives 1,983,915 / 4,515,713 = `0.439`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_audience():
    mo.md(r"""
    ### Exercise 4. What does the VRS compare?
    """)
    return


@app.cell(hide_code=True)
def choice_audience_cell():
    choice_audience = L.choice(
        "audience",
        question="Meta's Variance Reduction System for housing ads measures a difference between two groups of users. Which two?",
        options={
            "everyone": "All users and those who saw it",
            "right": "The targeted audience and those who saw it",
            "clicks": "Those who saw the ad and those who clicked it",
        },
        correct="right",
        why={
            "everyone": "The VRS starts from the advertiser's targeting, not from all users: the eligible audience is the users who fit that targeting. See *What was done about it* in *The idea*.",
            "right": "The eligible audience, the users who fit the advertiser's targeting, against the actual audience, those who saw the ad, by sex and by estimated race or ethnicity. It takes on the skew that the delivery adds; the targeting stays the advertiser's choice.",
            "clicks": "Clicks are not part of it: the VRS compares who could have seen the ad with who did. See *What was done about it* in *The idea*.",
        },
    )
    choice_audience
    return (choice_audience,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Bias in face analysis and image generators](/course/open/rai-bias-faces-images?lang=en), you counted errors per combination of groups. Speech recognition has its own error rate, the WER, and the same lesson: count per group and per combination, and look at a limit that matters, such as a transcript that cannot be used. With ads, the question is not how often a system is wrong, but who gets to see an opportunity: the advertiser's targeting can be neutral while the delivery is not. That is ad delivery bias, and it comes from optimizing for clicks and costs, without anyone choosing a group.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_audience, choice_cause, result_share, result_wer):
    L.wrap_up(result_wer, choice_cause, result_share, choice_audience)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
