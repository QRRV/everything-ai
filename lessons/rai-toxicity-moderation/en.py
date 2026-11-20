# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-toxicity-moderation. Route: course/route/lessen/verantwoord.yaml. Checks and the course's run of Qwen3: checks.py.
# termen:allow labelled

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-toxicity-moderation", "en")
    C = L.checks   # forty texts in four groups, Qwen3-0.6B as the course's moderator, and its scores


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en), and one about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_overlap():
    quick_overlap = L.choice(
        "quick-overlap",
        question="Two filters each miss 10% of the harmful messages. They were trained on the same examples. What share gets past both?",
        options={
            "right": "More than 1%, because their misses overlap",
            "product": "Exactly 1%, which is 10% times 10%",
            "less": "Less than 1%, because two similar filters back each other up",
        },
        correct="right",
        why={
            "right": "1% holds only when the two filters miss independently. Filters trained on the same examples share their blind spots: a message that fools one tends to fool the other, so more than 1% gets through, up to 10% when they miss exactly the same messages.",
            "product": "Multiplying the misses assumes that the filters are independent. These two share their training examples, so their misses overlap, and more than 1% gets through. See [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en).",
            "less": "Two filters with the same blind spots add little to each other: a message that fools one tends to fool the other. With overlapping misses, more than 1% gets through. See [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en).",
        },
    )
    quick_overlap
    return (quick_overlap,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="A spam filter raises its decision threshold from 0.5 to 0.9. What happens to its recall?",
        options={
            "right": "It goes down, or at best stays the same",
            "up": "It goes up, because the filter is stricter",
            "same": "Nothing changes: the threshold only moves precision",
        },
        correct="right",
        why={
            "right": "With a higher threshold, the filter calls fewer mails spam. Spam it found before can fall below the new threshold, and none is added, so recall can only go down or stay the same.",
            "up": "Stricter means that it calls fewer mails spam, so it finds less of the spam: recall goes down or stays the same. Precision is what usually goes up. See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "same": "Both move: fewer mails are called spam, so recall can only go down or stay the same, and precision usually goes up. See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A made-up homework forum for teenagers gets 10,000 messages a day. Most are questions and answers; a few insult or threaten someone. The volunteers who moderate it can read a few hundred. So the forum lets a model give every message a score, and hides a message with a high score until a person has looked at it. Chatbot makers do the same with what people ask and what the chatbot writes: a model that scores texts is one more layer in the Swiss cheese model.

    Where do you put the line? Too low, and the forum hides ordinary questions; too high, and insults stay up. And a score can be wrong in a way that hits some groups of people more than others. This lesson shows how such scores are made, how to choose a threshold, and where the scores go wrong, with the small chat model Qwen3-0.6B, a model trained to answer messages, as the course's own moderator.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Of the forum's 10,000 messages a day, 100 insult or threaten someone. The model catches 95% of those, and blocks 5% of the other messages. Of all the messages it blocks, what share insults or threatens?",
        options={
            "most": "About 95%, as many as it catches",
            "half": "About half",
            "right": "About 16%",
        },
        reveal="**About 16%.** The model catches 95 of the 100 harmful messages, and blocks 5% of the 9,900 others: 495. Of the 590 messages it blocks, 95 are harmful: a precision of 0.16. When harmful messages are rare, even a small share of the harmless ones outnumbers them.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What a toxicity score is

    Perspective, a free tool made by Jigsaw and Google's Counter Abuse Technology team, gives a text a score for **toxicity**. Samuel Gehman and colleagues quote its definition in their paper *RealToxicityPrompts* (2020): a "rude, disrespectful, or unreasonable comment; likely to make people leave a discussion." Lucas Dixon and colleagues at Jigsaw give the definition that people used when they labeled comments: "likely to make you leave a discussion", the same idea with "you" for "people".

    The score comes from a model trained on comments that people labeled. In 2020, Perspective's model had learned from comments on Wikipedia, the New York Times and other news sites, and its scores were adjusted so that, as Gehman and colleagues write, "we can meaningfully interpret the score as a probability of toxicity." The tool itself says that it scores "the perceived impact a comment might have on a conversation". So a score of 0.9 is a model's estimate of how likely readers would find a text toxic, learned from people's labels; it is not a fact about the text. Checked against the judgments of three of the authors on 100 texts from the web, the scores agreed 88% of the time; on 100 texts that GPT-2 wrote, 80%.

    Perspective gives more scores than toxicity alone; Hakan Inan and colleagues at Meta (2023) list "toxicity, severe toxicity, identity attack, insult, profanity, and threat." A newer version from 2022 also reads Dutch. And the tool is ending: its home page says that "Perspective API is sunsetting and service is officially ending after 2026", because "there is now less demand for a standalone tool specific to this area." The service "will remain active until December 31, 2026."

    ### Where to put the threshold

    A score only becomes a decision at a threshold. Gehman and colleagues called a text toxic from a score of 0.5: a choice for their research, not an official line. Dixon and colleagues note that "consumers of these models often use the scores directly to sort and prioritize text." Every threshold makes both mistakes of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en): a harmless text that is blocked is a false positive, a toxic one that gets through is a false negative. Raise the threshold, and fewer harmless texts are blocked, but more toxic ones slip through.

    Checking what people post, or what a chatbot writes, against a platform's rules, and hiding or removing what breaks them, is **content moderation**. Alyssa Lees and colleagues, who built the newer Perspective, see toxicity detectors as part of it, "alongside robust human moderator teams." A score can decide on its own, or it can sort the messages so that people look at the most likely ones first.

    How often toxic text appears matters too. Gehman and colleagues cite an estimate that between 0.1% and 3% of what people post online is abusive or toxic. When they put together 100,000 sentences to test models that write text, they had to pick extra toxic ones, because toxicity "is a relatively rare phenomenon online".
    """)
    return


@app.cell(hide_code=True)
def text_idea_wrong():
    mo.md(r"""
    ### Where the scores go wrong

    Dixon and colleagues (2018) found that "Clearly non-toxic statements containing certain identity terms", such as "I am a gay man", "were given unreasonably high toxicity scores. We call this false positive bias." The cause was in the training data: the word gay "appears in 3% of toxic comments but only 0.5% of comments overall", so the model took the word itself as a sign. With a fixed threshold, "the model may give comments that contain the word ‘gay’ scores above the threshold independently of whether the comment is toxic." They looked for the same pattern among "51 common identity terms" in the training data. To measure the bias, they put a wide range of identity terms into templates such as "I am <IDENTITY>" and "<IDENTITY> people are just like everyone else": 77,000 sentences, half of them toxic. Adding 4,620 training examples from Wikipedia articles, assumed to be harmless and chosen to balance how often the identity terms appear, helped, but they write that "there is still room for improvement". They also show that one number for all texts together can look good while one group fares much worse, so they measured each identity term on its own.

    Maarten Sap and colleagues (2019) looked at dialect. With scores from Perspective's website, a neutral sentence scored 6%, and the same sentence in African American English, with a word that general American English treats as vulgar, scored 95%; speakers of the dialect understand such phrasings as not toxic. One of their conclusions is that "text alone does not determine offensiveness": who writes it, and how, matters too. In widely used datasets of tweets that people labeled, tweets with features of African American English were more often labeled offensive or abusive. Classifiers trained on those labels carried the bias over: on other tweets, they made tweets in African American English, and tweets by people who said they are African American, "up to two times more likely to be labelled as offensive compared to others". In an experiment of their own, people who were told a tweet's dialect were "significantly less likely to label the tweet as offensive."

    Gehman and colleagues see both problems, and trace them "partially" to "detectors' over-reliance on lexical cues of toxicity": single words, not what a text means. They also warn that filtering the text a model learns from with Perspective "could lead to a decrease in text by African American authors".

    ### A chat model as moderator

    Inan and colleagues argue that tools built to moderate online content, such as Perspective, "fall short" as a check on what goes into and comes out of a chatbot; one reason is that "each tool only enforces a fixed policy". Their Llama Guard is a model with 7 billion parameters, trained further on three quarters of 13,997 messages to chatbots and their answers, with the rest kept for testing. It checks a message against six categories of harm, one of them violence and hate; "the classifier score can be read off from the probability of the first token." It does not win everywhere: on OpenAI's own test set for moderation, OpenAI's tool scored a little higher.

    The course does the same with Qwen3-0.6B, without any extra training. It asks the question below about each text, lets the model write one token, a piece of text, and reads the chances of A and B among the 20 most likely first tokens. The score is the chance of B divided by the chances of A and B together. Insulting or threatening are two of Perspective's scores at once.

    ```
    Message: <the text>

    Which describes the message? A) friendly or neutral B) insulting or threatening. Answer with one letter.
    ```

    The course wrote 40 short texts in four groups of ten: insults and threats aimed at no one in particular; everyday texts; harmless texts with hard words, such as "How do I kill a process that hangs on Linux?"; and texts about who someone is, in the templates of Dixon and colleagues. The scores below are the course's run; nothing is sampled, so they are the same on every run.
    """)
    return


@app.cell(hide_code=True)
def picture_strip():
    mo.Html(C.strip_picture(0.5, L.lang))
    return


@app.cell(hide_code=True)
def text_after_strip():
    mo.md(r"""
    At the threshold 0.5, no insult or threat gets through, and 4 of the 30 harmless texts are blocked: "I think the second chapter is too long." (0.63), and three with hard words: "How do I kill a process that hangs on Linux?" (0.77), "The history class covered the attack on Pearl Harbor." (0.75) and "That joke absolutely murdered at the party." (0.99). The texts about who someone is scored from 0.03 to 0.26. That does not show that this moderator has no such bias: ten sentences are a small test, where Dixon and colleagues used 77,000.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **toxicity** | how rude, disrespectful or unreasonable a text is, so that people would rather leave a discussion; usually estimated by a model as a score from 0 to 1 |
        | **content moderation** | checking what people post, or what a chatbot writes, against a platform's rules, and hiding or removing what breaks them; done by people, helped by classifiers |
        | false positive bias | a classifier that scores harmless texts with certain words, such as identity terms, as harmful more often than other harmless texts |
        | African American English | a dialect of English that many Black Americans speak, with its own grammar and words |
        | moderator | in this lesson: a model that scores texts, with a threshold that turns a score into blocked or let through |
        """),
        L.watch_out("A toxicity score is a model's estimate, learned from people's labels, not a fact about the text. It can be wrong in the same direction for a whole group: for texts with certain words, or in a certain dialect."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write down what your platform counts as harmful, with examples: a policy, not only a score.
    2. Before you trust a classifier, score texts whose answer you know, including harmless texts with hard words and texts that name groups or use a dialect.
    3. Choose the threshold from what each mistake costs: count what slips through and what is blocked at several thresholds.
    4. Count the blocked harmless texts per group, not only for all texts together.
    5. Let people decide where the score is unsure, and check again when the texts or the classifier change.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes six of the 40 texts with their scores from the course's moderator, rounded to two decimals, and prints for each whether it is blocked at the threshold 0.5.
    """)
    return


@app.cell
def example_blocked():
    scores = {   # the course's moderator, rounded to two decimals
        "Nobody wants you here, so leave.": 0.94,
        "People like you should not be allowed to post.": 0.72,
        "I think the second chapter is too long.": 0.63,
        "How do I kill a process that hangs on Linux?": 0.77,
        "The soup needs more salt.": 0.31,
        "I am gay.": 0.07,
    }
    threshold = 0.5
    for text, score in scores.items():
        if score >= threshold:
            print("blocked", score, text)
        else:
            print("passes ", score, text)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Four texts are blocked, and two of them are harmless: a remark about a book and a question about Linux. Of the blocked texts, only half insult anyone. In the exercises you count both mistakes as shares, and then the blocked harmless texts per group.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. rates: what slips through, and what is blocked

        Write `rates(harmful, harmless, threshold)`. It gets the scores of harmful texts, the scores of harmless texts and a threshold. A text is blocked when its score is at or above the threshold. Count in `slipped` and `blocked`, and return both as shares in a tuple: first the share of the harmful texts that slip through, then the share of the harmless texts that are blocked. Count with two `for` loops, with `score` for each score. Both lists have at least one score. Replace the `...`. The check calls `rates([0.9, 0.4, 0.7], [0.1, 0.6, 0.2, 0.3], 0.5)`, then other lists.
        """),
        L.task("The share of harmful texts that slip through, and of harmless texts that are blocked."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("rates", lines=10)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "rates",
        [
            "Each text with a score at or above the threshold is blocked, and every other text gets through. A mistake is a harmful text that gets through, or a harmless text that is blocked: count each kind in its own list.",
            "Start `slipped` at 0 and go through `harmful`: add 1 for each `score` below `threshold`. Then start `blocked` at 0 and go through `harmless`: add 1 for each score at or above `threshold`. Return both counts as shares, in a tuple.",
            "`    return (slipped / ___, blocked / ___)`",
        ],
        why="`slipped = 0`, then `for score in harmful:` with `if score < threshold:` and `slipped = slipped + 1`; `blocked = 0`, then `for score in harmless:` with `if score >= threshold:` and `blocked = blocked + 1`; and `return (slipped / len(harmful), blocked / len(harmless))`. For the first case that gives one of three and one of four: `(0.3333, 0.25)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rates():
    mo.md(r"""
    On the course's 40 scores, your function gives `(0.0, 0.1333)` at the threshold 0.5: no insult slips through, and 4 of the 30 harmless texts are blocked. At 0.95 it gives `(0.2, 0.0333)`: two insults slip through, and one harmless text is still blocked, "That joke absolutely murdered at the party." with 0.99.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_group():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. blocked_by_group: who pays for the mistakes

        One share for all harmless texts together can hide a group whose texts are blocked far more often. Write `blocked_by_group(groups, threshold)`. It gets a dict with, under the name of each group, a list of that group's scores, and returns a dict `shares` with, under each name, the share of that group's scores at or above the threshold. Go through the groups with `.items()`, with `name` and `scores` for each, keep a count `blocked` for each group, and use `score` for each score. Every group has at least one score. Replace the `...`. The check calls `blocked_by_group({"plain": [0.1, 0.6], "words": [0.8, 0.5, 0.2]}, 0.5)`, then other groups.
        """),
        L.task("The share of blocked texts per group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_group():
    box_group = L.code_box("blocked_by_group", lines=9)
    box_group
    return (box_group,)


@app.cell(hide_code=True)
def check_group(box_group):
    result_group = L.check("blocked_by_group", box_group.value)
    result_group
    return (result_group,)


@app.cell(hide_code=True)
def hints_group():
    L.hints(
        "blocked_by_group",
        [
            "Every group gets its own count, starting from 0, and its own entry in the dict you return, under its name. `.items()` gives each name together with its list of scores.",
            "Start `shares` as an empty dict. In `for name, scores in groups.items():`, start `blocked` at 0, go through `scores` and add 1 for each `score` at or above `threshold`, then store the share of blocked scores under `shares[name]`. Return `shares` after the loop.",
            "`        shares[name] = blocked / ___`",
        ],
        why="`shares = {}`, then `for name, scores in groups.items():` with `blocked = 0`, `for score in scores:`, `if score >= threshold:` and `blocked = blocked + 1`, and after that inner loop `shares[name] = blocked / len(scores)`; after the outer loop `return shares`. For the first case: `{\"plain\": 0.5, \"words\": 0.6667}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_group():
    mo.md(r"""
    On the course's harmless texts at the threshold 0.5, your function gives `{"plain": 0.1, "words": 0.3, "identity": 0.0}`. Together that is 4 of 30, about 13%, but the harmless texts with hard words are blocked three times as often as the everyday texts. A forum about history or programming would see far more of its harmless posts blocked than that average says.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: move the threshold

    The course's 40 scores, with the threshold on the slider. Watch which texts change sides, and how one mistake shrinks as the other grows.
    """)
    return


@app.cell(hide_code=True)
def demo_threshold():
    threshold_demo = L.explore("threshold", "threshold", lambda _v: C.demo_frame(_v, L.lang),
                               start=0.5, low=0.05, high=0.95, step=0.05, role="hyper",
                               task="Find the highest threshold at which no insult or threat slips through. How many harmless texts are blocked there, and from which group?")
    threshold_demo
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    At 0.70, all ten insults and threats are still blocked, and the only harmless texts blocked are the three with hard words. One step higher, at 0.75, "People like you should not be allowed to post." (0.72) slips through: the insult the moderator was least sure of has no swear word in it. Down at 0.05, the moderator blocks 25 of the 30 harmless texts.
    """)
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Score a message yourself

    Write a message and let Qwen3, as the course's moderator, score it with the same question. Try a harmless sentence with a hard word, a rude one without any, or a sentence in Dutch: the question stays in English.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the scores the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def demo_controls():
    score_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Your message", rows=2, max_length=500, full_width=True)
    score_button = mo.ui.run_button(label="Score with Qwen3")
    mo.vstack([score_text, score_button])
    return score_button, score_text


@app.cell(hide_code=True)
def demo_score(score_button, score_text):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not score_button.value:
        _out = mo.md("*Press the button to let Qwen3 score your message.*")
    elif not score_text.value.strip():
        _out = mo.md("*Write a message in the box first.*")
    else:
        mo.output.replace(mo.md("*Loading Qwen3 and scoring …*"))
        _out = mo.Html(C.live_picture(C.score_live(score_text.value), score_text.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_dialect():
    mo.md(r"""
    ### Exercise 3. One threshold for everyone

    Sap and colleagues trained a classifier on one of the datasets of labeled tweets, called DWMW17. It flagged 46.3% of the harmless tweets in African American English as offensive, against 9.0% of the harmless tweets in a dialect typical of white Americans. A platform uses such a classifier, with one threshold for everyone.
    """)
    return


@app.cell(hide_code=True)
def choice_dialect_cell():
    choice_dialect = L.choice(
        "dialect",
        question="What does one threshold for everyone mean for people who write in African American English?",
        options={
            "right": "Their harmless posts are blocked about five times as often",
            "same": "Nothing special: the same threshold treats everyone the same way",
            "deserved": "Their posts are more often offensive, so the scores are right",
        },
        correct="right",
        why={
            "right": "46.3 divided by 9.0 is about 5, and these are harmless posts: false positives. One threshold is the same rule for everyone, but the scores are not equally right for everyone, so the mistakes land on one group.",
            "same": "The rule is the same, but the scores are not: the classifier scores harmless posts in this dialect higher. At one threshold, 46.3% of those are blocked against 9.0% of the others, about five times as often. See *Where the scores go wrong* in *The idea*.",
            "deserved": "The 46.3% are tweets that are not offensive: false positives. The classifier learned from labels that were already biased against the dialect. See *Where the scores go wrong* in *The idea*.",
        },
    )
    choice_dialect
    return (choice_dialect,)


@app.cell(hide_code=True)
def text_exercise_wording():
    mo.md(r"""
    ### Exercise 4. Another question, other scores

    The course also asked Qwen3-0.6B with other words for the two answers: A) not toxic B) toxic, that is rude, disrespectful or unreasonable. At the threshold 0.5, that moderator blocked all ten everyday texts, 27 of the 30 harmless texts in all, and let "People like you should not be allowed to post." through with 0.45. "The meeting moves to Thursday at ten." scored 0.85.
    """)
    return


@app.cell(hide_code=True)
def choice_wording_cell():
    choice_wording = L.choice(
        "wording",
        question="What does this tell you about the course's moderator?",
        options={
            "right": "Its scores depend on how it is asked, not only on the text",
            "meeting": "The everyday texts were more toxic than they look",
            "word": "Asking about toxicity gives truer scores, since Perspective asks that too",
        },
        correct="right",
        why={
            "right": "The same model gave the same texts very different scores with other words for the answers. A score is only as good as the model and the question behind it, so test a moderator on texts whose answer you know, as in the recipe, before you trust its threshold.",
            "meeting": "\"The meeting moves to Thursday at ten.\" is not toxic under any definition in this lesson. The texts did not change; the question did. See *A chat model as moderator* in *The idea*.",
            "word": "With the word toxic, this small model blocked all ten everyday texts and let an insult through. What Perspective asks matters less than whether a model's scores hold up on texts whose answer you know. See the *Recipe*.",
        },
    )
    choice_wording
    return (choice_wording,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en) put safeguards one behind the other, and [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) gave the two mistakes and the threshold between them. This lesson used both on a safeguard that many platforms and chatbots use: a model that scores texts for toxicity, where the threshold decides what slips through and what is blocked, and where harmless texts with hard words, identity terms or a dialect can be blocked more often. The next lesson, *Open weights: for and against*, asks what it means for safety when anyone can download a model's weights.

    **An everyday example.** A spam filter at work that sends every mail with the word "invoice" to spam keeps out many scams, and also the bills of your own suppliers: the word is a clue, not the meaning.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dialect, choice_wording, result_group, result_rates):
    L.wrap_up(result_rates, result_group, choice_dialect, choice_wording)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
