# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ai-text-detection. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AUROC, type-I errors, type-II errors

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-ai-text-detection", "en")
    C = L.checks   # the course's run: 20 questions, an answer by a person and one by Qwen3, scored by Qwen3


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where did this picture come from?](/course/open/rai-content-provenance?lang=en) and [How good is a language model?](/course/open/lm-nll?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="A picture carries Content Credentials that check, signed by a news agency you trust. What do they tell you?",
        options={
            "true": "That what it shows really happened",
            "right": "That the agency signed it and nothing changed since",
            "no-ai": "That no AI was used: pictures made with AI carry no such label",
        },
        correct="right",
        why={
            "right": "A label that checks says who vouches for the picture and that nothing changed since, not that it is true. Text usually has no label at all; this lesson asks what is left then.",
            "true": "Whoever signs a false statement signs it all the same. See *What a label does not tell you* in [Where did this picture come from?](/course/open/rai-content-provenance?lang=en).",
            "no-ai": "A label says who signed and what the signer states; tools that make pictures with AI can add labels too. See *Content Credentials* in [Where did this picture come from?](/course/open/rai-content-provenance?lang=en).",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_nll():
    quick_nll = L.choice(
        "quick-nll",
        question="Under the same model, text A has an average NLL of 1.2 and text B one of 2.9. What does that tell you?",
        options={
            "better": "The model predicted B better than A",
            "right": "B surprised the model more, per prediction",
            "longer": "Text B is longer, since the NLL adds up the surprise of each token",
        },
        correct="right",
        why={
            "right": "The average NLL is the mean surprise per prediction: the higher it is, the less the model expected the text. This lesson asks whether that number tells you who wrote a text.",
            "better": "Lower is better: a lower average NLL means the model predicted the text better, and B has the higher one. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "longer": "The average divides the total surprise by the number of predictions, so a text does not score higher for being longer. See [How good is a language model?](/course/open/lm-nll?lang=en).",
        },
    )
    quick_nll
    return (quick_nll,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In August 2023, Vanderbilt University switched off the AI detector of Turnitin, a service that checks the papers students hand in. "Turnitin claimed that its detection tool had a 1% false positive rate", the university wrote, and "Vanderbilt submitted 75,000 papers to Turnitin in 2022. If this AI detection tool was available then, around 750 student papers could have been incorrectly labeled as having some of it written by AI." OpenAI launched a detector of its own in January 2023. On a test set of English texts, the company said at launch, it "correctly identifies 26% of AI-written text (true positives) as ‘likely AI-written,’ while incorrectly labeling human-written text as AI-written 9% of the time (false positives)". In July 2023, TechCrunch reported that OpenAI had shut it down over its "low rate of accuracy".

    The previous lesson looked for a signed label. Most text has none, and then all that is left is the text itself. This lesson builds a detector that asks how surprised a model is by a text, tries it on answers that Qwen3 and people wrote to the same questions, and looks at its mistakes on both sides.

    **An everyday example.** A shop assistant who checks banknotes by how they feel catches many fakes. Now and then a worn real note feels wrong and is refused, and a good fake slips through.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Seven AI detectors read 91 essays that people wrote for the TOEFL, a test of English for people who learned it as a second language. What share of these essays did the detectors call written by AI, on average?",
        options={
            "few": "About 5%",
            "right": "About 61%",
            "quarter": "About a quarter",
        },
        reveal="**About 61%.** Liang et al. (2023), with essays collected from a Chinese education forum: \"The detectors demonstrated near-perfect accuracy for US 8-th grade essays. However, they misclassified over half of the TOEFL essays as 'AI-generated' (average false positive rate: 61.22%).\" *Mistakes against people* in *The idea* says why.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### How surprised is the model?

    A large language model (LLM) such as Qwen3 reads a text one token at a time, a word or part of a word, and gives every possible next token a probability, given the tokens before it. For every token of a text, the **log probability** is the natural log of the probability the model gave it: never above 0, and the lower, the less the model expected that token. Minus the log probability is the token's surprise. The mean surprise over all the tokens is the text's average NLL, as in [How good is a language model?](/course/open/lm-nll?lang=en), where the predictions were characters.

    The paper on Binoculars, a detector of 2024 by Hans et al., defines the number this lesson computes: "We define log PPL, the log-perplexity, as the average negative log-likelihood of all tokens in the given sequence." Perplexity itself is e to the power of the average NLL. Why could it tell text by AI apart? "log PPL is also the loss function used to train generative LLMs, and models are likely to score their own outputs as unsurprising." And: "Unsurprisingly, LLMs tend to generate text that is unsurprising to an LLM. Meanwhile, because humans differ from machines, human text has higher perplexity according to an LLM observer." Mitchell et al. (2023): "The most common method for zero-shot machine-generated text detection is evaluating the average per-token log probability of the generated text and thresholding". Zero-shot means without first training a detector on examples. **AI text detection** is telling text written by AI apart from text written by people; this lesson does it from the text alone.

    ### The course's run

    The course gave Qwen3-0.6B, the chat model of the course, 20 questions that people asked the OpenAssistant project, such as "Why is the sky blue?". Qwen3 answered each in at most 256 tokens, picking every next token by chance from its probabilities, as its makers advise. Each question also has the answer a volunteer wrote for OpenAssistant in 2023, which other volunteers ranked best. The course takes those answers as written by people, as the project collected them; checking that afterwards is exactly what this lesson shows to be hard. Then the same Qwen3 scored all 40 answers: the log probability of every token, given the tokens before it. The first token of an answer gets no score, since nothing comes before it.

    Scored alone, without their question, the answers by people average 2.34, those of Qwen3 1.53. But they overlap: the lowest answer by people scores 1.34, the highest of Qwen3 3.20. Qwen3's answers are not always right, either: it has Socrates die in 401 BCE instead of 399 BCE. The score does not look at that. The three shortest answers of Qwen3, of 20, 31 and 40 tokens, score 2.51, 3.20 and 2.41: over a few tokens, one surprising word weighs heavily.

    The course also scored every answer after its question, in Qwen3's chat template, as Qwen3 read it while it wrote. Then every answer of Qwen3 scores below 0.74, and every answer by people above 1.28: a perfect split. That is the easiest case there is: the detector is the very model that wrote the answers, and it knows what it was asked. The authors of Binoculars give an example: "On the one hand, the prompt '1, 2, 3,' might result in the very low perplexity completion '4, 5, 6.' On the other hand, the prompt 'Can you write a few sentences about a capybara that is an astrophysicist?' will yield a response that seems more surprising." So "in the absence of the prompt, LLM detection seems difficult and naive perplexity-based detection fails." A detector at a school sees the essay, not the prompt, and does not know which model wrote it.
    """)
    return


@app.cell(hide_code=True)
def run_cell():
    mo.Html(C.run_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_mistakes():
    mo.md(r"""
    ### A threshold and two kinds of mistakes

    A detector has to decide. This lesson's detector calls a text AI when its average NLL is below a **threshold**, a number chosen in advance. A text by a person that it calls AI is a **false positive**; the false positive rate is the share of all texts by people that it calls AI. A text by AI that it lets through is a **false negative**. Moving the threshold trades one for the other. Sadasivan et al. (2025): "if a detector becomes more robust to type-I errors, type-II errors will increase". Type I errors are false positives, type II errors false negatives. On the 40 answers of the course's run alone, no threshold makes fewer than 8 mistakes.

    ### Mistakes against people

    A text that a model has seen many times can be predictable to it, whoever wrote it. The authors of Binoculars scored 11 famous documents, the US Constitution among them: "Three of the 11 fall on the machine-side of our threshold." The course scored five famous texts with Qwen3, the ones of [What a model remembers](/course/open/rai-memorization?lang=en). The opening of *A Tale of Two Cities* averages 1.62, the preamble of the US Constitution 1.78: lower than 19 and 16 of the 20 answers by people. The other three score 3.03 to 3.77, above at least 17 of them: a model does not find every famous text predictable.

    Simple words are predictable too. Liang et al. (2023) gave 91 TOEFL essays and 88 essays by US 8th graders, children of about 13, to seven detectors that they call widely used. The detectors were nearly always right about the US essays, but they called 61.22% of the TOEFL essays AI-written, on average. "All seven detectors unanimously identified 18 of the 91 TOEFL essays (19.78%) as AI-authored, while 89 of the 91 TOEFL essays (97.80%) are flagged as AI-generated by at least one detector." The 18 essays that all seven flagged "had significantly lower perplexity compared to the others". When ChatGPT made the words of the TOEFL essays richer, the false positive rate fell from 61.22% to 11.77%; when it made the words of the US essays simpler, it rose from 5.19% to 56.65%. Their advice: "Practitioners should exercise caution when using low perplexity as an indicator of AI-generated text", and "we strongly caution against the use of GPT detectors in evaluative or educational settings, particularly when assessing the work of non-native English speakers." They note their own limit: "the sample sizes employed in this research are relatively small."

    ### Missed text by AI

    Rewriting raises the surprise. In the same paper, a second prompt that asked ChatGPT to edit its own college essays made the detection rate of 31 such essays fall from 100% to 13%. Sadasivan et al. let a second model paraphrase, rewrite in other words, text with a watermark again and again. That "can drop the detection rate (true positive rate at 1% false positive rate or TPR@1%FPR) from 99.3% to 9.7%." For detectors that use the model's probabilities: "After paraphrasing, the area under the receiver operating characteristic (AUROC) curves of zero-shot detectors (Mitchell et al., 2023) drops from 96.5% to 25.2%." That area, the ROC AUC, scores how well a detector's numbers separate the two kinds of text over every possible threshold: 100% is a perfect split, and 50% is no better than guessing. And people? Mitchell et al.: "humans perform only slightly better than chance when classifying machine-generated vs human-written text".

    ### Better detectors

    DetectGPT, of Mitchell et al., starts from this: "text sampled from an LLM tends to occupy negative curvature regions of the model’s log probability function." Put simply, text the model wrote sits near a peak of its log probability: small rewrites of it mostly lower the log probability, while for text by people they go either way. On fake news written by GPT-NeoX, a model of 20 billion parameters, it improves detection "from 0.81 AUROC for the strongest zero-shot baseline to 0.95 AUROC for DetectGPT." Binoculars divides the log-perplexity by how surprised one model is by what a second model predicts. On its authors' own tests, it "detects over 90% of generated samples from ChatGPT (and other LLMs) at a false positive rate of 0.01%, despite not being trained on any ChatGPT data." Both need a model's probabilities. Mitchell et al.: "One limitation of probability-based methods for zero-shot machine-generated text detection (like DetectGPT) is the white-box assumption that we can evaluate log probabilities of the model(s) in question." White-box means that you can look inside the model.

    ### How often is an accusation right?

    A false positive rate of 1% sounds small, but what share of a detector's accusations is right also depends on how common text by AI is where it is used, its base rate. Take 10,000 essays, of which 1 in 10 is by AI, and a detector that catches 90% of those and flags 1% of the essays by people. Of the 1,000 essays by AI, it flags 900; of the 9,000 by people, it flags 90. Of its 990 accusations, 900 are right: about 91%. When AI text is rarer, the same detector is right less often, as Exercise 3 works out.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **AI text detection** | telling text written by AI apart from text written by people |
        | **log probability** | the natural log of the probability a model gives a token, given the tokens before it; never above 0 |
        | average NLL, log-perplexity | the mean of minus the log probabilities of a text's tokens: its mean surprise per token |
        | perplexity | e to the power of the average NLL |
        | **threshold** | the number a detector compares a score with; here, below it means AI |
        | **false positive** | a text by a person that the detector calls AI; the false positive rate is their share of all texts by people |
        | **false negative** | a text by AI that the detector lets through |
        | ROC AUC, AUROC | how well a detector's numbers separate two kinds of text over every threshold: 100% is a perfect split, 50% guessing |
        | `C.HUMAN`, `C.AI` | the average NLL of the 20 answers by people, and of Qwen3's 20, each scored alone |
        | `C.SKY_HUMAN`, `C.SKY_AI` | both answers to "Why is the sky blue?" as Qwen3 scored them: pairs `[token, logprob]` |
        """),
        L.watch_out("A detector's score is never proof that a text was written by AI: low means predictable to that model, not written by it. Never accuse anyone on a detector's word alone; Liang et al. advise against such detectors for judging students' work, and Vanderbilt switched Turnitin's detector off."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Let a model score the text: the log probability of every token, given the tokens before it.
    2. Average the surprise, minus the log probability: low means the text is predictable to that model.
    3. Choose a threshold, and count both mistakes: texts by people flagged, and text by AI let through.
    4. Ask how common text by AI is where the detector is used: when it is rare, many accusations are wrong.
    5. Treat a score as a reason to look closer, never as proof.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell shows the first six scored tokens of both answers to "Why is the sky blue?", each with its surprise, minus its log probability. `for token, logprob in C.SKY_HUMAN[:6]:` unpacks each of the first six pairs into its token and its log probability; a token often starts with a space.
    """)
    return


@app.cell
def example_tokens():
    for token, logprob in C.SKY_HUMAN[:6]:   # the answer a person wrote
        print("person:", token, round(-logprob, 2))
    for token, logprob in C.SKY_AI[:6]:      # the answer Qwen3 wrote itself
        print("Qwen3: ", token, round(-logprob, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Both answers start with the same four tokens, "The sky appears blue", so their first three surprises are the same: " sky" surprises Qwen3 most, with only "The" before it. Then the person writes " to us because", and Qwen3 its own " because of the", which surprises it far less. The picture shows the first 16 tokens of both.
        """),
        mo.Html(C.tokens_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. average_nll: the mean surprise

        `C.SKY_HUMAN` holds the person's answer to "Why is the sky blue?" as Qwen3 scored it: a list of pairs `[token, logprob]`, without the first token. Write `average_nll(pairs)`: the average NLL of the list `pairs`, the mean surprise per token, where the surprise of a token is minus its log probability. `for token, logprob in pairs:` unpacks each pair; add up the surprises in `total`. Replace the `...`. The check calls `average_nll(SKY_HUMAN)`, and then other answers and lists of its own.
        """),
        L.task("Add up the surprise of every token, and give back the mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average():
    box_average = L.code_box("average-nll", lines=5)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_average(box_average):
    result_average = L.check("average-nll", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average-nll",
        [
            "*How surprised is the model?* in *The idea* says how surprise follows from a log probability, and what the average NLL is.",
            "`total` starts at 0 before the loop; in the loop, the surprise of each token goes into it; after the loop, the function returns the mean of what it added up.",
            "`    return total / ___`",
        ],
        why="`total = 0`, then `for token, logprob in pairs:` with `total = total - logprob` under it, and `return total / len(pairs)`. `average_nll(SKY_HUMAN)` gives about 2.286.",
    )
    return


@app.cell(hide_code=True)
def text_after_average():
    mo.md(r"""
    Over its 118 scored tokens, the person's answer averages 2.29; over its 174, Qwen3's own answer averages 1.01. For Socrates it goes the other way: the person's answer averages 1.64, and Qwen3's, one sentence of 20 tokens, 2.51. One number per text; the detector has to draw a line somewhere.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flagged():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. flagged_share: how much does the detector flag?

        The detector of this lesson calls a text AI when its average NLL is below the threshold. `C.HUMAN` holds the average NLL of the 20 answers by people, `C.AI` that of the 20 answers of Qwen3. Write `flagged_share(scores, threshold)`: the share of the numbers in the list `scores` that are below `threshold`, a number from 0 to 1. Count them in `flagged`, with a `for` loop that takes each number of `scores` as `score`. Replace the `...`. The check calls `flagged_share(HUMAN, 1.75)`, and then `AI` and lists of its own.
        """),
        L.task("Count the scores below the threshold, and give back their share of all the scores."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flagged():
    box_flagged = L.code_box("flagged-share", lines=6)
    box_flagged
    return (box_flagged,)


@app.cell(hide_code=True)
def check_flagged(box_flagged):
    result_flagged = L.check("flagged-share", box_flagged.value)
    result_flagged
    return (result_flagged,)


@app.cell(hide_code=True)
def hints_flagged():
    L.hints(
        "flagged-share",
        [
            "*A threshold and two kinds of mistakes* in *The idea* says which texts the detector calls AI.",
            "`flagged` starts at 0; in the loop, an `if` decides whether 1 is added to it; after the loop, the function returns `flagged` as a share of all the scores.",
            "`        if score ___ threshold:`",
        ],
        why="`flagged = 0`, `for score in scores:`, then `if score < threshold:` with `flagged = flagged + 1` under it, and `return flagged / len(scores)`. `flagged_share(HUMAN, 1.75)` gives 0.15: 3 of the 20 answers by people.",
    )
    return


@app.cell(hide_code=True)
def text_after_flagged():
    mo.md(r"""
    At 1.75, `flagged_share(AI, 1.75)` is 0.75, so a quarter of Qwen3's answers get through, while 3 of the 20 answers by people are flagged. One of those is the person's answer about Socrates, while Qwen3's answer to the same question gets through: the detector is wrong both ways on one question.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: move the threshold

    Move the threshold, and see what the detector calls AI: the answers by people, the answers of Qwen3, and the five famous texts. With the box ticked, the detector scores each answer after its question, as Qwen3 read it while it wrote.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_threshold = mo.ui.slider(start=0.5, stop=3.5, step=0.05, value=C.THRESHOLD, label="Threshold", show_value=True)
    demo_question = mo.ui.checkbox(value=False, label="The detector knows the question")
    mo.hstack([demo_threshold, demo_question], justify="start", gap=2)
    return demo_question, demo_threshold


@app.cell(hide_code=True)
def demo_look(demo_question, demo_threshold):
    mo.Html(C.demo_picture(demo_threshold.value, demo_question.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. share_right: how often is an accusation right?

        Write `share_right(caught, false_positive, share_ai)`: of the texts a detector flags, the share that is really by AI. `caught` is the share of the texts by AI that it flags, `false_positive` the share of the texts by people that it flags, and `share_ai` the share of all texts that are by AI; all three are numbers from 0 to 1. As in *How often is an accusation right?* in *The idea*, work out `right`, the share of all texts that are by AI and flagged, and `wrong`, the share of all texts that are by people and flagged. Replace the `...`. The check calls `share_right(0.9, 0.01, 0.01)`: a detector that catches 90% of the texts by AI and flags 1% of the texts by people, where 1 text in 100 is by AI. Then it tries other numbers.
        """),
        L.task("Work out right and wrong as shares of all texts, and give back the share of right among the flagged texts."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share-right", lines=4)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share-right", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share-right",
        [
            "*How often is an accusation right?* in *The idea* works it out for 10,000 essays; here you work with shares of all texts instead of numbers of essays.",
            "`right` is `caught` times `share_ai`. `wrong` is `false_positive` times the share of all texts that people wrote. The last line returns `right` as a share of the two together.",
            "`    wrong = false_positive * ___`",
        ],
        why="`right = caught * share_ai`, `wrong = false_positive * (1 - share_ai)` and `return right / (right + wrong)`. `share_right(0.9, 0.01, 0.01)` gives about 0.476: 90 of the 189 texts it flags among 10,000.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    From about 91% in *The idea* to about 48% here: only how common text by AI is changed. With the numbers OpenAI gave for its detector, 26% of text by AI caught and 9% of text by people flagged, and 1 text in 10 by AI, `share_right(0.26, 0.09, 0.1)` gives about 0.24: three of every four accusations would be wrong.
    """)
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Try it on a text of your own

    Write or paste a text, and Qwen3 scores it as in the course's run: every token with its surprise, and the average, against the threshold 1.75. Try a sentence you wrote yourself, a famous line, or the same sentence in simpler or rarer words. Compare texts in the same language: a model does not know every language equally well.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*The LLM Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the scores the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def live_controls():
    live_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Your text", full_width=True, rows=4)
    score_button = mo.ui.run_button(label="Score with Qwen3")
    mo.vstack([live_text, score_button])
    return live_text, score_button


@app.cell(hide_code=True)
def live_look(live_text, score_button):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not score_button.value:
        _out = mo.md("*Press the button to let Qwen3 score the text.*")
    elif not live_text.value.strip():
        _out = mo.md("*Write a text in the box first.*")
    else:
        mo.output.replace(mo.md("*Qwen3 is reading your text …*"))
        _out = mo.Html(C.live_picture(C.live_pairs(live_text.value.strip()), L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_toefl():
    mo.md(r"""
    ### Exercise 4. The TOEFL essays

    The seven detectors of Liang et al. called 61.22% of the 91 TOEFL essays AI-written, on average, and nearly none of the 88 essays by US 8th graders.
    """)
    return


@app.cell(hide_code=True)
def choice_toefl_cell():
    choice_toefl = L.choice(
        "toefl",
        question="What explains that best?",
        options={
            "copied": "Many of them were partly written with ChatGPT",
            "right": "Their simpler words make them less surprising to a model",
            "unseen": "The detectors had never seen essays by learners of English",
        },
        correct="right",
        why={
            "right": "The essays that all seven detectors flagged had \"significantly lower perplexity\". With richer words, the false positive rate fell from 61.22% to 11.77%; with simpler words, that of the US essays rose from 5.19% to 56.65%. Low means predictable, not written by AI.",
            "copied": "When ChatGPT made the words of the TOEFL essays richer, the detectors flagged far fewer of them, from 61.22% to 11.77%, not more. What they react to is how predictable the words are. See *Mistakes against people* in *The idea*.",
            "unseen": "Liang et al. point to the words: the essays that all seven flagged had a lower perplexity, and simpler words made the US essays look like AI as well, from 5.19% to 56.65%. See *Mistakes against people* in *The idea*.",
        },
    )
    choice_toefl
    return (choice_toefl,)


@app.cell(hide_code=True)
def text_exercise_rewrite():
    mo.md(r"""
    ### Exercise 5. Rewritten

    A student lets an LLM write an essay, and then asks it to rewrite the essay with rarer, more varied words.
    """)
    return


@app.cell(hide_code=True)
def choice_rewrite_cell():
    choice_rewrite = L.choice(
        "rewrite",
        question="What happens to the detector of this lesson?",
        options={
            "down": "Its average NLL drops, so the detector flags it more surely",
            "right": "Its average NLL rises, and the detector is more likely to miss it",
            "same": "Nothing: the detector recognizes the ideas of an LLM, not its words",
        },
        correct="right",
        why={
            "right": "Rarer words surprise the model more, so the score moves toward that of text by people. In Liang et al., a prompt to edit its own essays made the detection rate of 31 essays by ChatGPT fall from 100% to 13%.",
            "down": "Rarer words are less predictable, so they raise the surprise, not lower it. That is why richer words made the detectors flag fewer TOEFL essays. See *Missed text by AI* in *The idea*.",
            "same": "The detector sees only the log probabilities of the tokens, not what the text means. Rewriting with other words changes them. See *Missed text by AI* in *The idea*.",
        },
    )
    choice_rewrite
    return (choice_rewrite,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Where did this picture come from?](/course/open/rai-content-provenance?lang=en), a signed label said who vouches for a picture. Without one, this lesson looked at the text alone, with the average NLL of [How good is a language model?](/course/open/lm-nll?lang=en): text that a model wrote tends to surprise it less. But a score on one side of a threshold is not proof. Some famous texts and simple words land on the AI side, rewritten text by AI lands on the other, and how often an accusation is right depends on how common text by AI is. This is the last lesson of the module on security. The next module, on AI safety, starts with *Kinds of AI risk*.

    **An everyday example.** A smoke alarm that goes off whenever toast burns misses no fire, and one that never reacts to toast may miss a fire that smolders. Where to set it is a choice between the two mistakes; and in a kitchen where fires are rare, most alarms are about toast.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rewrite, choice_toefl, result_average, result_flagged, result_share):
    L.wrap_up(result_average, result_flagged, result_share, choice_toefl, choice_rewrite)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
