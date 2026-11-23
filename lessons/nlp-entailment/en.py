# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-entailment. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-entailment", "en")
    C = L.checks   # the pairs of SNLI, three models of naive Bayes, the course's numbers, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en) and [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_every():
    quick_every = L.choice(
        "quick-every",
        question="In the small world of *Meaning: when is a sentence true?*, Tom and Mia are cats, and Mia and Rex sleep. Is *every cat sleeps* true in that world?",
        options={
            "mia": "Yes: Mia is a cat, and Mia sleeps",
            "two": "Yes: two animals sleep, and there are two cats",
            "right": "No: Tom is a cat and not among the sleepers",
            "rex": "It cannot be checked: Rex sleeps and is not a cat",
        },
        correct="right",
        why={
            "right": "*Every* asks that each cat be among the sleepers, and Tom is not. That lesson checked one sentence against one world. This lesson compares two sentences: when the first one is true, does the second one have to be true too?",
            "mia": "That makes *some cat sleeps* true: at least one cat sleeps. What does *every* ask of each cat? See [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
            "two": "*Every cat sleeps* is not about how many sleep, but about which ones. Is each cat among the sleepers? See [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
            "rex": "The sentence is about the cats; an animal that is not a cat does not change it. Is each cat among the sleepers? See [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
        },
    )
    quick_every
    return (quick_every,)


@app.cell(hide_code=True)
def quick_check_not():
    quick_not = L.choice(
        "quick-not",
        question="In *Logistic regression on words*, the model gives *Not bad at all.* a probability of 0.008 of being positive. Why does it get the sentence so wrong?",
        options={
            "short": "The sentence is too short for TF-IDF to count",
            "right": "Each word has its own weight, so *not* cannot turn *bad* around",
            "unknown": "The model never saw *bad* during training",
            "bias": "The bias is so negative that it decides every sentence on its own",
        },
        correct="right",
        why={
            "right": "The logit adds up what each word says on its own: *not* and *bad* both pull toward negative. A weight sees one word and not the words around it. This lesson gives such a model two sentences, and asks whether one follows from the other.",
            "short": "TF-IDF gives a value to each word of a short sentence too. What does the model do with the value of *not* and the value of *bad*? See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "unknown": "*bad* has a strong negative weight, so the model knows the word well. What can a weight of one word not see? See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "bias": "The bias is one number for every sentence, and other sentences do get a high probability. What do *not* and *bad* each add? See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
        },
    )
    quick_not
    return (quick_not,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Does an answer follow from a text, does a summary say what the source says, does a claim contradict a report? Each time, the question is whether one sentence follows from another. Samuel Bowman, Gabor Angeli, Christopher Potts and Christopher Manning, who built the dataset of this lesson: "Understanding entailment and contradiction is fundamental to understanding natural language, and inference about entailment and contradiction is a valuable testing ground for the development of semantic representations." A model that scores well on such a test seems to understand both sentences. This lesson looks at how well it can score while reading only one of them.

    **An everyday example.** A friend texts *I'm at the beach with the dog.* Then *my friend is outside* follows from it. *The dog is swimming* might be true, or not. *My friend is at home on the couch* cannot be true at the same time.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="The course trained naive Bayes, as in *Spam or not*, on 36,788 pairs of sentences, each labeled follows, contradicts or neither. It showed the model only the second sentence of each pair, never the first. On 9,824 new pairs, with the three labels about equally common, how many of every 100 does it label right?",
        options={
            "chance": "About 33 of every 100, as with guessing",
            "some": "About 45 of every 100",
            "right": "About 61 of every 100",
            "most": "About 90 of every 100",
        },
        reveal="**About 61:** 5,972 of the 9,824 test pairs, 0.608, without reading a single first sentence. Always giving the most common label, *follows*, gets 3,368 of them right, 0.343.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Premise, hypothesis, label

    **Natural language inference**, NLI for short, takes two sentences: a **premise** and a **hypothesis**. The question is whether the hypothesis follows from the premise (**entailment**), cannot be true at the same time (**contradiction**), or neither (**neutral**). The Stanford Natural Language Inference corpus, SNLI, holds 570,152 such pairs, with hypotheses written by about 2,500 people. Each premise is the caption of a photo. The people who wrote the hypotheses got these instructions: "We will show you the caption for a photo. We will not show you the photo. Using only the caption and what you know about the world:". Then each wrote three new captions: one that is "definitely a true description of the photo", one that "might be a true description of the photo", and one that is "definitely a false description of the photo". The instructions gave one example, in the picture.
    """)
    return


@app.cell(hide_code=True)
def guide_cell():
    mo.Html(C.guide_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_world():
    mo.md(r"""
    *There are animals outdoors* follows only with what you know about the world: dogs are animals, and a field is outdoors. [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en) checked one sentence against one small world. NLI asks something looser: whether a person who reads the premise would conclude the hypothesis.

    ### One scene

    When are two sentences in contradiction? Bowman and others give the pair *A boat sank in the Pacific Ocean* and *A boat sank in the Atlantic Ocean*: "The pair could be labeled as a contradiction if one assumes that the two sentences refer to the same single event, but could also be reasonably labeled as neutral if that assumption is not made." SNLI answers with the scene: in the words of Bowman and others, premise and hypothesis "were constrained to describe that scenario from the same perspective". So in SNLI, a boat in the premise and a boat in the hypothesis are usually taken to be the same boat. A footnote of theirs adds, about which words refer to the same thing: "Issues of coreference are not completely solved, but greatly mitigated."

    ### What the writers left behind

    Suchin Gururangan and others looked at SNLI in 2018: "We show that, in a significant portion of such data, this protocol leaves clues that make it possible to identify the label by looking only at the hypothesis, without observing the premise." They call such clues annotation artifacts, and give a reason: "A possible explanation for the formation and relative consistency of these artifacts is that crowd workers adopt heuristics in order to generate hypotheses quickly and efficiently." *Nobody* in a hypothesis points to contradiction, and so do *sleeping* and *cat*; *outdoors* and *animal* point to entailment, *tall* and *because* to neutral. The picture counts seven of these words in the training pairs of this lesson.
    """)
    return


@app.cell(hide_code=True)
def cue_cell():
    mo.Html(C.cue_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_shortcut():
    mo.md(r"""
    They write about the example of the instructions: "This example likely primed the annotators, inducing the specific heuristics of replacing dog with animal and mentions of scenery with outdoors." A model trained on such pairs can learn these words instead of the relation between the two sentences. That is **shortcut learning**: a model learns a cue that goes with the label in the training data, instead of the skill the task asks for. The cue works on test pairs written the same way, so the score looks good.

    The picture compares four ways to label the 9,824 test pairs of this lesson. The three models are naive Bayes, as in [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en), trained on the 36,788 training pairs; they differ only in the words they read. The first reads only the hypothesis. The second reads the words of both sentences in one bag, with each word marked by its sentence. The third reads the hypothesis, with each word marked as old when the premise has it and as new when it does not.
    """)
    return


@app.cell(hide_code=True)
def models_cell():
    mo.Html(C.models_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        Reading the premise as well adds nothing: 0.606 against 0.608. Naive Bayes scores each word on its own, and a word of the premise says little about the label by itself. What counts is how the words of the two sentences relate, and the third model gets a little of that: 0.693.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **natural language inference** (NLI) | deciding whether a hypothesis follows from a premise, contradicts it, or neither |
        | **premise** | the sentence that is given; in SNLI, the caption of a photo |
        | **hypothesis** | the sentence to decide about |
        | **entailment**, **contradiction**, **neutral** | the three labels: the hypothesis follows; it cannot be true at the same time; it might be true |
        | **shortcut learning** | a model learns a cue that goes with the label in the training data, instead of the skill the task asks for |
        | **overlap** | in this lesson: the share of the words of the hypothesis that the premise has |
        | `C.TRAIN`, `C.TEST` | the 36,788 training pairs and the 9,824 test pairs, each a tuple `(premise, hypothesis, label)`, with premise and hypothesis as tuples of words in lower case |
        | `C.LABELS` | `("entailment", "neutral", "contradiction")` |
        | `C.LOG_PRIOR` | a dict with, for each label, the log of its share of the training pairs |
        | `C.WORD_SCORES` | a dict with, for each label, a dict of the log probability of each word of the training hypotheses, with Laplace smoothing, as in *Spam or not* |
        """),
        L.watch_out("A **hypothesis** in NLI is a sentence to test against the premise. It is not a hypothesis as in research, a claim that an experiment tests. And a high score on SNLI does not show that a model relates the two sentences: in this lesson, a model that reads only one of them gets 0.608."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Labels of a word: go over the pairs, and for each hypothesis that has the word, add 1 to the count of its label and to the total. Divide each count by the total.
    2. A label from the hypothesis alone: for each label, start at its log prior and add the score of each word of the hypothesis that the dict of that label has. The label with the highest score wins.
    3. Overlap: count the words of the hypothesis that the premise has, and divide by the number of words of the hypothesis.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell counts the labels of the training pairs and prints the counts. Then it prints the first training pair.
    """)
    return


@app.cell
def example_count():
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    for premise, hypothesis, label in C.TRAIN:
        counts[label] = counts[label] + 1
    print(counts)
    print(C.TRAIN[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The three labels are about equally common, so a model that guesses gets about a third right. The first pair is labeled neutral: the premise says nothing about a park. Exercise 1 counts the labels of only the hypotheses that have one word.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. label_shares: which labels go with a word

        Write `label_shares(pairs, word)`: `pairs` holds pairs `(premise, hypothesis, label)`, with premise and hypothesis as lists or tuples of words. Of the hypotheses that have `word`, give back the share of each label, as a dict with the keys `"entailment"`, `"neutral"` and `"contradiction"`. Count each hypothesis once, also when it has the word twice. When no hypothesis has the word, each share is `0.0`. Replace the `...`. The check calls `label_shares([(["a", "dog", "runs"], ["an", "animal", "runs"], "entailment"), (["a", "dog", "runs"], ["nobody", "runs"], "contradiction"), (["two", "kids", "play"], ["the", "kids", "know", "nobody"], "neutral"), (["a", "man", "sings"], ["nobody", "sings"], "contradiction")], "nobody")`: three hypotheses have *nobody*, one neutral and two contradiction, which gives `{"entailment": 0.0, "neutral": 0.3333333333333333, "contradiction": 0.6666666666666666}`. Then cases of its own: a word that no sentence has, a word that only the premises have, a word twice in one hypothesis, and `C.TRAIN` with `"nobody"` and with `"outdoors"`.
        """),
        L.task("Give the share of each label among the hypotheses with a word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("label_shares", lines=14)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("label_shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "label_shares",
        [
            "Two counters: `counts`, which the start gives, and `total`, 0 at the start. A loop `for premise, hypothesis, label in pairs:` with an `if` that asks whether `word` is in `hypothesis`. Then a new dict `shares`, and a loop `for label in counts:` that fills `shares[label]`.",
            "Inside the `if`, add 1 to the count of the label of the pair, and 1 to `total`. A share is a count divided by the total; when `total` is 0, every share is `0.0`.",
            "The line under `if total > 0:`: `            shares[label] = ___`",
        ],
        why="`total = 0`, then `for premise, hypothesis, label in pairs:` with `if word in hypothesis:` and under it `counts[label] = counts[label] + 1` and `total = total + 1`. After the loop, `shares = {}` and `for label in counts:` with `if total > 0:` and `shares[label] = counts[label] / total` under it, and `else:` with `shares[label] = 0.0`; then `return shares`. The example gives `{\"entailment\": 0.0, \"neutral\": 0.3333333333333333, \"contradiction\": 0.6666666666666666}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_shares():
    mo.md(r"""
    The picture under the check runs your function on the training pairs, for the seven words of the idea. Gururangan and others: "Negation words such as nobody, no, never and nothing are strong indicators of contradiction." And: "sleeping contradicts any activity". The many cats have a reason of their own, in their words: "The high frequency of cat probably stems from the many dog images in the original dataset." Exercise 2 uses all the words of a hypothesis at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_guess():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. guess: a label from the hypothesis alone

        Write `guess(hypothesis, log_prior, word_scores)`: naive Bayes as in [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en), with three labels instead of two, on the words of the hypothesis alone. `log_prior` holds the log prior of each label, and `word_scores` holds, for each label, a dict with the score of each word it knows. The score of a label starts at its log prior. Add the score of each word of the hypothesis that the dict of that label has, as often as the word comes; skip a word it does not have. The lines under the `...` keep the label with the highest score. Replace the `...`. The check calls `guess(["nobody", "is", "outdoors"], {"entailment": -1.0, "neutral": -1.2, "contradiction": -1.1}, {"entailment": {"nobody": -9.0, "outdoors": -4.0, "dog": -6.0}, "neutral": {"nobody": -9.5, "outdoors": -6.0, "dog": -6.1}, "contradiction": {"nobody": -5.0, "outdoors": -6.5, "dog": -6.2}})`. The dicts do not have *is*; contradiction gets −1.1 − 5.0 − 6.5 = −12.6, the highest of the three, so it gives `"contradiction"`. Then cases of its own: a hypothesis without words and one with only words the dicts do not have, both with a prior in which neutral is highest; a word twice; two known words; one known word with a prior in which neutral is a whole point higher than the rest; one known word three times; and the three hypotheses of the example of SNLI with `C.LOG_PRIOR` and `C.WORD_SCORES`.
        """),
        L.task("Choose a label from the words of the hypothesis alone."),
    ])
    return


@app.cell(hide_code=True)
def exercise_guess():
    box_guess = L.code_box("guess", lines=14)
    box_guess
    return (box_guess,)


@app.cell(hide_code=True)
def check_guess(box_guess):
    result_guess = L.check("guess", box_guess.value)
    result_guess
    return (result_guess,)


@app.cell(hide_code=True)
def hints_guess():
    L.hints(
        "guess",
        [
            "Let `score` start at the log prior of `label`. Then go over the words of `hypothesis`, and for each word that the dict of `label` has, add the score of that word to `score`.",
            "The dict of a label is `word_scores[label]`, and `word in word_scores[label]` asks whether it has the word. The loop over the words comes right under the line with the log prior, at the same depth. A word that comes twice goes through the loop twice.",
            "`                score = score + ___`",
        ],
        why="Replace `score = ...` with `score = log_prior[label]`, then `for word in hypothesis:` with `if word in word_scores[label]:` and under it `score = score + word_scores[label][word]`. The lines under it stay as they are. The example gives `\"contradiction\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_guess():
    mo.md(r"""
    On the whole test set, your function gets 0.608 right: the answer to the guess at the start. Gururangan and others report about 67% for their own model of the hypothesis alone, trained on all of SNLI, against 34.3% for the most common label. Adam Poliak and others report 69.00% against 34.28% on the test set of SNLI. But the second model of the idea, which reads both in one bag, got 0.606: a word of the premise does not say much by itself. Exercise 3 looks at the two sentences together.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. overlap: how much of the hypothesis is in the premise

        Write `overlap(premise, hypothesis)`: the share of the words of the hypothesis that the premise has. Count every word of the hypothesis, so a word that comes twice counts twice. A hypothesis without words gives `0.0`; the start does that. Replace the `...`. The check calls `overlap(["two", "dogs", "are", "running", "through", "a", "field"], ["there", "are", "animals", "outdoors"])`: only *are* is in the premise, which gives `0.25`. Then cases of its own: every word inside, a hypothesis without words, a word twice in the hypothesis, a word twice in the premise, the pair *The doctor was paid by the actor.* and *The doctor paid the actor.*, and no word inside.
        """),
        L.task("Give the share of the words of the hypothesis that the premise has."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overlap():
    box_overlap = L.code_box("overlap", lines=9)
    box_overlap
    return (box_overlap,)


@app.cell(hide_code=True)
def check_overlap(box_overlap):
    result_overlap = L.check("overlap", box_overlap.value)
    result_overlap
    return (result_overlap,)


@app.cell(hide_code=True)
def hints_overlap():
    L.hints(
        "overlap",
        [
            "Make `in_premise`, the words of `premise` as a set, and a counter `inside`, 0 at the start. Go over the words of `hypothesis`, and count the ones that are in `in_premise`.",
            "A set keeps each word once: fine for the premise, which you only ask about. Count the hypothesis word by word, and after the loop, divide `inside` by the number of words of the hypothesis.",
            "`    in_premise = ___`",
        ],
        why="`in_premise = set(premise)` and `inside = 0`, then `for word in hypothesis:` with `if word in in_premise:` and `inside = inside + 1` under it; after the loop `return inside / len(hypothesis)`. The example gives `0.25`.",
    )
    return


@app.cell(hide_code=True)
def text_after_overlap():
    mo.md(r"""
    On the test pairs, entailment has the highest overlap, 0.598 on average, against 0.423 for neutral and 0.402 for contradiction. Gururangan and others, who take each sentence as a bag of words: "8.8% of entailed hypotheses in SNLI are fully contained within their premise, while only 0.2% of neutrals and contradictions exhibit the same property." In the test pairs of this lesson, 330 of the 3,368 entailed hypotheses have overlap 1, against 3 neutral and 3 contradiction ones. Marking which words of the hypothesis are old and which are new, as the third model of the idea does, brings the score from 0.608 to 0.693.

    Gururangan and others also split the test set of SNLI in two: the pairs that their model of the hypothesis alone labels right, and the rest, which they call Hard. Three well-known models of 2016 to 2018, which do read both sentences, get 84.7 to 86.5% of the whole test set right, and 69.4 to 72.7% of the Hard part. They write: "This result implies that the ability of NLI models to recognize textual entailment is lower than previously perceived, and that such models rely heavily on annotation artifacts in the hypothesis to make their predictions." And they add: "Importantly, artifacts do not render any particular example incorrect".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_doctor():
    mo.md(r"""
    ### Exercise 4. Every word comes back

    Tom McCoy, Ellie Pavlick and Tal Linzen describe a rule that a model can learn from such data, the lexical overlap heuristic: "Assume that a premise entails all hypotheses constructed from words in the premise". In the test pairs of this lesson, it works well: 330 of the 336 pairs with overlap 1 are entailment.
    """)
    return


@app.cell(hide_code=True)
def choice_doctor_cell():
    choice_doctor = L.choice(
        "doctor",
        question="What does the rule say about the premise *The doctor was paid by the actor.* and the hypothesis *The doctor paid the actor.*, and is it right?",
        options={
            "neutral": "Neutral: the hypothesis adds nothing new",
            "right": "Entailment, and it is wrong: in the premise, the actor pays",
            "fine": "Entailment, and it is right: every word of the hypothesis is in the premise",
            "short": "Contradiction, since the hypothesis is shorter than the premise",
        },
        correct="right",
        why={
            "right": "Every word of the hypothesis is in the premise, so the overlap is 1, and the rule says entailment. But the order of the words says who pays whom, and a bag of words throws the order away. McCoy, Pavlick and Linzen built a test set full of such pairs, HANS. The models they tested, trained on another large dataset of NLI, \"nearly always predicted entailment for the examples in HANS\", and so got almost every pair wrong whose label was not entailment.",
            "neutral": "The rule only asks whether the words of the hypothesis are in the premise, and here all of them are. Which label does the rule give then?",
            "fine": "The rule does say entailment. But read both sentences: who gives the money to whom in each of them?",
            "short": "The rule does not look at length; it asks whether the words of the hypothesis are in the premise. What does it give when all of them are?",
        },
    )
    choice_doctor
    return (choice_doctor,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: three models on your own pair

    Type a premise and a hypothesis, and see what the three models of the idea say. Change only the premise: the model of the hypothesis alone keeps its label. Try *A cat is sleeping on a couch.* with *A cat is sleeping.*: the hypothesis follows, and that model says contradiction, for *cat* and *sleeping*. Then try the doctor and the actor.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_premise = mo.ui.text(value=C.DEMO[0], label="Premise", full_width=True)
    demo_hypothesis = mo.ui.text(value=C.DEMO[1], label="Hypothesis", full_width=True)
    mo.vstack([demo_premise, demo_hypothesis])
    return demo_hypothesis, demo_premise


@app.cell(hide_code=True)
def demo_look(demo_hypothesis, demo_premise):
    mo.Html(C.demo_picture(demo_premise.value, demo_hypothesis.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en) checked one sentence against a world; NLI asks whether one sentence follows from another, with what people know about the world. The naive Bayes of [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en), like the weights of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en), scores each word on its own. On SNLI, that gets 0.608 from the hypothesis alone, against 0.343 for always entailment, since the writers of the hypotheses left cues such as *nobody*, *outdoors* and *tall*. That is shortcut learning: a good score without the skill. A test set that the shortcut does not fit, such as the Hard part of SNLI or the pairs of HANS, shows the difference. SNLI used one scene to mostly settle whether two sentences speak of the same boat; the next lesson, *What does 'it' refer to? Coreference*, works out which words of a text refer to the same thing.

    **An everyday example.** In a test of multiple choice where the longest answer is nearly always right, a student who has spotted that scores well without knowing the subject. A new test, written without that pattern, shows what the student knows.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_doctor, result_guess, result_overlap, result_shares):
    L.wrap_up(result_shares, result_guess, result_overlap, choice_doctor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
