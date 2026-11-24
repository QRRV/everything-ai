# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-sentiment. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sentiment", "en")
    C = L.checks   # the review sentences in English and Dutch, the classifier, the course's reviews, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_not():
    quick_not = L.choice(
        "quick-not",
        question="In *Logistic regression on words*, *I did not expect this to be so good!* is labeled positive, and gets a probability of positive of 0.423. Where does that come from?",
        options={
            "unknown": "*good* is not one of the words of the training sentences, so it adds nothing",
            "right": "*not* pushes down, whatever stands next to it",
            "start": "The classifier reads only the first few words of a sentence, and those are neutral",
            "mark": "An exclamation mark lowers the probability of every sentence",
        },
        correct="right",
        why={
            "right": "*not* has a weight of −1.78, and a weight belongs to one word, wherever it stands: *not* adds −1.852 here, more than the +1.505 of *good*. The classifier never sees that *not* goes with *expect*. This lesson looks at a word that does change what the words around it mean: *but*.",
            "unknown": "*good* is in many training sentences, and adds +1.505 here. Which word pulls the sentence the other way? See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "start": "The classifier adds up a number for every word of the sentence, wherever it stands. Which word adds the most toward negative? See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "mark": "The words are runs of letters, so the classifier never sees an exclamation mark, and it adds nothing. Which word pulls the sentence down? See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
        },
    )
    quick_not
    return (quick_not,)


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="The TF-IDF value of a word in a text is its count in the text times $\\log_{10}$ of 2,400 divided by the number of training sentences with the word. For *great*, that logarithm is 1.15. What is the TF-IDF value of *great* in *Great food, great service.*?",
        options={
            "one": "1.15, as in any text that has the word",
            "half": "0.575, since the text has two parts",
            "right": "2.30",
            "square": "1.32, which is 1.15 times 1.15, for two times the same word",
        },
        correct="right",
        why={
            "right": "*great* occurs twice, so its value is 2 × 1.15 = 2.30. In this lesson a whole review becomes one text, and a word in three of its sentences counts three times.",
            "one": "The value counts how often the word occurs in the text. How often does *great* occur here? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "half": "The value does not depend on the parts of a text, only on the count of the word in it. How often does *great* occur here? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "square": "The count multiplies the logarithm once. What is the count of *great* in this text? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before you buy a kettle or book a table, you read what others wrote. A shop that wants to show what thousands of buyers think cannot read every review by hand. Jurafsky and Martin, in *Speech and Language Processing*, call this task **sentiment analysis**: "the extraction of sentiment, the positive or negative orientation that a writer expresses toward some object." They add: "The simplest version of sentiment analysis is a binary classification task, and the words of the review provide excellent cues."

    You built such a classifier in [Logistic regression on words](/course/open/nlp-logistic-text?lang=en): it gives 491 of the 600 held-out sentences their own label. But a review is more than one sentence, and its words do not all point the same way. In a 2002 study that sorted film reviews by machine, Pang, Lee and Vaithyanathan describe "a kind of “thwarted expectations” narrative, where the author sets up a deliberate contrast to earlier discussion", and note that "there are many words indicative of the opposite sentiment to that of the entire review."

    This lesson goes from the labels of the sentences to the label of a review, and looks at a word that can turn a sentence around: *but*. The course also translated the 3,000 sentences into Dutch and trained the same classifier on the translation; near the end, the lesson says what changes in Dutch.

    **An everyday example.** A friend back from a holiday says: the hotel was lovely, the beach was beautiful and the food was good, but it rained all week. Three things were fine and one was not, and still you know the holiday was a letdown.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_but = L.guess(
        "guess",
        question="50 of the 600 held-out sentences have the word *but*, and the classifier of *Logistic regression on words* gives 36 of those 50 their own label. What happens when it counts only the words after the last *but*, and leaves out the words before it?",
        options={
            "all": "It gets all 50 right",
            "more": "It gets more right: about 45",
            "same": "Nothing changes: 36",
            "right": "It gets fewer right: 34",
        },
        reveal="**Fewer: 34.** Four sentences come out right that were wrong, such as *The building itself seems pretty neat; the bathroom is pretty trippy, but I wouldn't eat here again.* But six that were right go wrong, such as *Love this place, hits the spot when I want something healthy but not lacking in quantity or flavor.*: on its own, *not lacking in quantity or flavor* sounds like a complaint to the classifier, and the praise before the *but* is gone. A rule that throws away half a sentence also throws away what that half says.",
    )
    guess_but
    return (guess_but,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Sentiment can be read at more than one level. Bing Liu, in *Sentiment Analysis and Opinion Mining* (2012), writes that "sentiment analysis has been investigated mainly at three levels". At the **document level**: "The task at this level is to classify whether a whole opinion document expresses a positive or negative sentiment". The **sentence level** "determines whether each sentence expressed a positive, negative, or neutral opinion", and "Neutral usually means no opinion." The third level looks at parts of a product, such as the battery or the screen; this lesson stays with the first two.

    The 3,000 sentences of this course have two labels and no neutral one. Kotzias, Denil, de Freitas and Smyth, who collected them in 2015, write for each of three websites: "we extracted and manually labeled 1000 sentences from the test set, with 50% positive sentiment and 50% negative sentiment". The page of the set adds: "the goal was for no neutral sentences to be selected." The set keeps the sentences, not the reviews they came from. So the course wrote 18 short reviews of its own, about products, films and restaurants, three to five sentences each, and gave every sentence and every review a label.

    From the labels of the sentences to the label of the review, two simple ways:

    - **Majority voting**: give each sentence a label, and give the review the label that most of its sentences have. On a tie, this lesson lets the last sentence decide. Becker and Aharonson (2010) had readers rate 16 reviews, and the last and the second sentence of each on their own. They write: "Due to the small sample size, we cautiously infer that last sentences express polarity better than second sentences".
    - **Whole text**: join the sentences into one text, and give that text to the classifier. Each word adds its TF-IDF value times its weight to $z$, whichever sentence it stands in, so $z$ of the whole text is the bias plus what the words of each sentence add: $z = b + \sum_s (z_s - b)$, with $z_s$ the $z$ of sentence $s$ on its own.

    The picture shows both for the second review of the course, about a kettle.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Three sentences of praise and one complaint: the majority vote says positive, and the review is negative. The whole text gets it right, only just, because the last sentence pulls down harder than the other three push up; the worked example below shows the numbers. Over all 18 reviews, the majority vote gives 14 their own label, and the whole text 16.

    **What *but* does.** The complaint in the kettle review starts with *But*. Polanyi and Zaenen (2004) write about such a sentence: "By using But the author chooses to give more weight to the second point of view". Socher and colleagues (2013) write about their model, the RNTN: "RNTNs also learn that sentiment of phrases following the contrastive conjunction ‘but’ dominates." And Liu warns: "As in the case of negation, not every but means contrary, e.g., “not only … but also.”"

    A simple way to give the words after *but* more say: give every word before the last *but* half its weight. In this lesson, that is a **scale** for each word: 0.5 before the last *but*, 1 from the *but* on, and the course multiplies what each word adds to $z$ by its scale. On the 50 held-out sentences with *but*, the classifier then gets 38 right instead of 36, and on all 600, 493 instead of 491: it fixes three sentences and breaks one. Two out of 50 is too few to say the rule is better. On the course's reviews, it fixes the one about a steak: three complaints, and then *But the steak was perfect, and that is what we came for.* Its probability of positive goes from 0.447 to 0.569, and the whole text with scales gets 17 of the 18 reviews right.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **sentiment analysis** | deciding whether a text is positive or negative (sometimes also neutral) about its subject |
        | **document level** | one label for a whole text, such as a review |
        | **sentence level** | one label for each sentence |
        | **majority voting** | giving a text the label that most of its sentences have; in this lesson, the last sentence decides a tie |
        | **scale** | a number for each word that the course multiplies what the word adds to $z$ by: 0.5 before the last *but*, 1 elsewhere |
        | `C.reviews(lang)` | the course's 18 reviews in English (`"en"`) or Dutch (`"nl"`), as a list of (label, sentences), with each sentence as (label, text) |
        | `C.probability(text, lang)` | the probability of positive that the classifier of that language gives a text |
        | `C.whole(sentences)` | the text of a review: its sentences joined with a space |
        """),
        L.watch_out("The 18 reviews and their labels are the course's own, written to show how sentences and a review can disagree. What they count shows what can happen, not how often it happens. The counts on the 600 held-out sentences come from real reviews."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Give each sentence of a review its probability of positive, and its label: positive when the probability is 0.5 or more.
    2. Majority voting: count the positive and the negative sentences. More positive ones make the review positive, more negative ones negative; on a tie, the last sentence decides.
    3. Whole text: join the sentences into one text, and give that its probability. A word that occurs in several sentences counts each time.
    4. With *but*: give every word a scale, 0.5 before the last *but* and 1 from the *but* on, and multiply what each word adds to $z$ by its scale.
    5. Compare the labels with the label of the review, and count how many reviews each way gets right.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code takes the second review of the course, the one about a kettle, from the course's `reviews`. For each sentence it prints the probability of positive from the course's `probability`, the label the course gave the sentence and the sentence itself. Then it prints the probability of the whole review as one text, made by the course's `whole`, and the label of the review.
    """)
    return


@app.cell
def example_cell():
    label, sentences = C.reviews(L.lang)[1]
    for sentence_label, text in sentences:
        print(round(C.probability(text, L.lang), 3), sentence_label, text)
    print(round(C.probability(C.whole(sentences), L.lang), 3), label)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The classifier gives each sentence the label the course gave it, and the majority vote still says positive. Each sentence adds $z_s - b$ to the whole: 0.515, 0.275 and 1.242 for the three sentences of praise, together about 2.03, and −2.050 for the last one. With the bias of −0.181, $z$ of the whole text is −0.200, and its probability of positive 0.450: negative, as the course labeled it, but only just.

    The classifier does not even know *leak*: no training sentence has the word, so it adds nothing. The last sentence pulls down through *started* (−1.15), *sent* (−0.47), *two* (−0.30), *after* (−0.28) and *back* (−0.26): words that stood more often in negative training sentences than in positive ones.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_majority():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. majority: one label from the sentences

        Write `majority(labels)`, where `labels` holds the label of each sentence of a review, in order: 1 for positive and 0 for negative. Give back 1 when more sentences are positive than negative, 0 when more are negative, and on a tie the label of the last sentence. Replace the `...`. The check calls `majority([1, 1, 1, 0])`, the labels the classifier gives the four sentences of the kettle, which gives `1`, and `majority([0, 1])`, a tie, which gives `1`. Then cases of its own: the tie the other way round, `[1, 0]`; one positive and one negative sentence alone; two and two in both orders; five sentences with more negative ones; six with more positive ones; and three negative sentences before a positive one.
        """),
        L.task("Give a review the label of most of its sentences."),
    ])
    return


@app.cell(hide_code=True)
def exercise_majority():
    box_majority = L.code_box("majority", lines=12)
    box_majority
    return (box_majority,)


@app.cell(hide_code=True)
def check_majority(box_majority):
    result_majority = L.check("majority", box_majority.value)
    result_majority
    return (result_majority,)


@app.cell(hide_code=True)
def hints_majority():
    L.hints(
        "majority",
        [
            "Two counters, `positive` and `negative`, both 0 at the start. A loop `for label in labels:` with an `if` and an `else`, so that each label adds 1 to one of the two. After the loop, an `if` that gives back 1, an `if` that gives back 0, and one more `return` for a tie.",
            "On a tie, the last sentence decides. Which position always points to the last item of a list, however long the list is?",
            "`    return labels[___]`",
        ],
        why="`positive = 0` and `negative = 0`; `for label in labels:` with `if label == 1:` and `positive = positive + 1` under it, and `else:` with `negative = negative + 1`. After the loop: `if positive > negative:` with `return 1`, `if negative > positive:` with `return 0`, and last `return labels[-1]`. The examples give `1` and `1`.",
    )
    return


@app.cell(hide_code=True)
def text_after_majority():
    mo.md(r"""
    The majority vote goes wrong where a review turns. On the kettle, three sentences of praise outvote the complaint, and on the steak, three complaints outvote *But the steak was perfect, and that is what we came for.* On the headphones, the classifier calls *Very comfortable, even after hours.* negative (0.413), so two sentences stand against two, and the last one, *My only complaint: the cable is short.*, makes the positive review negative: the last sentence is often a good guide, not always. On the film with cheap effects, the classifier gives *But the characters are so likable that I did not care.* a probability of positive of 0.261, and the majority vote, two against one, cannot get the review right.

    Kotzias and colleagues describe a review rated 8 out of 10, where "most of the words in the review are negative, hence the naive strategy of a simple count of words or sentences would misclassify this".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scales():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. scales: half weight before the last *but*

        Write `scales(words, word)`, where `words` is a list of the words of a text, in small letters, and `word` the word that turns a sentence around: `"but"` in English, `"maar"` in Dutch. Give back a list with one number for each word: `0.5` for each word before the last `word`, and `1.0` for that `word` and every word after it. When `word` is not in the list, every word gets `1.0`. The course then multiplies what each word adds to $z$ by its number. Replace the `...`. The check calls `scales(["good", "food", "but", "slow", "service"], "but")`, which gives `[0.5, 0.5, 1.0, 1.0, 1.0]`. Then cases of its own: two words without *but*; *but* as the first word; two times *but*; *but* as the last word; five Dutch words with *maar*; three words with *but* and the word `"maar"`; an empty list; and the eleven words of the held-out sentence *This item worked great, but it broke after 6 months of use.*
        """),
        L.task("Give each word its scale: half before the last but."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scales():
    box_scales = L.code_box("scales", lines=12)
    box_scales
    return (box_scales,)


@app.cell(hide_code=True)
def check_scales(box_scales):
    result_scales = L.check("scales", box_scales.value)
    result_scales
    return (result_scales,)


@app.cell(hide_code=True)
def hints_scales():
    L.hints(
        "scales",
        [
            "First the position of the last `word`: start with `last = -1`, go over the positions with `for i in range(len(words)):`, and set `last = i` each time `words[i] == word`. Then a new list `out`, a second loop over the positions, and an `if` and an `else` that each do `out.append(...)` with `0.5` or `1.0`.",
            "The word itself and every word after it keep `1.0`. Which comparison of a position `i` with `last` says that the word at `i` comes before the last `word`?",
            "`        if i ___ last:`",
        ],
        why="`last = -1`; `for i in range(len(words)):` with `if words[i] == word:` and `last = i` under it. Then `out = []`; `for i in range(len(words)):` with `if i < last:` and `out.append(0.5)` under it, and `else:` with `out.append(1.0)`; after the loop `return out`. Without `word`, `last` stays −1, so every word gets `1.0`. The example gives `[0.5, 0.5, 1.0, 1.0, 1.0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_scales():
    mo.md(r"""
    With the course's scales, three held-out sentences come out right that were wrong, such as *This place has a lot of promise but fails to deliver.* (from 0.539 to 0.350), and one goes wrong: *Love this place, hits the spot when I want something healthy but not lacking in quantity or flavor.* (from 0.566 to 0.152). There the *but* turns nothing around: *healthy but not lacking* praises both sides, as Liu warned for *not only … but also*. With half weight, *love* adds 0.94 instead of 1.88, while *not* (−1.852) and *lacking* (−0.771) after the *but* count in full. Among the course's reviews, the scales fix the steak and leave the film with cheap effects wrong: the classifier already misreads the sentence after its *But*.

    In Dutch, the word to weigh is *maar*, and it does more than turn a sentence around: in *De bezorging duurde maar twee dagen.* (Delivery took only two days.) it means *only*. Of the 65 held-out Dutch sentences with *maar*, 16 have no *but* in English. The Dutch classifier gets 57 of the 65 right, and 58 with scales, and the steak review stays wrong.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 3. Where is the judgment?

    Eight held-out sentences with *but*. Each one comes with its label. Put each one where its judgment, positive or negative, lies: after the *but*, before it, or in both parts, where the *but* turns nothing around.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_judgment = L.sort("where-judgment")
    sort_judgment
    return (sort_judgment,)


@app.cell(hide_code=True)
def check_sort(sort_judgment):
    result_sort = L.check("where-judgment", sort_judgment)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: a review of your own

    Write a review of a few sentences in English, about a product, a film or a meal. The picture gives each sentence its probability of positive, then the majority vote, the whole review as one text, and the whole review with half weight before the last *but*. Try a review that praises three things and ends with *But it broke after a week.*, and then one that complains three times and ends with *But the food was worth it.*
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.text_area(value=C.TRY_START[L.lang], label="Your review", rows=4, full_width=True)
    try_box
    return (try_box,)


@app.cell(hide_code=True)
def try_look(try_box):
    mo.Html(C.try_picture(try_box.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The classifier of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) labels one sentence at a time. A review has several: a majority vote counts their labels, and the whole text lets every word add to one $z$, so one sentence with heavy words can outweigh several light ones. On the course's 18 reviews, the majority vote gets 14 right and the whole text 16. A word such as *but* changes what the words around it count for. Scales that halve the words before the last *but* help where the judgment comes after it, and pull the wrong way where it comes before it or where *but* turns nothing around: on the held-out sentences with *but*, that comes out at 38 right against 36. The next lesson, *A list of good and bad words*, gives each word a score from how often it occurs in positive and in negative reviews, and turns a score around after a negation.

    **An everyday example.** A reference letter that calls someone *a pleasant colleague, but often late* is read as a letter about being late, however many kind sentences came before. Readers of such a letter weigh the sentences: they do not count them.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_majority, result_scales, result_sort):
    L.wrap_up(result_majority, result_scales, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
