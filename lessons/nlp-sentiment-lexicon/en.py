# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-sentiment-lexicon. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sentiment-lexicon", "en")
    C = L.checks   # the review sentences in English and Dutch, the two lexicons, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Sentiment: positive or negative?](/course/open/nlp-sentiment?lang=en), and one about [WordNet: a network of meanings](/course/open/nlp-wordnet?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scales():
    quick_scales = L.choice(
        "quick-scales",
        question="In *Sentiment: positive or negative?*, each word gets a scale: 0.5 before the last *but*, and 1 from that *but* on. Which scales does *Nice but loud, but worth it.* give its six words?",
        options={
            "first": "0.5, 1, 1, 1, 1, 1",
            "right": "0.5, 0.5, 0.5, 1, 1, 1",
            "but": "0.5, 0.5, 0.5, 0.5, 1, 1",
            "after": "1, 1, 1, 1, 0.5, 0.5",
        },
        correct="right",
        why={
            "right": "The last *but* is the fourth word, so *nice*, the first *but* and *loud* get 0.5, and the last *but*, *worth* and *it* get 1. A scale changes how much a word counts. In this lesson, a word gets its score from a list, and the scores of the two words after *not* are turned around.",
            "first": "Half weight stops at the last *but*, not at the first. Where does the last *but* stand in this sentence? See [Sentiment: positive or negative?](/course/open/nlp-sentiment?lang=en).",
            "but": "The last *but* itself already counts in full. Which words come before it? See [Sentiment: positive or negative?](/course/open/nlp-sentiment?lang=en).",
            "after": "The words after *but* carry the judgment, so they keep their full weight. Which words get half? See [Sentiment: positive or negative?](/course/open/nlp-sentiment?lang=en).",
        },
    )
    quick_scales
    return (quick_scales,)


@app.cell(hide_code=True)
def quick_check_senses():
    quick_senses = L.choice(
        "quick-senses",
        question="In *WordNet: a network of meanings*, the adjective *ridiculous* stands in three synsets: `pathetic.s.03`, `absurd.s.02` and `farcical.s.01`. What does that tell you?",
        options={
            "hyper": "*ridiculous* has three hypernyms, each one step more general",
            "same": "Three words mean exactly what *ridiculous* means",
            "right": "*ridiculous* has three senses",
            "count": "*ridiculous* occurs three times in the texts WordNet was made from",
        },
        correct="right",
        why={
            "right": "A synset holds the words that share one sense, so a word in three synsets has three senses: pitiful, absurd and farcical. A list of good and bad words then has to choose. Pattern's English list, which this lesson uses, gives *ridiculous* a score for each sense: −1.0 for *inspiring scornful pity*, +1.0 for *broadly or extravagantly humorous* and −1.0 for *incongruous*.",
            "hyper": "Adjectives have no hypernyms in WordNet at all, and these three are the synsets *ridiculous* itself stands in. What does each synset of a word stand for? See [WordNet: a network of meanings](/course/open/nlp-wordnet?lang=en).",
            "same": "Each of the three synsets holds other words as well, such as *absurd* and *ludicrous*, and each one is different. What does each of the three stand for? See [WordNet: a network of meanings](/course/open/nlp-wordnet?lang=en).",
            "count": "WordNet does not count how often a word occurs: it lists the senses of words, and the words that share each sense. What does a word in three synsets have? See [WordNet: a network of meanings](/course/open/nlp-wordnet?lang=en).",
        },
    )
    quick_senses
    return (quick_senses,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The classifier of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) learned a weight for each of its words from 2,400 labeled sentences. Long before such classifiers, people wrote lists of words by hand. Jurafsky and Martin, in *Speech and Language Processing*, describe "an alternative model, in which instead of using every word as a feature, we focus only on certain words, ones that carry particularly strong cues to affect or sentiment. We call these lists of words affective lexicons or sentiment lexicons." The oldest they name, the General Inquirer of 1966, "has a lexicon of 1915 positive words and a lexicon of 2291 negative words".

    Such lists are still in use. Tausczik and Pennebaker (2010) describe LIWC as "a transparent text analysis program that counts words in psychologically meaningful categories", and write that its first goal was modest: "We simply wanted the computer to calculate the percentage of positive and negative emotion words within a text." LIWC is sold, with a license for academic research and one for everyone else. VADER (Hutto and Gilbert, 2014), made for social media, collected ratings for every word "from ten independent human raters", on a scale from −4 to +4, and kept "just over 7,500 lexical features"; it is free, under the MIT license. Pattern, a Python package from the University of Antwerp, has a list for English and one for Dutch, with a score for each sense of a word; this lesson uses the English one.

    A list needs no labeled sentences of your own, and you can read it: you see why a sentence gets its label. Jurafsky and Martin add when it helps most: "when training data is sparse or not representative of the test set, using dense lexicon features instead of sparse individual-word features may generalize better." This lesson builds a list from the labeled review sentences, compares it with Pattern's, and looks at a word that turns other words around: *not*.

    **An everyday example.** Looking at a used car, you scan the ad for words: *well maintained* and *one owner* are good signs, *sold as is* and *needs work* bad ones. You carry a list of good and bad words in your head, and you add up what you find.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_lexicon = L.guess(
        "guess",
        question="Count, for each word, how often it occurs in the positive and in the negative training sentences, and give it a score from the two counts. A sentence then gets the sum of the scores of its words: above 0 positive, below 0 negative. The classifier of *Logistic regression on words* gives 491 of the 600 held-out sentences their own label. How many does such a list get right?",
        options={
            "half": "About 300, hardly better than a coin",
            "fewer": "About 430: clearly fewer, since nothing is trained",
            "right": "About 490: as many as the classifier",
            "all": "About 560, since the words come from the same reviews",
        },
        reveal="**495: four more than the classifier.** No weights are learned together here: each word gets its score from its own two counts. The classifier weighs TF-IDF values and learns all its weights together, and still the two end up close. Pattern's English list, which started from its Dutch list and was partly scored again by hand, gets 365 right, and says nothing about 130 sentences, where the sum is 0: in 98 of those, not one word is on the list.",
    )
    guess_lexicon
    return (guess_lexicon,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **sentiment lexicon** is a list of words, each with a score: above 0 for positive, below 0 for negative. Hutto and Gilbert define it as "a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative". There are two ways to make one.

    - **By hand.** People give each word a score. For Pattern's Dutch list, De Smedt and Daelemans (2012) took the most frequent adjectives of 14,000 Dutch book reviews: "We took the top 1,100 most frequent adjectives, i.e. all adjectives that occurred more than four times." Seven people scored each sense of each adjective, from −1 to +1, and the list keeps the mean. Pattern calls that score the **polarity**. Later the list grew with adjectives whose score a program inferred: the file marks 1,595 of its 3,918 senses as scored by hand and 2,113 as inferred. The course gives a word the mean over its senses.
    - **From labeled reviews.** Count how often each word occurs in the positive and in the negative training sentences. Jurafsky and Martin: "positive words are more likely to appear in 5-star reviews; negative words in 1-star reviews." This lesson gives a word the score $\log_{10}\frac{p + 1}{n + 1}$, with $p$ its count in positive and $n$ its count in negative sentences. A word as frequent on both sides gets 0, a word that stands more often in positive sentences gets more than 0. The +1 keeps a word that occurs on one side only from a score without end.

    A sentence gets the sum of the scores of its words; a word the list does not have adds nothing. Above 0 is positive, below 0 negative, and at exactly 0 the list says nothing.

    Then **negation**: *not good* is not good. Pang, Lee and Vaithyanathan (2002) write: "clearly “good” and “not very good” indicate opposite sentiment orientations." Their fix: "we added the tag NOT_ to every word between a negation word (“not”, “isn’t”, “didn’t”, etc.) and the first punctuation mark following the negation word." VADER looks back from each word: "By examining the tri-gram preceding a sentiment-laden lexical feature, we catch nearly 90% of cases where negation flips the polarity of the text." For Dutch, De Smedt and Daelemans looked for *niet*, *nooit* and *geen*: "Reversing the polarity of the successive adjective can be used to raise precision by 2-3%."

    In this lesson, a negation turns the score of the next two words around: they count with the opposite sign. The negation word itself keeps its own score. The picture shows one held-out sentence in both lists.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Pattern's list knows one word of the sentence, *good*, and says positive; with the score of *good* turned around, it says negative, as the label of the sentence does. The list from the reviews says negative without any rule: *not* itself has a score of −0.64, since it occurs 189 times in negative sentences and 43 times in positive ones, and *money* has −0.74. With the rule, *good* and *for* count the other way around, and the sum goes from −0.81 to −1.88.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **sentiment lexicon** | a list of words, each with a score: above 0 for positive, below 0 for negative |
        | **polarity** | Pattern's name for the score of a word, from −1 (negative) to +1 (positive) |
        | **negation** | a word such as *not*, *no* or *never*, which turns around what the words after it say |
        | $\log_{10}\frac{p + 1}{n + 1}$ | the score of a word in the list from the reviews, with $p$ its count in positive and $n$ its count in negative training sentences |
        | `C.training(lang)` | the 2,400 training sentences in English (`"en"`) or Dutch (`"nl"`), as a list of `[label, words]`: the label 1 for positive and 0 for negative, the words in small letters |
        | `C.counts(lang)` | for each word of the training sentences, a list of its two counts: first in negative sentences, then in positive ones |
        | `C.lexicon(lang)` | the course's list from the training sentences: a dictionary with a score for each word |
        | `C.pattern(lang)` | Pattern's list: a dictionary with a polarity for each word, the mean over its senses, per word class first, as Pattern does |
        | `C.NEGATIONS[lang]` | the negation words of this lesson: in English *not*, *no*, *never* and words such as *didn't*; in Dutch *niet*, *geen*, *nooit*, *niets*, *niemand* and *nergens* |
        """),
        L.watch_out("Pattern's own code does more than this lesson. When it loads its Dutch list, it adds the form of each adjective that stands before a noun, such as *slechte* beside *slecht*; it takes a score after a negation times −0.5 instead of turning it around; and it lets words such as *very* make the next score stronger. The course looks words up as they are written, so its numbers for Pattern's lists differ from what Pattern itself would give."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Count, for each word, how often it occurs in negative and in positive training sentences: every time it occurs, also twice in one sentence.
    2. Give each word the score $\log_{10}\frac{p + 1}{n + 1}$.
    3. Give a sentence the sum of the scores of its words that the list has.
    4. With negation: the two words right after a negation count with the opposite sign.
    5. Positive above 0, negative below 0; at 0 the list says nothing. Count how many held-out sentences get their own label.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code takes the four words in `C.EXAMPLE_WORDS` for the lesson's language, and prints each with its two counts from the course's `counts`, negative first, and its score in the course's `lexicon`, rounded to three decimals.
    """)
    return


@app.cell
def example_cell():
    counts = C.counts(L.lang)
    scores = C.lexicon(L.lang)
    for word in C.EXAMPLE_WORDS[L.lang]:
        print(word, counts[word], round(scores[word], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *excellent* never occurs in a negative sentence. Without the +1, the fraction would divide by 0; with it, the score is $\log_{10}\frac{41 + 1}{0 + 1} = \log_{10} 42 = 1.623$. *the* occurs about as often on both sides, 747 and 807 times, so its score is close to 0: $\log_{10}\frac{808}{748} = 0.034$. *not* and *money* lean negative: $\log_{10}\frac{44}{190} = -0.635$ and $\log_{10}\frac{4}{22} = -0.740$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lexicon():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. lexicon: a list from labeled sentences

        Write `lexicon(sentences)`, where `sentences` is a list of sentences, each a pair `[label, words]`: the label, 1 for positive and 0 for negative, and the list of its words in small letters. Give back a dictionary with a score for each word: $\log_{10}\frac{p + 1}{n + 1}$, with $p$ its count in positive and $n$ its count in negative sentences. A word counts every time it occurs. You can use `np`. Replace the `...`. The check calls `lexicon([[1, ["great", "food"]], [0, ["bad", "food"]], [1, ["great", "service"]]])`, which gives `{"great": 0.477, "food": 0, "bad": -0.301, "service": 0.301}` (rounded here), and `lexicon([[0, ["bad", "bad", "food"]], [1, ["good", "food"]]])`, which gives `{"bad": -0.477, "food": 0, "good": 0.301}`. Then cases of its own: no sentences at all; one positive sentence without words; three negative sentences of one word, *slow*; one positive sentence with *nice* three times; and three Dutch sentences that start with a negative one.
        """),
        L.task("Build a list with a score for each word from labeled sentences."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lexicon():
    box_lexicon = L.code_box("lexicon", lines=16)
    box_lexicon
    return (box_lexicon,)


@app.cell(hide_code=True)
def check_lexicon(box_lexicon):
    result_lexicon = L.check("lexicon", box_lexicon.value)
    result_lexicon
    return (result_lexicon,)


@app.cell(hide_code=True)
def hints_lexicon():
    L.hints(
        "lexicon",
        [
            "Two parts. First the counts: a dictionary `counts`, empty at the start, a loop `for label, words in sentences:` and inside it `for word in words:`. Each word gets a list of two counts, `[0, 0]` the first time, and the label of the sentence is the place in that list to add 1 to. Then the scores: a dictionary `scores`, a loop `for word in counts:`, the count in positive sentences as `positive = counts[word][1]`, and `np.log10` of a fraction for `scores[word]`.",
            "Place 0 in the list of a word counts the sentences with the other label. Which count goes under the fraction, and what does the recipe add to it?",
            "`        scores[word] = np.log10((positive + 1) / (___ + 1))`",
        ],
        why="`counts = {}`; `for label, words in sentences:` with `for word in words:` under it, and in there `if word in counts:` with `counts[word][label] = counts[word][label] + 1`, and `else:` with `counts[word] = [0, 0]` and `counts[word][label] = 1`. Then `scores = {}`; `for word in counts:` with `negative = counts[word][0]`, `positive = counts[word][1]` and `scores[word] = np.log10((positive + 1) / (negative + 1))`; after the loop `return scores`. The examples give `{\"great\": 0.477, \"food\": 0, \"bad\": -0.301, \"service\": 0.301}` and `{\"bad\": -0.477, \"food\": 0, \"good\": 0.301}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_lexicon():
    mo.md(r"""
    The words at the two ends are words a person would also put on a list: *nice* and *excellent* at the top, *bad* and *poor* at the bottom. In between stand words nobody puts on a list by hand: *works* gets +1.16, from sentences such as *Works great.*, and *waste* −1.10. The list knows what these reviewers praise and complain about. Over the 600 held-out sentences, it gets 495 right, and the classifier of *Logistic regression on words* 491.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. score: the sum, with negation

        Write `score(words, lexicon, negations)`, where `words` is the list of words of a sentence, `lexicon` a dictionary with a score for each word, and `negations` the list of negation words. Give back the sum of the scores of the words that the lexicon has, where each of the two words right after a negation counts with the opposite sign: subtract its score instead of adding it. A negation word that the lexicon has keeps its own score. Replace the `...`. The check calls `score(["not", "good", "for", "the", "money"], {"not": -0.6, "good": 0.5, "money": -0.7}, ["not", "no", "never"])`, which gives `-1.8`: *not* adds −0.6, *good* comes right after it and subtracts 0.5, *for* is the second word after *not* but the lexicon does not have it, and *money* adds −0.7. And `score(["never", "bad", "and", "not", "slow"], {"bad": -0.7, "slow": -0.5, "good": 0.5}, ["not", "never"])`, which gives `1.2`. Then cases of its own: an empty list; one word without a negation; a word two places after a negation; a word three places after one; a negation with a score of its own; two negations in a row; Dutch negations; and a negation as the last word.
        """),
        L.task("Add up the scores, and turn around the two words after a negation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=14)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "A `total`, 0 at the start, and a counter `left`, also 0: how many words still count with the opposite sign. A loop `for word in words:`; for a word the lexicon has, an `if` on `left` that subtracts its score from `total` or adds it. Then `left` goes down by 1, and an `if word in negations:` sets `left` again. After the loop, `return total`.",
            "The word right after a negation must see `left` above 0, and so must the word after that, but not the third. `left` goes down by 1 after every word, and a negation sets it after that. To which number should a negation set it?",
            "`            left = ___`",
        ],
        why="`total = 0` and `left = 0`; `for word in words:` with `if word in lexicon:` under it, and in there `if left > 0:` with `total = total - lexicon[word]`, and `else:` with `total = total + lexicon[word]`. Then, still in the loop, `left = left - 1`, and `if word in negations:` with `left = 2`; after the loop `return total`. The examples give `-1.8` and `1.2`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    With negation, Pattern's list goes from 365 to 381 right: *Not impressed.*, for one, now comes out negative. The list from the reviews goes from 495 to 494: the rule fixes 14 sentences and breaks 15. One it breaks is *THAT one didn't work either.* In the list from the reviews, *work* already has a negative score, since it often stands in the two words after a negation (19 of its 47 times), and *either* has a negative score too. Turning them around counts the negation twice. A list learned from the reviews has the negation in it already; a list made by hand does not.

    And turning a score around is not always what a negation does. In *Do not waste your time.*, *not* does belong to *waste*, but the sentence is a warning, so still negative: with the rule, Pattern's list makes *waste* count as positive.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 3. What do the two lists say?

    Eight words from the review sentences. Put each one where it belongs: do Pattern's list and the list from the reviews give it a score with the same sign, does only one of the two lists have it, or do they give it scores with opposite signs?
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_lists = L.sort("words-lists")
    sort_lists
    return (sort_lists,)


@app.cell(hide_code=True)
def check_sort(sort_lists):
    result_sort = L.check("words-lists", sort_lists)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    Pattern's English list holds adjectives: 2,643 of its 2,918 senses are tagged as adjectives. A list from the reviews also holds the nouns and verbs that people praise or complain about, but only for the kind of reviews it learned from. Turney (2002) gave an example: "the adjective “unpredictable” may have a negative orientation in an automotive review, in a phrase such as “unpredictable steering”, but it could have a positive orientation in a movie review, in a phrase such as “unpredictable plot”." Jurafsky and Martin note about lexicons: "the affective meaning of a word is generally fixed, irrespective of the linguistic context in which a word is used". So *impressed* gets +1.0 in Pattern's list and −0.11 from the reviews, where it stands in the two words after a negation 7 of its 14 times, as in *Not impressed.*
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: a sentence of your own

    Write a sentence in English. The picture gives each word its score in both lists, and the sum without and with negation. The box starts with *Not bad at all, and the screen is great.*: without negation, the list from the reviews calls it negative, because *not* and *bad* both pull down. Try a negation such as *never*, and a sentence where *not* belongs to a word further on, such as *I would not say the sound is bad.*
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.text_area(value=C.TRY_START[L.lang], label="Your sentence", rows=2, full_width=True)
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

    **How this connects.** [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) learned all its weights together. A sentiment lexicon gives each word a score of its own: by hand, as in Pattern's lists, or from counts in labeled reviews, as your `lexicon` does. A sum of scores, each from one word's own two counts, does as well as the classifier on these reviews, and Pattern's list, made without these sentences, does much worse. A negation turns the words after it around: a window of two words helps a list made by hand, and hardly matters for a list that learned *not* from the reviews. The next lesson, *Project: sentiment in English and Dutch*, puts your own classifier and your own lexicon side by side, in both languages, with macro F1 and the reviews each one gets wrong.

    **An everyday example.** *Cold* is praise for a beer and a complaint about soup. A list of good and bad words for a café would need both: one for the drinks and one for the kitchen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_lexicon, result_score, result_sort):
    L.wrap_up(result_lexicon, result_score, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
