# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-pos-tagging. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-pos-tagging", "en")
    C = L.checks   # the two treebanks with their training and test sets, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en) and [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_spans():
    quick_spans = L.choice(
        "quick-spans",
        question="Tags for names have a `B-` and an `I-`, such as `B-LOC` and `I-LOC`. The tags for parts of speech have neither. Why can they do without?",
        options={
            "fewer": "There are fewer parts of speech than types of names",
            "right": "A part of speech belongs to one token, so no span needs marking",
            "neighbors": "Two words of the same part of speech never stand next to each other, so nothing has to be kept apart",
            "english": "B and I only work for English",
        },
        correct="right",
        why={
            "right": "Each token gets its own part of speech, and no tag stretches over several tokens. Jurafsky and Martin put it this way: \"Unlike part-of-speech tagging, where there is no segmentation problem since each word gets one tag, the task of named entity recognition is to find and label spans of text\". This lesson builds a tagger for those tags, one per token.",
            "fewer": "Universal Dependencies has 17 tags for parts of speech, and Universal NER three types of names: more parts of speech, not fewer. What does a B mark in a name of two tokens, and does a part of speech ever stretch over two tokens? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
            "neighbors": "They often do: in *a big red car*, *big* and *red* are both `ADJ`. But each keeps its own tag, so nothing has to say where one ends. What does a B mark in a name? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
            "english": "Universal NER marks names with B and I in 13 languages. The difference lies in what a tag belongs to: a whole name, or one token. See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
        },
    )
    quick_spans
    return (quick_spans,)


@app.cell(hide_code=True)
def quick_check_upos():
    quick_upos = L.choice(
        "quick-upos",
        question="In a CoNLL-U file of Universal Dependencies, every word has a line with columns. Which column holds its part of speech?",
        options={
            "form": "`FORM`",
            "right": "`UPOS`",
            "head": "`HEAD`",
            "deprel": "`DEPREL`, the relation",
        },
        correct="right",
        why={
            "right": "`UPOS`, the universal part-of-speech tag, such as `NOUN` or `VERB`. People filled in that column for every word of a treebank; a tagger has to fill it in for a sentence that nobody tagged.",
            "form": "`FORM` holds the word itself, as it stands in the sentence. The tag of the word is in another column. See [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
            "head": "`HEAD` holds the number of the word's head, with 0 for the root. Which column holds tags such as `NOUN` and `VERB`? See [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
            "deprel": "`DEPREL` holds the relation to the head, such as `nsubj` or `det`. Which column holds tags such as `NOUN` and `VERB`? See [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
        },
    )
    quick_upos
    return (quick_upos,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The dependency trees of [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en) come from a treebank, where people gave every word its tag and its head. For a new sentence, nobody has done that, and a program that draws the tree often uses the parts of speech. In the words of Jurafsky and Martin: "Knowing whether a word is a noun or a verb tells us about likely neighboring words (nouns in English are preceded by determiners and adjectives, verbs by nouns) and syntactic structure (verbs have dependency links to nouns), making part-of-speech tagging a key aspect of parsing." Finding names, the next lesson, also starts from the words and their tags. This lesson builds a tagger by counting, and measures how good it is.

    **An everyday example.** Read *I will record a record* aloud. You say the verb *reCORD* and the noun *REcord* differently, so before you say the word, you have to know which one it is. The word *will* before the first one tells you.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="The English treebank EWT has a training set of 8,774 sentences and a test set of 1,471 other sentences. A tagger counts, for every word of the training set, which tag it has most often, and gives that word that tag every time; a word it never saw gets `NOUN`. How many of every 100 words of the test set get the tag that people gave them?",
        options={
            "half": "About 55: half the words can have more than one tag",
            "seventy": "About 70",
            "right": "About 86",
            "all": "About 99: most words have only one tag",
        },
        reveal="**About 86:** 16,374 of the 18,987 words of the test set, 86.2 in 100. Counting alone gets most words right. The rest of the lesson looks at where it goes wrong: at words the training set does not have, and at words with more than one tag.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Part-of-speech tagging

    Jurafsky and Martin: "Part-of-speech tagging is the process of assigning a part-of-speech to each word in a text." It is sequence labeling with one tag per token, as in [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en), and the tags are the 17 of Universal Dependencies from [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en). The hard part is that the same word can have more than one tag: the goal, they write, "is to find the correct tag for the situation". A **tagger** is a program that does this.

    ### Learn from one part, measure on another

    A treebank of Universal Dependencies comes in parts. A tagger learns only from the **training set**, and is measured on the **test set**: sentences it never saw, as a new text would be. The English treebank EWT has a training set of 8,774 sentences with 158,323 words and a test set of 1,471 sentences with 18,987 words; the Dutch treebank Alpino has 12,289 sentences with 186,026 words, and 596 sentences with 11,046 words. Each also has a third part, which this lesson leaves alone.

    The measure is **accuracy**: the share of the words of the test set that get the tag people gave them in the treebank. Jurafsky and Martin call it "the percentage of test set tags that match human gold labels". This lesson writes it as a number between 0 and 1, and in the text often as a number of every 100.

    ### The first tagger: the most frequent tag

    For each word, count in the training set how often it has each tag, and keep the tag it has most often. The tagger gives the word that tag every time, wherever it stands. A word that the training set does not have, an **unknown word**, gets `NOUN`, the most frequent tag of all: 26,562 of the 158,323 English words, about 1 in 6. The words stay as they are written, with their capitals.

    Such a simple method is a **baseline**: a better method has to beat it before you trust its score. Jurafsky and Martin give it as a rule for any classifier, a program that gives each item one of a fixed set of labels, as a tagger does with words: "Always compare a classifier against a baseline at least as good as the most frequent class baseline (assigning each token to the class it occurred in most often in the training set)."
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_hard():
    mo.md(r"""
    *Friendliest* is not in the training set, so it gets `NOUN`. *have* is in it, as a verb more often than as an auxiliary, so the first tagger gives it `VERB`, also here, after *I*.

    ### Where the first tagger goes wrong

    Of the 16,599 different words in the English training set, 1,596 have more than one tag: about 1 in 10. But they are common words, such as *that*, *to* and *have*, and they make up 80,783 of the 158,323 words of running text: half. Jurafsky and Martin found the same for an older English tag set with 45 tags: "most word types (85-86%) are unambiguous", and yet "55-67% of word tokens in running text are ambiguous". In the Dutch training set, only 765 of the 26,665 different words have more than one tag, but they too make up about half of the text.

    ### The words around decide

    Jurafsky and Martin write that "knowing the previous or following words might be a useful feature (if the previous word is *the*, the current tag is unlikely to be a verb)." Take *to*. After *want* it is a `PART`, as in *want to go*; after *go* it is an `ADP`, as in *go to school*. A tagger can count tags per **pair** of the word before and the word, just as [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) counted pairs of characters, and give a word the most frequent tag of its pair.
    """)
    return


@app.cell(hide_code=True)
def pairs_cell():
    mo.Html(C.pairs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        Many pairs never occur in the training set: the test set has new combinations of words. Then the tagger falls back to the word alone, and for an unknown word to `NOUN`. Taggers that look at much more, such as the tags they gave the words before and the letters at the end of a word, get about 97 of every 100 words right. In the words of Jurafsky and Martin: "Accuracies on various English treebanks are also 97% (no matter the algorithm; HMMs, CRFs, BERT perform similarly). This 97% number is also about the human performance on this task, at least for English (Manning, 2011)." On news text of the *Wall Street Journal*, they give the first tagger "an accuracy of about 92%". Later lessons build such taggers.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **part-of-speech tagging** | giving every word of a text its part of speech, as one tag per token |
        | **tagger** | a program that gives every token a tag |
        | **training set** | the part of a treebank a tagger learns from |
        | **test set** | the part a tagger is measured on: sentences it never saw |
        | **accuracy** | the share of words that get the tag of the treebank, between 0 and 1 |
        | **baseline** | a simple method that a better one has to beat, such as the most frequent tag of each word |
        | **unknown word** | a word that the training set does not have |
        | `C.TRAIN_EN`, `C.TEST_EN` | the training set and the test set of EWT: tuples of sentences, each a tuple of pairs `(word, tag)` |
        | `C.TRAIN_NL`, `C.TEST_NL` | the same for Alpino |
        | `C.BEST_EN`, `C.BEST_NL` | the course's table of the most frequent tag of each word of a training set; you can read it, not change it |
        | `C.PAIRS_EN`, `C.PAIRS_NL` | the course's table of the most frequent tag of each pair `(word before, word)` of a training set; you can read it, not change it |
        | `C.SAMPLE`, `C.SAMPLE_BEST` | two short sentences with their tags, and a small table of most frequent tags, for Exercise 2 |
        | a tuple as a key | a pair of words can be a key of a dict, as a string can: `pairs[("go", "to")]` |
        """),
        L.watch_out("The lesson about parts of speech put every word in small letters. The tagger here keeps the capitals, and that matters: in the English training set, *US* is a `PROPN` 149 times, and *us* a `PRON` 105 times. A word with a capital at the start of a sentence can be unknown even when the training set has it in small letters."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Count: for each word of the training set, how often it has each tag.
    2. Choose: for each word, keep the tag with the highest count.
    3. Measure: tag every word of the test set, with `NOUN` for an unknown word, and give back the share that gets the tag of the treebank.
    4. Look at the word before: when the training set has the pair of the word before and the word, the pair decides; otherwise the word alone; otherwise `NOUN`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell does step 1 of the recipe for both training sets, and prints the counts of *that* in EWT and of *dat* in Alpino.
    """)
    return


@app.cell
def example_counts():
    def tag_counts(sentences):
        counts = {}
        for sentence in sentences:
            for word, tag in sentence:
                if word not in counts:
                    counts[word] = {}
                if tag not in counts[word]:
                    counts[word][tag] = 0
                counts[word][tag] = counts[word][tag] + 1
        return counts


    counts_en = tag_counts(C.TRAIN_EN)
    counts_nl = tag_counts(C.TRAIN_NL)
    print(counts_en["that"])
    print(counts_nl["dat"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *that* has five tags in the training set: `SCONJ` 733 times, as in *I think that it works*, `PRON` 641 times, as in *that is true*, `DET` 123 times, as in *that book*, and two more that are rare. Dutch *dat* has three, and `SCONJ` is the most frequent there too. Exercise 1 does step 2 for every word.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. best_tags: the most frequent tag of each word

        Write `best_tags(counts)`. `counts` is a table of the kind that `tag_counts` of the worked example makes: a dict with for each word a dict of its tags and how often the word has each. Give back a dict with for each word the tag it has most often. When two tags of a word tie, either one is fine. Replace the `...`. The check calls `best_tags({"that": {"PRON": 641, "SCONJ": 733, "DET": 123}, "book": {"VERB": 6, "NOUN": 18}})`, which gives `{'that': 'SCONJ', 'book': 'NOUN'}`; then cases of its own: no words, one word with one tag, the most frequent tag first, and two words where the second has lower counts than the first.
        """),
        L.task("For each word, find the tag with the highest count."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best_tags", lines=10)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best_tags", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best_tags",
        [
            "Two loops, as in the worked example: one over the words of `counts`, and inside it one over the tags of `counts[word]`. For each word, keep the best tag so far and its count in two names of your own, set before the inner loop; after that loop, the table gets the word with its best tag.",
            "A tag becomes the new best tag of its word when its count is higher than something. Higher than what, and what changes when it is?",
            "`            if counts[word][tag] > ___:`",
        ],
        why="`best = {}`, then `for word in counts:` with `top = \"\"` and `most = 0` under it, and `for tag in counts[word]:` with `if counts[word][tag] > most:` under that; under the `if` come `most = counts[word][tag]` and `top = tag`. After the inner loop `best[word] = top`, and after the outer loop `return best`. The example gives `{'that': 'SCONJ', 'book': 'NOUN'}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_best():
    mo.md(r"""
    The picture under the check runs your function on the counts of both training sets. Look at *like*: `ADP` wins with 137, so the first tagger calls *like* a preposition every time, also in *I like it*, where it is a verb. Exercise 2 measures how often the most frequent tag is right.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. accuracy: how many words are right?

        Write `accuracy(sentences, best)`. `sentences` holds sentences, each a tuple of pairs `(word, tag)` with the tag of the treebank, and `best` is a table of most frequent tags of the kind that `best_tags` makes. For each word, the guess of the first tagger is its tag in `best`, or `"NOUN"` when `best` does not have the word. Give back the share of all words whose guess is the tag of the treebank: a number between 0 and 1. Replace the `...`. The check calls `accuracy(C.SAMPLE, C.SAMPLE_BEST)`: the two sentences *Book that flight.* and *Hand me that book.*, and a small table that does not have *Hand* and gives *that* the tag `SCONJ`. That gives `0.6666666666666666`, 6 of the 9 tokens; then cases of its own: an empty table, a word in the table, an unknown word that is a noun, two sentences of different length, and both whole test sets with the tables of the course.
        """),
        L.task("Guess the tag of every word with the table, and give back the share of guesses that are right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("accuracy", lines=12)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Count in `right` the words whose guess is the tag of the treebank, and count every word in a second name. Two loops, as in the worked example: over the sentences, and over the pairs `(word, tag)` of each sentence. At the end, the function gives back a share.",
            "What is the guess for a word that `best` does not have? And which number counts all the words, the wrong ones too?",
            "`    return right / ___`",
        ],
        why="`right = 0` and `total = 0`, then `for sentence in sentences:` and `for word, tag in sentence:`; under that, `if word in best:` with `guess = best[word]`, and `else:` with `guess = \"NOUN\"`; then `if guess == tag:` with `right = right + 1`, and `total = total + 1` for every word. After the loops `return right / total`. The example gives `0.6666666666666666`.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracy():
    mo.md(r"""
    In the picture, the first tagger gets 86.2 of every 100 words of the English test set right, and 88.1 of the Dutch one. The table splits the English words into three groups. Words with one tag in the training set are almost always right. Words with more than one tag go wrong where their most frequent tag does not fit. And of the unknown words, only a third get the right tag: they are 1 in 11 words, but 1,178 of the 2,613 mistakes. Most of those are names, `PROPN`, which the tagger calls `NOUN`. Jurafsky and Martin mention a clue a tagger can use there: "words starting with capital letters are likely to be proper nouns".

    Accuracy counts words, not sentences. Only 404 of the 1,471 sentences of the English test set have every word right.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tag():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. tag_words: the word before decides

        Write `tag_words(words, best, pairs)`: give back a list with one tag for each word of the list `words`. `best` is a table of most frequent tags as before, and `pairs` is a table with a tag for each pair `(word before, word)`, made by counting in the same way: its keys are tuples of two words. For each word, take the tag of its pair with the word before it when `pairs` has that pair; otherwise its tag in `best`; and otherwise `"NOUN"`. The first word has no word before it. Replace the `...`. The check calls `tag_words(["go", "to", "school"], {"go": "VERB", "to": "PART"}, {("go", "to"): "ADP"})`, which gives `['VERB', 'ADP', 'NOUN']`; then cases of its own: *to* as the first word, a pair for the word after *to*, no words, an unknown word, a pair for a word that `best` does not have, and a sentence of each treebank with the tables of the course, `C.BEST_EN` with `C.PAIRS_EN` and `C.BEST_NL` with `C.PAIRS_NL`.
        """),
        L.task("Tag each word with its pair when you can, else with the word alone, else with NOUN."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tag():
    box_tag = L.code_box("tag_words", lines=10)
    box_tag
    return (box_tag,)


@app.cell(hide_code=True)
def check_tag(box_tag):
    result_tag = L.check("tag_words", box_tag.value)
    result_tag
    return (result_tag,)


@app.cell(hide_code=True)
def hints_tag():
    L.hints(
        "tag_words",
        [
            "One loop over the positions `i` of `words`, with `word = words[i]`, and three branches: `if`, `elif` and `else`. The first branch asks whether the pair `(words[i - 1], word)` is in `pairs`, the second whether `word` is in `best`.",
            "Which word has no word before it? And what does `words[i - 1]` give for that word in Python?",
            "`        if ___ and (words[i - 1], word) in pairs:`",
        ],
        why="`tags = []`, then `for i in range(len(words)):` with `word = words[i]`; under `if i > 0 and (words[i - 1], word) in pairs:` comes `tags.append(pairs[(words[i - 1], word)])`, under `elif word in best:` comes `tags.append(best[word])`, and under `else:` comes `tags.append(\"NOUN\")`. After the loop `return tags`. The example gives `['VERB', 'ADP', 'NOUN']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tag():
    mo.md(r"""
    In the picture, the word before turns *that* in *Is that a money maker?* into a pronoun, and *have* in *I should have asked* into an auxiliary. On the whole English test set, the tagger with pairs gets 88.0 of every 100 words right, and on the words with more than one tag 90.3 instead of 86.6. It cannot help the unknown words: their pairs are not in the training set either.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: your own sentence

    Type a sentence and see the tags of both taggers, with the tables of EWT. Put a space between the words and before punctuation, as the treebank does. Try *I want to go to the store .* and *I like it .*, and then *Book that flight .*: which tags go wrong there, and what in the training set could cause that?
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value=C.DEMO_TEXT[L.lang], label="A sentence", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Exercise 4. Which neighbor decides?

    *to* is a `PART` in *I want to go*, and an `ADP` in *I went to the store*. In the English test set, *to* occurs 413 times. The first tagger gets it right 260 times, and your tagger with the word before 321 times.
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after",
        question="Which other word would most likely help a tagger more to choose between `PART` and `ADP` for *to*?",
        options={
            "first": "The first word of the sentence",
            "right": "The word after *to*",
            "two_before": "The word two places before *to*, since the word right before it says too little",
            "capital": "Whether *to* starts with a capital",
        },
        correct="right",
        why={
            "right": "After *to* comes a verb, as in *to go*, or a noun phrase, as in *to the store*, and that tells the two apart. A tagger that counts pairs of a word and the word after it gets *to* right 347 of the 413 times. Jurafsky and Martin mention both neighbors: \"the previous or following words\". Taggers that look at both sides, and at the tags they choose, come in later lessons.",
            "first": "The first word of a sentence is often far from *to*, and says little about it. In *I want to go* and *I went to the store*, the sentences start the same. Which word right next to *to* differs between the two?",
            "two_before": "Two places back, *I want to go* and *I went to the store* both have *I*. What comes right after *to* in each?",
            "capital": "*to* almost never has a capital, and when it does, at the start of a sentence, it can still be either. Which word next to *to* differs between *to go* and *to the store*?",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en) showed that the same word can have more than one tag, and [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en) called giving every token a label sequence labeling. Here you built a tagger by counting: the most frequent tag of each word gets 86 of every 100 English words right, measured on a test set it never saw. The mistakes fall on unknown words and on words with more than one tag, and the word before helps with the second kind, as the counts of pairs did in [A bigram model by counting](/course/open/lm-bigram-counts?lang=en). The next lesson, *Finding names: named entity recognition*, tags names with the B, I and O of the lesson before, with rules, lists and a tagger that learns.

    **An everyday example.** Someone who has only met *record* as a noun reads *I will record it* with the wrong stress at first. The *will* before it is the clue that it is a verb this time.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_after, result_accuracy, result_best, result_tag):
    L.wrap_up(result_best, result_accuracy, result_tag, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
