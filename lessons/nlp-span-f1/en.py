# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-span-f1. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-span-f1", "en")
    C = L.checks   # the tags and names of the test set, five taggers, the course's scores, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en) and [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_woodward():
    quick_woodward = L.choice(
        "quick-woodward",
        question="In *Finding names*, the tagger gave *Bob Woodward* the tags `B-PER I-PER`, although the training set never has *Woodward*. Where did the `I-PER` come from?",
        options={
            "gazetteer": "From the gazetteer, which has *Bob Woodward* as a person",
            "word": "From the table of words: *Woodward* after `B-PER`",
            "capital": "From a rule that people wrote: a word with a capital after a first name is a surname",
            "right": "From the table of word shapes: `Xx` after `B-PER`",
        },
        correct="right",
        why={
            "right": "The table of words has no pair for a word the training set never has, so the tagger looked up the short word shape: `Xx` after `B-PER` gets `I-PER`. That lesson counted the names right, the wrong finds and the missed names. This lesson turns those counts into a score.",
            "gazetteer": "The gazetteer has only names of the training set, and *Woodward* is not in it. Which table did the tagger use for a word it never saw? See [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).",
            "word": "The table of words has only pairs from the training set, and the training set never has *Woodward*. Where did the tagger look next? See [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).",
            "capital": "The tagger of that lesson has no rules that people wrote: it counts tags in the training set. Which of its two tables knows something about a word it never saw? See [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).",
        },
    )
    quick_woodward
    return (quick_woodward,)


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="A detector has a precision of 1.0 and a recall of 0.1. What is its F1 score?",
        options={
            "mean": "0.55, halfway between the two numbers",
            "low": "0.1, the smaller of the two",
            "right": "About 0.18",
            "high": "1.0, the larger of the two",
        },
        correct="right",
        why={
            "right": "F1 is the harmonic mean: 2 · 1.0 · 0.1 / (1.0 + 0.1) = 0.2 / 1.1, about 0.18. It stays close to the smaller of the two, so one high number alone does not give a high F1 score. This lesson uses it on names.",
            "mean": "Halfway is the plain mean. F1 stays close to the smaller of the two numbers. Which mean does it use? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "low": "F1 stays close to the smaller number, but not on it: a precision of 1.0 still adds something. What does the formula give? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "high": "F1 does not reward one high number: it stays close to the smaller of the two. What does the formula give? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A tagger comes with a score, and the score decides which tagger gets used. The same tagger can look almost perfect or rather poor, depending on what you count. Jurafsky and Martin: "Part-of-speech taggers are evaluated by the standard metric of accuracy. Named entity recognizers are evaluated by recall, precision, and F1 measure." This lesson shows why names need the second way: most words of a text are not part of any name, so a score per word mostly measures how well a tagger leaves words alone.

    **An everyday example.** A smoke alarm that never goes off is right on almost every day of the year, since most days have no fire. Still, nobody wants it: what counts are the few days with smoke, and whether it went off on those days and only on those.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="The test set of Universal NER has 1,471 sentences with 18,990 tokens, and 802 names. A tagger gives every token the tag `O`, so it finds no name at all. What share of the tokens does it tag right?",
        options={
            "coin": "About 50 of every 100, as with tossing a coin",
            "quarter": "About 75 of every 100",
            "right": "About 94 of every 100",
            "almost": "About 99 of every 100",
        },
        reveal="**About 94:** 17,758 of the 18,990 tokens have the tag `O`, so the tagger that never finds a name gets 93.5 of every 100 tokens right. Only 1,232 tokens, about 6 in 100, are part of a name, and this tagger finds none of the 802 names.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Accuracy per token

    The **accuracy per token** of a tagger is the share of all tokens whose tag is the tag people gave. For word classes it fits: every token has a word class, and every tag matters. For names, most of that share is `O`. The tagger that tags every token `O` gets 0.935 without finding a single name, and the tagger of [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en) gets 0.958. By accuracy, finding 343 names adds 0.023.

    ### The name as the unit

    Jurafsky and Martin: "For named entity tagging, the entity rather than the word is the unit of response." So count names, as [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) counted digits. Call each span that a tagger marks a **find**. A find that has the same start, the same end and the same type as a name people marked is an **exact match**, a true positive. Any other find is a false positive. A name that no find matches is a false negative. Precision is the share of the finds that are right, recall the share of the names that were found, and their F1 score is the **span F1**.

    Erik Tjong Kim Sang and Fien De Meulder ran the shared task of CoNLL-2003, in which sixteen systems were compared on the same English and German news. They wrote: "Precision is the percentage of named entities found by the learning system that are correct. Recall is the percentage of named entities present in the corpus that are found by the system. A named entity is correct only if it is an exact match of the corresponding entity in the data file."

    The tagger of *Finding names* makes 443 finds on the test set, 343 of them right, and the test set has 802 names. Its precision is 343 / 443 = 0.774, its recall 343 / 802 = 0.428, and its span F1 is 2 · 343 / (443 + 802) = 0.551. That is the formula in counts from *Precision and recall in one number*, 2 · tp / (2 · tp + fp + fn), since the finds (tp + fp) and the names (tp + fn) together make 443 + 802. The picture puts both scores of five taggers side by side.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    By accuracy, all five lie between 0.897 and 0.958; by span F1, between 0 and 0.551. The two scores do not even agree on the order. The **capitals rule** marks every word with a capital that is not the first word of its sentence as part of a person's name. It puts a wrong tag on many words, so its accuracy, 0.897, is lower than that of the tagger that finds nothing. But 146 of its finds are right, so its span F1 is 0.136, and not 0.

    ### One wrong tag, two errors

    Jurafsky and Martin: "The fact that named entity tagging has a segmentation component which is not present in tasks like text categorization or part-of-speech tagging causes some problems with evaluation. For example, a system that labeled Jane but not Jane Villanueva as a person would cause two errors". The picture shows their example sentence with a tagger that gives *Villanueva* the tag `O`.
    """)
    return


@app.cell(hide_code=True)
def slip_cell():
    mo.Html(C.slip_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        One tag of the eleven is wrong, so the accuracy is 0.909. Over names, the find *Jane* is not an exact match, so it is a false positive, and the name *Jane Villanueva* is not found, a false negative. They add one more mismatch: "using entities as the unit of response but words as the unit of training means that there is a mismatch between the training and test conditions." The tagger of *Finding names* learned tag by tag, and is scored name by name.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **accuracy per token** | the share of all tokens whose tag is the tag people gave |
        | **find** | a span that a tagger marks as a name, with its type |
        | **exact match** | a find with the same start, the same end and the same type as a name people marked |
        | **span F1** | precision, recall and their F1 score over names, where only an exact match counts as right |
        | **capitals rule** | a tagger that marks every word with a capital that is not the first word of its sentence as part of a person's name |
        | `C.GOLD` | the tags people gave in the test set: a list with a list of tags for each sentence |
        | `C.TAGGERS` | the tags of the five taggers on the test set, in a dict with the keys `"all_o"`, `"one_word"`, `"gazetteer"`, `"tagger"` and `"capitals"`; each value is built like `C.GOLD` |
        | `C.GOLD_SPANS`, `C.FOUND` | the same as spans: for each sentence a list of spans `(start, end, kind)`; `C.FOUND` holds the finds of each tagger, under the same keys |
        | `C.span_f1` | the course's version of Exercise 2 |
        | `gold[s][k]` | item `k` of item `s`: here the tag of token `k` in sentence `s` |
        """),
        L.watch_out("A score is only as good as the names people marked. In Universal NER, part of the sentences was marked by two people, and its makers, Mayhew and others, write that the two generally agree less on organizations and places than on persons, since some names can take either type: \"a restaurant is a LOC when you go there to eat, but it is an ORG when it hires a new chef.\" A tagger that gives such a name the other type counts as wrong."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Accuracy per token: go over all tokens of all sentences, count them, and count the tokens whose two tags are the same. Divide the second count by the first.
    2. Span F1: for each sentence, count the finds, the names, and the finds that are also a name in the same sentence. Add the counts up over the sentences. Precision is right finds / finds, recall right finds / names, and F1 is 2 · precision · recall / (precision + recall). When no find is right, all three are 0.
    3. Per type: keep, for each sentence, only the spans of one type, in the names and in the finds, and compute span F1 on what is left.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell counts the tokens of the test set and the tokens with the tag `O`, and prints both and their share. Then it counts the names people marked.
    """)
    return


@app.cell
def example_count():
    tokens = 0
    outside = 0
    for sentence_tags in C.GOLD:
        for tag in sentence_tags:
            tokens = tokens + 1
            if tag == "O":
                outside = outside + 1
    print(tokens, outside, outside / tokens)

    names = 0
    for sentence_spans in C.GOLD_SPANS:
        names = names + len(sentence_spans)
    print(names)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The share of tokens with `O` is the accuracy of the tagger that tags every token `O`. Exercise 1 compares two lists of tags instead: those people gave, and those of a tagger.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. token_accuracy: the share of right tags

        Write `token_accuracy(gold, guess)`: `gold` and `guess` each hold one list of tags for each sentence; the two lists of a sentence are equally long. Give back the share of all tokens of all sentences whose tag in `guess` is the tag in `gold`. Go over the sentences with a position `s`, and over the tokens of a sentence with a position `k`: `gold[s][k]` is the tag people gave token `k` of sentence `s`. Replace the `...`. The check calls `token_accuracy([["B-PER", "I-PER", "O"], ["O", "B-LOC"]], [["B-PER", "O", "O"], ["O", "B-LOC"]])`, which gives `0.8`; then cases of its own: every tag right, no tag right in two sentences, one token with another type, a sentence of nine tokens and one of two, and `C.GOLD` with `C.TAGGERS["all_o"]` and with `C.TAGGERS["tagger"]`.
        """),
        L.task("Give the share of tokens whose tag is right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("token_accuracy", lines=9)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("token_accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "token_accuracy",
        [
            "Two counters, `same` and `total`, both 0 at the start. A loop `for s in range(len(gold)):` with a loop `for k in range(len(gold[s])):` inside it. `total` goes up for every token, `same` only when the tag in `gold` and the tag in `guess` at that position are equal.",
            "Inside the inner loop, an `if` compares the two tags of token `k` in sentence `s`. After both loops, the share is the right tokens divided by all tokens.",
            "`            if gold[s][k] == ___:`",
        ],
        why="`same = 0` and `total = 0`, then `for s in range(len(gold)):` with `for k in range(len(gold[s])):` under it; inside it `if gold[s][k] == guess[s][k]:` with `same = same + 1` under it, and `total = total + 1`; after the loops `return same / total`. The example gives `0.8`.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracy():
    mo.md(r"""
    The picture under the check runs your function on the tags of the five taggers. A sentence of nine tokens weighs more than a sentence of two, since every token counts once. Exercise 2 counts names instead.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_f1():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. span_f1: precision, recall and F1 over names

        Write `span_f1(gold, found)`: `gold` holds the names people marked, and `found` the finds of a tagger, each as one list of spans `(start, end, kind)` for each sentence. Count the finds, the names, and the right finds: a find is right when it is one of the names of the same sentence, with the same start, end and type. Give back `(precision, recall, f1)`, and `(0.0, 0.0, 0.0)` when no find is right. A tagger finds each span once. Replace the `...`. The check calls `span_f1([[(0, 2, "PER"), (3, 5, "ORG"), (7, 9, "LOC")]], [[(0, 1, "PER"), (3, 5, "ORG")]])`: the names of *Jane Villanueva of United Airlines flew to San Diego .*, with *Jane* found as a person, *United Airlines* right and *San Diego* missed, which gives `(0.5, 0.3333333333333333, 0.4)`. Then cases of its own: nothing found, every name found in two sentences, the same words with another type, a find that matches a name of another sentence, more finds than names, a sentence with no names and no finds, and `C.GOLD_SPANS` with `C.FOUND["gazetteer"]` and with `C.FOUND["tagger"]`.
        """),
        L.task("Score the finds of a tagger by precision, recall and F1 over names."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("span_f1", lines=16)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("span_f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "span_f1",
        [
            "Three counters: `right`, `n_found` and `n_gold`. A loop `for s in range(len(gold)):`, and inside it a loop `for span in found[s]:` with an `if` that asks whether `span` is one of the names. After the inner loop, add the number of finds and the number of names of that sentence to the counters. Before you divide, handle `right == 0`.",
            "Which names may a find of sentence `s` be compared with: the names of all sentences, or only those of one sentence?",
            "`            if span in ___:`",
        ],
        why="`right = 0`, `n_found = 0` and `n_gold = 0`, then `for s in range(len(gold)):` with `for span in found[s]:` and `if span in gold[s]:` with `right = right + 1` under it; after that inner loop `n_found = n_found + len(found[s])` and `n_gold = n_gold + len(gold[s])`. After the loop, `if right == 0:` with `return (0.0, 0.0, 0.0)`; then `precision = right / n_found`, `recall = right / n_gold`, `f1 = 2 * precision * recall / (precision + recall)` and `return (precision, recall, f1)`. The example gives `(0.5, 0.3333333333333333, 0.4)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_f1():
    mo.md(r"""
    By span F1, the five taggers lie between 0 and 0.551 instead of within 0.061 of each other. Of the 100 wrong finds of the tagger, 26 have the words of a name with another type, 40 overlap a name without matching it, and 34 lie where no name is; it misses 459 names. Exercise 3 splits the score by type.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kind():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. of_kind: the names of one type

        Write `of_kind(per_sentence, kind)`: `per_sentence` holds one list of spans for each sentence, such as `C.GOLD_SPANS` or `C.FOUND["tagger"]`. Give back a list with, for each sentence, a list of only the spans of type `kind`, in their order. A sentence without such spans gets an empty list, so the lists stay in line with the sentences. Then `C.span_f1` can score one type: `C.span_f1(of_kind(C.GOLD_SPANS, "ORG"), of_kind(C.FOUND["tagger"], "ORG"))`. Replace the `...`. The check calls `of_kind([[(0, 2, "PER"), (3, 5, "ORG")], [(1, 2, "LOC")]], "PER")`, which gives `[[(0, 2, 'PER')], []]`; then cases of its own: a type that no span has, no sentences, two names of the type in one sentence with another type between them, a sentence without names between two others, and `C.GOLD_SPANS` with `"ORG"`.
        """),
        L.task("Keep, for each sentence, only the names of one type."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kind():
    box_kind = L.code_box("of_kind", lines=5)
    box_kind
    return (box_kind,)


@app.cell(hide_code=True)
def check_kind(box_kind):
    result_kind = L.check("of_kind", box_kind.value)
    result_kind
    return (result_kind,)


@app.cell(hide_code=True)
def hints_kind():
    L.hints(
        "of_kind",
        [
            "Begin with an empty list `out` and a loop `for sentence_spans in per_sentence:`. For each sentence, make a list of the spans whose type is `kind`, with a list comprehension that goes over each `span` of the sentence and has an `if` at the end, and append it to `out`, also when it is empty.",
            "A span is `(start, end, kind)`. At which position in the span does its type stand?",
            "`        out.append([span for span in sentence_spans if ___ == kind])`",
        ],
        why="`out = []`, then `for sentence_spans in per_sentence:` with `out.append([span for span in sentence_spans if span[2] == kind])` under it; after the loop `return out`. The example gives `[[(0, 2, 'PER')], []]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_kind():
    mo.md(r"""
    Per type, the two come out differently. The tagger is better at persons, 0.527 against 0.334 for the gazetteer: the tag before carries a name on into words it never saw. The gazetteer is better at organizations, 0.491 against 0.349: an organization it knows, it finds whole. Over all types, the two are close: 0.551 for the tagger, 0.521 for the gazetteer. Mayhew and others, who made Universal NER, give both kinds of numbers for a far larger model trained on the same English training set: a span F1 of 0.858 over all names on their test set, which also has the emails this course leaves out, and 0.950 for persons, 0.871 for places and 0.709 for organizations.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cut():
    mo.md(r"""
    ### Exercise 4. Half a name

    In *I have used Bright Futures for the last 7 years .*, people marked *Bright Futures* as an organization. Tagger A marks *Bright Futures for the* as an organization. Tagger B marks nothing in the sentence.
    """)
    return


@app.cell(hide_code=True)
def choice_cut_cell():
    choice_cut = L.choice(
        "near-miss",
        question="What does each add to the counts of span F1?",
        options={
            "same": "Both add one missed name, nothing more",
            "right": "A adds a wrong find and a missed name; B adds only a missed name",
            "half": "A gets half a right find, since its find holds both words of the name",
            "b": "B adds a missed name and a wrong find; A adds a missed name",
        },
        correct="right",
        why={
            "right": "The find of A is not an exact match, so it counts as a wrong find, and the name itself is still missed: precision and recall both drop. B only misses the name. So span F1 punishes a name cut wrong more than a name not found at all. For a reader of the result, *Bright Futures for the* is not the name of the organization.",
            "same": "That holds for B. But A also marks something that people did not mark as a name. What does that find count as?",
            "half": "Span F1 has no half finds: a find is right with the same start, end and type, and wrong otherwise. Which of the two is the find of A?",
            "b": "B marks nothing, so B has no finds to count. What does the find of A count as, beside the name that stays unfound?",
        },
    )
    choice_cut
    return (choice_cut,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: one sentence, per token and per name

    Choose a tagger and a sentence of the test set, and see its tags beside those people gave, and its finds beside the names. Try *Dr. White is the best !* with the tagger: one wrong tag, and two errors over names. Then choose *Every token O*, and the capitals rule.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_tagger = mo.ui.dropdown(options=C.tagger_options(L.lang), value=C.TAGGER_NAMES[L.lang]["tagger"], label="Tagger")
    demo_sentence = mo.ui.dropdown(options=C.sentence_options(), value=list(C.sentence_options())[1], label="Sentence")
    mo.hstack([demo_tagger, demo_sentence], justify="start", gap=1.5, wrap=True)
    return demo_sentence, demo_tagger


@app.cell(hide_code=True)
def demo_look(demo_sentence, demo_tagger):
    mo.Html(C.demo_picture(demo_tagger.value, demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en) counted the names right, the wrong finds and the missed names; here those counts became a score. Accuracy per token, the score of [Tagging word classes](/course/open/nlp-pos-tagging?lang=en), puts five taggers between 0.897 and 0.958, and gives the tagger that finds nothing 0.935. Span F1 uses the precision and recall of [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) with a name as the unit, right only as an exact match: the tagger gets 0.551, and a name cut wrong costs a wrong find and a missed name. Per type, the tagger is better at persons and the gazetteer at organizations. The next lesson, *Does one sentence follow from another?*, goes from names to the meaning of whole sentences.

    **An everyday example.** An envelope addressed to *Villanueba* has 9 of the 10 letters of *Villanueva* right, and it can still end up with the wrong person. For a name, only the whole name counts.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cut, result_accuracy, result_f1, result_kind):
    L.wrap_up(result_accuracy, result_f1, result_kind, choice_cut)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
