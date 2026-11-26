# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-rouge. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-rouge", "en")
    C = L.checks   # the course's six abstracts and Qwen3-0.6B's summaries, ROUGE, SciTLDR, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en), and one about [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bleu():
    quick_bleu = L.choice(
        "quick-bleu",
        question="In *Scoring a translation: BLEU and chrF*, `clipped` counts the n-grams of a translation that the reference has, each at most as often as the reference has it. The translation *the cat the cat* has the reference *the cat sat*: its clipped count of unigrams is 2, one *the* and one *cat*. What is BLEU's precision of unigrams, $p_1$?",
        options={
            "three": "2 / 3: the reference has 3 unigrams",
            "right": "2 / 4",
            "four": "4 / 4: each word of the translation is also in the reference",
            "seven": "2 / 7: the words of both sentences together",
        },
        correct="right",
        why={
            "right": "The translation has 4 unigrams, and 2 of them count: $p_1$ = 2 / 4 = 0.5, a precision, the share of the translation that the reference has. ROUGE, in this lesson, divides the same count by the unigrams of the reference instead: 2 / 3, a recall.",
            "three": "2 / 3 divides by the reference: the share of the reference that comes back in the translation, a recall. This lesson's ROUGE does exactly that. BLEU divides by the unigrams of the translation: how many does it have? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
            "four": "4 / 4 counts each word as often as the translation has it, without clipping: *the* twice, while the reference has it once. How often may *the* count? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
            "seven": "BLEU never adds the two sentences together. It divides the clipped count by the unigrams of one of the two: which one? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
        },
    )
    quick_bleu
    return (quick_bleu,)


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="In *How far apart are two words? Edit distance*, the table `D` has a row for each first part of `a` and a column for each first part of `b`, from no letters to all of them. What does the cell `D[i, j]` hold?",
        options={
            "letters": "Whether letter `i` of `a` is the same as letter `j` of `b`",
            "right": "The edit distance between the first `i` letters of `a` and the first `j` letters of `b`",
            "whole": "The edit distance between all of `a` and all of `b`, written again in every cell of the table",
            "plus": "`i + j`, the number of letters of the two first parts together",
        },
        correct="right",
        why={
            "right": "Each cell answers the question for two first parts, and each new cell builds on cells that are already filled: the answer for all of `a` and all of `b` ends up in the last cell. The longest common subsequence of this lesson fills a table in the same way.",
            "letters": "Whether two letters are the same decides only what a substitution costs at that cell. What does the cell itself hold, for the first parts of `a` and `b`? See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
            "whole": "The edit distance of all of `a` and all of `b` is in one cell only: the last one. What do the cells on the way there hold? See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
            "plus": "`i + j` is an upper limit of the cell: deleting all `i` letters and inserting all `j`. What does the cell hold when some letters are shared? See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    *Summarizing: pick or write* left one question open: how do you score a summary? People can read each one, but that takes time for every summary and for every new version of a summarizer. In 2004, Chin-Yew Lin of the Information Sciences Institute of the University of Southern California published a package of measures that needs no person at all, once people have written a few reference summaries: "ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation." Lin: "The measures count the number of overlapping units such as n-gram, word sequences, and word pairs between the computer-generated summary to be evaluated and the ideal summaries created by humans." SciTLDR chose its oracle sentences with ROUGE.

    In this lesson you build the three main parts of ROUGE, score Qwen3-0.6B's summaries and four kinds of summary of 618 real papers, and find out what ROUGE cannot see.

    **An everyday example.** A teacher grades a short answer by ticking off the key words of a model answer. An answer with all the key words gets full marks, even when it says something else; an answer that says the same in other words loses marks.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_longer = L.guess(
        "guess",
        question="SciTLDR has, for each of its 618 papers, the TLDR of its authors and one or more TLDRs that experts wrote from the peer reviews. Score the authors' TLDR against each TLDR of the experts by the share of its words that come back, and keep the best share of the paper. Do the same for the one sentence that the title method of *Summarizing: pick or write* picks from the abstract. Which gets the higher share, on average?",
        options={
            "far": "The authors' TLDR, by far: it was written as a TLDR, just as those of the experts were",
            "little": "The authors' TLDR, by a few points",
            "right": "The sentence picked from the abstract",
            "same": "Both get exactly the same share",
        },
        reveal="The picked sentence: on average 31.7% of the words of the best-matching TLDR of the experts come back in it, against 28.7% in the authors' TLDR, over the 618 papers. The picked sentence is longer: 25.5 words on average, against 19.6 for the authors' TLDR and 22.4 for a TLDR of the experts. A longer summary has more chances to hold a word of the reference, and a score that only counts how much of the reference comes back does not look at what the summary adds: the longest sentence of each abstract, 35.6 words on average, gets 31.1%. The demo further on in this lesson shows a score that does look at it.",
    )
    guess_longer
    return (guess_longer,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Lin: "Formally, ROUGE-N is an n-gram recall between a candidate summary and a set of reference summaries." The **candidate summary** is the one to score; a **reference summary** is one that people wrote, such as the TLDRs of SciTLDR.

    1. **ROUGE-N** counts the n-grams of the reference that come back in the candidate, clipped as in BLEU: an n-gram counts at most as often as both have it. It divides by the number of n-grams of the reference, where BLEU divides by those of the translation: that division makes it a recall, as Lin points out. ROUGE-1 counts words, ROUGE-2 bigrams.
    2. **More than one reference.** "When multiple references are used, we compute pairwise summary-level ROUGE-N between a candidate summary s and every reference, ri, in the reference set. We then take the maximum of pairwise summary-level ROUGE-N scores as the final multiple reference ROUGE-N score."
    3. **ROUGE-L** looks at the order of the words. Lin: "Given two sequences X and Y, the longest common subsequence (LCS) of X and Y is a common subsequence with maximum length." A **subsequence** keeps the order of a sentence but may leave out words: *the bees hive* is a subsequence of *the bees at the hive*. "One advantage of using LCS is that it does not require consecutive matches but in-sequence matches that reflect sentence level word order as n-grams." ROUGE-L recall is the length of the LCS divided by the length of the reference, and its precision the same length divided by the length of the candidate. Lin combines the two in an F-score with a β, as chrF does, and writes that in DUC, the yearly comparison of summarizers of those years, β was set so large that only recall counted. The LCS comes from a table, as the edit distance does.
    4. **ROUGE-S** counts **skip-bigrams**. Lin: "Skip-bigram is any pair of words in their sentence order, allowing for arbitrary gaps." A sentence of $n$ words has $n(n - 1) / 2$ of them. In Lin's example, the reference is S1, *police killed the gunman*, and three sentences are scored against it: S2, *police kill the gunman*; S3, *the gunman kill police*; and S4, *the gunman police killed*. Each has 6 skip-bigrams, of which S2 shares 3 with S1, S3 one and S4 two: "S2's ROUGE-S score is 0.5, S3 is 0.167, and S4 is 0.333." Lin computes these scores as an F-score with β = 1 of the recall and the precision of the skip-bigrams, as for ROUGE-L; with 6 skip-bigrams in every sentence, recall, precision and F-score are the same here. And: "One advantage of skip-bigram vs. BLEU is that it does not require consecutive matches but is still sensitive to word order."

    All three compare words, and only words. In 2019, Wojciech Kryściński and colleagues of Salesforce Research wrote: "ROUGE scores are based on exact token matches, meaning that computing overlap between synonymous phrases is not supported." Synonymous phrases are phrases with the same meaning. And of the way summaries were scored, with ROUGE at its center, they wrote that the "current evaluation protocol is weakly correlated with human judgment and does not account for important characteristics such as factual correctness".

    The picture scores Qwen3-0.6B's summary of the abstract about bees, from *Summarizing: pick or write*, against the TLDR that the course wrote, with ROUGE-1, ROUGE-2 and ROUGE-L. As there, words are runs of letters in small letters: *4%* holds no word, and *count* and *counts* are two different words.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **ROUGE** | a family of measures of how much of a reference summary comes back in a summary (Lin, 2004) |
    | **candidate summary** | the summary that is scored |
    | **reference summary** | a summary that people wrote, which the candidate is compared with |
    | **ROUGE-N** | the recall of the n-grams of the reference: how many come back in the candidate, clipped, divided by how many the reference has |
    | **subsequence** | words of a sentence in their order, with any words left out between them |
    | **longest common subsequence** (LCS) | the longest subsequence that two sentences share |
    | **ROUGE-L** | the length of the LCS divided by the length of the reference (recall), or by the length of the candidate (precision) |
    | **skip-bigram** | a pair of words of a sentence in their order, with any gap between them |
    | **ROUGE-S** | a score from the skip-bigrams that the candidate shares with the reference: Lin combines their recall and their precision in an F-score |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To score a summary against a reference with ROUGE:

    1. Split both into words, in small letters.
    2. ROUGE-N: count the n-grams of each. Count each n-gram of the reference as often as both have it: the smaller of the two counts. Divide the result by the number of n-grams of the reference.
    3. ROUGE-L: fill a table with a row for each first part of the summary and a column for each first part of the reference. A cell gets the cell up and to the left plus 1 when its two words are the same, and else the larger of the cell above and the cell to the left. Divide the last cell by the length of the reference.
    4. ROUGE-S: make every pair of words of each in their order, count the pairs of the reference that come back as in step 2, and divide by the number of pairs of the reference: the recall. Lin also divides by the number of pairs of the summary, for the precision, and combines the two in an F-score.
    5. With more than one reference, score against each, and keep the highest score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take Lin's S1 as the reference and S2 as the summary. `C.ngram_counts` counts the n-grams of a list of words, each n-gram written as its words with a space between them, as in *Scoring a translation: BLEU and chrF*. The cell prints the unigrams and the bigrams of the reference, each line followed by those of the summary:
    """)
    return


@app.cell
def example_cell():
    _reference = "police killed the gunman".split()
    _summary = "police kill the gunman".split()
    for _n in (1, 2):
        print(C.ngram_counts(_reference, _n))
        print(C.ngram_counts(_summary, _n))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Three of the four words of the reference come back: *police*, *the* and *gunman*, so ROUGE-1 recall is 3 / 4 = 0.75; *kill* is not *killed*. Of the three bigrams of the reference, only *the gunman* comes back: ROUGE-2 recall is 1 / 3, about 0.33. The LCS is *police the gunman*, 3 words, so ROUGE-L recall is 3 / 4 too. Both sentences have 6 skip-bigrams, and S2 shares 3 with S1: *police the*, *police gunman* and *the gunman*. ROUGE-S recall is 3 / 6 = 0.5; S2 has 6 pairs too, so its precision and Lin's F-score are 0.5 as well.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rouge():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. rouge_n: how much of the reference comes back

        Write `rouge_n`, which gets `mine`, the n-gram counts of a summary, and `theirs`, those of a reference, both dictionaries such as `C.ngram_counts` gives; the reference has at least one n-gram. Give back ROUGE-N recall: count each n-gram of the reference as often as both dictionaries have it, and divide by the number of n-grams of the reference. In the start code, the loop goes over the n-grams of the reference: `total` adds up how often each one comes, and `matched` grows for each n-gram that the summary has too. Replace the two `...`.

        The check calls `rouge_n({"the": 3, "cat": 1}, {"the": 2, "cat": 1, "sat": 1, "down": 1})`, which gives `0.6`: *the* counts twice, as the reference has it twice, and *cat* once, so 3 of the 5 unigrams of the reference come back. Then `rouge_n({"een": 1, "klein": 1, "netwerk": 1}, {"een": 1, "netwerk": 1, "telt": 1, "bijen": 1})`, with the words of the Dutch lesson, which gives `0.5`, and cases of its own: a summary without n-grams, a word that the summary has fewer times than the reference, bigrams, and Qwen3-0.6B's summaries against the TLDRs of the course.
        """),
        L.task("Give the share of the n-grams of a reference that come back in a summary, clipped."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rouge():
    box_rouge = L.code_box("rouge_n", lines=8)
    box_rouge
    return (box_rouge,)


@app.cell(hide_code=True)
def check_rouge(box_rouge):
    result_rouge = L.check("rouge_n", box_rouge.value)
    result_rouge
    return (result_rouge,)


@app.cell(hide_code=True)
def hints_rouge():
    L.hints(
        "rouge_n",
        [
            "`gram` is one n-gram of the reference, and `theirs[gram]` is how often the reference has it; `mine` holds the counts of the summary. `matched` grows one n-gram at a time, as a running total does. `min` gives the smallest of the numbers it gets: `min(3, 2)` is 2.",
            "An n-gram counts as often as the summary has it, but never more often than the reference has it: the smaller of the two counts. Look up the count of the summary in its own dictionary. Recall divides what is matched by all the n-grams of the reference, which `total` adds up.",
            "`            matched = matched + min(___, theirs[gram])`",
        ],
        why="`matched = matched + min(mine[gram], theirs[gram])` and `return matched / total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rouge():
    mo.md(r"""
    Qwen3-0.6B's summary has a higher ROUGE-1 recall than the sentence that the title method picks in 3 of the six abstracts. On the abstract about leaks, the written summary gets 57%, more than any sentence of the abstract: the closest sentence gets 48%. The summary also says that the network outperforms traditional methods, which the abstract does not say; to ROUGE-1 recall, words that the TLDR does not have cost nothing. In all six abstracts, the closest sentence of *Summarizing: pick or write* is the sentence with the highest ROUGE-1 recall.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lcs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. lcs: the longest common subsequence

        Write `lcs`, which gets `a` and `b`, two lists of words, and gives back the length of their longest common subsequence. The table works like the table of the edit distance: `L[i, j]` is the length of the LCS of the first `i` words of `a` and the first `j` words of `b`, and the first row and the first column stay 0, as a part without words shares nothing. The start code makes the table with `np.zeros`, fills it row by row, and gives back the last cell. When word `i` of `a` and word `j` of `b`, counted from 1, are the same, the LCS of the two parts can end with that word; when they differ, the LCS leaves out the last word of `a` or the last word of `b`. Replace the two `...`.

        The check calls `lcs(["police", "kill", "the", "gunman"], ["police", "killed", "the", "gunman"])`, which gives `3`, for *police the gunman*. Then `lcs(["de", "bijen", "tellen"], ["het", "netwerk", "telt", "de", "bijen"])`, with the words of the Dutch lesson, which gives `2`, and cases of its own: an empty list on each side, Lin's S3 and S4 against S1, a word that comes twice, and Qwen3-0.6B's summaries against the TLDRs of the course.
        """),
        L.task("Fill the table of longest common subsequences, and give back the last cell."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lcs():
    box_lcs = L.code_box("lcs", lines=9)
    box_lcs
    return (box_lcs,)


@app.cell(hide_code=True)
def check_lcs(box_lcs):
    result_lcs = L.check("lcs", box_lcs.value)
    result_lcs
    return (result_lcs,)


@app.cell(hide_code=True)
def hints_lcs():
    L.hints(
        "lcs",
        [
            "`L[i, j]` is the length of the LCS of the first `i` words of `a` and the first `j` words of `b`, as `D[i, j]` held an edit distance. `max` gives the largest of the numbers it gets, as `min` gives the smallest.",
            "When the two words are the same, the LCS of the parts without these two words gets one word longer: the cell one row up and one column to the left, plus 1. When they differ, the LCS without the word of `a` is in the cell above, and the LCS without the word of `b` is in the cell to the left; the cell keeps the larger of the two.",
            "`                L[i, j] = max(L[i - 1, j], ___)`",
        ],
        why="`L[i, j] = L[i - 1, j - 1] + 1` and `L[i, j] = max(L[i - 1, j], L[i, j - 1])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_lcs():
    mo.md(r"""
    The summary of the abstract about leaks gets ROUGE-L recall 24%, against ROUGE-1 recall 57%: 12 of the 21 words of the TLDR come back, but only 5 of them in the order of the TLDR. The summary starts with the words of the title, *Finding leaks in water pipes by listening*, where the TLDR names the network first. The summary of the abstract about rivers keeps the order of the TLDR: its ROUGE-L recall and its ROUGE-1 recall are both 52%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_skip():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. skip_bigrams: every pair in its order

        Write `skip_bigrams`, which gets `words`, a list of words, and gives back every pair of words in the order of the list, each pair a tuple of two: the word at place `i` with each word at a later place. In the start code, the outer loop takes each place `i`, the inner loop should take each later place `j`, and `pairs.append` adds one pair. Replace the two `...`.

        The check calls `skip_bigrams(["police", "killed", "the", "gunman"])`, which gives the 6 pairs `[("police", "killed"), ("police", "the"), ("police", "gunman"), ("killed", "the"), ("killed", "gunman"), ("the", "gunman")]`. Then `skip_bigrams(["de", "politie", "schoot"])`, with the words of the Dutch lesson, which gives 3 pairs, and cases of its own: no words, one word, a word that comes twice, five letters, and the TLDR about plastic. The pairs may come in any order.
        """),
        L.task("Make every pair of words of a list, in the order of the list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_skip():
    box_skip = L.code_box("skip_bigrams", lines=6)
    box_skip
    return (box_skip,)


@app.cell(hide_code=True)
def check_skip(box_skip):
    result_skip = L.check("skip_bigrams", box_skip.value)
    result_skip
    return (result_skip,)


@app.cell(hide_code=True)
def hints_skip():
    L.hints(
        "skip_bigrams",
        [
            "`pairs.append` adds one item to `pairs`, as in *Lists*, and `words[i]` is the word at place `i`. A tuple of two is written with parentheses: `(\"police\", \"the\")`. `range(start, stop)` gives the whole numbers from `start` up to `stop`, without `stop`.",
            "The second word of a pair stands at a later place than `i`: the first later place is the one right after `i`, and the last one is the last place of the list. The pair holds the word at `i` first, then the word at the later place.",
            "`            pairs.append((words[i], ___))`",
        ],
        why="`for j in range(i + 1, len(words)):` and `pairs.append((words[i], words[j]))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_skip():
    mo.md(r"""
    S4, *the gunman police killed*, has every word of S1: its ROUGE-1 recall is 1. But only 2 of its 6 pairs keep the order of S1, *the gunman* and *police killed*, so its ROUGE-S is 0.333, as Lin computed. S3, *the gunman kill police*, says the opposite of S1 and keeps 1 pair. Lin also warns that without a limit on the gap, pairs of small words that have nothing to do with each other, "such as ‘the the’ or ‘of in’ might be counted as valid matches."

    ### Try it: 618 papers, four summaries each

    SciTLDR, from *Summarizing: pick or write*, has for each paper the TLDR of its authors and one or more TLDRs that experts wrote from the peer reviews. The demo scores four summaries of each paper against the TLDRs of the experts: the TLDR of the authors, the sentence that the title method picks, the first sentence of the abstract and its longest sentence. Each summary gets the best of its scores over those TLDRs. Besides recall, it shows the **F-score** of ROUGE-1 with β = 1, as in *Scoring a translation: BLEU and chrF*: precision and recall count the same. Choose a paper with the slider.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciTLDR, 1.2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scitldr_cell(get_button):
    sci_papers = C.scitldr_or_stop(get_button, {
        "ask": "*The test set of SciTLDR is not on your computer yet. The button downloads it once from its repository on GitHub, 1.2 MB, under the license Apache-2.0. The rest of the lesson works without it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: GitHub could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk has no room for 1.2 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    sci_paper = mo.ui.slider(1, len(sci_papers), value=C.SCI_START, label="Paper", show_value=True)
    sci_paper
    return sci_paper, sci_papers


@app.cell(hide_code=True)
def scitldr_look(sci_paper, sci_papers):
    mo.Html(C.scitldr_picture(L.lang, sci_papers, sci_paper.value))
    return


@app.cell(hide_code=True)
def text_after_scitldr():
    mo.md(r"""
    On ROUGE-1 recall, the sentence of the title method beats the authors' TLDR in 323 of the 618 papers, and the authors' TLDR beats the sentence in 229. With the F-score, the authors' TLDR comes first, at 28.5%, just ahead of the title method's sentence at 27.8%, and the longest sentence, which was nearly as good as the best on recall, falls to 23.3%, as most of its words are not in a TLDR of the experts. Recall alone rewards a summary for being long; precision takes that reward away.

    ### Try it: your own summary

    Choose an abstract, and write a summary of it in the box; the box starts with Qwen3-0.6B's summary. The course scores yours against the TLDR, next to Qwen3-0.6B's summary and the closest sentence. First try to get a high ROUGE-1 recall with a summary that is wrong: put the words of the TLDR in another order, or add *not*. Then write a good summary in words of your own, and see what it scores.
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    try_abstract = mo.ui.dropdown(options={C.ABSTRACT_NAMES[L.lang][_key]: _key for _key in C.ABSTRACTS},
                                  value=C.ABSTRACT_NAMES[L.lang]["leaks"], label="Abstract")
    try_abstract
    return (try_abstract,)


@app.cell(hide_code=True)
def try_write(try_abstract):
    try_text = mo.ui.text_area(value=C.QWEN[try_abstract.value], label="Your summary", rows=3, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_abstract, try_text):
    mo.Html(C.try_picture(L.lang, try_abstract.value, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ### Exercise 4. What does ROUGE-1 see?

    The reference is *A small network on a cheap board counts the bees at a hive.*, and the summary *A small network counts the bees at a hive.*: 9 of the 13 words of the reference come back, so ROUGE-1 recall is about 0.69. Each change below makes a new summary. Put each change with what it does to ROUGE-1 recall.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en) clipped the n-grams of a translation and divided by the translation, a precision; ROUGE clips the same way and divides by the reference, a recall, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en). [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en) filled a table of first parts with a minimum; the longest common subsequence fills one with a maximum. [Summarizing: pick or write](/course/open/nlp-summarize?lang=en) looked for the sentence closest to the TLDRs, and in the six abstracts of the course, that turns out to be the sentence with the highest ROUGE-1 recall. The next lesson, *Simple language and readability*, scores a text without any reference at all, by the length of its sentences and its words.

    **An everyday example.** Software that looks for copied homework works like ROUGE: it counts the n-grams that two texts share, and flags a text with many shared n-grams, whatever the text says.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rouge, result_lcs, result_skip, result_changes):
    L.wrap_up(result_rouge, result_lcs, result_skip, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
