# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-bleu. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-bleu", "en")
    C = L.checks   # the translations of nlp-classic-mt and those by people, BLEU and chrF, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [A bigram model by counting](/course/open/lm-bigram-counts?lang=en), and one about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bigrams():
    quick_bigrams = L.choice(
        "quick-bigrams",
        question="In *A bigram model by counting*, a bigram is two tokens next to each other, and you counted the bigrams of a text character by character. How many bigrams does *banana* have, counting each one every time it occurs?",
        options={
            "distinct": "3, one for each different pair: *ba*, *an* and *na*",
            "right": "5",
            "letters": "6, one for each letter",
            "four": "4",
        },
        correct="right",
        why={
            "right": "*banana* has 6 letters, and a bigram starts at each of the first 5: *ba*, *an*, *na*, *an*, *na*; *an* and *na* occur twice. This lesson counts such n-grams, of words and of letters, and how many of them a translation shares with a translation by people.",
            "distinct": "3 counts each different pair once, and *an* and *na* each occur twice. At how many places does a bigram start? See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
            "letters": "The last letter, *a*, starts no bigram: nothing follows it. At how many places does a bigram start? See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
            "four": "*banana* has 6 letters, and every letter but the last starts a bigram. How many is that? See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
        },
    )
    quick_bigrams
    return (quick_bigrams,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="In *Beyond accuracy: precision and recall*, recall is the share of what is positive that a model finds. A spam filter marks 10 messages as spam, and 8 of them are spam. In all, 20 messages are spam. What is its recall?",
        options={
            "precision": "0.8, as 8 of the 10 marked messages are spam",
            "half": "0.5",
            "right": "0.4",
            "two": "0.2",
        },
        correct="right",
        why={
            "right": "Recall is 8 / 20 = 0.4: of the 20 spam messages, the filter finds 8. Its precision is 8 / 10 = 0.8. This lesson works out both for the letters of a translation: precision is the share of its groups of letters that the translation by people has too, and recall the share of those of the translation by people that it has.",
            "precision": "8 / 10 = 0.8 is the precision: of what the filter marks, the share that is spam. Of the 20 spam messages, what share does it find? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "half": "10 / 20 compares how many messages it marks with how many are spam, and 2 of the 10 it marks are not spam. Of the 20 spam messages, how many does it find? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "two": "2 of the 10 marked messages are not spam: that is the share of mistakes in what it marks. Of the 20 spam messages, how many does it find? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In *Translating by rules*, a translation was exactly a translation by people, had its words in another order, or had other words. Of 21,636 word-for-word translations, 17,620 had other words: *i have hunger*, where people write *I'm hungry*, counts the same as a text without a single right word. People can judge every translation, but that is slow and costs a lot. In 2002, Kishore Papineni and three colleagues at IBM proposed BLEU, a score a computer works out in a moment, from one idea: "The closer a machine translation is to a professional human translation, the better it is." They measured how close by counting the groups of words that a translation shares with translations by people, and compared the score with how people judged the same translations: "The high correlation coefficient of 0.99 indicates that BLEU tracks human judgment well." A correlation coefficient of 1 would mean that BLEU and the judgments of people rise and fall in perfect step.

    They also warned: "BLEU only needs to match human judgment when averaged over a test corpus;" and "scores on individual sentences will often vary from human judgments." In this lesson you build BLEU, and chrF, a score that counts groups of letters, and score the translations of *Translating by rules* with both.

    **An everyday example.** A teacher grading a dictation does not mark the whole text right or wrong, but counts the words spelled right: one wrong letter costs a little, not everything.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_zero = L.guess(
        "guess",
        question="The lesson has 3,707 Dutch sentences of Tatoeba with more than one English translation by people. Take the first translation of each, and score it against the others, sentence by sentence, with BLEU as Papineni and colleagues defined it. BLEU counts the groups of 1 to 4 words in a row that a translation shares with the others, and gives 0 when it shares no group of four. For how many of the 3,707 does BLEU give 0?",
        options={
            "almost": "Almost none of them",
            "tenth": "About one in ten",
            "right": "More than half",
            "all": "Nearly all of them",
        },
        reveal="More than half: 2,200 of the 3,707, or 59.3%. 775 of the first translations have fewer than four words, so no group of four at all. 193 of those match another translation exactly: *Ik ben moe.* has *I'm tired.* and *I am tired.*, the same words once *I'm* is written out, and BLEU gives 0. The other 1,425 have four words or more, but share no group of four with another translation: *Kan ik u helpen?* has *Can I help you?* and *May I help you?*, which share *i help you*, three words. Both are good English: people say one thing in many good ways.",
    )
    guess_zero
    return (guess_zero,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A translation by people that a translation is compared with is a **reference**; a sentence can have more than one. BLEU, short for *bilingual evaluation understudy*, counts the **word n-grams** that a translation shares with its references: groups of 1, 2, 3 and 4 words in a row, as in *A bigram model by counting*. Four steps turn the counts into a score:

    1. **Clipped counts.** For each n, count the n-grams of the translation that a reference has, but each at most as often as any single reference has it. In the words of Papineni and colleagues, "one clips the total count of each candidate word by its maximum reference count," and longer n-grams go the same way. The clipped count divided by how many n-grams the translation has is the **n-gram precision** $p_n$.
    2. **The geometric mean** of $p_1$ to $p_4$: multiply the four and take the fourth root, or, the same, take *e* to the power of the mean of their logarithms. "BLEU uses the average logarithm with uniform weights, which is equivalent to using the geometric mean of the modified n-gram precisions." Modified is their word for clipped. Unlike the ordinary mean, it is 0 as soon as one $p_n$ is 0.
    3. **The brevity penalty** (BP). Precision alone rewards a translation that leaves words out: *the cat is on* has only words of *the cat is on the mat*. With $c$ the length of the translation in words and $r$ that of the reference closest to it in length: $\text{BP} = 1$ when $c > r$, and $\text{BP} = e^{1 - r/c}$ when $c \le r$, which is below 1 when $c < r$.
    4. **BLEU** is BP times the geometric mean, from 0 to 1. Tools such as sacreBLEU, a Python package that computes BLEU and chrF, and this lesson, write it times 100: 22.9 for 0.229.

    For many sentences, BLEU first adds up the counts and lengths of all sentences, and only then divides: BLEU at the **corpus level**. Papineni and colleagues: "Instead, we compute the brevity penalty over the entire corpus to allow some freedom at the sentence level."

    BLEU compares whole words. Dutch writes many **compounds** as one word: *insectensoorten* is *insecten* and *soorten*, and to BLEU it is a word unlike both. In 2015, Maja Popović proposed **chrF**, which counts **character n-grams**: groups of 1 to 6 letters in a row, with the spaces left out. For each length, precision is the share of the translation's character n-grams that the reference has, clipped as in BLEU, and recall the share of the reference's character n-grams that the translation has. chrF averages each over the six lengths, into $P$ and $R$, and combines them in an **F-score**:

    $$\text{chrF}_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

    Popović: "β is a parameter which assigns β times more importance to recall than to precision." With β = 1, precision and recall count the same. In 2016 Popović wrote "we found out that CHRF2 is the most promising version": β = 2, which this lesson uses, as sacreBLEU does.
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
    | **reference** | a translation by people that a translation is compared with; a sentence can have more than one |
    | **word n-gram**, **character n-gram** | n words in a row, or n characters in a row; this lesson leaves the spaces out of character n-grams |
    | **clipped count** | how often the n-grams of a translation are in a reference, each at most as often as any single reference has it |
    | **n-gram precision** ($p_n$) | the clipped count of the n-grams of a translation, divided by how many n-grams it has |
    | **geometric mean** | of k numbers: their product to the power 1/k; the same as *e* to the power of the mean of their logarithms |
    | **brevity penalty** (BP) | a factor of at most 1 that lowers BLEU for a translation shorter than its reference: $e^{1 - r/c}$ when $c \le r$ |
    | **corpus level** | adding up the counts of all sentences before dividing, instead of scoring sentence by sentence |
    | **BLEU** | the brevity penalty times the geometric mean of $p_1$ to $p_4$ |
    | **compound** | a word made of two or more words, such as *insectensoorten* |
    | **F-score** ($F_\beta$) | one number from precision $P$ and recall $R$ that counts recall β times as much as precision |
    | **chrF** | the F-score with β = 2 of the character n-grams of length 1 to 6, with precision and recall averaged over the lengths |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To score translations with BLEU:

    1. Cut each translation and each reference into words.
    2. For n = 1 to 4, count the n-grams of each translation and clip each count by the largest count in any single reference.
    3. Add up the clipped counts, the numbers of n-grams, the lengths of the translations and the lengths of the closest references over all sentences.
    4. Divide: $p_1$ to $p_4$. Take their geometric mean, and multiply by the brevity penalty.

    To score them with chrF, leave the spaces out of the words, count the character n-grams of length 1 to 6 in each translation and its reference, and clip. Add up over all sentences, work out precision and recall for each length, average both over the lengths, and take the F-score with β = 2. A sentence with more than one reference takes the one that gives it the highest chrF.

    This lesson scores words in small letters without punctuation, with *I'll* written out as *i will*, as *Translating by rules* did. On those words, sacreBLEU gives the same scores for all sentences together as the course; for a single sentence it does something extra, as the text after exercise 2 says.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take the example of Papineni and colleagues, as printed: "Candidate: the the the the the the the." with "Reference 1: The cat is on the mat." and "Reference 2: There is a cat on the mat." `C.words` cuts a text into words in small letters, and `C.ngram_counts` counts the n-grams of a list of words in a dictionary, each n-gram written as one text, with spaces between its words:
    """)
    return


@app.cell
def example_cell():
    _mine = C.ngram_counts(C.words("the the the the the the the."), 1)
    _first = C.ngram_counts(C.words("The cat is on the mat."), 1)
    _second = C.ngram_counts(C.words("There is a cat on the mat."), 1)
    print(_mine)
    print(_first["the"], _second["the"])
    print(C.ngram_counts(C.words("The cat is on the mat."), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The translation has 7 unigrams, all *the*, and both references have *the*: counted plainly, its precision is 7/7. But the first reference has *the* twice and the second once, so *the* counts at most twice: "Modified Unigram Precision = 2/7." Its 6 bigrams are all *the the*, which no reference has: $p_2 = 0/6$, so BLEU is 0. With more than one reference, the course gives your `clipped` one dictionary with the largest count of each n-gram in any single reference: here *the* 2 times, from the first reference, and *there* once, from the second.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_clipped():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. clipped: count what the reference has

        Write `clipped`, which gets `mine`, the n-gram counts of a translation, and `theirs`, the largest count of each n-gram in any single reference, both dictionaries such as `C.ngram_counts` gives. Give back how many n-grams of the translation count: an n-gram of `mine` that `theirs` has counts as often as `mine` has it, but never more often than `theirs` has it; an n-gram that `theirs` lacks counts 0 times. In the start code, `matched` starts at 0, and the loop takes one n-gram `gram` at a time. `matched += x` adds `x` to `matched`, and `min` gives the smallest of the numbers it gets: `min(7, 2)` is 2. Replace the two `...`.

        The check calls `clipped({"the": 7}, {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1})`, which gives 2: the example of Papineni and colleagues, with the first reference. Then `clipped({"the": 1, "cat": 1, "sat": 1}, {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1})`, which gives 2, for *the* and *cat*. And cases of its own: an empty dictionary, n-grams that `theirs` lacks, bigrams, and the n-grams of the worked example and of *i will the next bus take*.
        """),
        L.task("Count the n-grams of a translation that a reference has, each at most as often as the reference has it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clipped():
    box_clipped = L.code_box("clipped", lines=6)
    box_clipped
    return (box_clipped,)


@app.cell(hide_code=True)
def check_clipped(box_clipped):
    result_clipped = L.check("clipped", box_clipped.value)
    result_clipped
    return (result_clipped,)


@app.cell(hide_code=True)
def hints_clipped():
    L.hints(
        "clipped",
        [
            "The `if` asks with `in` whether `gram` is a key of `theirs`, as in *Dictionaries*. Under it, `matched +=` adds how often `gram` counts: `mine[gram]` is how often the translation has it, `theirs[gram]` how often the reference has it at most, and `min` gives the smaller of two numbers.",
            "Without the `if`, `theirs[gram]` stops with a `KeyError` at an n-gram that the reference lacks, such as *sat*. And take the smaller count, not the larger: *the* 7 times in the translation and twice in the reference counts twice.",
            "`            matched += ___`",
        ],
        why="In the `if`: `gram in theirs`; under it: `matched += min(mine[gram], theirs[gram])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_clipped():
    mo.md(r"""
    Word for word, on the 21,636 sentences of *Translating by rules*: 82,092 of its 118,401 words are in a translation by people, 69.3%. Of its 96,765 bigrams, 28,737 are, 29.7%; of its trigrams 15.5%; and of its groups of four only 8.7%, 4,752 of 54,366. Most single words are right, but a longer group also needs its words in the right order, and that is where word for word fails.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bleu():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. bleu: from counts to a score

        Write `bleu`, which gets `matched` and `total`, two lists with the clipped count and the number of n-grams for each n, from 1 up; `c`, the length of the translation in words; and `r`, the length of the reference. Give back BLEU, a number from 0 to 1. The start code gives 0.0 at once when a clipped count is 0, as the geometric mean is then 0 and the logarithm of 0 does not exist. Then it puts the logarithm of each precision in the list `logs`, and gives back `penalty` times *e* to the power `mean`. Replace the two `...`: `mean` is the mean of `logs`, and `penalty`, in the `else`, the brevity penalty of a translation that is not longer than its reference.

        The check calls `bleu([6, 4, 2, 1], [6, 5, 4, 3], 6, 6)`, which gives about 0.6043: the geometric mean of 6/6, 4/5, 2/4 and 1/3, without a penalty. Then `bleu([4, 3, 2, 1], [4, 3, 2, 1], 4, 6)`, which gives about 0.6065: every n-gram counts, but the translation has 4 words where the reference has 6. And cases of its own, among them the counts of the three ways to translate in the picture below, added up over all their sentences.
        """),
        L.task("Compute BLEU from the clipped counts, the numbers of n-grams and the two lengths."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bleu():
    box_bleu = L.code_box("bleu", lines=13)
    box_bleu
    return (box_bleu,)


@app.cell(hide_code=True)
def check_bleu(box_bleu):
    result_bleu = L.check("bleu", box_bleu.value)
    result_bleu
    return (result_bleu,)


@app.cell(hide_code=True)
def hints_bleu():
    L.hints(
        "bleu",
        [
            "`np.mean` gives the mean of a list, as in *The loss you train on and the metric you report*, and `np.exp(x)` gives *e* to the power `x`, as in *From score to probability: the sigmoid*. In the `else`, `penalty` is *e* to the power 1 minus the length of the reference divided by the length of the translation.",
            "Put `r` on top and `c` below: a translation shorter than its reference has `r / c` above 1, so the power is below 0 and the penalty below 1. And take the mean of the logarithms, not their total: four logarithms added up are the logarithm of the product.",
            "`        penalty = np.exp(___)`",
        ],
        why="`mean = np.mean(logs)` and `penalty = np.exp(1 - r / c)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_bleu():
    mo.md(r"""
    Word for word scores 22.9 on the 21,636 sentences. Its 118,401 words are a few fewer than the 118,658 of the closest references, so the brevity penalty is 0.998. With the rule of *Translating by rules*, BLEU drops to 22.7, though the rule gave 27 more exact translations. The rule only moves words, so the unigrams that count stay the same, 82,092, and the bigrams that count drop from 28,737 to 28,381. Of the 5,361 sentences the rule changes, 920 now share more groups of two to four words with a translation by people, and 1,293 fewer. 595 of those 1,293 are questions, such as *Kan ik helpen?*, which becomes *can help i*. Two ways to measure, two answers: the score you choose decides what counts as better.

    People against people score 57.8: far above word for word, and far below 100. 787 of the 3,707 first translations match another translation exactly; the others say the same thing in other words. Sentence by sentence, BLEU is 0 for 2,200 of those 3,707, and for 18,498 of the 21,636 word-for-word translations, 85.5%. Those zeros come from the groups of four: one $p_n$ of 0 makes the geometric mean 0, and adding up the counts of all sentences first avoids that. Papineni and colleagues add up the lengths for the brevity penalty first too: "if we computed the brevity penalty sentence by sentence and averaged the penalties, then length deviations on short sentences would be punished harshly." For a single sentence, sacreBLEU smooths: an n-gram length with no match gets a small precision instead of 0. With that, *Can I help you?* against *May I help you?* gets 59.5 instead of 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chars():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. char_ngrams: count groups of letters

        Write `char_ngrams`, which gets `letters`, a text without spaces, and a length `n`. Give back a dictionary with each group of `n` letters in a row that occurs in `letters`, and how often it occurs. In the start code, the loop goes over the places `i` where such a group starts, and the `if` counts the group `gram` in the dictionary `counts`; `counts[gram] += 1` adds 1 to the count of `gram`. Replace the two `...`.

        The check calls `char_ngrams("banana", 2)`, which gives `{"ba": 1, "an": 2, "na": 2}`, and `char_ngrams("hand", 3)`, which gives `{"han": 1, "and": 1}`. Then cases of its own: a text shorter than `n`, an empty text, groups that overlap, and the letters of Dutch sentences with a compound, such as `C.letters(C.words("Alles hier is handgemaakt."))`: `C.letters` puts the words of a list together without spaces.
        """),
        L.task("Count every group of n letters in a row in a text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chars():
    box_chars = L.code_box("char_ngrams", lines=9)
    box_chars
    return (box_chars,)


@app.cell(hide_code=True)
def check_chars(box_chars):
    result_chars = L.check("char_ngrams", box_chars.value)
    result_chars
    return (result_chars,)


@app.cell(hide_code=True)
def hints_chars():
    L.hints(
        "char_ngrams",
        [
            "A text of 6 letters has 5 bigrams: the last group of `n` starts at place `len(letters) - n`, and `range(k)` stops just before `k`: how far must `range` go to reach that place? A slice `letters[a:b]` gives the letters from place `a` up to, but not including, place `b`, as in *Lists*; `gram` is the slice of `n` letters that starts at place `i`.",
            "The slice stops at place `i + n`, not at `n`: `n` is a length, and the slice needs a place. And with `range(len(letters))`, the last places give groups shorter than `n`, such as the *a* at the end of *banana*.",
            "`        gram = ___`",
        ],
        why="`for i in range(len(letters) - n + 1):` and `gram = letters[i:i + n]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_chars():
    mo.md(r"""
    On the four Dutch pairs, BLEU gives 0 three times. *Tom is een rechtenstudent* shares *tom is een*, three words, with *Tom is een student rechten*, and no group of four. chrF gives those three 71.7 to 80.6, as most groups of letters still line up: *rechten* and *student* are both inside *rechtenstudent*. Only *Ik had vandaag een wiskundeproefwerk* shares a group of four words with its reference, *ik had vandaag een*: BLEU 54.8, chrF 82.3. Popović wrote that chrF "takes into account some morpho-syntactic phenomena" and that "it is absolutely language independent and also tokenisation independent." In 2016: "the main advantage of CHRF is that it does not penalise too hard acceptable variations in high quality translations."

    On the translations into English, chrF is 51.1 for word for word, 50.9 with the rule and 71.4 for people against people: the same order as BLEU, with the three closer together.

    ### Try it: score a translation

    Type a reference and a translation, and see their word n-grams, the brevity penalty, BLEU, and the character n-grams of chrF. Try *the cat is on* against *the cat is on the mat*, a translation that leaves words out, or a Dutch sentence with and without a compound, such as *Ik heb een proefwerk wiskunde* and *Ik heb een wiskundeproefwerk*.
    """)
    return


@app.cell(hide_code=True)
def box_texts():
    try_reference = mo.ui.text(value="May I help you?", label="The reference", full_width=True)
    try_translation = mo.ui.text(value="Can I help you?", label="The translation", full_width=True)
    mo.vstack([try_reference, try_translation])
    return try_reference, try_translation


@app.cell(hide_code=True)
def box_cell(try_reference, try_translation):
    mo.Html(C.box_picture(L.lang, try_reference.value, try_translation.value))
    return


@app.cell(hide_code=True)
def text_exercise_beta():
    mo.md(r"""
    ### Exercise 4. Which does chrF prefer?

    Take a reference and two translations. Translation A has every character n-gram of the reference, and as many again that are not in it: precision 0.5, recall 1. Translation B has half the character n-grams of the reference and nothing else: precision 1, recall 0.5.
    """)
    return


@app.cell(hide_code=True)
def choice_beta_cell():
    beta = L.choice(
        "beta",
        question="Which of the two gets the higher chrF, with β = 2?",
        options={
            "b": "B, which adds nothing wrong",
            "right": "A, which misses nothing",
            "same": "They tie: both have a half",
            "zero": "Neither: both get 0",
        },
        correct="right",
        why={
            "right": "A: (1 + 4) × 0.5 × 1 / (4 × 0.5 + 1) = 2.5 / 3, about 0.83. B: (1 + 4) × 1 × 0.5 / (4 × 1 + 0.5) = 2.5 / 4.5, about 0.56. With β = 2, recall counts twice as much as precision: leaving out half of the reference costs more than adding words. With β = 1, both would get 2 × 0.5 × 1 / 1.5, about 0.67. Popović in 2016: \"recall is even more important for morphologically rich(er) languages.\"",
            "b": "B has nothing wrong, but misses half of the reference. Which does chrF count more heavily, precision or recall?",
            "same": "With β = 1 they would tie, at about 0.67 each. With β = 2, which of the two counts more?",
            "zero": "chrF is 0 only when no character n-gram matches, and both translations match at least half. What does the formula give for each?",
        },
    )
    beta
    return (beta,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Translating by rules](/course/open/nlp-classic-mt?lang=en) said right or wrong; BLEU and chrF give the same translations partial credit, and, on the rule, disagree with counting exact matches. The n-grams of [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) are what BLEU and chrF count, and the precision and recall of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) are what chrF combines. The next lesson, *Translators and machines together*, looks at people who correct what a machine translated, and measures how much they change.

    **An everyday example.** Two people give directions to the station: *turn left at the church* and *go left by the church*. A score that counts shared words gives the second little credit against the first, and a listener could follow either.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_clipped, result_bleu, result_chars, beta):
    L.wrap_up(result_clipped, result_bleu, result_chars, beta)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
