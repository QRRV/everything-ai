# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-zipf-heaps. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-zipf-heaps", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Log scales and power laws](/course/open/math-log-scales?lang=en), and one about [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="On a log-log plot, a straight line goes from $x$ = 1, $y$ = 1,000 down to $x$ = 100, $y$ = 10. What is its slope?",
        options={
            "normal": "−10",
            "gridlines": "−2",
            "right": "−1",
        },
        correct="right",
        why={
            "right": "Two factors of 10 down, from 1,000 to 10, over two factors of 10 along, from 1 to 100: −2 divided by 2 is −1. A slope of about −1 comes back in this lesson, for words counted against their rank.",
            "normal": "That is the slope on normal axes: (10 − 1,000) / (100 − 1). On a log-log plot, you divide the change in the logs. See *The idea* in [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "gridlines": "The line goes two factors of 10 down, but also two along, and the slope is up divided by along. See *The idea* in [Log scales and power laws](/course/open/math-log-scales?lang=en).",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question="GPT-2's tokenizer meets a word that is not one of its 50,257 tokens. What does it do with the word?",
        options={
            "unknown": "It gives the word one token that means unknown",
            "right": "It cuts the word into pieces that are tokens",
            "skip": "It leaves the word out",
        },
        correct="right",
        why={
            "right": "A rarer word falls into subwords, pieces that are in its list. This lesson shows why a tokenizer needs them: however long a list of whole words is, new words keep turning up.",
            "unknown": "GPT-2's tokenizer has no such token: every text becomes tokens of its list. What happens to a word it does not have whole? See *New words and symbols* in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "skip": "A tokenizer turns every text into tokens and back, so nothing goes missing. What happens to a word it does not have whole? See *New words and symbols* in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Count the words of any text, and the same pattern turns up: a handful of words take a large part of it, and most of the different words occur once or twice. Read on, and new words keep coming, however much you have read already. Both patterns matter for every model of language. A model meets most words only a few times, too few to learn much about each, and a list of words made from one text always misses words of the next. Jurafsky and Martin, in their chapter on words and tokens: "No matter how big our vocabulary, we will never have a vocabulary that captures all the possible words that might occur!" That is one reason why the tokenizers of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) cut rare words into subwords. Search engines meet the same growth: in *Introduction to Information Retrieval*, Manning, Raghavan and Schütze point out that, by Heaps' law, one of the two laws of this lesson, the list of words a search engine can look up keeps growing with every document it takes in, instead of leveling off at a maximum size.

    **An everyday example.** Think of the messages you send in a week: *I*, *you*, *the* and *ok* come back all the time, while most other words turn up once. And a trip to a new city brings words you never typed before: the names of its streets and its dishes.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The course's five English books hold 551,664 words. How many of the most common words, taken together, make up half of them?",
        options={
            "right": "About 75",
            "mid": "About 750",
            "many": "About 7,500",
        },
        reveal="**About 75**: the 76 most common words, from *the*, *and* and *of* down to *other*, make up half of the 551,664 words, out of 23,131 different ones. At the other end, 8,659 different words occur only once. *Alice* on its own needs 58 words for half of its text, and in Mark Twain's *Tom Sawyer*, Manning and Schütze found that the 100 most common words make up 50.9% of the text.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A few words, very often: Zipf's law

    Count every word of a text, and put the different words in order, from the most to the least common. The place of a word in that list is its **rank**: the most common word has rank 1, the next one rank 2, and so on. In *Human Behavior and the Principle of Least Effort*, from 1949, George Kingsley Zipf wrote down how the count $f$ of a word goes with its rank $r$: rank times count stays roughly the same, $r \times f = C$. So the count is roughly a fixed number divided by the rank, $f \approx C / r$: the word at rank 2 occurs about half as often as the word at rank 1, and the word at rank 10 about a tenth as often. On a log-log plot, as in [Log scales and power laws](/course/open/math-log-scales?lang=en), that is a straight line with slope −1. Zipf's main example was James Joyce's novel *Ulysses*: 260,430 words with 29,899 different ones, counted for a word index of the book published in 1937. The word at rank 10 occurs 2,653 times and the word at rank 100 only 265 times: ten times further down the list, a tenth of the count.

    Zipf was not the first to see it. From its fourth edition of 1916 on, *Gammes sténographiques*, a book on shorthand by the French stenographer Jean-Baptiste Estoup, held counts of French: in a text of 20,000 words, about a dozen words made up 40% of it. Zipf named Estoup as the first, to the best of Zipf's knowledge. In 1928, Edward Condon, then at Bell Telephone Laboratories, put the counts of two English word lists, each made from 100,000 words, on log scales, and the points lay close to a straight line with slope −1. Zipf made the pattern widely known, and the law is named after Zipf. In 1953 and 1954, Benoit Mandelbrot added two numbers to the formula, so that it fits the most common words better.

    Why would words follow such a law? Zipf saw a balance between two kinds of effort: the speaker would rather use a few words for everything, and the listener would rather have a different word for every meaning. Zipf called that balance a hypothesis, and no explanation has been settled since.

    The picture shows the ten most common words of two corpora: the five English books of the course, and 1,301,039 words of today's Dutch from Tatoeba, a collection of short sentences that volunteers write and translate, released under CC BY 2.0 FR. The course took its Dutch sentences on November 21, 2026, and left out the sentences with a crude, sexual or violent word. The open dots are what Zipf's law predicts from the first word: the count of the first word divided by the rank.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_tail():
    mo.md(r"""
    ### Most words, rarely

    The other end of the list is long. A word that occurs only once in a text or a corpus is a **hapax legomenon**, Greek for *said once*; the plural is *hapax legomena*. They are not a few odd words. In *Ulysses*, 16,432 of the 29,899 different words occur once, 55%; in *Tom Sawyer*, Manning and Schütze counted 3,993 of 8,018, 49.8%. A larger corpus does not solve that: it brings new rare words of its own. So a model meets most of the different words of its corpus only once or twice, too rarely to learn much about each of them.

    And the law holds only roughly. In a review of the research from 2014, Steven Piantadosi calls the counts of language only near-Zipfian: on a log-log plot they run roughly straight, but even the most common words do not follow the law exactly, and the rarest words bend away from the line. A curve much like Zipf's even comes out of random typing, letters and spaces hit at random, as Miller noted in 1957 and Li worked out in 1992, so a straight line by itself shows little about language; whether random text matches the details of real counts is disputed. Still, Piantadosi writes, such counts have been found in every language that has been looked at, even in Meroitic, an extinct language that nobody can translate yet.
    """)
    return


@app.cell(hide_code=True)
def text_idea_heaps():
    mo.md(r"""
    ### New words keep coming: Heaps' law

    Now read a text from the start, and keep track of how many different words you have met. At first, almost every word is new; later, more and more words are ones you met before, but new ones keep coming. Call the number of words read $N$, and the number of different words among them $V$. In the chapter *Words and Tokens* of *Speech and Language Processing*, Jurafsky and Martin write the pattern as $V = k \cdot N^\beta$, with $k$ and $\beta$ positive numbers and $\beta$ between 0 and 1. They call it Herdan's law or Heaps' law, after Gustav Herdan, who described it in linguistics in 1960, and Harold Heaps, who described it for search in 1978; this course says Heaps' law. On a log-log plot it is again a straight line, now with slope $\beta$. For $\beta$, numbers "from 0.44 to 0.56 or even higher" have often been reported, they write: the number of different words grows a little faster than the square root of the length. *Introduction to Information Retrieval* shows how well that can fit: for news stories of Reuters, Manning, Raghavan and Schütze find $k$ = 44 and $\beta$ = 0.49, which predicts 38,323 different words in the first 1,000,020 words. The real number is 38,365.

    The two laws belong together. If most words are rare, as Zipf's law says, a longer text keeps meeting rare words it has not seen before, so the number of different words keeps growing, but more and more slowly. Lü, Zhang and Zhou showed in 2010 how Heaps' law follows from Zipf's law; how fast the different words grow depends on the exponent of Zipf's law and on the length of the text. Nor is the growth one straight line from the very start: in books of Project Gutenberg, Tria, Loreto and Servedio found that the different words first grow almost as fast as the text, and only later more slowly, close to the square root. Jurafsky and Martin put the first part down to common words that still turn up for the first time.

    For a model of language, this is the problem in one formula. A list of whole words, however long, misses words that the next text brings: names, typing errors, new words, and in Dutch the compounds that are written as one word. In 2016, Sennrich, Haddow and Birch called translation an open-vocabulary problem, especially for languages that make new words by joining words, and they let a translation model cut rare words into subwords. Today's tokenizers do the same with a list of a fixed length: 50,257 tokens for GPT-2, 151,669 for Qwen3.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **rank** | the place of a word in a list from the most to the least common word: the most common word has rank 1 |
        | **Zipf's law** | the count of a word is roughly proportional to 1 over its rank: $f \approx C / r$, so rank times count, $r \cdot f$, stays roughly the same |
        | **hapax legomenon** | a word that occurs only once in a text or a corpus; plural *hapax legomena* |
        | **Heaps' law** | the number of different words in a text grows roughly as a power of its length: $V \approx k \cdot N^\beta$, with $\beta$ between 0 and 1 |
        | $N$, $V$ | the number of words of a text, counting every time a word occurs, and the number of different words in it |
        | `sorted(xs, reverse=True)` | the values of `xs` from the largest to the smallest; `reverse=True` is a keyword argument |
        | `np.arange(a, b)` | an array of the whole numbers from `a` up to `b`, without `b` itself, as `range(a, b)` counts |
        | `d.values()` | the values of the dict `d`, without their keys |
        | `seen.add(x)` | puts `x` in the set `seen`; a set holds each value once, so adding a value that is already in it changes nothing |
        | `n % step` | the remainder of `n` divided by `step`: 0 when `n` is a whole number of steps |
        """),
        L.watch_out("Zipf's law and Heaps' law are patterns that counts follow roughly, not rules that every count obeys. Their numbers also depend on what you count as a word: with *The* and *the* kept apart, or with *don't* cut in two, every count changes."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut the text into words, and decide what counts as the same word: here, the same letters in small letters.
    2. Count how often each word occurs, and sort the counts from the highest to the lowest. The place in that list is the rank.
    3. Draw count against rank on a log-log plot. A line that runs roughly straight, with a slope of about −1, is Zipf's law; rank times count then stays roughly the same.
    4. For Heaps' law, go through the text once with a set of the words seen so far, and write down the size of the set every so many words.
    5. Draw the different words against the words read on a log-log plot. The slope between two points is the exponent $\beta$, worked out as in [Log scales and power laws](/course/open/math-log-scales?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell gives you the words of *Alice's Adventures in Wonderland*, as in [Corpora and treebanks](/course/open/nlp-corpora?lang=en), and, from `C.counts_by_rank`, the course's own function, the counts of its different words, from the most to the least common: what your own function of exercise 1 will give. It prints, for a few ranks, the count and rank times count. The second cell counts the different words in the first 1,000 words, the first 10,000 and the whole book, each time with a new set of the words in small letters, and works out the exponent $\beta$ between the first two points, as step 5 of the recipe says.
    """)
    return


@app.cell
def book_cell():
    import numpy as np

    ALICE = C.book_words("alice")
    _counts = C.counts_by_rank(ALICE)
    print(len(ALICE), len(_counts))
    print(_counts[:5])
    for _rank in [1, 2, 3, 10, 100, 1000]:
        print(_rank, _counts[_rank - 1], _rank * _counts[_rank - 1])
    return ALICE, np


@app.cell
def example_growth(ALICE, np):
    _v_a = len(set([w.lower() for w in ALICE[:1000]]))
    _v_b = len(set([w.lower() for w in ALICE[:10000]]))
    print(_v_a, _v_b, len(set([w.lower() for w in ALICE])))
    _beta = (np.log10(_v_b) - np.log10(_v_a)) / (np.log10(10000) - np.log10(1000))
    print(round(_beta, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Alice* has 26,776 words and 2,636 different ones. The most common word, *the*, occurs 1,651 times. Zipf's law predicts about half of that at rank 2, 826, and the book has 874; at rank 10, a tenth, 165, and the book has 386. Rank times count goes from 1,651 at rank 1 to 3,860 at rank 10 and 4,900 at rank 100, and back to 3,000 at rank 1,000: it changes by a factor of 3, while the count itself falls from 1,651 to 3. That is how roughly the law holds. The second cell finds 389 different words in the first 1,000 words, 1,552 in the first 10,000 and 2,636 in the whole book: ten times as much text, about four times as many different words, so $\beta$ is 0.6, a little more than a square root, which would be 0.5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rank_counts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. rank_counts: words by rank

        Write `rank_counts(words)`: count how often each word of `words` occurs, in small letters, so that *The* and *the* are one word, and give back the counts in a list, from the highest to the lowest. The words themselves stay out of the list: place 0 holds the count of the word at rank 1, place 1 the count at rank 2, and so on. Use a dict from word to count, and `sorted(..., reverse=True)` for the order. For `rank_counts(ALICE)`, the list starts with `[1651, 874, 729]`. Replace the `...`. The check calls `rank_counts(ALICE)`, then cases of its own: a few short lists, one with capitals and one with words that occur equally often, a list of one word and an empty list.
        """),
        L.task("Count the words of a text, and give their counts from the highest to the lowest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rank_counts():
    box_rank_counts = L.code_box("rank-counts", lines=9)
    box_rank_counts
    return (box_rank_counts,)


@app.cell(hide_code=True)
def check_rank_counts(box_rank_counts):
    result_rank_counts = L.check("rank-counts", box_rank_counts.value)
    result_rank_counts
    return (result_rank_counts,)


@app.cell(hide_code=True)
def hints_rank_counts():
    L.hints(
        "rank-counts",
        [
            "Start with an empty dict `counts`. Go through the words, put each in small letters, and update its count, or give it the count 1 when it is not in the dict yet. After the loop, sort the values of the dict from the highest to the lowest.",
            "A word is already in `counts`. What is its new count?",
            "`            counts[w] = ___`",
        ],
        why="`counts = {}`, then `for w in words:` with `w = w.lower()` under it, and `if w in counts:` with `counts[w] = counts[w] + 1`, `else:` with `counts[w] = 1`. After the loop: `return sorted(counts.values(), reverse=True)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rank_counts():
    mo.md(r"""
    The picture under the check runs your function on two whole corpora: the five English books, 551,664 words, and the Dutch sentences of Tatoeba, 1,301,039 words. On log scales both lines run roughly straight from rank 10 on, close to the dashed line of 1 over the rank, and both end in steps: thousands of words at the end of the list share the count 2, and then thousands more the count 1. Those last ones are the hapax legomena.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_product():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. rank_times_count: Zipf's product, with NumPy

        `COUNTS` below holds the counts of *Alice* by rank as a NumPy array. Write `rank_times_count(counts)`: give back an array with, for every rank, the rank times its count. The ranks go 1, 2, 3 and on, one for each count; `np.arange` makes them, and `*` multiplies two arrays of the same length place by place. For `COUNTS`, the array starts with 1,651, 1,748 and 2,187. Replace the `...`. The check calls `rank_times_count(COUNTS)`, then cases of its own: short arrays, one of them empty.
        """),
        L.task("Multiply every count by its rank, with NumPy."),
    ])
    return


@app.cell
def counts_cell(ALICE, np):
    COUNTS = np.array(C.counts_by_rank(ALICE))
    print(COUNTS[:5])
    return (COUNTS,)


@app.cell(hide_code=True)
def exercise_product():
    box_product = L.code_box("rank-times-count", lines=6)
    box_product
    return (box_product,)


@app.cell(hide_code=True)
def check_product(box_product):
    result_product = L.check("rank-times-count", box_product.value)
    result_product
    return (result_product,)


@app.cell(hide_code=True)
def hints_product():
    L.hints(
        "rank-times-count",
        [
            "Make an array `ranks` with one rank for every count, then give back `ranks` times `counts`.",
            "Where does `np.arange(a, b)` start, and where does it stop?",
            "`    ranks = np.arange(___)`",
        ],
        why="`ranks = np.arange(1, len(counts) + 1)` gives 1, 2, 3, up to and including `len(counts)`; then `return ranks * counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_product():
    mo.md(r"""
    Rank times count is Zipf's own way of writing the law: if it stays the same, the count is a constant divided by the rank. In the picture, the count of the English books falls by a factor of more than 30,000 down the list, and rank times count changes by a factor of 6: much flatter, though not flat. It rises over the first ranks, and it drops in steps at the end, where thousands of words share one count and their order among each other is only a place in a list of equals.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_growth():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. growth: new words keep coming

        Write `growth(words, step)`: go through `words` once, with a set of the words seen so far, in small letters. After every `step` words, add a pair to a list: how many words you have read, and how many different words the set holds. A few words left at the end, fewer than `step`, give no point. Give back the list. The worked example made a new set for every point; on a corpus of a million words, that takes too long, so here one set grows as you go. `n % step` gives the remainder of `n` divided by `step`. For `growth(ALICE, 5000)`, the first two points are `(5000, 1066)` and `(10000, 1552)`. Replace the `...`. The check calls `growth(ALICE, 5000)`, then cases of its own: short lists, one with capitals, a step that does not fit the length exactly, a step longer than the list, and an empty list.
        """),
        L.task("Write down how many different words a text holds, every few words."),
    ])
    return


@app.cell(hide_code=True)
def exercise_growth():
    box_growth = L.code_box("growth", lines=8)
    box_growth
    return (box_growth,)


@app.cell(hide_code=True)
def check_growth(box_growth):
    result_growth = L.check("growth", box_growth.value)
    result_growth
    return (result_growth,)


@app.cell(hide_code=True)
def hints_growth():
    L.hints(
        "growth",
        [
            "Start with an empty set `seen` and an empty list `points`. Go through the places `i` with `range(len(words))`, and add each word in small letters to `seen`. When the number of words read so far is a whole number of steps, add the pair of that number and `len(seen)` to `points`.",
            "After the word at place `i`, how many words have you read, since `i` counts from 0?",
            "`        if ___ % step == 0:`",
        ],
        why="`seen = set()` and `points = []`, then `for i in range(len(words)):` with `seen.add(words[i].lower())` under it, and `if (i + 1) % step == 0:` with `points.append((i + 1, len(seen)))`. After the loop: `return points`.",
    )
    return


@app.cell(hide_code=True)
def text_after_growth():
    mo.md(r"""
    Both curves bend on normal axes: the different words come more and more slowly. On log-log axes they run almost straight, a power law, with $\beta$ about 0.64 for the English books and 0.65 for the Dutch sentences between 10,000 and 100,000 words: a little above the 0.44 to 0.56 that Jurafsky and Martin give, though they add "or even higher". The English line jumps where a new book starts, such as *Pride and Prejudice* after the 26,776 words of *Alice*: each book brings names and words of its own.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: the words of a corpus

    Choose a corpus: one of the eight books, the five English books together, the three Dutch books together, or the Dutch sentences of Tatoeba. The picture gives its numbers, its counts by rank and its growth on log-log axes, and the words at a few ranks. Compare *Alice* with *Moby-Dick*, the English books with the Dutch ones, and the old spelling of the Dutch books with the Dutch of today in Tatoeba.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    _names = C.corpus_names(L.lang)
    demo_corpus = mo.ui.dropdown(options=_names, value=list(_names)[0], label="Corpus")
    demo_corpus
    return (demo_corpus,)


@app.cell(hide_code=True)
def demo_look(demo_corpus):
    mo.Html(C.demo_picture(demo_corpus.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Every corpus shows both laws, and every corpus has numbers of its own. In the three Dutch books, 51.0% of the different words occur only once, against 37.4% in the English ones. One reason, among others: Dutch writes a compound as one word, so *treinkaartje* is one rare word where English writes *train ticket*, two common ones. In Tatoeba, *tom* is the seventh most common word, since 12.7% of its Dutch sentences use the name Tom, and *ik* (I) and *je* (you) stand high, as in conversation; in the books, *en* and *de* lead. A count from a corpus is a count of that corpus, as [Corpora and treebanks](/course/open/nlp-corpora?lang=en) said.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_laws():
    mo.md(r"""
    ### Exercise 4. Which law?

    Each bold statement is about one of the two laws of this lesson. Put it under that law.
    """)
    return


@app.cell(hide_code=True)
def exercise_laws():
    sort_laws = L.sort("laws")
    sort_laws
    return (sort_laws,)


@app.cell(hide_code=True)
def check_laws(sort_laws):
    result_laws = L.check("laws", sort_laws)
    result_laws
    return (result_laws,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Corpora and treebanks](/course/open/nlp-corpora?lang=en) put together collections of text; this lesson counted their words. In every corpus, a few words are very common and most are rare, roughly as Zipf's law says, and new words keep coming as the text goes on, roughly as Heaps' law says. You counted words by rank, multiplied rank by count with NumPy, followed the different words through a text, and told the two laws apart. Because new words never stop, the tokenizers of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) cut rare words into subwords; a later lesson has you build such a tokenizer. The next lesson of the domain is about languages themselves: *About 7,000 languages*.

    **An everyday example.** The next time you meet a word you have never seen, in a book, a game or on a menu, think of Heaps: however much you read, there is always one more.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_growth, result_laws, result_product, result_rank_counts):
    L.wrap_up(result_rank_counts, result_product, result_growth, result_laws)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
