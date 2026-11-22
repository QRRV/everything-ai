# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-language-change. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-language-change", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Language, groups and dialects](/course/open/nlp-language-and-society?lang=en) and one about [Corpora and treebanks](/course/open/nlp-corpora?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_variety():
    quick_variety = L.choice(
        "quick-variety",
        question="A word list made from a treebank of standard Dutch misses nearly half of the words of Gronings sentences, and about 1 in 16 words of their Dutch translations. What does that say?",
        options={
            "right": "A tool built on the list knows fewer of the words of Gronings",
            "mistakes": "People who write Gronings make more spelling mistakes than people who write Dutch",
            "topics": "The Gronings sentences are about other things than their translations",
        },
        correct="right",
        why={
            "right": "The list knows the words of the variety it was made from. This lesson adds a second reason why a list can miss words: time, since a language changes.",
            "mistakes": "An unknown word says something about the word list, not about the speaker: *nait* is the Gronings word for *niet*, not a misspelling. See *What a tool knows* in [Language, groups and dialects](/course/open/nlp-language-and-society?lang=en).",
            "topics": "Each pair says the same thing, so the topic is the same on both sides. See *What a tool knows* in [Language, groups and dialects](/course/open/nlp-language-and-society?lang=en).",
        },
    )
    quick_variety
    return (quick_variety,)


@app.cell(hide_code=True)
def quick_check_concordance():
    quick_concordance = L.choice(
        "quick-concordance",
        question="You want to see how a word is used in a corpus. What does a concordance of the word give you?",
        options={
            "counts": "How often each word of the corpus occurs, from the most common word down",
            "right": "Every place where the word stands, with the words around it",
            "trees": "The tree of each sentence the word is in",
        },
        correct="right",
        why={
            "right": "A concordance lines up every place where the word stands, with its neighbors on both sides. This lesson uses one to compare how a word was used in novels of the 19th century and how it is used now.",
            "counts": "That is a word list with counts. What does a concordance line up? See *Every use of a word: the concordance* in [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
            "trees": "That is what a treebank adds to its sentences. What does a concordance line up? See *Every use of a word: the concordance* in [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
        },
    )
    quick_concordance
    return (quick_concordance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every text was written by someone, somewhere, at some time. Jurafsky and Martin write in the chapter *Words and Tokens* of *Speech and Language Processing*: "And finally, time matters too. Language changes over time, and for some languages we have good corpora of texts from different historical periods." A model learns the language of its data, and that language moves on. In 2021, Angeliki Lazaridou and others trained models on news and scientific abstracts up to September 2017, and tested them on text of 2018 and 2019: the models "perform up to 16% worse when predicting articles that are published up to 2 years after the end of the training period." In one of their collections of news, about 27% of the different words of each month had never occurred in the training period, such as *Brexiteers* and *MeToo*. Languages also meet. People who know two languages often use both, even in one sentence: "Code switching is enormously common across the world," Jurafsky and Martin write in the same chapter, and a tool that expects one language per text can stumble over such sentences. This lesson looks at sentences that switch language, and counts how Dutch changed between three novels of the 19th century and sentences written today.

    **An everyday example.** Old letters, recipes and newspaper pages of a hundred years ago write *menschen*, *tusschen* and *oogen*: words you know, in a spelling you would not use.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Three Dutch novels of the 19th century, 469,251 words together, have the word *zoo* 2,587 times. How often does *zoo* occur in the 1,301,039 words of the Dutch sentences of Tatoeba, written by volunteers today?",
        options={
            "zero": "0 times",
            "right": "15 times",
            "few": "About 300 times",
            "same": "About 2,500 times",
        },
        reveal="**15 times**, and each time it is another word: *zoo* for a *dierentuin*, as in English, in sentences such as *Ik ben naar de zoo geweest*, I went to the zoo. The *zoo* of the novels is the old spelling of *zo*, so, which the novels write only 3 times, and today's sentences 3,154 times. One form, two words: a spelling that changed, and a word for a place with animals.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Borrowing

    Languages take words from each other. The World Loanword Database (WOLD), edited by Martin Haspelmath and Uri Tadmor, puts it this way: "A loanword is a word that was copied from another language, either by adoption or by retention, at some point in the history of the language." English has *cookie* from Dutch *koekje*, *coleslaw* from *koolsla*, *yacht* from an older Dutch *jaght*, and *boss* from *baas*, Merriam-Webster's dictionary says. Dutch took *trottoir* and *cadeau* from French, and *tram*, *weekend* and *computer* from English. In WOLD's word list for Dutch, 1,513 words for a fixed list of common meanings, 19% are clearly or probably borrowed; in its list for English, 42%. How much of a language is borrowed depends on what you count. In a count by Nicoline van der Sijs of four pages of the newspaper NRC Handelsblad of 1994, 30.7% of the 2,144 different words had once come into Dutch from another language, and counted at every place where a word occurred, the share fell to 16.2%. Borrowing goes both ways: the book *Nederlandse woorden wereldwijd* of Van der Sijs, of 2010, lists 17,560 Dutch words borrowed by one or more of 138 languages.

    ### Code-switching

    People who know two languages often use both in one conversation, or in one sentence: **code-switching**. Shana Poplack defined it in 1980: "Code-switching is the alternation of two languages within a single discourse, sentence or constituent." The title of that article is a sentence of a speaker in Poplack's study: *Sometimes I'll start a sentence in Spanish y termino en español* (and I finish in Spanish). Poplack named three kinds: a switch between sentences, a short insert such as an interjection or a fixed phrase, which Poplack calls a tag, and a switch within a sentence, which needs the most skill, Poplack wrote. Ad Backus studied how Turkish speakers in the Netherlands use Turkish and Dutch together, in sentences such as *Babama bir smoes verzinnen yaptik, schoolfeestje var diye*: we made up an excuse for my father (*smoes verzinnen*), we said there was a school party (*schoolfeestje*). A switch can even fall inside a word: in *nieuwkomers'a*, the Dutch word *nieuwkomers*, newcomers, gets the Turkish ending *-a*, where Dutch has *naar*, to.

    Borrowing and code-switching are not always easy to tell apart. A. Seza Doğruöz and others write in a survey of 2021: "While C-S implies active alternation between grammatical systems, borrowing does not." C-S is their short form for code-switching. For a single word from another language, whether it is a loanword or a switch is hard to tell without looking at how the word fits into the grammar of the other language, they write. For tools, code-switching is hard: "language technologies are not yet capable of processing C-S as seamlessly as monolingual data," the survey says. In 2014, Thamar Solorio and others held a competition of the kind called a shared task, in which research teams solve the same task on the same data: give every word of texts with code-switching one of six labels, such as the first language, the second language and mixed. Seven teams took part. Working out which language a text or a word is in is called language identification, and the organizers wrote that "language identification in code-switched data is still far from solved and warrants further research."

    ### Semantic change

    Words also change what they stand for, while their form stays: **semantic change**. A word can come to mean less: Old English *dēor* could refer to many kinds of animals, and became English *deer*. It can also come to mean something else entirely: English *silly* goes back to Old English *sǣlig*, lucky or blessed, *nice* once meant foolish, and *awful* once meant full of awe. Dutch has the same. According to the *Etymologisch Woordenboek van het Nederlands*, *slim* first meant crooked, then bad, and from 1602 on it is found meaning clever. *Gemeen* meant shared, as it still does in *iets gemeen hebben*, to have something in common, then ordinary, and only from the 18th century low or mean. *Erg* meant bad; the dictionary first finds it meaning *very* in 1867. *Zeer*, which meant painful, as it still does in *zeer doen*, to hurt, was already a word for *very* in Middle Dutch. And *schoon* and the words related to it first meant beautiful in all Germanic languages; in the Netherlands it is now old-fashioned in that meaning, except in fixed phrases such as *de schone kunsten*, the fine arts, and means clean, while in Belgium the meaning beautiful is still common, as in German.

    Embeddings, which a later lesson of this domain builds in *Word embeddings: neighbors and analogies*, can measure such change. William Hamilton, Jure Leskovec and Dan Jurafsky trained embeddings on the texts of each decade of six historical corpora in four languages, and found two laws: "The law of conformity: Rates of semantic change scale with a negative power of word frequency." and "The law of innovation: After controlling for frequency, polysemous words have significantly higher rates of semantic change." Frequent words change their meaning more slowly; words with many meanings change faster.

    ### Spelling

    Spelling changes too, often by decision. Dutch got its first official spelling in 1804, by Siegenbeek; the word list of De Vries and Te Winkel followed in the 1860s. In 1934, Minister Marchant simplified the spelling for schools, and a law of 1947 in the Netherlands, after a decree of 1946 in Belgium, made a spelling with those changes official. The first *Groene Boekje*, the official word list, appeared in 1954, and new spelling rules came in 1995 and 2005. The explanatory memorandum of 2005 to the Dutch spelling law sums up what changed in 1947: a double *e* and *o* at the end of a syllable became single, *sch* was written *s* where the *ch* is not pronounced, and the case endings became optional. So *oogen* became *ogen*, eyes, and *menschen* became *mensen*, people; and *den* and *eene*, case forms of *de* and *een*, the and a, are rare in the sentences of Tatoeba.

    The three Dutch novels of the course first came out in 1839, 1860 and 1889, and the course's files hold later printings: *Camera Obscura* is the 29th printing, of 1917, and *Max Havelaar* an edition of 1881 or later, which writes *y* for *ij*, as in *hy* and *zyn*. All three are in spelling from before 1934. English spelling changes too: most of the English novels of the course write *to-day* and *to-morrow*.

    ### A corpus of its time

    So a corpus shows the language of its time and its people. The three novels hold the Dutch of three writers of the 19th century, in print from before 1934; the sentences of Tatoeba hold the Dutch of volunteers today, short sentences of 6.5 words on average. When you compare the two, part of the difference comes from time, and part from who wrote what, and for whom. Jurafsky and Martin advise, in the same chapter, describing every corpus with questions such as "Situation: When and in what situation was the text written/spoken?" and "Language variety: What language (including dialect/region) was the corpus in?" The picture shows a sentence that switches language, and a sentence of one of the novels in its old spelling.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **loanword** | a word that a language took over from another language, such as English *cookie* from Dutch *koekje* |
        | **code-switching** | using two languages in one conversation or one sentence, between sentences, as a short insert, or within a sentence |
        | **language identification** | working out which language a text or a word is in |
        | **semantic change** | a change in what a word stands for, while its form stays, such as Dutch *slim*: crooked, then bad, then clever |
        | **official spelling** | the way of writing a language that a government lays down, such as the spelling of the *Groene Boekje* |
        | **per 10,000 words** | how often a word occurs for every 10,000 words of a corpus: its count divided by all words of the corpus, times 10,000, so that corpora of different sizes can be compared |
        | **language tag** | a short code for the language of a word, such as `"nl"` for Dutch and `"tr"` for Turkish |
        """),
        L.watch_out("A difference between two corpora is not all change over time: the novels and the sentences of Tatoeba also differ in who wrote them, about what, and for whom. And an old spelling is not a mistake: *menschen* was the right spelling when these novels were printed."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. To find code-switching, give each word of a sentence a language tag, and walk along the sentence comparing each tag with the one before it.
    2. To compare a word between then and now, count it in a corpus of each time, and divide each count by the size of its own corpus, and multiply by 10,000: per 10,000 words.
    3. To see how a word is used, count which words follow it in each corpus, and read a few places where it stands.
    4. Read the numbers with what the corpora are: who wrote them, when, and for whom.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell holds the title sentence of Poplack's article as a list of words, `POPLACK`, and the language of each word as a list of tags, `POPLACK_TAGS`: `"en"` for English and `"es"` for Spanish. The second cell counts how many words each language has, in a dict from the tag to its count.
    """)
    return


@app.cell
def poplack_cell():
    POPLACK = C.POPLACK
    POPLACK_TAGS = C.POPLACK_TAGS
    print(POPLACK)
    print(POPLACK_TAGS)
    return (POPLACK_TAGS,)


@app.cell
def example_per_language(POPLACK_TAGS):
    def per_language(tags):
        counts = {}
        for tag in tags:
            if tag in counts:
                counts[tag] = counts[tag] + 1
            else:
                counts[tag] = 1
        return counts

    print(per_language(POPLACK_TAGS))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sentence has seven English words and four Spanish ones. The counts say which languages a sentence has, but not where it switches: they would be the same if the words were shuffled. To find the switches, look at each word together with the word before it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_switches():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. switches: where a sentence changes language

        `BACKUS` below holds the sentence of Ad Backus's research from the picture as a list of words, and `BACKUS_TAGS` the language of each word: `"tr"` for Turkish and `"nl"` for Dutch. Write `switches(tags)`: the number of places where a word has another tag than the word before it. For `switches(BACKUS_TAGS)`, the result is 4: from Turkish to Dutch, back to Turkish, to Dutch again, and back to Turkish. A tag `"mixed"`, for a word that is half one language and half the other, counts as a tag of its own. Replace the `...`. The check calls `switches(BACKUS_TAGS)`, then cases of its own: Poplack's title, a sentence with a mixed word, one word, no words, a switch at every word, and a sentence in one language.
        """),
        L.task("Count where a sentence switches from one language to another."),
    ])
    return


@app.cell
def backus_cell():
    BACKUS = C.BACKUS
    BACKUS_TAGS = C.BACKUS_TAGS
    print(BACKUS)
    print(BACKUS_TAGS)
    return


@app.cell(hide_code=True)
def exercise_switches():
    box_switches = L.code_box("switches", lines=6)
    box_switches
    return (box_switches,)


@app.cell(hide_code=True)
def check_switches(box_switches):
    result_switches = L.check("switches", box_switches.value)
    result_switches
    return (result_switches,)


@app.cell(hide_code=True)
def hints_switches():
    L.hints(
        "switches",
        [
            "Start a count `n` at 0. Go through the places of the list, from the second place to the end, and compare the tag at each place with the tag at the place before it. When they differ, add 1 to `n`. After the loop, give back `n`.",
            "At place `i`, which place holds the tag of the word before it?",
            "`        if tags[i] != tags[___]:`",
        ],
        why="`n = 0`, then `for i in range(1, len(tags)):` with `if tags[i] != tags[i - 1]:` under it, and `n = n + 1`. After the loop: `return n`.",
    )
    return


@app.cell(hide_code=True)
def text_after_switches():
    mo.md(r"""
    Poplack's title switches once, from English to Spanish. Backus's first sentence switches four times, and the second, *Geldiğimde iki ay nieuwkomers'a gittim*, goes from Turkish to a mixed word and back: a tag per word catches a switch inside a sentence, and the label mixed one inside a word. A tool that gives a whole text one language cannot show where a sentence switches; in the competition of Solorio and others, every word got a label of its own.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. per_10k: comparing corpora of different sizes

        `NOVEL_COUNTS` below holds how often each word occurs in the three novels, in small letters, and `TATOEBA_COUNTS` the same for the Dutch sentences of Tatoeba. The novels have 469,251 words, the sentences 1,301,039: with bare counts, a word can look more common in the larger corpus only because that corpus is larger. Write `per_10k(counts, word)`: how often `word` occurs for every 10,000 words of the corpus, that is, its count divided by the number of all words of the corpus, the sum of all counts, times 10,000. A word that is not in `counts` occurs 0 times. For `per_10k(NOVEL_COUNTS, "zoo")`, the result is 55.13. Replace the `...`. The check calls `per_10k(NOVEL_COUNTS, "zoo")`, then cases of its own: *zoo*, *zo* and *menschen* in Tatoeba, a small dict of counts, a word that is not in it, and an empty dict.
        """),
        L.task("Work out how often a word occurs per 10,000 words, to compare corpora of different sizes."),
    ])
    return


@app.cell
def counts_cell():
    NOVEL_COUNTS = C.novel_counts()
    TATOEBA_COUNTS = C.tatoeba_counts()
    print(len(NOVEL_COUNTS), NOVEL_COUNTS["zoo"], len(TATOEBA_COUNTS), TATOEBA_COUNTS["zo"])
    return


@app.cell(hide_code=True)
def exercise_per():
    box_per = L.code_box("per-10k", lines=7)
    box_per
    return (box_per,)


@app.cell(hide_code=True)
def check_per(box_per):
    result_per = L.check("per-10k", box_per.value)
    result_per
    return (result_per,)


@app.cell(hide_code=True)
def hints_per():
    L.hints(
        "per-10k",
        [
            "Add up all counts in a variable `total`: go through the words of `counts` and add the count of each. Then give back 0 when `word` is not in `counts`. Otherwise divide the count of `word` by `total`, and scale the result up.",
            "Inside the loop, `w` is one word of `counts`. How do you get how often it occurs?",
            "`        total = total + ___`",
        ],
        why="`total = 0`, then `for w in counts:` with `total = total + counts[w]` under it. After the loop: `if word not in counts:` with `return 0`, and then `return counts[word] / total * 10000`.",
    )
    return


@app.cell(hide_code=True)
def text_after_per():
    mo.md(r"""
    Each old spelling is common in the novels and rare or absent in today's sentences, and each new spelling the other way round: *oogen* occurs 10.7 times per 10,000 words in the novels and not once in Tatoeba, *ogen* not once in the novels and 3.9 times in Tatoeba. That is the change of 1934 and 1947 in numbers. Per 10,000 words, the novels also use *zoo* more than Tatoeba uses *zo*, 55.1 against 24.2 times: those are other texts, with another style, and the count does not say why. Without dividing, *nodig* (1,959 times in Tatoeba) would look 11 times as common as *noodig* (176 times in the novels); per 10,000 words, it is 4 times.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. next_words: the company a word keeps

        `NOVELS` below holds the sentences of the three novels, cut at every full stop, question mark and exclamation mark, each a list of its words; `TATOEBA` holds each Dutch sentence of Tatoeba, from tatoeba.org, as a list of its words. Write `next_words(sentences, target)`: count which words come right after `target`, in a dict from the next word, in small letters, to how often it follows. Compare each word with `target` in small letters, so that *Schoon* at the start of a sentence counts too; `target` itself is given in small letters. The last word of a sentence has no next word. For `next_words(TATOEBA, "schoon")`, the dict holds 23 words, and `"te"` follows 20 times. Replace the `...`. The check calls `next_words(TATOEBA, "schoon")`, then cases of its own: *schoon* and *erg* in the novels, the word with a capital and at the end of a sentence, a next word with a capital, the word twice in one sentence, a sentence of one word, and no sentences.
        """),
        L.task("Count which words follow a word in a corpus, to see how its use changed."),
    ])
    return


@app.cell
def sentences_cell():
    NOVELS = C.novels()
    TATOEBA = C.tatoeba()
    print(len(NOVELS), NOVELS[100])
    print(len(TATOEBA), TATOEBA[100])
    return


@app.cell(hide_code=True)
def exercise_next():
    box_next = L.code_box("next-words", lines=11)
    box_next
    return (box_next,)


@app.cell(hide_code=True)
def check_next(box_next):
    result_next = L.check("next-words", box_next.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next-words",
        [
            "Start with an empty dict `counts`. Go through the sentences, and inside that loop through the places of each sentence, up to the place before the last word. When the word at a place is `target` in small letters, take the word at the next place, in small letters, and update its count. After both loops, give back the dict.",
            "The word `target` stands at place `i`. Which place holds the word that follows it?",
            "`                w = sentence[___].lower()`",
        ],
        why="`counts = {}`, then `for sentence in sentences:` and inside it `for i in range(len(sentence) - 1):`, with `if sentence[i].lower() == target:` and under it `w = sentence[i + 1].lower()`, then `if w in counts:` with `counts[w] = counts[w] + 1`, and `else:` with `counts[w] = 1`. After both loops: `return counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_next():
    mo.md(r"""
    In the novels, *schoon* is most often followed by *ik*, *en* and *was*. Followed by *ik*, it means although, as in *schoon ik gaarne beken*, though I gladly admit, a use that goes back to a meaning *schoon* took on towards 1500, the etymological dictionary of De Vries says; elsewhere it means beautiful, as in *hoe schoon en voortreffelijk*, how beautiful and excellent. In Tatoeba, *schoon* is followed by *te*, as in *schoon te maken*, to clean, and by *houden* and *drinkwater*: there it means clean. *Erg* is followed by a word 41 times in the novels, most often by *best* and *boos*, and 1,824 times in Tatoeba, most often by *goed*, *van* and *druk*. Per 10,000 words, *erg* went from 1.0 in the novels to 14.3 in Tatoeba, and *zeer* from 18.2 to 3.6: in the novels, *zeer* is the usual word for *very*, as in *zeer goed*, in Tatoeba *erg* and *heel*. *Erg* took the path from a word for something bad to a word for *very*, as *zeer*, painful, had done centuries earlier.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a word then and now

    Type a word. The picture shows how often it occurs per 10,000 words in the three novels and in the sentences of Tatoeba, and a few of the places where it stands in each, with the words around it: a concordance. Try *schoon*, *erg*, *zeer*, *gaarne*, *mevrouw*, *leuk* or a word of your own.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="schoon", label="Word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    *Gaarne*, gladly, occurs 165 times in the novels and not once in Tatoeba, where *graag* occurs 1,505 times. *Mevrouw* goes from 18.0 to 0.5 per 10,000 words, and *leuk* from none in the novels to 6.9 in Tatoeba. Some of this is change, and some is the difference between novels of the 19th century and short everyday sentences.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. Borrowed, switched or changed?

    Each bold statement shows one way in which languages meet or change. Put it under what it shows.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_change = L.sort("borrow-switch-change")
    sort_change
    return (sort_change,)


@app.cell(hide_code=True)
def check_sort(sort_change):
    result_sort = L.check("borrow-switch-change", sort_change)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Language, groups and dialects](/course/open/nlp-language-and-society?lang=en) showed that a language varies between regions, groups and situations; this lesson added time and contact. Languages borrow words from each other, and how much of a language is borrowed depends on what you count. People who know two languages often switch between them, even inside a sentence, which tools still handle less well than text in one language. Words change what they stand for, as *slim* and *schoon* did, and spelling changes, often by decision, as in 1934 and 1947. You counted where sentences switch language, compared old and new spellings per 10,000 words in two corpora of different sizes, and counted the words after *schoon* and *erg* then and now. The concordance of [Corpora and treebanks](/course/open/nlp-corpora?lang=en) showed you a word in its places; counting its neighbors over a whole corpus shows its use in numbers. A corpus shows the language of its time and its people, so describe when, by whom and for whom it was written. This lesson ends the module on linguistics; the next lesson of the domain starts processing text: *Finding patterns: regular expressions*.

    **An everyday example.** When a translation app gives an old-fashioned word, or does not know a new one, ask from which years its data came.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_next, result_per, result_sort, result_switches):
    L.wrap_up(result_switches, result_per, result_next, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
