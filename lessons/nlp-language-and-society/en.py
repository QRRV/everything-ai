# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-language-and-society. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-language-and-society", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_family():
    quick_family = L.choice(
        "quick-family",
        question="Glottolog puts Dutch, English and Frisian in one branch of a family: West Germanic. What does that say about the three languages?",
        options={
            "order": "They put the subject, the verb and the object in the same order",
            "right": "They grew out of one older language",
            "understand": "Their speakers can understand each other",
        },
        correct="right",
        why={
            "right": "A family is about descent: Dutch, English and Frisian grew out of one older language. Whether their speakers understand each other is another question, and this lesson comes back to it.",
            "order": "That would be a type, not a family: related languages can differ in their word order, as Dutch and English do. What does a branch of a family stand for? See *Families* in [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
            "understand": "Related languages can be far apart: an English speaker does not follow Frisian without learning it. What does a branch of a family stand for? See *Families* in [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
        },
    )
    quick_family
    return (quick_family,)


@app.cell(hide_code=True)
def quick_check_dialect():
    quick_dialect = L.choice(
        "quick-dialect",
        question="When does Glottolog count two ways of speaking as two languages, and not as two dialects of one language?",
        options={
            "countries": "When they are spoken on either side of a national border",
            "spelling": "When each has a spelling of its own",
            "right": "When their speakers do not understand each other",
        },
        correct="right",
        why={
            "right": "Glottolog's measure is whether the speakers understand each other. Other people decide by other measures, such as a law or a spelling, and this lesson shows where those measures part ways.",
            "countries": "A border does not split a language for Glottolog: Dutch is spoken in the Netherlands and in Belgium. What does Glottolog look at? See *How many languages?* in [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
            "spelling": "A spelling is a decision of people, and Glottolog looks at the speakers themselves. What does Glottolog look at? See *How many languages?* in [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
        },
    )
    quick_dialect
    return (quick_dialect,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In November 1962, William Labov asked salespeople in three department stores in New York for a department that was on the fourth floor, and listened to how they said *fourth floor*: with or without the *r*. The stores were ranked from high to low: Saks Fifth Avenue, Macy's and S. Klein. At Saks, 62 percent of the salespeople pronounced the *r* at least once, at Macy's 51 percent and at S. Klein 20 percent: the same city and the same two words, and still the way people spoke went with the rank of their store. Such patterns are what sociolinguistics studies, and text for NLP holds them too. Jurafsky and Martin write in the chapter *Words and Tokens* of *Speech and Language Processing*: "most languages also have multiple varieties, often spoken in different regions or by different social groups." A tool learns the varieties of its data, and works less well for the others. In 2016, Su Lin Blodgett, Lisa Green and Brendan O'Connor ran langid.py, a tool that tells which language a text is in, on tweets: "Of the AA-aligned tweets, 13.2% were classified by langid.py as non-English; in contrast, 7.6% of white-aligned tweets were classified as such." The AA-aligned tweets came from users whose words their model linked, through census figures for the places the tweets were sent from, to African Americans; the authors checked that this language shows known traits of African-American English, a variety of American English spoken by millions of people. In 2020, Allison Koenecke and others tested the systems of five companies that turn speech into text, on interviews with 42 white and 73 black speakers: "the error rates for black speakers are nearly twice as large in every case." The white speakers were recorded in California and the black speakers in the east of the United States, so at least some of the difference may come from where the speakers live rather than from their ethnic group, they add. This lesson makes the same kind of count for Dutch, with words you can see: what a word list of standard Dutch misses in Frisian, Gronings and Limburgish.

    **An everyday example.** Type a word of your own dialect or of your city on your phone, and see whether it gets a red line under it, or turns into another word.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Tatoeba, a collection of sentences written and translated by volunteers, has 13,206 sentences in Gronings, which is spoken in the province of Groningen, that come with a Dutch translation. Of the words of the Dutch translations, about 1 in 16 is missing from a word list made from the Dutch treebank of the course. And of the words of the Gronings sentences?",
        options={
            "eight": "About 1 in 8: twice as many",
            "four": "About 1 in 4",
            "right": "Nearly 1 in 2",
        },
        reveal="**Nearly 1 in 2**: 32,492 of the 68,803 words, 47.2%, against 4,539 of 71,152, 6.4%, in their Dutch translations. The sentences say the same, so the difference comes from the variety, not from what the sentences are about. Among the ten words the list misses most often are small, common ones such as *n*, *nait*, *veur* and *noar*, where Dutch has *een*, *niet*, *voor* and *naar*.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Varieties

    Nobody speaks a language in only one way. In the first lines of their introduction, Blodgett, Green and O'Connor write: "regional and social dialects exist within languages across the world. These varieties or dialects differ from the standard variety in syntax (sentence structure), phonology (sound structure), and the inventory of words and phrases (lexicon)." Each such form of a language is a **language variety**, or **variety** for short; the word does not judge which form is the real language. A variety that goes with a region is a **dialect**, one that goes with a social group a **sociolect**, also called a social dialect, and the way of speaking or writing that goes with a situation a **register**: you write differently to a friend than to a city council. An **accent** is only a way of pronouncing. One person can use more than one variety, from one moment to the next. **Sociolinguistics** studies how language goes with regions, groups and situations, and how people show with it who they are.

    ### Standard Dutch in two countries

    One variety has a special place: the **standard language**. Taaladvies.net, the language advice of the Dutch Language Union (Nederlandse Taalunie), describes it as the Dutch that can be used in all important parts of public life, such as government, administration, the courts, education and the media. It is a variety too, and not quite the same everywhere: for the most part, Taaladvies writes, standard Dutch is the same in the Netherlands and in Belgium, but some words are standard in only one of the two, labeled *standard language in Belgium* or *standard language in the Netherlands*. A Dutch person speaks of a *pinpas*, a debit card, and a Belgian of a *bankkaart*, and both words are standard language. The mayor and aldermen of a municipality are *burgemeester en wethouders* in the Netherlands, and *burgemeester en schepenen* in Belgium. Dutch is "the mother tongue of more than 25 million people," the Taalunie writes on its website. The Taalunie goes back to a treaty between Belgium and the Netherlands of 1980, and signed an association agreement with Suriname in 2003.

    Between dialect and standard language, Flanders has **tussentaal**. Team Taaladvies of the Flemish government describes it as both the careful speech of Flemish speakers of a dialect who cannot or will not speak the standard language, and the relaxed speech of Flemish speakers of the standard language who cannot or will not speak a dialect in more personal or informal moments. It shows the steps with one question, in standard language, tussentaal and a dialect: *Heb je nog een kop koffie voor me?*, *'Ebde nog een tas koffie voor mij?* and *Édde nog een zjat kaffe veu maai?* (do you have another cup of coffee for me?). The steps run into each other. Tussentaal has a bad name: "The prevailing attitude towards tussentaal is negative," Robertus de Louw wrote in 2016, and Geert van Istendael coined the name *Verkavelingsvlaams* for it in 1989.

    ### Languages beside Dutch

    The Netherlands also has languages of its own. By the Wet gebruik Friese taal, in force since January 1, 2014, Dutch and **Frisian** are the official languages of the province of Fryslân; in two surveys of 2025, 53% of the people of Fryslân gave Frisian as their first language. In the tree of Glottolog, Frisian is not a branch of Dutch: it shares the branch Anglo-Frisian with English. The Netherlands recognized Low Saxon in 1996, and Limburgish in 1997, as **regional languages** under the European Charter for Regional or Minority Languages. Gronings is one of the varieties of Low Saxon, as the Dutch government and Glottolog both say. By the Charter's own definition, a regional language is not a dialect of the official language: "it does not include either dialects of the official language(s) of the State or the languages of migrants." So calling Limburgish a regional language, and not a group of dialects of Dutch, is a decision of people, and the words stay the same either way. Max Weinreich made famous a saying about this, in Yiddish; in English it is usually given as: "A language is a dialect with an army and navy." In 1945, Weinreich wrote that a listener at a series of Weinreich's lectures had said it.

    ### What a tool knows

    A tool for text knows the words of the text it is built on. In this lesson that is a **word list**: every word of a treebank in standard Dutch, in small letters, with how often it occurs. A word that is not in the list is an **unknown word**: a tool that looks its words up in the list does not find it. The fewer words of a variety the list holds, the more such words a tool built on it meets. The picture lays two sentences of Tatoeba, a collection of sentences written and translated by volunteers, beside their Dutch translations, with the unknown words underlined, the way a spelling checker marks them.
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
        | **sociolinguistics** | the study of how language goes with regions, social groups and situations, and how people show with it who they are |
        | **language variety** | any form of a language that its speakers share, such as a dialect, a sociolect or the standard language |
        | **dialect** | a variety that goes with a region: a local form of a language |
        | **sociolect** | a variety that goes with a social group |
        | **register** | the way of speaking or writing that goes with a situation, such as a chat with friends or a letter to a city council |
        | **accent** | a way of pronouncing a language, typical of a region, a group or a person |
        | **standard language** | the variety for all of public life, such as government, the courts, education and the media; in Dutch, some words are standard only in Belgium or only in the Netherlands |
        | **tussentaal** | in Flanders: the speech between dialect and standard language |
        | **regional language** | a language that the state recognizes in a region and that is not a dialect of the official language, such as Low Saxon and Limburgish in the Netherlands |
        | **word list** | the words a tool knows: here every word of a treebank, in small letters, with how often it occurs |
        | **unknown word** | a word that is not in the word list |
        | `w in counts` | `True` when `w` is a key of the dict `counts`: when the word is in the list |
        | `x not in y` | `True` when `x` is not in `y`: the opposite of `x in y` |
        """),
        L.watch_out("A variety is not a broken form of the standard language: each variety has rules of its own, and the standard language is one variety among others, with a special place in public life. An unknown word says something about the word list, not about the speaker. And the sentences of Tatoeba show a variety as volunteers write it, not as its speakers talk with each other."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Make a word list from text in the standard language: every word, in small letters, with how often it occurs.
    2. Take sentences in a variety, and the same sentences in the standard language, each as a list of words.
    3. Count the words of each side that are not in the list, and divide by all words of that side.
    4. Compare the two shares: the sentences say the same, so the difference comes from the variety.
    5. Count which words the list misses most often, to see where a tool built on the list would stumble.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell holds a tiny word list of the course, `SMALL`, as a set, and a Frisian proverb from Tatoeba with its Dutch translation, each as a list of words: *De appel falt net fier fan de beam* and *De appel valt niet ver van de boom* (the apple does not fall far from the tree). The second cell gives back the words of a sentence that are not in the list, looked up in small letters, so that *De* at the start counts as *de*.
    """)
    return


@app.cell
def sentences_cell():
    SMALL = {"de", "appel", "valt", "niet", "ver", "van", "boom"}
    FRISIAN_EXAMPLE = ["De", "appel", "falt", "net", "fier", "fan", "de", "beam"]
    DUTCH_EXAMPLE = ["De", "appel", "valt", "niet", "ver", "van", "de", "boom"]
    return DUTCH_EXAMPLE, FRISIAN_EXAMPLE, SMALL


@app.cell
def example_missing(DUTCH_EXAMPLE, FRISIAN_EXAMPLE, SMALL):
    def missing(words, known):
        found = []
        for word in words:
            if word.lower() not in known:
                found.append(word)
        return found

    print(missing(FRISIAN_EXAMPLE, SMALL))
    print(missing(DUTCH_EXAMPLE, SMALL))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the Frisian proverb, the tiny list misses five of the eight words, and none of the Dutch translation; *De* is found, since the list is asked in small letters. With the word list of the whole treebank, in the picture above, the Frisian sentence keeps only three unknown words: *falt*, *fan* and *beam*. That list knows *net* and *fier*, because Dutch has those words too, with another meaning: in Frisian, *net* is Dutch *niet*, not, and *fier* is *ver*, far; in Dutch, *net* means neat or just, and *fier* proud. A word list knows forms, not meanings.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. word_counts: the word list of a treebank

        `ALPINO` below holds the 13,603 sentences of the Dutch treebank of [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en), each a list of its words as the treebank writes them, with capitals and punctuation. Write `word_counts(sentences)`: count how often each word occurs over all sentences, in small letters, so that *De* and *de* are one word, in a dict from the word to its count. The keys of the dict are the word list: `w in counts` asks whether the word `w` is in it. For `word_counts(ALPINO)`, the dict holds 27,169 words, and `"de"` occurs 13,035 times. Replace the `...`. The check calls `word_counts(ALPINO)`, then cases of its own: short lists of sentences, one word in capitals and in small letters, an empty sentence, and an empty list.
        """),
        L.task("Count the words of a treebank in standard Dutch: the word list of a tool."),
    ])
    return


@app.cell
def treebank_cell():
    ALPINO = C.treebank()
    print(len(ALPINO), ALPINO[0])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("word-counts", lines=10)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("word-counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "word-counts",
        [
            "Start with an empty dict `counts`. Go through the sentences, and inside that loop through the words of each sentence. Put each word in small letters, `w`, and update its count, or give it the count 1 when it is not in the dict yet. After both loops, give back the dict.",
            "The word `w` is already in `counts`. What is its new count?",
            "`                counts[w] = ___`",
        ],
        why="`counts = {}`, then `for sentence in sentences:` and inside it `for word in sentence:`, with `w = word.lower()` and `if w in counts:` under it, with `counts[w] = counts[w] + 1`, and `else:` with `counts[w] = 1`. After both loops: `return counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    Each Belgian word in the picture is standard language in Belgium, and still the treebank has none of *hospitaal*, *plezant*, *proper* and *bankkaart*, while it has *ziekenhuis* 27 times and *leuk* 19 times. Only *schepen* occurs, 11 times, and every time as the plural of *schip*, ship, and not once as a Belgian alderman: here, too, the list knows the form and not the meaning. In these pairs, a tool built on this treebank knows the words of the whole language area and of the Netherlands, and not those that are standard only in Belgium.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. share_unknown: how much the list misses

        Below, `COUNTS` holds the counts of the treebank, as `word_counts(ALPINO)` gives them: the word list. `FRISIAN` holds the 506 Frisian sentences of Tatoeba that have a Dutch translation, and `FRISIAN_NL` those translations, in the same order; `GRONINGS` and `GRONINGS_NL` the same for Gronings. Each sentence is a list of its words. Write `share_unknown(sentences, known)`: the share of all words of all sentences together that are not in `known`, looked up in small letters, as a number between 0 and 1. A long sentence weighs more than a short one, since it has more words. With no words at all, give back 0. For `share_unknown(FRISIAN, COUNTS)`, the result is 0.5877. Replace the `...`. The check calls `share_unknown(FRISIAN, COUNTS)`, then cases of its own: the Dutch translations, short sentences with a word list of its own, words with a capital, a long sentence and a short one, an empty sentence, and no sentences.
        """),
        L.task("Work out the share of the words of a variety that a word list of the standard language does not know."),
    ])
    return


@app.cell
def pairs_cell():
    COUNTS = C.counts()
    FRISIAN, FRISIAN_NL = C.sentences("fry")
    GRONINGS, GRONINGS_NL = C.sentences("gos")
    print(len(COUNTS), FRISIAN[3], FRISIAN_NL[3])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share-unknown", lines=11)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share-unknown", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share-unknown",
        [
            "Keep two counts, both starting at 0: `total`, for all words, and `unknown`, for the words that are not in `known` in small letters. Go through the sentences, keep `total` up to date at each sentence, and go through its words to keep `unknown` up to date. After the loops, give back 0 when there were no words, and else the share.",
            "How many words does one sentence add to `total`?",
            "`        total = total + ___`",
        ],
        why="`total = 0` and `unknown = 0`, then `for sentence in sentences:` with `total = total + len(sentence)` under it, and `for word in sentence:` with `if word.lower() not in known:` and `unknown = unknown + 1`. After the loops: `if total == 0:` with `return 0`, and then `return unknown / total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    The sentences of a pair say the same thing, so what the list misses on one side and not on the other comes from the variety, not from the topic. The list misses 58.8% of the Frisian words and 47.2% of the Gronings ones, against 6.1% and 6.4% in their Dutch translations. Even in standard Dutch, it misses about 1 in 16 words: names, and everyday words that the 13,603 sentences of a treebank never happened to meet, such as *fles* and *soep*. Limburgish has only 18 pairs, short sentences such as greetings, so its bars say little.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unknown():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. unknown_counts: the words the list misses

        Write `unknown_counts(sentences, known)`: count the words of the sentences with `word_counts`, and keep, in a new dict, only the words that are not in `known`, each with how often it occurs in the sentences. `word_counts` is ready for your function: the check gives it the course's own. For `unknown_counts(FRISIAN, COUNTS)`, the dict holds 745 words, and `"it"` occurs 68 times. Replace the `...`. The check calls `unknown_counts(FRISIAN, COUNTS)`, then cases of its own: the Gronings sentences, short sentences with a word list of its own, words with a capital, a word that comes back, only known words, and no sentences.
        """),
        L.task("Count which words of a variety a word list of the standard language misses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unknown():
    box_unknown = L.code_box("unknown-counts", lines=7)
    box_unknown
    return (box_unknown,)


@app.cell(hide_code=True)
def check_unknown(box_unknown):
    result_unknown = L.check("unknown-counts", box_unknown.value)
    result_unknown
    return (result_unknown,)


@app.cell(hide_code=True)
def hints_unknown():
    L.hints(
        "unknown-counts",
        [
            "Ask `word_counts(sentences)` for the counts. Make a new, empty dict `unknown`, and go through the words `w` of the counts: when `w` is not in `known`, give it a place in `unknown`. After the loop, give back `unknown`.",
            "Which number belongs with a word that you keep?",
            "`            unknown[w] = ___`",
        ],
        why="`counts = word_counts(sentences)` and `unknown = {}`, then `for w in counts:` with `if w not in known:` and `unknown[w] = counts[w]` under it. After the loop: `return unknown`.",
    )
    return


@app.cell(hide_code=True)
def text_after_unknown():
    mo.md(r"""
    In Frisian and Gronings, the words the list misses most often are small and common: Frisian *it*, *fan* and *yn*, where Dutch has *het*, *van* and *in*, and in Gronings, among others, *n*, *nait* and *veur*, for *een*, *niet* and *voor*. In Gronings, the 20 most common unknown words make up 24.8% of all unknown words in the sentences: with those 20 words added, the list would miss a quarter fewer words. On the Dutch side, the list misses names such as *Sami*, and everyday words such as *lawaai*, *zussen* and *fles*, and 71% of these words occur only once. The treebank does not even know the words *gronings* and *dialect*.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: sentence by sentence

    Choose a variety, and where to start. The picture shows six sentences of Tatoeba with their Dutch translations, the words the word list of the treebank does not know underlined, and counts what it misses in those six and in all pairs. Look at which words get a line in the Dutch translations, and which in the variety.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_variety = mo.ui.dropdown(options=C.variety_options(L.lang), value=list(C.variety_options(L.lang))[0], label="Variety")
    demo_variety
    return (demo_variety,)


@app.cell(hide_code=True)
def demo_slider(demo_variety):
    demo_start = mo.ui.slider(start=1, stop=C.last_start(demo_variety.value), step=1, value=1, label="Start at pair", full_width=True)
    demo_start
    return (demo_start,)


@app.cell(hide_code=True)
def demo_look(demo_start, demo_variety):
    mo.Html(C.demo_picture(demo_variety.value, demo_start.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In Gronings, the same small words come back in pair after pair, while the Dutch translations get a line only now and then, at a name or a word the treebank happens not to hold. Some Frisian pairs are proverbs whose translation is another proverb, such as *In keap is in keap* beside *Gedane zaken nemen geen keer*: a pair of Tatoeba says the same thing, but not always in the same words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. Region, group or situation?

    Each bold statement is about a difference in how people speak. Put it under what the difference goes with.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_variation = L.sort("region-group-situation")
    sort_variation
    return (sort_variation,)


@app.cell(hide_code=True)
def check_sort(sort_variation):
    result_sort = L.check("region-group-situation", sort_variation)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en) counted languages and families; this lesson looked inside one language. Nobody speaks a language in only one way: varieties go with a region, a social group or a situation, and the standard language is one variety among them, with a special place in public life. Standard Dutch differs a little between the Netherlands and Belgium, Flanders has tussentaal between dialect and standard language, and Frisian, Low Saxon and Limburgish beside it; whether something counts as a language or a dialect is a decision of people. You built a word list from a treebank in standard Dutch, worked out how much of three varieties it misses against their own translations, and counted which words it misses: in Frisian and Gronings most often small, common words, in Dutch names and everyday words the treebank never met. A tool built on one variety works less well on the others, so write down which variety your data is in: one of the questions Jurafsky and Martin list for describing a corpus. The next lesson of the domain is about how languages borrow from each other and change over time: *Languages meet and change*.

    **An everyday example.** When a voice assistant or a translation app gets your words wrong, ask whether its data held enough text by people who speak like you.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_counts, result_share, result_sort, result_unknown):
    L.wrap_up(result_counts, result_share, result_unknown, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
