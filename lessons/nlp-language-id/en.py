# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-language-id. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-language-id", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [A bigram model by counting](/course/open/lm-bigram-counts?lang=en), and one about [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bigrams():
    quick_bigrams = L.choice(
        "quick-bigrams",
        question="How many bigrams does `count_bigrams` count in *banana*, repeats included?",
        options={
            "three": "3: *ba*, *an* and *na*",
            "right": "5",
            "six": "6: one for each letter of the word",
        },
        correct="right",
        why={
            "right": "*banana* has six letters, and `count_bigrams` takes a pair at every place that still has a next letter: *ba*, *an*, *na*, *an*, *na*. Two of them come twice, and the counts keep both. This lesson cuts a text into runs of three characters in the same way, with a space before and after the text, so that the first and the last letters get runs of their own too.",
            "three": "Those are the different bigrams. With repeats there are five: *ba*, *an*, *na*, *an*, *na*, and `count_bigrams` counts *an* and *na* twice. See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
            "six": "A bigram needs a next letter, and the last *a* has none: six letters give five bigrams. See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
        },
    )
    quick_bigrams
    return (quick_bigrams,)


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="*Huis* and *huis* are the same Dutch word, *house*. What does lowercasing do before you count words?",
        options={
            "drop": "It drops every word that starts with a capital",
            "nothing": "Nothing: Python sees them as equal",
            "right": "It counts *Huis* and *huis* as one word",
        },
        correct="right",
        why={
            "right": "For Python, *H* and *h* are different characters, so without lowercasing the counts hold two keys for one word. This lesson lowercases too: a capital at the start of a sentence should not make runs of letters of its own.",
            "drop": "Lowercasing changes letters, not which words stay: *Huis* becomes *huis*. Leaving words out is what a list of stop words does. See [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
            "nothing": "For Python, `\"Huis\" == \"huis\"` is `False`: a capital is another character. See [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before a program can look up words, correct spelling or translate a text, it has to know which language the text is in. Tommi Jauhiainen, Marco Lui, Marcos Zampieri, Timothy Baldwin and Krister Lindén open their survey of the field, in 2019, with a definition: "Language identification (“LI”) is the task of determining the natural language that a document or part thereof is written in." Search engines use it to sort pages, translation services to pick a direction, and the people who collect text from the web to train language models use it to put every page in the pile of its language. This lesson builds such a program in the course's own code, after the method that William Cavnar and John Trenkle published in 1994: count the short letter sequences of a text, and compare them with those of each language. The same counts also show text that is not language at all, such as a line of HTML.

    **An everyday example.** A browser that offers to translate a page from Italian has guessed the language of the page from its text.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The program of this lesson learns twelve languages from short sentences of Tatoeba, a collection of sentences that volunteers write and translate: Dutch, Afrikaans, Frisian, German, English, Danish, Norwegian Bokmål (one of the two written forms of Norwegian), Swedish, French, Spanish, Portuguese and Italian. It then meets 1,190 sentences it has not seen. Which two languages does it mix up most often?",
        options={
            "afrikaans": "Dutch and Afrikaans",
            "right": "Danish and Bokmål",
            "romance": "Spanish and Portuguese",
            "frisian": "Dutch and Frisian",
        },
        reveal="**Danish and Bokmål**: 57 of the 192 sentences that get a wrong language. 31 Danish sentences come out as Bokmål and 26 Bokmål sentences as Danish, so only 54 of the 100 Danish sentences and 52 of the 90 Bokmål ones come out right. The two are written much alike: both have *jeg* for *I* and *er* for *is*. Spanish and Portuguese are mixed up 11 times, Dutch and Afrikaans 8 times. Exercise 3 runs your own program on the same 1,190 sentences.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Letter sequences give a language away

    Cavnar and Trenkle define, in their section *N-Grams*: "An N-gram is an N-character slice of a longer string." A **character n-gram** is a run of n characters of a text, spaces and punctuation included, and a **trigram** is one of three characters. *the cat* holds the trigrams *the* and *cat*, but also *he␣* and *e␣c*, where ␣ stands for a space. Cavnar and Trenkle also pad the text: "We also append blanks to the beginning and ending of the string in order to help with matching beginning-of-word and ending-of-word situations." Their paper pads each word and drops digits and punctuation; this lesson keeps it simpler, with one space before the whole text and one after, and every character kept. *the cat* then gives seven trigrams, from *␣th* to *at␣*: *␣th* says that a word starts with *th*, and *at␣* that a word ends in *at*.

    Some trigrams are far more common in one language than in another. The picture shows the ten most common trigrams of four languages, counted in the sentences of the course. English has *␣th*, *the* and *he␣*: the start, the whole word and the end of *the*. Dutch has *␣de* and *en␣*, German *ich* and *sch*, and Afrikaans *␣di* and *ie␣*, the start and the end of *die*, the Afrikaans word for *the*.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### A profile

    Count every trigram in the sentences of a language, and sort them from the most common down. The **profile** of the language is the top of that list: each trigram with its **rank**, 0 for the most common, then 1, 2, and so on. The profile of a text is made in the same way, from the text alone. How long should a profile be? Cavnar and Trenkle noticed that "The top 300 or so N-grams are almost always highly correlated to the language", and add: "There is nothing special about rank 300 itself"; "we arrived at this number mostly by inspection." This lesson keeps 300 trigrams for each language, from 400 sentences each (364 for Bokmål, which has fewer). It joins those sentences into one text, with a space between them, so a few trigrams span the end of one sentence and the start of the next, such as *.␣t* in English.

    ### Out of place

    To compare two profiles, Cavnar and Trenkle use what they call the "out-of-place" measure: "This measure determines how far out of place an N-gram in one profile is from its place in the other profile." Their example: "the N-gram “ING” is at rank 2 in the document, but at rank 5 in the category. Thus it is 3 ranks out of place." A *category* is their word for a language. The **out-of-place measure** adds up those numbers for every n-gram of the text's profile. An n-gram that the language lacks gets a penalty: "If an N-gram (such as “ED” in the figure) is not in the category profile, it takes some maximum out-of-place value." The paper gives no number for it. This lesson takes the size of the language's profile, 300: more than any trigram in the profile can be out of place, since ranks run from 0 to 299.

    A long text has more trigrams than a short one, so its sum is larger. This lesson therefore divides the sum by the most it can be, the number of trigrams of the text times 300, and calls that the **distance**: 0 when every trigram of the text sits at the same rank as in the language, 1 when the language lacks all of them.

    ### The nearest language, or none

    The program measures the distance to each of the twelve languages and takes the nearest. But the nearest is not always near: for a line of HTML, every language lies far away, and the program would still name one. Cavnar and Trenkle saw this in their conclusion: a normalized score "would allow the system to reject some documents on the grounds that their normalized scores were so low that the documents did not have good matches at all." The survey of Jauhiainen and colleagues says it in one line: "The easiest way to implement unknown language detection is through thresholding." This lesson sets a **limit** of 0.9: a language counts only when its distance lies below the limit, and when none does, the program answers *?*.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **language identification** | working out which language a text is written in |
        | **character n-gram** | a run of n characters of a text, spaces and punctuation included |
        | **trigram** | a character n-gram of three characters, such as *␣th*; ␣ is a space |
        | **profile** | the most common n-grams of a language or a text, each with its rank |
        | **rank** | the place of an n-gram in a profile: 0 for the most common, then 1, and so on |
        | **out-of-place measure** | for each n-gram of the text, how far its rank lies from its rank in the language, or a penalty when the language lacks it, all added up |
        | **distance** | in this lesson: that sum divided by the most it can be, between 0 and 1 |
        | **limit** | the distance from which the program answers *?* instead of a language |
        | **mojibake** | text whose letters came out wrong because its bytes were read with the wrong encoding, such as *cafÃ©* for *café* |
        | `max(a, b)` | the larger of two numbers; `min(a, b)` gives the smaller |
        """),
        L.watch_out("A distance is not a probability. A distance of 0.67 to Dutch does not mean that the program is 33% sure: it only says how far the trigrams of the text lie from where Dutch puts them. The program compares the distances with each other and with the limit, nothing more."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Make the profile of each language once: lowercase its sentences, join them with a space between them, cut that text into trigrams with a space before and after it, count them, and keep the 300 most common with their ranks.
    2. Make the profile of the text in the same way.
    3. For each trigram of the text's profile, add how far its rank lies from its rank in the language's profile, or 300 when the language lacks it.
    4. Divide the sum by the number of trigrams of the text times 300: that is the distance to the language.
    5. Give back the nearest language, or *?* when no distance lies below the limit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below makes the profile of *I am going home now.* with the course's `profile`, takes the profile of English from the course's `language_profiles`, which gives the profiles of all twelve languages, and looks up a few trigrams in both. `profile` lowercases the text first.
    """)
    return


@app.cell
def example_cell():
    text = "I am going home now."
    doc = C.profile(text)
    lang = C.language_profiles()["en"]
    print(len(doc), len(lang))
    print(list(doc)[:6])
    print(doc["ing"], lang["ing"])
    print(doc[" i "], lang[" i "])
    print("w. " in lang)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sentence has 20 trigrams, each once, so its profile ranks them in alphabetical order: *␣am* first, at rank 0. *ing* has rank 11 in the sentence and rank 11 in English: 0 out of place. *␣i␣*, the word *I*, has rank 3 in the sentence and 6 in English: 3 out of place. *w.␣*, the end of *now.*, is not in the profile of English, so it adds 300. Ten of the 20 trigrams are missing, which adds 3,000; the other ten add 1,072, so the sum is 4,072. Divided by 20 × 300 = 6,000, that gives a distance of 0.679 to English, the nearest of the twelve.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ngrams():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. ngrams: runs of characters

        Write `ngrams(text, n)`: put a space before `text` and one after it, and give back, as a list, every run of `n` characters of that, from left to right, repeats included. For `ngrams("the cat", 3)`, the result is `[" th", "the", "he ", "e c", " ca", "cat", "at "]`. Replace the `...`. The check calls that and `ngrams("de kat", 3)`, then cases of its own: texts of one or two letters, an empty text, `n` of 1 and 2, a letter that repeats, capitals, an `n` of 4, and an `n` longer than the text with its spaces.
        """),
        L.task("Cut a text into its character n-grams, with a space before and after it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ngrams():
    box_ngrams = L.code_box("ngrams", lines=6)
    box_ngrams
    return (box_ngrams,)


@app.cell(hide_code=True)
def check_ngrams(box_ngrams):
    result_ngrams = L.check("ngrams", box_ngrams.value)
    result_ngrams
    return (result_ngrams,)


@app.cell(hide_code=True)
def hints_ngrams():
    L.hints(
        "ngrams",
        [
            "Start with `padded = \" \" + text + \" \"` and `grams = []`. Go over the places with `for i in range(...)`, append the slice `padded[i:i + n]` at each place, and give back `grams` after the loop.",
            "*the cat* with its two spaces has 9 characters. At which place does the last run of 3 characters start, and how many places is that, counting from 0?",
            r"`    for i in range(___):`",
        ],
        why=r"`padded = \" \" + text + \" \"` and `grams = []`; `for i in range(len(padded) - n + 1):` and inside it `grams.append(padded[i:i + n])`; after the loop, `return grams`.",
    )
    return


@app.cell(hide_code=True)
def text_after_ngrams():
    mo.md(r"""
    An `n` longer than the text with its spaces gives an empty list: `range` then gets a number below 1. In the picture, your function also counted the trigrams of 400 English sentences, and the top of that count is the start of the profile of English. The course's `profile` does the same with its own `ngrams`, after lowercasing the text and turning every run of spaces, tabs and line breaks into one space.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_oop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. out_of_place: how far apart

        `profiles` holds the profile of each of the twelve languages, under its code, such as `"en"` for English and `"nb"` for Bokmål; each profile is a dictionary from trigram to rank. Write `out_of_place(doc, lang)`, where `doc` is the profile of a text and `lang` the profile of a language. Go over the trigrams of `doc`. For a trigram that `lang` has, add how far apart its two ranks are: the larger minus the smaller. For a trigram that `lang` lacks, add `len(lang)`. Give back the sum.

        For the small profiles `{" de": 0, "de ": 1, "en ": 2, " ka": 3}` and `{"en ": 0, " de": 1, "de ": 2, "et ": 3, " he": 4}`, the result is 9: *␣de* is 1 out of place, *de␣* 1 and *en␣* 2, and *␣ka* is missing, which adds 5. Replace the `...`. The check calls that and `out_of_place(profile("Ik fiets naar huis."), profiles["nl"])`, then cases of its own: an English sentence against English, an empty `doc`, the same profile twice, profiles with nothing in common, ranks out of place in both directions, profiles of different sizes, and a Dutch sentence against Italian.
        """),
        L.task("Measure how far the profile of a text lies from the profile of a language."),
    ])
    return


@app.cell
def profiles_cell():
    profiles = C.language_profiles()
    profile = C.profile
    print(sorted(profiles))
    print(len(profiles["en"]), profiles["en"][" th"], profiles["nl"]["en "])
    print(profile("the cat"))
    return profile, profiles


@app.cell(hide_code=True)
def text_after_profiles():
    mo.md(r"""
    The codes are the two-letter codes of ISO 639-1: *af* for Afrikaans, *fy* for Frisian, *sv* for Swedish, and so on. *␣th* has rank 0 in English and *en␣* rank 0 in Dutch. The profile of *the cat* holds its seven trigrams; each occurs once, so they are ranked in alphabetical order.
    """)
    return


@app.cell(hide_code=True)
def exercise_oop():
    box_oop = L.code_box("out-of-place", lines=8)
    box_oop
    return (box_oop,)


@app.cell(hide_code=True)
def check_oop(box_oop):
    result_oop = L.check("out-of-place", box_oop.value)
    result_oop
    return (result_oop,)


@app.cell(hide_code=True)
def hints_oop():
    L.hints(
        "out-of-place",
        [
            "Start with `total = 0` and go over the trigrams with `for gram in doc:`. Ask with `if gram in lang:` whether the language has the trigram; then add the larger of the ranks `doc[gram]` and `lang[gram]` minus the smaller, and otherwise add `len(lang)`. After the loop, give back `total`.",
            "Rank 3 and rank 5 lie 2 apart, and so do rank 5 and rank 3. Which call gives the smaller of two numbers?",
            r"`            total = total + max(doc[gram], lang[gram]) - ___`",
        ],
        why=r"`total = 0`; `for gram in doc:` and inside it `if gram in lang:` with `total = total + max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])`, `else:` with `total = total + len(lang)`; after the loop, `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_oop():
    mo.md(r"""
    For *Ik fiets naar huis.*, Dutch for *I am cycling home.*, your sum against Dutch is 3,841: ten of its 19 trigrams are missing from the profile of Dutch and add 300 each, and the nine others add 841 together, the most *␣hu*, at rank 1 in the sentence and 293 in Dutch. Against Italian the sum is 5,697: only one trigram of the sentence is in the profile of Italian. The picture does the same for *I am going home now.*: what each trigram adds against English with your function, and your sums divided by the most they can be, for each of the twelve languages.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_identify():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. identify: the nearest language, or "?"

        Write `identify(text, profiles, limit)`: make the profile of `text` with `profile`, measure its `distance` to the profile of each language in `profiles`, and give back the code of the nearest language, such as `"nl"`. When no distance lies below `limit`, give back `"?"`. The check gives your function the course's `profile`, `distance` and `out_of_place`; `distance(doc, lang)` divides the sum of `out_of_place` by `len(doc) * len(lang)`, and gives 1.0 when `doc` is empty. For `identify("Ik fiets naar huis.", profiles, 0.9)`, the result is `"nl"`. Replace the `...`. The check calls that and `identify('<div class="nav"><a href="/home">Home</a></div>', profiles, 0.9)`, a line of HTML, which should give `"?"`. Then come cases of its own: an English sentence, a lower limit, an empty text, a Russian sentence, a Dutch sentence with broken letters, small profiles of its own where the nearest language comes second or none lies below the limit, and a limit of 1.
        """),
        L.task("Give the nearest language of a text, or \"?\" when no language comes near enough."),
    ])
    return


@app.cell
def distance_cell(profile, profiles):
    distance = C.distance
    print(distance(profile("Ik fiets naar huis."), profiles["nl"]))
    print(distance(profile("Ik fiets naar huis."), profiles["it"]))
    print(distance(profile("Я люблю читать книги."), profiles["nl"]))
    return


@app.cell(hide_code=True)
def text_after_distance():
    mo.md(r"""
    The Dutch sentence lies at 0.674 from Dutch and at 0.999 from Italian. The Russian sentence, *I like to read books.*, lies at exactly 1 from Dutch: not one of its trigrams is in the profile, since none of the twelve languages is written in Cyrillic.
    """)
    return


@app.cell(hide_code=True)
def exercise_identify():
    box_identify = L.code_box("identify", lines=10)
    box_identify
    return (box_identify,)


@app.cell(hide_code=True)
def check_identify(box_identify):
    result_identify = L.check("identify", box_identify.value)
    result_identify
    return (result_identify,)


@app.cell(hide_code=True)
def hints_identify():
    L.hints(
        "identify",
        [
            "Make `doc = profile(text)`, start with `best = \"?\"` and with `best_distance` at a value that only a language below the limit can beat, and go over the codes with `for lang in profiles:`. Measure `d = distance(doc, profiles[lang])`; when `d` is smaller than `best_distance`, put `lang` in `best` and `d` in `best_distance`. After the loop, give back `best`.",
            "A language at 0.95 should never be kept when the limit is 0.9, and one at 0.62 never when the limit is 0.6. With which start for `best_distance` is `d < best_distance` false for both, while a language just below the limit still gets in?",
            r"`    best_distance = ___`",
        ],
        why=r"`doc = profile(text)`, `best = \"?\"` and `best_distance = limit`; `for lang in profiles:` with `d = distance(doc, profiles[lang])`, and `if d < best_distance:` with `best = lang` and `best_distance = d`; after the loop, `return best`.",
    )
    return


@app.cell(hide_code=True)
def text_after_identify():
    mo.md(r"""
    ### How good is it?

    In the picture, your program gives the right language for 992 of the 1,190 sentences it had not seen: 83.4%. Cavnar and Trenkle did much better: with profiles of 400 n-grams, of lengths 1 to 5 together, their program got about 99.7% of 3,478 Usenet articles in eight languages right (the paper says 99.8%). But their articles averaged "around 1700 bytes", and the course's sentences from Tatoeba average about 30 characters. The survey of Jauhiainen and colleagues says: "LI of short strings is known to be challenging for existing LI techniques." The survey reports that Mandl and colleagues, in 2006, found that every program they tried did much worse on texts of 25 characters than on texts of 125. The first table below puts the held-out sentences together, two, three or five at a time: with five sentences in one text, the program gets 95.4% right.

    Closely related languages are the other problem. The survey says: "Similar languages are a known problem for existing language identifiers". The second table shows Danish, Bokmål and Swedish. René Haas and Leon Derczynski tested langid.py, a well-known tool for 97 languages, on Danish, Swedish, Norwegian, Faroese and Icelandic from Wikipedia in 2021: it got 78.3% right, and it labeled about a quarter of the Bokmål sentences as Danish. On sentences from Tatoeba it got 80.9% right. In 2008, the textbook *Introduction to Information Retrieval* by Christopher Manning, Prabhakar Raghavan and Hinrich Schütze still wrote, in section 2.5: "Written language identification is regarded as a fairly easy problem, whereas spoken language identification remains more difficult". That holds for long, clean texts in a few languages that differ a lot.
    """)
    return


@app.cell(hide_code=True)
def length_picture():
    mo.Html(C.length_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_junk():
    mo.md(r"""
    ### Text that is not language

    The web is full of text that is not language at all: markup, code, numbers, and letters that came out broken. Isaac Caswell, Theresa Breiner, Daan van Esch and Ankur Bapna at Google worked in 2020 on a web corpus for a thousand languages, and found that "human-judged LangID accuracy for web-crawl text corpora created using these models is only around 5% for many lower-resource languages". In their crawl, the corpus of the language Dimli "had mostly invalid HTML", and Fula "was awash in URLs". They note: "Many error modes seem to be direct consequences of n-gram count based models". In 2022, Julia Kreutzer and many coauthors checked samples of 100 lines from 205 web corpora by hand: "In total, 15 of the 205 language-specific samples (7.3%) contained not a single correct sentence." In one collection, CCAligned, 31.42% of the lines were not language, on average over the corpora they checked; the authors call their numbers "only rough estimates".

    **Mojibake** is a special case. The documentation of ftfy, a tool by Robyn Speer that repairs it, says: "Most mojibake comes from decoding correct UTF-8 as if it were some other encoding." The letter *é* takes two bytes in UTF-8, and an encoding with one byte per letter reads them as two letters: *café* becomes *cafÃ©*. A Dutch sentence with a few broken letters keeps most of its trigrams, so it still lies nearest to Dutch. The picture shows seven texts outside the twelve languages, with the distance to the nearest language and the limit of 0.9.
    """)
    return


@app.cell(hide_code=True)
def junk_picture():
    mo.Html(C.junk_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_junk():
    mo.md(r"""
    The limit catches the HTML, the data, the date and both versions of the Russian sentence. The line of code and the Turkish sentence slip under it: both lie at 0.88 from Bokmål, and the program cannot answer *Turkish*, since Turkish has no profile. The survey calls such languages "unseen" languages, "languages that we do not have training data for but may nonetheless be encountered by a LI system when applied to real-world data." A lower limit catches more: at 0.85, the line of code and the Turkish sentence get *?* too, but so do 12 of the 1,190 held-out sentences instead of 6. Where to put the limit is a choice between letting junk through and turning real sentences away.

    Mixed text is a problem of its own. *Introduction to Information Retrieval* gives the example that "a French email might quote clauses from a contract document written in English." The survey reports that "Stupar et al. (2011) found that 4% of the documents they collected contained text in more than one language." A program that gives one language for each text gives one answer for a mix.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: your own text

    Type a text, and set the limit. The picture shows the distance to each of the twelve languages, the answer, and which trigrams of your text are in the profile of the nearest language. The text starts as a mix of Dutch and English. Try a sentence in a thirteenth language, a name, a web address, or a text with broken letters.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Your text", full_width=True)
    demo_limit = L.slider("demo-limit", "limit", start=0.9, low=0.5, high=1.0, step=0.01, role="hyper")
    mo.vstack([demo_text, demo_limit])
    return demo_limit, demo_text


@app.cell(hide_code=True)
def demo_look(demo_limit, demo_text):
    mo.Html(C.demo_picture(demo_text.value, demo_limit.widget.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The mix of Dutch and English comes out as English, at 0.684, with Dutch second at 0.751: five of the nine words are English, and the program has one answer for the whole text. Jurafsky and Martin write in *Speech and Language Processing*: "It’s also quite common for speakers or writers to use multiple languages in a single utterance, a phenomenon called code switching." To label each part of such a text, a program has to look at smaller pieces, such as each word with its neighbors.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: other profiles

    This lesson chose trigrams and profiles of 300. Choose another size for the profiles and another n, and see how the program does on the 1,190 held-out sentences, with the same limit of 0.9.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_size = mo.ui.dropdown(options=[str(s) for s in C.SIZES], value="300", label="N-grams in a profile, at most")
    explore_n = mo.ui.slider(1, 5, value=3, label="n", show_value=True)
    mo.vstack([explore_size, explore_n])
    return explore_n, explore_size


@app.cell(hide_code=True)
def explore_look(explore_n, explore_size):
    mo.Html(C.explore_picture(int(explore_size.value), explore_n.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Single characters get only half of the sentences right: every one of the twelve languages uses *e*, *a* and *n* a lot. Bigrams already get 81.6%. Longer n-grams are more telling, but a short sentence shares fewer of them with the profiles, so more sentences end above the limit: with 5-grams, 520 of the 1,190 get *?*. Larger profiles help: with 3,000 trigrams, 89.5% come out right. The survey has a warning for anyone who compares such numbers: "The relative performance of different n-gram lengths is hugely dependent on the overall setting in which the evaluation was done."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. What comes out?

    Each bold line is a text. Put it under what your `identify` gives for it, with the twelve profiles of this lesson and a limit of 0.9.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_comes_out = L.sort("what-comes-out")
    sort_comes_out
    return (sort_comes_out,)


@app.cell(hide_code=True)
def check_sort(sort_comes_out):
    result_sort = L.check("what-comes-out", sort_comes_out)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) counted pairs of characters to predict the next one; this lesson counted runs of three characters to tell languages apart, after lowercasing the text as in [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en). A profile keeps only the ranks of the most common trigrams, the out-of-place measure compares two profiles, and a limit lets the program say that it does not know. Short texts, closely related languages, languages without a profile and text that is not language at all remain hard. Tools such as langid.py turn the counts into probabilities instead: Marco Lui and Timothy Baldwin write that "langid.py is trained over a naive Bayes classifier with a multinomial event model (McCallum and Nigam, 1998), over a mixture of byte n-grams (1≤n≤4)." The next lesson of the domain, *Spam or not: naive Bayes*, builds such a classifier, for spam.

    **An everyday example.** An online translator that shows *Dutch, detected* above your text has done the same job as your `identify`: it worked out the language from the text alone.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_identify, result_ngrams, result_oop, result_sort):
    L.wrap_up(result_ngrams, result_oop, result_identify, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
