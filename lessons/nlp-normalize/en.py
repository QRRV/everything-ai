# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-normalize. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-normalize", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en), and one about [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="What does `re.findall(r\"\\w+|[^\\w\\s]\", \"Café!\")` give?",
        options={
            "split": "`[\"Caf\", \"é\", \"!\"]`",
            "right": "`[\"Café\", \"!\"]`",
            "whole": "`[\"Café!\"]`",
        },
        correct="right",
        why={
            "right": "`\\w` matches a letter with an accent too, such as é, so *Café* is one token, and `!` is a token of its own. For a program, the token *Café* and the token *cafe* are still two different strings; this lesson makes them one.",
            "split": "`\\w` matches letters with an accent too, so é belongs to the run of letters. See *More pieces of a pattern* in [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "whole": "`!` is not a letter or a digit, so the run of letters stops there, and the second choice takes `!` as a token of its own. See *More pieces of a pattern* in [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def quick_check_ids():
    quick_ids = L.choice(
        "quick-ids",
        question="GPT-2's tokenizer turns `\" cafe\"`, with the space before it, into one token. What does it do with `\" Cafe\"`?",
        options={
            "same": "It gives the same token ID: GPT-2 makes every letter small before it looks up a token",
            "right": "Another token ID: to GPT-2, *Cafe* and *cafe* are different tokens",
            "letters": "Two token IDs: one for the capital C and one for the rest",
        },
        correct="right",
        why={
            "right": "GPT-2 keeps the text as it is: `\" cafe\"` is token 26725 and `\" Cafe\"` token 26965, and `\" café\"` and `\" Café\"` are two more tokens, 40304 and 42151. That makes four tokens for what a reader takes as one word. This lesson folds such forms into one, and shows what that costs.",
            "same": "GPT-2 does not change the text before it cuts it: `\"the\"` and `\" the\"` are different tokens, and so are a word with a capital and the same word without one. See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "letters": "A common word is one token, often with the space before it, also when it starts with a capital: `\" Cafe\"` is one token, 26965. See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
        },
    )
    quick_ids
    return (quick_ids,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A program compares strings character by character, by their code points: for Python, `"Café" == "cafe"` is `False`. So a search for *cafe* misses a text that writes *Café*, and a program that counts words sees two words where a reader sees one. **Text normalization** rewrites text into one standard form before a program works with it. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze describe it in *Introduction to Information Retrieval* as making tokens the same "so that matches occur despite superficial differences in the character sequences of the tokens". This lesson takes three common steps: every letter small, the accents off, and the most common words out. The first two steps put together forms that belong together, and each step also puts together or throws away something that mattered: *US*, the United States, becomes *us*, *één* (one) becomes *een* (a), and *niet* (not) disappears.

    **An everyday example.** Someone looks for *Zoe* in the contacts of a phone and hopes to find *Zoë*: only a search that ignores the two dots on the e finds that name.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Snowball, a project that makes tools for search in many languages, keeps a list of 101 very common Dutch words, such as *de*, *en* and *van*. What share of all the words in the 199,970 Dutch sentences of Tatoeba is one of these 101?",
        options={
            "few": "Only about 10%",
            "quarter": "About 25%",
            "right": "51.1%",
            "most": "About 75%",
        },
        reveal="**51.1%**: 664,197 of the 1,301,039 words. In the five English books of the course, Snowball's English list of 174 words takes between 48.0% of the words (*Moby-Dick*) and 54.9% (*The Adventures of Sherlock Holmes*). A small set of very common words makes up about half of a text. Leaving them out halves the text, and the sections below show what goes with them.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Capitals

    The simplest step is **case folding**. Manning and colleagues: "A common strategy is to do case-folding by reducing all letters to lower case." In Python, `.lower()` does it. Whether that is a good idea depends on what a capital tells. Jurafsky and Martin ask in the chapter *Words and Tokens* of *Speech and Language Processing*: "should we consider a capitalized string (like They) and one that is uncapitalized (like they) to be the same word type?" Their answer: "The answer is that it depends on the task!" In Dutch and English, the first word of a sentence gets a capital because of its place, and that capital tells nothing. But a capital can also tell two words apart. Manning and colleagues warn that "such case folding can equate words that might better be kept apart." In Dutch, *Wenen* is the city of Vienna and *wenen* the verb to weep. And Snowball's English list of very common words, the stop words that a section below is about, leaves out the word *us*, with the comment "care is required here because US = United States. It is usually safe to remove it if it is in lower case."

    Between all and nothing, a program can make only some words small. Manning and colleagues: "The simplest heuristic is to convert to lowercase words at the beginning of a sentence and all words occurring in a title that is all uppercase or in which most or all words are capitalized." A capital in the middle of a sentence stays, "which is usually correct", they add. A machine learning model can make the same choice "more accurately", and they call that **truecasing**: deciding for each capital whether it stays. At the start of a sentence, a program cannot see whether a word is a name. Exercise 2 looks it up in a list of names that the course made from Tatoeba: the words that the middle of a sentence writes with a capital more often than in small letters. But when the people who search type small letters anyway, getting the capitals right probably does not help much, Manning and colleagues write: "Thus, lowercasing everything often remains the most practical solution." Python has `str.casefold()` too, which its documentation calls "similar to lowercasing but more aggressive": it makes the German *ß* into *ss*.

    ### Accents

    In *Text is a sequence of numbers*, é is the code point 233. Unicode can also write é as two code points: e (101), followed by code point 769, a **combining mark**: a code point that puts a mark on the character before it, here an acute accent. The two look the same. The Unicode standard, in its annex *Unicode Normalization Forms*, calls such strings canonically equivalent: they stand for the same character, and should look and behave the same. The Python documentation, in its *Unicode HOWTO*, shows it with ê, written as one code point or as e and a combining circumflex: "These will produce the same output when printed, but one is a string of length 1 and the other is of length 2."

    `unicodedata.normalize` puts a string in one of the **normalization forms** of Unicode. In the form **NFD**, a letter with marks on it, such as é, is taken apart into the letter and its combining marks. In the form **NFC**, they are put together again where Unicode has one code point for the whole. Most text with accents is already in NFC, the annex notes, and the Dutch sentences of Tatoeba are, all of them. The documentation says of `unicodedata.combining(chr)`: "Returns the canonical combining class assigned to the character chr as integer. Returns 0 if no combining class is defined." For the acute accent it gives 230. So removing accents takes two steps: put the text in the form NFD, and keep only the characters for which `unicodedata.combining` gives 0.

    Accents, the two dots of the trema in *ë* and the hook of the cedilla in *ç* are all **diacritics**: marks on a letter. Manning and colleagues: "Diacritics on characters in English have a fairly marginal status, and we might well want cliché and cliche to match, or naive and naïve." And: "Occasionally words are distinguished only by their accents." Their example is Spanish: *peña* "a cliff" and *pena* "sorrow". Dutch has such pairs too. *één* is the number one: Taaladvies.net explains that the accents go on *een* only when it is the number and could otherwise be read as the article a. *vóór* and *voor* can make a sentence mean something else: Taaladvies.net gives *Ik voelde me vóór de yogatherapie erg zenuwachtig*, nervous before the therapy began, against *voor de yogatherapie*, nervous about going to it. A trema shows that a vowel is said apart from the vowel before it, as in *ruïne* and *poëzie*, says the Leidraad, the guide to the spelling of the Woordenlijst Nederlandse Taal.

    ### Stop words

    Manning and colleagues: "Sometimes, some extremely common words that would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words." The list of them is a **stop list**. Snowball's list for English has 174 words, and its list for Dutch 101; of the Dutch one, the file says: "This is a ranked list (commonest to rarest) of stopwords derived from a large sample of Dutch text." The English list leaves out common words that have a second meaning, such as *can* and *may*, with examples in the comments: "old tin CAN", "merry month of MAY". The Dutch list keeps such words, and points them out in its comments: *meer* is more, but also a lake, and *bij* is by, but also a bee.

    A stop list throws away words that can carry the meaning. Jurafsky and Martin, in the chapter *Retrieval-based Models* of the same book: "For example, common stop lists would reduce the phrase to be or not to be to the phrase not." Manning and colleagues: "The meaning of flights to London is likely to be lost if the word to is stopped out." The Dutch list holds *niet* (not), *geen* (no) and *niets* (nothing). Most search engines have moved away from stop lists. Manning and colleagues: "The general trend in IR systems over time has been from standard use of quite large stop lists (200–300 terms) to very small stop lists (7–12 terms) to no stop list whatsoever. Web search engines generally do not use stop lists." Jurafsky and Martin write that "In modern IR systems, the use of stop lists is much less common, partly due to improved efficiency", and partly because a way of weighing words, which the lesson *TF-IDF: weighing the words* covers later, already does much of the same work. They add: "Nonetheless, stop word removal is occasionally useful in various NLP tasks so is worth keeping in mind."

    ### And language models?

    BERT, a language model that Google published in 2018, comes in an uncased and a cased version. Its README, where WordPiece is BERT's tokenizer: "`Uncased` means that the text has been lowercased before WordPiece tokenization, e.g., `John Smith` becomes `john smith`. The `Uncased` model also strips out any accent markers." It goes on: "Typically, the `Uncased` model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging)": tasks such as finding the names in a text, or the part of speech of each word. GPT-2 does none of this. Alec Radford and colleagues write in the paper of GPT-2, *Language Models are Unsupervised Multitask Learners*: "Since our model operates on a byte level and does not require lossy pre-processing or tokenization, we can evaluate it on any language model benchmark." So for GPT-2, *café* written in the form NFD is another text: `" café"` with one code point for é is token 40304, and with e and the combining accent it becomes the token of `" cafe"` and two more tokens, each holding one byte of the accent.
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
        | **text normalization** | rewriting text into one standard form before a program works with it |
        | **case folding** | making every letter small, so that *Café* and *café* become one string |
        | **truecasing** | deciding for each capital whether it stays: one that a word has because of its place goes, the capital of a name stays |
        | **combining mark** | a code point that puts a mark on the character before it, such as the acute accent 769 |
        | **diacritic** | a mark on a letter: an accent as in é or è, a trema as in ë, a cedilla as in ç |
        | **normalization form**: **NFC**, **NFD** | a standard way to write the characters of Unicode: every letter with its marks as one code point where possible; letters such as é taken apart from their marks |
        | **stop word**, **stop list** | a very common word that a program leaves out, such as *de* or *the*; the list of such words |
        | `import unicodedata` | Python's module with what Unicode says about every character |
        | `unicodedata.normalize("NFD", s)` | the string `s` in the form NFD: `len(unicodedata.normalize("NFD", "é"))` is 2 |
        | `unicodedata.combining(c)` | 0 for a character that is not a combining mark, and a whole number above 0 for most combining marks, such as 230 for the acute accent |
        | `"\u0301"` | `\u` and a code point in four digits of base 16, the way Unicode writes them: `0301` is 769, the combining acute accent, and `ord("\u0301")` gives 769 |
        | `s.casefold()` | like `s.lower()`, but more aggressive: `"Straße".casefold()` is `"strasse"` |
        | `words[1:]` | the list from its second item on: the list without its first item |
        """),
        L.watch_out("Making a word small can give a form that the spelling does not have. Dutch puts no accent on a capital at the start of a sentence: it writes *Eén*, not *Één*, Taaladvies.net says. But `\"Eén\".lower()` is *eén*, which is neither *één* (one) nor *een* (a), and Tatoeba starts 75 sentences with *Eén*. Fold, in Exercise 1, takes the accents off after the small letters, and makes it *een*."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Ask what the task needs: does a capital, an accent or a small word tell two things apart here?
    2. Cut the text into tokens.
    3. Make the letters small, or only the first word of each sentence, and keep the names.
    4. Take the accents off: the form NFD, then leave out the combining marks.
    5. Leave out the stop words only when the task does not need them, and count what goes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell puts *Café* in the form NFD, and shows the code points before and after, and what `unicodedata.combining` gives for each character.
    """)
    return


@app.cell
def example_nfd():
    import unicodedata

    W = "Café"
    decomposed = unicodedata.normalize("NFD", W)
    print(len(W), len(decomposed))
    print([ord(c) for c in W])
    print([ord(c) for c in decomposed])
    print([unicodedata.combining(c) for c in decomposed])
    print(unicodedata.normalize("NFC", decomposed) == W)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the form NFD, é (233) becomes e (101) and the combining acute accent (769), so the string gets one character longer. `unicodedata.combining` gives 0 for the four letters and 230 for the accent, and the form NFC puts the two together again. Exercise 1 leaves the combining marks out, and makes the word small.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fold():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. fold: small letters, no accents

        Write `fold(word)`: `word` in small letters, without accents or other marks on its letters. Make the word small with `.lower()`, take it apart with `unicodedata.normalize` in the form NFD, and glue the characters that are not a combining mark together again with `"".join` around a list comprehension. For `fold("Café")`, the result is `"cafe"`. Replace the `...`. The check calls `fold("Café")`, then cases of its own: the number *één*, capitals with a trema, two tremas, the capital IJ, an apostrophe, *café* written with e and the combining accent, `"cafe\u0301"`, a cedilla, and an empty string.
        """),
        L.task("Make a word small and take the accents off: the form NFD, then leave out the combining marks."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fold():
    box_fold = L.code_box("fold", lines=5)
    box_fold
    return (box_fold,)


@app.cell(hide_code=True)
def check_fold(box_fold):
    result_fold = L.check("fold", box_fold.value)
    result_fold
    return (result_fold,)


@app.cell(hide_code=True)
def hints_fold():
    L.hints(
        "fold",
        [
            "Make the word small and put it in the form NFD in one line, and give the result a name, such as `decomposed`. Then give back its characters without the combining marks, glued together with `\"\".join` around a list comprehension with an `if`.",
            "What does `unicodedata.combining` give for a character that is not a combining mark? Which test is true for exactly those characters?",
            r'`    return "".join([c for c in decomposed if ___])`',
        ],
        why=r'`decomposed = unicodedata.normalize("NFD", word.lower())`, then `return "".join([c for c in decomposed if unicodedata.combining(c) == 0])`.',
    )
    return


@app.cell(hide_code=True)
def text_after_fold():
    mo.md(r"""
    With your fold, the 42,673 different word forms of Tatoeba become 38,591. Small letters alone make 38,682 of them, so most of the merging comes from the capitals. In Tatoeba, 791 forms carry an accent or another mark, and they occur 5,065 times. The most common is *Australië* (852 times). Next come *één* (671 times), which becomes *een* just as *Eén* (77 times) does, then *geïnteresseerd* (169) and *skiën* (137). For *Australië* and *australie*, nothing is lost. But *één* and *een* are two words, and fold makes them one. The Leidraad does not count the trema as an accent; fold takes it off all the same. Manning and colleagues counted what case folding does to the English news of Reuters-RCV1, after taking out the numbers: 17% fewer different terms. In Tatoeba, small letters give 9.4% fewer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_truecase():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. truecase: only the capital of the first word

        `NAMES` below holds the course's list of names: the 3,951 words that the middle of a sentence of Tatoeba writes with a capital more often than in small letters, such as *Tom*, *Boston* and *IJsland*. Write `truecase(words, names)`, for the words of one sentence: when `words` is empty, or when its first word is in the set `names`, give back `words` as it is; otherwise a new list with the first word in small letters, followed by the other words as they are. For `truecase(["Het", "regent", "in", "Wenen"], NAMES)`, the result is `["het", "regent", "in", "Wenen"]`. Replace the `...`. The check calls that, then cases of its own: a name at the start, a sentence without words, a name in the middle, *Eén* at the start, the capital IJ, a word in capitals, a list of names of its own, and an empty list of names.
        """),
        L.task("Make the first word of a sentence small, unless it is a name."),
    ])
    return


@app.cell
def names_cell():
    NAMES = C.names()
    print(len(NAMES), "Tom" in NAMES, "Het" in NAMES)
    return


@app.cell(hide_code=True)
def exercise_truecase():
    box_truecase = L.code_box("truecase", lines=4)
    box_truecase
    return (box_truecase,)


@app.cell(hide_code=True)
def check_truecase(box_truecase):
    result_truecase = L.check("truecase", box_truecase.value)
    result_truecase
    return (result_truecase,)


@app.cell(hide_code=True)
def hints_truecase():
    L.hints(
        "truecase",
        [
            "First the case where nothing changes, with one `if` and `or`: a list without words, or a sentence that starts with a name. Then give back a new list: the first word small, followed by the other words.",
            "Which item of `words` is the first word, and how do you ask whether a value is in a set?",
            r'`    if len(words) == 0 or ___:`',
        ],
        why=r'`if len(words) == 0 or words[0] in names:` gives back `words`; otherwise `return [words[0].lower()] + words[1:]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_truecase():
    mo.md(r"""
    With your truecase, the words of Tatoeba make 39,788 different words, against 40,603 when only the first word is made small without the names: the 815 more are names made small, such as *mary* and *boston*, which the sentences write nowhere else in small letters. *Tom*, which starts 17,626 sentences, keeps its capital. But the list knows only what the middle of a sentence shows. Five sentences start with *Wenen*: three use the verb, such as *Wenen zal niet helpen* (crying will not help), and two the city, such as *Wenen is de hoofdstad van Oostenrijk*. In the middle of a sentence, Tatoeba writes *wenen* 42 times and *Wenen* 10 times, so *Wenen* is not on the list, and your function makes the city small. A name that only starts sentences, such as *Esther*, is not on it either. And a line of Tatoeba can hold more than one sentence, so the list also holds words such as *Ja*, *Nee* and *Eén*: the middle of such a line writes them with a capital. A model that learns truecasing can look at more than the word itself, such as the words after it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_drop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. drop_stop_words: leave the common words out

        `STOP_WORDS` below holds the 101 Dutch stop words of Snowball, and `STOP_WORDS_EN` its 174 English ones, all in small letters. Write `drop_stop_words(words, stop_words)`: the words of `words`, in their order and as they are written, without the ones whose small-letter form is in the set `stop_words`. Use a list comprehension with an `if`. For `drop_stop_words(["Tom", "is", "niet", "thuis"], STOP_WORDS)`, the result is `["Tom", "thuis"]`. Replace the `...`. The check calls that, then cases of its own: a stop word with a capital, an empty list, a word that is a stop word with another meaning, a line of English with `STOP_WORDS_EN`, *US* and *us*, a punctuation mark, and a list of stop words of its own.
        """),
        L.task("Leave out the words of a list whose small-letter form is a stop word."),
    ])
    return


@app.cell
def stop_words_cell():
    STOP_WORDS = C.stop_words("nl")
    STOP_WORDS_EN = C.stop_words("en")
    print(len(STOP_WORDS), len(STOP_WORDS_EN), "niet" in STOP_WORDS, "us" in STOP_WORDS_EN)
    return


@app.cell(hide_code=True)
def exercise_drop():
    box_drop = L.code_box("drop-stop-words", lines=2)
    box_drop
    return (box_drop,)


@app.cell(hide_code=True)
def check_drop(box_drop):
    result_drop = L.check("drop-stop-words", box_drop.value)
    result_drop
    return (result_drop,)


@app.cell(hide_code=True)
def hints_drop():
    L.hints(
        "drop-stop-words",
        [
            "Build the new list with a list comprehension over `words`, with an `if` that keeps a word only when it is not a stop word.",
            "The lists hold small letters. Which form of the word do you look up, and which form do you keep?",
            r'`    return [word for word in words if ___]`',
        ],
        why=r'`return [word for word in words if word.lower() not in stop_words]`: the small-letter form to look up, the word as written to keep.',
    )
    return


@app.cell(hide_code=True)
def text_after_drop():
    mo.md(r"""
    Your function leaves out 51.1% of the words of Tatoeba, and between 44.6% and 54.9% of the words of each of the eight books of the course, five English and three Dutch. All 29,667 sentences with *niet*, *geen* or *niets* lose that word, and 844 sentences lose every word, such as *Ik kan niet meer!* (I can't take any more!). Jurafsky and Martin's example comes out even emptier: with Snowball's English list, *to be or not to be* is gone altogether, since *not* is on that list too. And in the English books, the apostrophe gets in the way: they write *don’t* with a curly apostrophe, and the list writes *don't* with a straight one, so 1,744 words such as *don’t* and *it’s* stay. Normalizing the apostrophe would be one more step.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: your own text, step by step

    Type a text and choose a list of stop words. The picture takes it through the steps of this lesson: tokens, small letters, accents off, stop words out. Try *naïef*, *Eén*, *IJsland*, a sentence with *niet*, or *to be or not to be* with the English list.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="Is Tom niet in het café in Wenen?", label="Text", full_width=True)
    demo_list = mo.ui.dropdown(options={"Dutch": "nl", "English": "en"}, value="Dutch", label="Stop words")
    mo.vstack([demo_text, demo_list])
    return demo_list, demo_text


@app.cell(hide_code=True)
def demo_look(demo_list, demo_text):
    mo.Html(C.demo_picture(demo_text.value, demo_list.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    *Is Tom niet in het café in Wenen?* (Isn't Tom in the café in Vienna?) comes out as *tom cafe wenen ?*: the question is still there, but not what it asks. The steps are the same for every text, and what they throw away is different in each one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. What does the task need?

    Each bold task asks something of a program that searches or counts words. Put it under what it needs.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_keep = L.sort("what-to-keep")
    sort_keep
    return (sort_keep,)


@app.cell(hide_code=True)
def check_sort(sort_keep):
    result_sort = L.check("what-to-keep", sort_keep)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en) cut a text into tokens. Here the tokens became fewer: with case folding, *Café* and *café* are one; with NFD and without the combining marks, *café* and *cafe* are one; and without the stop words, about half of every text is gone. Each step also threw something away: *Wenen* the city and *wenen* the verb, *één* and *een*, and the *niet* that turns a sentence around. Manning and colleagues call case folding, leaving out stop words and stemming, which the next lesson covers, "forms of lossy compression": what they take away does not come back. [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) showed a tokenizer that keeps every form as it is; BERT's `Uncased` model folds case and accents, its `Cased` model keeps them. The next lesson of the domain, *Stems and lemmas*, goes one step further, and puts the forms of one word together, such as *walks*, *walked* and *walking*.

    **An everyday example.** The search of a web shop that finds *Café Noir* when you type *cafe noir* folds the case and the accents, both in your words and in its own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_drop, result_fold, result_sort, result_truecase):
    L.wrap_up(result_fold, result_truecase, result_drop, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
