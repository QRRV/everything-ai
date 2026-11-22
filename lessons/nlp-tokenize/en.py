# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-tokenize. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-tokenize", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_digits():
    quick_digits = L.choice(
        "quick-digits",
        question="What does `re.findall(r\"\\d+\", \"Om 8.30 uur\")` give?",
        options={
            "whole": "`[\"8.30\"]`",
            "right": "`[\"8\", \"30\"]`",
            "dots": "`[\"8\", \".\", \"30\"]`",
        },
        correct="right",
        why={
            "right": "`\\d` matches one digit and `+` makes a run of them, as long as it can be. The period is not a digit, so the run stops there, and a new one starts after it. This lesson cuts whole texts into pieces with such runs.",
            "whole": "The period is not a digit, so `\\d+` cannot take it: the run of digits stops at the period. See *Patterns in Python* in [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
            "dots": "`re.findall` gives only the matches, and the period does not match `\\d+`. See *Patterns in Python* in [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
        },
    )
    quick_digits
    return (quick_digits,)


@app.cell(hide_code=True)
def quick_check_doctor():
    quick_doctor = L.choice(
        "quick-doctor",
        question="A program says that a sentence ends after every word that ends with a period. In *Morgen komt dr. Visser langs.*, it says that a sentence ends after *dr.* too. Which kind of mistake is that?",
        options={
            "right": "A false positive: the program sees an end where there is none",
            "negative": "A false negative: the program misses an end that is really there",
            "none": "No mistake, since the word *dr.* does end with a period",
        },
        correct="right",
        why={
            "right": "*dr.* is an abbreviation, and the sentence goes on after it: the program finds an end that should not be there, a false positive. This lesson keeps abbreviations and initials from ending a sentence.",
            "negative": "A false negative is an end that the program misses. Here it finds one end too many. See *Two kinds of mistakes* in [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
            "none": "The period of *dr.* belongs to the abbreviation, and the sentence goes on after it. See *Two kinds of mistakes* in [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
        },
    )
    quick_doctor
    return (quick_doctor,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before a program can count words, look them up or find the names in a text, it has to know where each word starts and stops, and where each sentence ends. In Dutch and English, a space does much of that work, but not all of it. A period can end a sentence, or an abbreviation such as *dr.*, or both at once; whether *zo'n* and *Zuid-Afrika* are one word or more depends on the rules you follow. And Jurafsky and Martin write in the chapter *Words and Tokens* of *Speech and Language Processing* that "the writing systems of languages like Chinese, Japanese, and Thai simply don’t have orthographic words at all! That is, they don’t use spaces to mark potential word-boundaries." This lesson cuts Dutch sentences into words with one pattern, decides with a list and a pattern where a sentence ends, and cuts text without spaces with a dictionary. It compares the tokens and the sentence ends with a **treebank**: a collection of sentences in which every sentence comes with its words and its grammar written out. The treebank here is UD_Dutch-Alpino of **Universal Dependencies**, a project that writes treebanks for many languages in the same way; the lesson calls it Alpino, and uses only its words.

    **An everyday example.** A program that reads a text aloud has to know where each sentence ends: if it took the period of *dr.* in *dr. Visser* for the end of a sentence, it would pause in the middle of a name.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A rule cuts a text at the spaces, and splits off every punctuation mark as a token of its own. In how many of the 13,603 sentences of Alpino does it give exactly the words of the treebank?",
        options={
            "few": "About 2,000",
            "half": "About 6,800",
            "right": "11,074",
            "all": "All 13,603 of them",
        },
        reveal="**11,074**, about 4 in 5. In the other sentences, the treebank keeps together what the rule splits: words with a hyphen such as *Zuid-Afrika*, initials such as *J.*, abbreviations such as *mr.*, words with an apostrophe such as *zo'n* and *'t*, and numbers such as *10.000*. Cutting only at the spaces gives exactly the treebank's words in 1,635 sentences. Which pieces are words is set by a standard, and a rule has to follow it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Tokens and standards

    A **token** is a piece of text that a program treats as one unit, such as a word, a number or a punctuation mark, and cutting a text into tokens is **tokenization**. Which pieces are one token is a choice. Jurafsky and Martin: "In rule-based tokenization, we pre-define a standard and implement rules to implement that kind of tokenization." The Penn Treebank standard for English, they write, "separates out clitics (doesn’t becomes does plus n’t), keeps hyphenated words together, and separates out all punctuation". A **clitic** is "a part of a word that can’t stand on its own, and can only occur when it is attached to another word." Universal Dependencies makes other choices. Its English treebank EWT splits *don't* into *do* and *n't* as well, but it also splits most words with a hyphen: its guidelines turn *search-engine* into *search*, *-* and *engine*. In Alpino, *zo'n* and *'t* are one word each, and the guidelines for Dutch say: "No multiword tokens occur". Words there "may contain arbitrary punctuation signs", as in *o.a.* and *vroeg-renaissance*. The picture shows one sentence of Alpino cut at the spaces, cut at the spaces with every punctuation mark on its own, and cut by the treebank.

    ### More pieces of a pattern

    A **choice** `|` between two patterns matches what either of them matches: `kat|hond` finds *kat* and *hond*. At each place in the text, `re.findall` tries the choices from left to right, and takes the first one that fits. A character class with `^` right after the `[` matches one character that is *not* in the set: `[^aeiou]` matches any character except these five vowels, also a space or a digit. `\s` matches one white-space character, such as a space, a tab or a newline, and `\S` one character that is not white space, so `\S+` is a run of characters between spaces. And `\w` matches a letter, a digit or `_`, also a letter with an accent, such as é. Inside square brackets, `\d`, `\w` and `\s` keep their meaning: `[\d.]` matches a digit or a period.

    ### Where does a sentence end?

    Finding where the sentences of a text end is **sentence segmentation**. Question marks and exclamation marks are, in Jurafsky and Martin's words, "relatively unambiguous markers of sentence boundaries". They go on: "The period character “.”, on the other hand, is ambiguous between a sentence boundary marker and a marker of abbreviations like Dr. or Inc." In Dutch, an **abbreviation** is written with one or more periods, as in *bijv.*, *blz.* and *m.a.w.*, says rule 17.A of the Leidraad, the guide to the spelling of the Woordenlijst Nederlandse Taal. An **initial**, the first letter of a name, such as *J.*, has a period too. And when a sentence ends with an abbreviation, no second period follows, Taaladvies.net says: the period of *enz.* at the end of a sentence ends the abbreviation and the sentence at once.

    A rule for sentence ends can go wrong in two ways, as a pattern can: a **false positive** is an end the rule names that is not an end, and a **false negative** a real end that it misses. Its **precision** is the share of the ends it names that are real, and its **recall** the share of the real ends that it finds. A list of abbreviations raises precision. Jurafsky and Martin again: "An abbreviation dictionary can help determine whether the period is part of a commonly used abbreviation; the dictionaries can be hand-built or machine-learned (Kiss and Strunk, 2006)". The program of Tibor Kiss and Jan Strunk, Punkt, learns the abbreviations from the text it cuts, without a list. On text from the Dutch newspaper *De Limburger*, Punkt made the wrong decision for 0.97% of the periods at the end of a token, and the rule that every such period ends a sentence for 6.50%.

    ### Text without spaces

    Cutting a text without spaces into words is **word segmentation**. Even people who know the words do not all cut them the same way. Jurafsky and Martin show the Chinese sentence 姚明进入总决赛, "Yao Ming reaches the finals", which is 3 words by the standard of the Chinese Treebank, and 5 by that of Peking University. Richard Sproat and three colleagues asked six native speakers to cut 100 Chinese sentences into words. Their agreement, measured as the average of precision and recall between two speakers, was .76 on average, where 1 means that two speakers cut every sentence the same way.

    A simple way to cut is **maximum matching**, or MaxMatch. The 2017 draft of the same book describes it: "The maximum matching algorithm starts by pointing at the beginning of a string. It chooses the longest word in the dictionary that matches the input at the current position. The pointer is then advanced to the end of that word in the string. If no word matches, the pointer is instead advanced one character (creating a one-character word)." It adds that the algorithm "works better in Chinese than English, because Chinese has much shorter words than English." **Backward MaxMatch** does the same from the end of the text: it takes the longest word that the text ends with.
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
        | **token** | a piece of text that a program treats as one unit: a word, a number or a punctuation mark |
        | **tokenization** | cutting a text into tokens |
        | **treebank** | a collection of sentences, each with its words and its grammar written out |
        | **clitic** | a part of a word that cannot stand on its own, such as *n't* in *doesn't* |
        | **abbreviation**, **initial** | a shortened word with a period, such as *bijv.*; the first letter of a name with a period, such as *J.* |
        | **sentence segmentation** | finding where each sentence of a text ends |
        | **false positive**, **false negative** | an end that a rule names and that is not an end; a real end that the rule misses |
        | **precision**, **recall** | the share of the ends a rule names that are real; the share of the real ends that it finds |
        | **word segmentation** | cutting a text without spaces into words |
        | **maximum matching** (MaxMatch) | cutting a text into words by taking, again and again, the longest word of a dictionary that the text starts with |
        | **backward MaxMatch** | the same from the end: again and again the longest word of the dictionary that the text ends with |
        | **dictionary** | here: the collection of words that MaxMatch may choose from, in the code a set of strings, not a Python dict |
        | `A|B` | what A matches, or else what B matches; at each place, the first choice from the left that fits |
        | `[^...]` | one character that is not in the set: `[^aeiou]` is any character except these five vowels |
        | `\w` | one letter, digit or `_`, also a letter with an accent, such as é |
        | `\s`, `\S` | one white-space character: a space, a tab or a newline; one character that is not white space |
        | `word.lower()` | the string in small letters: `"Dr.".lower()` is `"dr."` |
        | `range(n, 0, -1)` | the whole numbers from n down to 1: `range(3, 0, -1)` gives 3, 2 and 1 |
        | `text[:k]` | the first k characters of `text`: `"kat"[:2]` is `"ka"` |
        """),
        L.watch_out("MaxMatch is greedy: at each place it takes the longest word, and one word that is too long throws off the words after it. In the chapter *Tokenisation and Sentence Segmentation* of the *Handbook of Natural Language Processing*, David Palmer gives the English example *thetabledownthere*: it \"would be segmented by the greedy algorithm as theta bled own there\", and backward MaxMatch, Palmer adds, gets *the table down there*. With the words of Tatoeba as the dictionary, *ikbenzoterug* (ik ben zo terug, I'll be right back) becomes *ik ben zot er u g* forward, and *ik ben zo terug* backward. But backward is not always better: *neeikweethetniet* (nee, ik weet het niet, no, I don't know) becomes *ne eik weet het niet* backward, and forward gets it right."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Choose a standard: which pieces of text count as one token, such as *dr.*, *zo'n* or *10.000*.
    2. Cut with a pattern, a run of letters and digits or one punctuation mark, and compare the result with the words of a treebank.
    3. Keep abbreviations and initials from ending a sentence: a list for the words, a pattern for the form.
    4. Where there are no spaces, cut with a dictionary: take the longest word that fits, again and again, and compare forward with backward.
    5. Count what goes wrong, and read the cases.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `S` holds a sentence of Alpino. The cell cuts it at the spaces with `\S+`, and then shows the words whose last character is a period, a question mark or an exclamation mark: each of them could end a sentence.
    """)
    return


@app.cell
def example_cut():
    import re

    S = C.example()
    print(S)
    words = re.findall(r"\S+", S)
    print(words)
    print([w for w in words if re.search(r"[.?!]$", w)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `\S+` keeps every period on the word before it, so four words end with a period, and only the last one ends the sentence: *drs.* is an abbreviation, and *J.* and *D.* are initials. A rule that ends a sentence after every period would cut this one into four. Exercise 2 keeps the first three from ending it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tokens():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. tokens: words and punctuation marks

        `SENTENCE` below holds a sentence of Alpino with the names of three cities. Write `tokens(text)`: the tokens of `text`, in order, where a token is a run of letters, digits and `_`, or one character that is not a letter, a digit or white space. So a comma or a question mark is a token of its own, and white space is not a token. Use `re.findall` with one pattern of two choices, joined by `|`. For `tokens(SENTENCE)`, the result is `["Maar", "waar", "blijven", "Amsterdam", ",", "Rotterdam", "en", "Den", "Haag", "?"]`. Replace the `...`. The check calls `tokens(SENTENCE)`, then cases of its own: a clock time, an apostrophe, two marks in a row, three periods, letters with accents, brackets, extra spaces, a newline, and an empty text.
        """),
        L.task("Cut a text into words and punctuation marks with one pattern and re.findall."),
    ])
    return


@app.cell
def sentence_cell():
    SENTENCE = C.cities()
    print(SENTENCE)
    return


@app.cell(hide_code=True)
def exercise_tokens():
    box_tokens = L.code_box("tokens", lines=4)
    box_tokens
    return (box_tokens,)


@app.cell(hide_code=True)
def check_tokens(box_tokens):
    result_tokens = L.check("tokens", box_tokens.value)
    result_tokens
    return (result_tokens,)


@app.cell(hide_code=True)
def hints_tokens():
    L.hints(
        "tokens",
        [
            "Give back what `re.findall` finds in `text`, with one pattern of two choices joined by `|`: first a run of letters and digits, then one character that is something else, but no white space.",
            "Which piece of the table matches one character that is not in a set? And which two short pieces belong in that set here?",
            r'`    return re.findall(r"\w+|___", text)`',
        ],
        why=r'`return re.findall(r"\w+|[^\w\s]", text)`: a run of letters, digits and `_`, or else one character that is none of these and no white space.',
    )
    return


@app.cell(hide_code=True)
def text_after_tokens():
    mo.md(r"""
    The picture shows where your tokens and the treebank part ways. Most often it is a hyphen: Alpino keeps *Zuid-Afrika* and *1-0* whole, and the rule splits them, in 1,305 sentences. Next come abbreviations (465 sentences), words with an apostrophe such as *'t* and *zo'n* (432), initials (320) and numbers such as *10.000* and *2,5* (227). Some sentences of Alpino write an opening quotation mark as two commas, `,,`, and a closing one as two apostrophes, `''`; the treebank keeps each pair as one word. On the English treebank EWT, the same rule cuts 7,808 of the 11,723 sentences exactly: EWT splits *don't* into *do* and *n't*, and the rule cuts it into *don*, *'* and *t*. A rule fits a standard, and each treebank has its own.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ends():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ends_sentence: an end, or an abbreviation?

        When a text is cut at the spaces, as in the worked example, a word ends a sentence if its last character is `.`, `?` or `!`, unless it is an abbreviation or an initial. Write `ends_sentence(word, abbreviations)`: `False` when `word` in small letters is in the set `abbreviations`, or when `word` is one capital letter from A to Z followed by a period, such as *J.*; otherwise `True` when the last character of `word` is `.`, `?` or `!`, and `False` when it is not. `ABBREVIATIONS` below is the course's own short list of 30 Dutch abbreviations, in small letters. Use `.lower()`, `in` and `re.search`, and compare what `re.search` gives with `None`. For `ends_sentence("drs.", ABBREVIATIONS)`, the result is `False`. Replace the `...`. The check calls `ends_sentence("drs.", ABBREVIATIONS)`, then cases of its own: a word at the end of a sentence, an initial, an abbreviation with a capital, a word in capitals, a question, an exclamation, a number with a period inside, a word without a mark, another abbreviation with a capital, and a word with an empty set of abbreviations.
        """),
        L.task("Say whether a word ends a sentence, with a list of abbreviations and a pattern for initials."),
    ])
    return


@app.cell
def abbreviations_cell():
    ABBREVIATIONS = C.abbreviations()
    print(sorted(ABBREVIATIONS))
    return


@app.cell(hide_code=True)
def exercise_ends():
    box_ends = L.code_box("ends-sentence", lines=6)
    box_ends
    return (box_ends,)


@app.cell(hide_code=True)
def check_ends(box_ends):
    result_ends = L.check("ends-sentence", box_ends.value)
    result_ends
    return (result_ends,)


@app.cell(hide_code=True)
def hints_ends():
    L.hints(
        "ends-sentence",
        [
            "First give back `False` for an abbreviation or an initial, with one `if` and `or`: is the word in small letters in `abbreviations`, or does `re.search` find an initial in it? Then give back whether the word ends with one of the three marks.",
            "An initial is the whole word, not a part of a longer one: which anchors does its pattern need?",
            r'`    if word.lower() in abbreviations or re.search(r"___", word):`',
        ],
        why=r'`if word.lower() in abbreviations or re.search(r"^[A-Z]\.$", word):` gives back `False`, and then `return re.search(r"[.?!]$", word) is not None`.',
    )
    return


@app.cell(hide_code=True)
def text_after_ends():
    mo.md(r"""
    With the list and the initials, precision goes from 89.2% to 96.4%: of the 1,326 false ends of the simplest rule, 415 are left, such as *v.*, *nl.* and *zg.*, abbreviations that are not on the list, and a few real ends inside what Alpino counts as one sentence, such as *Cryptogram.* Recall goes down a little, from 80.8% to 80.7%: in 13 sentences, an abbreviation or an initial is the last word, as in *etc.*, *pct.* and *N.V.*, and its period ends the sentence too. Most missed ends have no mark at all: 1,790 sentences of Alpino end without one, such as headlines, and 409 end with a colon. Kiss and Strunk point out that new abbreviations keep being made, so no list can hold them all; Punkt learns them from the text instead.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_start():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. longest_start: the step of MaxMatch

        MaxMatch takes one step again and again: the longest start of the text that is a word, and then it goes on after that word. Write `longest_start(text, words)`: the longest start `text[:end]` of `text` that is in the set `words`; when no start of `text` is in `words`, the first character of `text`. Try the longest start first, so that the first start you find in `words` is the one to give back. `WORDS` below holds the 38,682 words of the Dutch sentences of Tatoeba, in small letters. For `longest_start("zoterug", WORDS)`, the result is `"zot"`: *zo* is a word too, but *zot* is longer. Replace the `...`. The check calls `longest_start("zoterug", WORDS)`, then cases of its own: a whole sentence, a text that is a word itself, a text that starts with a word with shorter words in it, a text that starts with no word, three letters, a long sentence, a euro sign, and one letter. In the picture under the check, the course's own MaxMatch calls your function.
        """),
        L.task("Find the longest start of a text that is in a dictionary: the step of MaxMatch."),
    ])
    return


@app.cell
def words_cell():
    WORDS = C.dictionary()
    print(len(WORDS))
    return


@app.cell(hide_code=True)
def exercise_start():
    box_start = L.code_box("longest-start", lines=5)
    box_start
    return (box_start,)


@app.cell(hide_code=True)
def check_start(box_start):
    result_start = L.check("longest-start", box_start.value)
    result_start
    return (result_start,)


@app.cell(hide_code=True)
def hints_start():
    L.hints(
        "longest-start",
        [
            "Go through the possible ends with a `for` loop over a `range` that counts down, so that the longest start comes first. Give back the first `text[:end]` that is in `words`. After the loop, give back `text[:1]`.",
            "How many characters does the longest start of `text` have?",
            r'`    for end in range(___, 0, -1):`',
        ],
        why=r'`for end in range(len(text), 0, -1):`, then `if text[:end] in words:` with `return text[:end]`, and after the loop `return text[:1]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_start():
    mo.md(r"""
    With your function, the course's MaxMatch cuts 1,247 of the 2,000 sentences right, 62.4%; backward MaxMatch gets 1,162 of them. Every word of these sentences is in the dictionary, so every mistake comes from taking the longest word: *ikwastelaatopschool* (ik was te laat op school, I was late for school) becomes *ik waste laat op school*. Chinese has shorter words, and there MaxMatch comes close to people: Sproat and colleagues compared a greedy MaxMatch with their six native speakers, and it agreed with them .73 on average, against .76 between the speakers themselves.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: MaxMatch on a text of your own

    Type a text, with or without spaces: the spaces are taken out. The picture cuts it with MaxMatch forward and backward, with the words of Tatoeba as the dictionary. Try *neemdeanderestoel*, *dekatisweggelopen*, *gisterenwasunietopschool*, or a hashtag of your own.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="ikbenzoterug", label="Text", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    *neemdeanderestoel* (neem de andere stoel, take the other chair) comes out forward as *neem dean der es toe l*, and right backward. *dekatisweggelopen* (de kat is weggelopen, the cat has run away) becomes *dek at is weggelopen* forward. And *gisterenwasunietopschool* (gisteren was u niet op school, you were not at school yesterday) goes wrong both ways, as *gisteren was unie top school*. A dictionary says which words exist, not which ones the writer meant.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. Which fix fits?

    Each bold case asks something of a program that cuts text into words and sentences. Put it under the fix that it needs.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_fix = L.sort("which-fix")
    sort_fix
    return (sort_fix,)


@app.cell(hide_code=True)
def check_sort(sort_fix):
    result_sort = L.check("which-fix", sort_fix)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en) found pieces of text with a fixed form. Here a pattern cut a whole text into tokens: a run of letters and digits, or one punctuation mark, with `|` for the choice and `[^...]` for one character outside a set. Which pieces are one token is set by a standard, and the standard of Alpino keeps together what a simple rule splits: words with a hyphen or an apostrophe, abbreviations, initials, numbers. For sentences, the period is the problem: a list keeps abbreviations from ending a sentence, a pattern does it for initials, and you measured what that does to precision and recall. Where a text has no spaces, MaxMatch cuts it with a dictionary and the longest word, and goes wrong where a longer word fits by accident. The next lesson of the domain, *Normalizing: case, accents and stop words*, makes *Café* and *cafe* one word, leaves out very common words, and shows what that throws away.

    **An everyday example.** The word count of a word processor cuts text into words too: whether *zo'n* or *e-mail* counts as one word or as two depends on its rules.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ends, result_sort, result_start, result_tokens):
    L.wrap_up(result_tokens, result_ends, result_start, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
