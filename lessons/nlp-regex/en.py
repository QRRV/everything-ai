# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-regex. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-regex", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [What NLP is](/course/open/nlp-what-is-nlp?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_keywords():
    quick_keywords = L.choice(
        "quick-keywords",
        question="The bike shop's program found what a request was about with lists of keywords, and took *The lock on my bike won't open.* for a question about opening hours. What did the program look at?",
        options={
            "meaning": "What the customer wanted to know, read from the whole sentence",
            "right": "Whether certain letters occur in the request",
            "order": "The order of the words, from first to last",
        },
        correct="right",
        why={
            "right": "The program checks whether a word such as *open* occurs among the letters of the request, and those letters do not say what the request means. This lesson writes patterns that find more than one fixed word, such as every year or every clock time in a text, and they too find letters, not meaning.",
            "meaning": "Reading what a sentence means is what the keywords could not do. See Exercise 4, *A lock that won't open*, in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "order": "The program did not look at the order of the words. See Exercise 4, *A lock that won't open*, in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_keywords
    return (quick_keywords,)


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="A program reads your e-mails and puts every date and clock time it finds in your calendar. By the rule of [What NLP is](/course/open/nlp-what-is-nlp?lang=en), which side of NLP is that?",
        options={
            "right": "Understanding: text goes in, and data comes out",
            "making": "Making: data goes in, and text comes out",
            "neither": "Neither side, since the program puts nothing but numbers in a calendar",
        },
        correct="right",
        why={
            "right": "Text goes in, and dates and times come out as data for the calendar: understanding. This lesson writes the first step of such a program: patterns that find clock times and other numbers in text.",
            "making": "Making goes from data to text, as a weather report written from measurements does. Which way does this program go? See *Understanding and making* in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "neither": "What comes out is not text, but what goes in is: the e-mails. See *Understanding and making* in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Text is full of things with a fixed form: dates, clock times, amounts, postcodes, phone numbers. A **regular expression**, regex for short, is a pattern that describes such a form, and a program that has one can find every piece of a text that fits it. Jurafsky and Martin write in the chapter *Words and Tokens* of *Speech and Language Processing*: "Formally, a regular expression is an algebraic notation for characterizing a set of strings. Practically, we can use a regex to search for a string in a text and to specify how to change the string, both of which are key to tokenization." And: "All tokenization systems also depend on regular expressions as a processing step." The code of GPT-2, for example, first cuts a text into pieces with one regular expression, such as a word with the space before it, a run of digits or a run of punctuation, and only then splits each piece further. The program ELIZA, which turns phrases such as "I need X" into answers such as "What would it mean to you if you got X?", works, Jurafsky and Martin write, "by having a series or cascade of regex substitutions each of which matches and changes some part of the input lines." This lesson writes patterns for numbers, postcodes and clock times, tries the numbers and the times on the Dutch sentences of Tatoeba, and counts what they find, and what they get wrong.

    **An everyday example.** A web form that turns down your postcode because you typed a space can be checking it against a pattern of what a postcode looks like.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In the 199,970 Dutch sentences of Tatoeba, a pattern finds 472 numbers of exactly four digits. How many of them lie between 1000 and 2099, where most years in a text would be?",
        options={
            "few": "About 50",
            "half": "About 240",
            "right": "431",
            "all": "All 472 of them",
        },
        reveal="**431**, about 9 in 10. Most are years, such as 2013, which alone occurs 42 times. But a number between 1000 and 2099 is not always a year, as in *bijna 2000 nieuwe leraren* (nearly 2,000 new teachers) and *Tussen de 1490 en 1635 mensen* (between 1,490 and 1,635 people), and a year can lie outside it, as in *het jaar 2345*. A pattern finds the form of a piece of text, not what it stands for.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A pattern for a set of strings

    The simplest pattern is a piece of text itself: the pattern `de` fits every place where a d and an e stand side by side. Each place where a pattern fits is a **match**. Other characters in a pattern stand for more than themselves, so that one pattern can describe many strings. The Python documentation calls regular expressions "essentially a tiny, highly specialized programming language embedded inside Python and made available through the re module." This lesson uses four kinds of pieces.

    A **character class**, between square brackets, matches one character out of a set: `[dD]` matches d or D, `[A-Z]` one capital letter from A to Z, and `[0-9]` one digit; `\d` is short for a digit. A dot `.` matches any one character except a newline.

    **Repetition** says how often the piece before it comes: `*` zero times or more, `+` once or more, `?` zero times or once, `{3}` exactly three times, and `{1,3}` one to three times. Jurafsky and Martin name the first two after Stephen Kleene: "The Kleene star means 'zero or more occurrences of the immediately previous character or regular expression'." A repetition takes as much as it can: "we say that patterns are greedy, expanding to cover as much of a string as they can."

    An **anchor** matches a place, not a character: `^` the start of the text, `$` the end, and `\b` a word boundary: a place with a letter, a digit or `_` on one side, and on the other side a character that is none of these, or the start or end of the text.

    A **group**, between parentheses, holds a part of the pattern together, and takes out the part of a match that you want, such as the hours of a clock time.

    ### Patterns in Python

    Python's module `re` works with patterns. `re.search(pattern, text)` looks for the first match. The documentation: "Scan through string looking for the first location where the regular expression pattern produces a match, and return a corresponding Match. Return None if no position in the string matches the pattern". `re.findall(pattern, text)` gives every match: "Return all non-overlapping matches of pattern in string, as a list of strings or tuples." With groups, it gives what the groups matched: "If there is exactly one group, return a list of strings matching that group. If multiple groups are present, return a list of tuples of strings matching the groups."

    Patterns are written as **raw strings**, with an `r` in front of the quotes. In a normal string, Python itself already reads some backslashes: `"\n"` is a newline, and `"\b"` a backspace. The documentation: "backslashes are not handled in any special way in a string literal prefixed with 'r'." So `r"\n"` is two characters, a backslash and an n, while `"\n"` is one character, a newline. And `r"\b"` reaches `re` as a backslash and a b, a word boundary, and `"\b"` as a backspace, which no sentence contains.

    ### Two kinds of mistakes

    A pattern can go wrong in two ways. Jurafsky and Martin build a pattern for the English word *the*, from `the` to `[tT]he` to `\b[tT]he\b`, and write: "The simple process we just went through was based on fixing two kinds of errors: false positives, strings that we incorrectly matched like other or there, and false negatives, strings that we incorrectly missed, like The." A **false positive** is a match that should not be one; a **false negative** is a piece of text that the pattern should have matched, and missed. Making fewer mistakes, they write, "involves two antagonistic efforts": increasing **precision**, which means fewer false positives, and increasing **recall**, which means fewer false negatives. The picture does the same for the Dutch word *de*, in a sentence of Tatoeba, and counts the matches of each pattern in all Dutch sentences.

    ### Where they come from

    In a memorandum of 1951 for the RAND Corporation, *Representation of Events in Nerve Nets and Finite Automata*, Stephen Kleene wrote: "We shall presently describe a class of events which we will call 'regular events.'" And: "We would welcome any suggestions as to a more descriptive term." The memorandum was printed again in 1956, in the book *Automata Studies*, and Jurafsky and Martin date the first definition of regular expressions to that work of 1951 and 1956. In 1968, Ken Thompson described in *Communications of the ACM* a compiler that turns a regular expression into a program that searches text, for the IBM 7094 computer. Regular expressions went from Thompson's version of the editor QED to the Unix editor ed, whose command `g/re/p`, for global regular expression print, printed the lines with a match. When Thompson was asked how to search a file that was too big for the editor, Doug McIlroy writes in *A Research UNIX Reader*, Thompson made a separate program of it and named it after that command: grep. "It was an instant hit and soon entered our language as a verb." grep is in the Unix manual of November 1973.
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
        | **regular expression** (regex) | a pattern that describes a set of strings, such as every number of four digits |
        | **match** | a piece of text that a pattern fits |
        | **character class** `[...]` | one character out of a set: `[aeiou]` a vowel, `[A-Z]` a capital letter, `[1-9]` a digit from 1 to 9 |
        | `\d` | one digit, short for `[0-9]`; in Python also a digit of another script, such as ٣ |
        | `.` | any one character except a newline; `\.` is a dot itself |
        | `*`, `+`, `?` | the piece before it zero times or more, once or more, zero times or once: `colou?r` matches color and colour |
        | `{3}`, `{1,3}` | the piece before it exactly three times, or one to three times |
        | **anchor** | a place instead of a character: `^` the start of the text, `$` the end, `\b` a word boundary |
        | **group** `( )` | a part of a pattern between parentheses, held together; `re.findall` gives what each group matched |
        | **raw string** `r"..."` | a string in which Python leaves every backslash alone, so that `re` gets the pattern as written |
        | `re.search(pattern, text)` | the first match in the text, or `None` when there is none |
        | `m.group()` | the text that the match `m` covers |
        | `re.findall(pattern, text)` | every match, as a list of strings, or of tuples when the pattern has more than one group |
        | `x is not None` | `True` when `x` is something, `False` when it is `None` |
        | `int("08")` | the whole number that a string of digits stands for, here 8 |
        | **false positive**, **false negative** | a match that should not be one; a piece of text that should have matched and was missed |
        | **precision**, **recall** | fewer false positives means more precision; fewer false negatives means more recall |
        """),
        L.watch_out("A pattern can take very long. When it can try to fit a text in many ways, the program that runs the pattern tries one way after another, which is called backtracking. On July 2, 2019, Cloudflare deployed a rule with \"a regular expression that backtracked enormously\", in the words of its report: the processors that serve web traffic went to nearly 100% across its network, and the service went down for 27 minutes. On July 20, 2016, Stack Overflow was down for 34 minutes when a post with roughly 20,000 characters of white space in a row met a pattern that trims white space, simplified in its report as `\\s+$`, where `\\s` is any white space. The spaces were not at the end, so `\\s+` started at the first space, took all of them, found no end, and started again at the second space, and so on for every space. Try a pattern on long and odd text too."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Describe what you look for as a form: which characters, how many of them, and where they are.
    2. Write that form as a pattern in a raw string: character classes for the characters, repetition for how many, anchors and word boundaries for where.
    3. Use `re.search` to know whether and where there is a match, and `re.findall` for every match; put the parts you want in groups.
    4. Look at what the pattern finds and misses in real text: its false positives and its false negatives.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `ECLIPSE` holds a sentence of Tatoeba about an eclipse of the moon, with two clock times. The cell looks for a run of digits in it with `re.search` and shows the match with `m.group()`, finds every run of digits with `re.findall`, and looks for a euro sign.
    """)
    return


@app.cell
def example_search():
    import re

    ECLIPSE = C.eclipse()
    print(ECLIPSE)
    m = re.search(r"\d+", ECLIPSE)
    print(m.group())
    print(re.findall(r"\d+", ECLIPSE))
    print(re.search(r"€", ECLIPSE))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `\d+` takes a run of digits as long as it can: 21, not 2. `re.search` stops at the first match, and `re.findall` gives all four runs. But the times come apart: 21 and 30 belong together. Exercise 3 keeps them together with groups. The sentence has no euro sign, so `re.search` gives `None`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_four():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. four_digits: every number of four digits

        `SENTENCE` below holds a sentence of Tatoeba with a year and the number of a star. Write `four_digits(text)`: every number of exactly four digits in `text`, as strings, in the order in which they appear. A number that is part of a longer number does not count, so 28678 gives nothing. Use `re.findall`, and write the pattern as a raw string. For `four_digits(SENTENCE)`, the result is `["2019"]`. Replace the `...`. The check calls `four_digits(SENTENCE)`, then cases of its own: a phone number, a year after 2099, a toll-free number with hyphens, two years, a share with a decimal comma, a text without numbers, and an empty text.
        """),
        L.task("Find every number of exactly four digits in a text with re.findall."),
    ])
    return


@app.cell
def sentence_cell():
    SENTENCE = C.star()
    print(SENTENCE)
    return


@app.cell(hide_code=True)
def exercise_four():
    box_four = L.code_box("four-digits", lines=4)
    box_four
    return (box_four,)


@app.cell(hide_code=True)
def check_four(box_four):
    result_four = L.check("four-digits", box_four.value)
    result_four
    return (result_four,)


@app.cell(hide_code=True)
def hints_four():
    L.hints(
        "four-digits",
        [
            "Give back what `re.findall` finds in `text`. Write the pattern as a raw string: a word boundary, then the digits, then a word boundary again.",
            "Which piece of the table says how many times the piece before it comes?",
            r'`    return re.findall(r"\b___\b", text)`',
        ],
        why=r'`return re.findall(r"\b\d{4}\b", text)`: a word boundary, a digit exactly four times, and a word boundary.',
    )
    return


@app.cell(hide_code=True)
def text_after_four():
    mo.md(r"""
    Of the 472 numbers, 431 lie between 1000 and 2099, and most of those are years: 2013 alone occurs 42 times. The 41 others include amounts and sizes, such as *3000 yen*, which occurs 6 times, and the parts of phone numbers such as *9876-5432*, where the hyphen is a word boundary too. Without the word boundaries, `\d{4}` finds 487 matches: 15 of them are four digits inside a longer number or word, such as 2867 in 28678, false positives.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_postcode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. is_postcode: is this a postcode?

        A Dutch postcode consists of four digits and two letters, PostNL says. The library python-stdnum, which checks numbers such as postcodes, lets the four digits start with 1 to 9, not with 0. Statistics Netherlands (CBS) writes a postcode with a space, as in 2514 AT, and without one, as in 1234AB. Write `is_postcode(s)`: `True` when the whole string `s` is a postcode, and `False` otherwise: four digits of which the first is not 0, then one space or none, then two capital letters, and nothing before or after. Use `re.search` with a pattern tied to the start and to the end of the text, and compare what it gives with `None`. For `is_postcode("1234 AB")`, the result is `True`. Replace the `...`. The check calls `is_postcode("1234 AB")`, then cases of its own: no space, another postcode, small letters, a first digit 0, too few and too many digits, too many letters, two spaces, a hyphen, a word before the postcode, and nothing.
        """),
        L.task("Say whether a whole string is a Dutch postcode, with re.search and anchors."),
    ])
    return


@app.cell(hide_code=True)
def exercise_postcode():
    box_postcode = L.code_box("is-postcode", lines=4)
    box_postcode
    return (box_postcode,)


@app.cell(hide_code=True)
def check_postcode(box_postcode):
    result_postcode = L.check("is-postcode", box_postcode.value)
    result_postcode
    return (result_postcode,)


@app.cell(hide_code=True)
def hints_postcode():
    L.hints(
        "is-postcode",
        [
            "Build the pattern from the left: a digit from 1 to 9, three digits more, the space that may be missing, and two capital letters. Put an anchor for the start in front of it and one for the end after it. Then give back whether `re.search` finds a match, by comparing what it gives with `None`.",
            "A space is a character too. Which sign makes the piece before it optional?",
            r'`    return re.search(r"^[1-9]\d{3}___[A-Z]{2}$", s) is not None`',
        ],
        why=r'`return re.search(r"^[1-9]\d{3} ?[A-Z]{2}$", s) is not None`: the start, a digit from 1 to 9, three digits, a space zero times or once, two capital letters, and the end.',
    )
    return


@app.cell(hide_code=True)
def text_after_postcode():
    mo.md(r"""
    `^` and `$` tie the pattern to the whole text, so *Postcode: 1234 AB* and *12345 AB* are turned down. Python also has `re.fullmatch`, which only matches when the whole string matches, and it is stricter in one place: `$` also matches just before a newline at the very end, so `re.search(r"^[1-9]\d{3} ?[A-Z]{2}$", "1234 AB\n")` finds a match, and `re.fullmatch(r"[1-9]\d{3} ?[A-Z]{2}", "1234 AB\n")` gives `None`. The pattern checks only the form: whether a postcode of that form exists, a pattern cannot know. For that, you need the list of postcodes itself.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_times():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. clock_times: every time in a text

        In ordinary Dutch text, a clock time is written with a dot, as in *8.30 uur*, Taaladvies.net says; technical texts can use a colon, as in *17:03:16*. The sentences of Tatoeba have both. Write `clock_times(text)`: every clock time in `text`, as a pair of whole numbers `(hours, minutes)`, in order. A time is one or two digits for the hours, a dot or a colon, and two digits for the minutes, with no digit right before or after. Use `re.findall` with a group around the hours and a group around the minutes, and turn each pair of strings into whole numbers with `int`. For `clock_times(ECLIPSE)`, the result is `[(21, 30), (23, 10)]`. Replace the `...`. The check calls `clock_times(ECLIPSE)`, then cases of its own: a time with a dot, two times in one sentence, minutes that start with a zero, a year before a time, the number of a version, a number with a dot between the thousands, an amount in dollars, and an empty text.
        """),
        L.task("Find every clock time in a text, and take it apart into hours and minutes with groups."),
    ])
    return


@app.cell(hide_code=True)
def exercise_times():
    box_times = L.code_box("clock-times", lines=5)
    box_times
    return (box_times,)


@app.cell(hide_code=True)
def check_times(box_times):
    result_times = L.check("clock-times", box_times.value)
    result_times
    return (result_times,)


@app.cell(hide_code=True)
def hints_times():
    L.hints(
        "clock-times",
        [
            "Write the pattern from the left: a word boundary, a group with one or two digits, what comes between the hours and the minutes, a group with two digits, and a word boundary. `re.findall` then gives a pair of strings for each time. Make a list of pairs of whole numbers from those, with a list comprehension and `int`.",
            "Two different signs may come between the hours and the minutes. How does one place in a pattern take either of them?",
            r'`    found = re.findall(r"\b(\d{1,2})___(\d\d)\b", text)`',
        ],
        why=r'`found = re.findall(r"\b(\d{1,2})[.:](\d\d)\b", text)`, and then `return [(int(h), int(m)) for h, m in found]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_times():
    mo.md(r"""
    Inside square brackets a dot is just a dot, so `[.:]` takes a dot or a colon; outside them, a dot is any character. In Tatoeba, the pattern finds 67 clock times, 50 with a colon and 17 with a dot, and 31 of them are 2:30. One of them is not a time: the 3.11 of *Windows voor werkgroepen 3.11* is the number of a version, a false positive. And the pattern misses every time without a dot or a colon, such as *om 8 uur*: *om*, a number and *uur* occur together 52 times in Tatoeba, false negatives. A second pattern could find those, and would bring mistakes of its own.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a pattern of your own

    Type a pattern. The picture shows how many matches it has in the Dutch sentences of Tatoeba, in how many sentences, and a few of those sentences with each match marked. Two more pieces you can use: `\w` matches a letter, a digit or `_`, and `{2,}` means two times or more. Try `\bTom\b`, `\b\w+heid\b`, `\b[A-Z]{2,}\b`, `\d+%`, `€ ?\d+` or a pattern of your own.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_pattern = mo.ui.text(value=r"\b\d{1,2} uur\b", label="Pattern", full_width=True)
    demo_pattern
    return (demo_pattern,)


@app.cell(hide_code=True)
def demo_look(demo_pattern):
    mo.Html(C.demo_picture(demo_pattern.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    `\bTom\b` matches in 25,328 sentences: Tatoeba's sentences use that name a lot. `\b\w+heid\b` finds *waarheid* 440 times and *gezondheid* 166 times, and also *afscheid*, 40 times, which only ends in the same letters. Each pattern you try finds letters in a form; whether the matches are what you meant, only reading them tells you.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. Which tool fits?

    Each bold task asks something of a text. Put it under the tool that fits it.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_tool = L.sort("which-tool")
    sort_tool
    return (sort_tool,)


@app.cell(hide_code=True)
def check_sort(sort_tool):
    result_sort = L.check("which-tool", sort_tool)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What NLP is](/course/open/nlp-what-is-nlp?lang=en) found what a request was about with keywords: fixed pieces of text. A regular expression describes a whole set of strings: character classes for which characters, repetition for how many, anchors and word boundaries for where, and groups for the parts you want. `re.search` says whether and where a pattern matches, `re.findall` gives every match, and a raw string keeps the backslashes for `re`. You found every number of four digits, checked postcodes as a form would, and took clock times apart into hours and minutes. On real text, every pattern had false positives or false negatives, and looking at what a pattern finds and misses is part of writing it. The next lesson of the domain, *Cutting text into sentences and words*, cuts text into sentences and words with rules, and shows where simple rules fall short.

    **An everyday example.** In a text editor that can search with regular expressions, `\b\d{4}\b` finds every number of four digits in a document at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_four, result_postcode, result_sort, result_times):
    L.wrap_up(result_four, result_postcode, result_times, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
