# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-dutch-compounds. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dutch-compounds", "en")
    C = L.checks   # the course's list of Dutch words, its linking sounds and its own splitter


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Words and their parts](/course/open/nlp-morphemes?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="In *Words and their parts*, `split_compound` gives back the first cut, from the left, into two known words, or the word as one part. What does `split_compound(\"voetbalveld\", [\"voet\", \"bal\", \"voetbal\", \"veld\"])` give?",
        options={
            "right": "`['voetbal', 'veld']`",
            "three": "`['voet', 'bal', 'veld']`, the three words it is made of",
            "whole": "`['voetbalveld']`, since that word is not in the list",
        },
        correct="right",
        why={
            "right": "The cuts go from left to right. After *voet*, the rest *balveld* is not in the list; after *voetbal*, both pieces are, and `split_compound` stops there, with two pieces. This lesson splits the pieces again.",
            "three": "`split_compound` cuts once, so it gives two pieces at most. See Exercise 2 in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "whole": "`split_compound` gives the word as one part only when no cut works, and here *voetbal* and *veld* are both in the list. See Exercise 2 in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def quick_check_write():
    quick_write = L.choice(
        "quick-write",
        question="How does Dutch write the compound that English writes as *science fiction*?",
        options={
            "right": "As one word, *sciencefiction*",
            "space": "With a space, as in English",
            "hyphen": "With a hyphen between the two parts, *science-fiction*",
        },
        correct="right",
        why={
            "right": "The spelling rules of the Taalunie say that the parts of a compound are written together. So a Dutch program meets a compound as one string, with no space to cut at: this lesson finds where to cut.",
            "space": "English writes many compounds with a space, and Dutch writes them as one word. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "hyphen": "Dutch writes compounds as one word, with a hyphen in a few special cases. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_write
    return (quick_write,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Dutch keeps making new compounds, and writes each one as one word. A search for *fiets* (bike) that only compares whole words misses a page about a *fietsenmaker* (bike repairer). And no list of words holds them all: the Dutch reference grammar, the e-ANS, gives *kindercarnavalsoptochtvoorbereidingswerkzaamheden* (the work of preparing a children's carnival parade) as an example of a long compound.

    Programs that split compounds do better. Christof Monz and Maarten de Rijke (2002) split the compounds in a collection of Dutch documents and in the search queries, and added the parts; on the measure they used, the quality of their search engine's results went up by slightly more than 6 percent. Philipp Koehn and Kevin Knight (2003) write that a translation program can translate a compound it does not know by translating its parts.

    This lesson splits Dutch compounds into more than two parts, with the sounds that Dutch puts between them, and in every way the letters allow.

    **An everyday example.** A long train is a row of coupled carriages. To see what it carries, you look for the couplings and take the carriages one by one.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Monz and de Rijke took a list of arbitrarily chosen nouns from a collection of Dutch texts for testing search engines, and split each noun into its parts. What share of the nouns had two parts or more?",
        options={
            "few": "About 5 in 100",
            "right": "About a quarter",
            "most": "About three quarters",
        },
        reveal="**About a quarter.** In their table, 125 of the 535 Dutch nouns have two or three parts: 23.4 in 100. They write \"approximately 25%\", for German too, and add themselves that their samples are far too small for a strong claim. Their text speaks of 553 Dutch nouns, but the numbers in the table add up to 535.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Parts within parts

    *Words and their parts* split *doorbell* into two known words. A Dutch compound can have more parts, because a part can itself be a compound: *voetbalveld* (football field) is *voetbal* and *veld*, and *voetbal* is *voet* and *bal*. The e-ANS calls compounding recursive: you can keep joining, and that is how its long examples come about. **Decompounding**, or compound splitting, is splitting a compound into its parts; Ben Verhoeven and colleagues (2014) describe compound splitting as "identifying the boundaries of the components". A program that splits *voetbalveld* has to split the parts too: the same job on a shorter word. A function that calls itself does that, as in [A function that calls itself](/course/open/py-recursion?lang=en).

    ### Linking sounds

    Between the two words of a Dutch compound you sometimes hear an extra sound: *stad* (city) and *muur* (wall) make *stadsmuur*, *hond* (dog) and *hok* (shed) make *hondenhok* (doghouse), and *zon* (sun) and *bril* (glasses) make *zonnebril* (sunglasses). The e-ANS names three such **linking sounds**: -en-, -e- and -s-; the language advice site of the Taalunie, taaladvies.net, adds -er-, as in *eiergerecht* (egg dish). It also writes that there are no hard rules for whether a compound gets one, and that the same word can take different ones: *zonwering* (sunshade), *zonneschijn* (sunshine), *zonsverduistering* (solar eclipse).

    How the sound is spelled does have a rule, from the Taalunie: write -en- when the left part is a noun with a plural in -en and none in -es, and -e- otherwise. So since 1996 it is *pannenkoek* (pancake), as *pan* has the plural *pannen*, and no longer *pannekoek*. The rule has exceptions: *zonnebril* keeps -e-, since there is only one sun.

    German has linking sounds too. Koehn and Knight allowed them between any two words, since "there are no simple rules for when such letters may be inserted": *Aktionsplan* (action plan) is *Aktion*, *s* and *Plan*.

    ### More than one split

    The same letters can often be split in more ways. *valkuil* (pitfall) is *val* (fall) and *kuil* (pit), but also *valk* (falcon) and *uil* (owl); the Taalunie's spelling rules say that you may write *valk-uil* when you mean such a bird. Roeland Ordelman, Arjan van Hessen and Franciska de Jong (2003) found two or more splits for some 2 percent of the compounds their program detected. That happened, for example, when a compound had three or more parts, which is usually no problem; but some, such as *politiekringen*, also have a split that makes no sense: *politie-kringen* (police circles), and not *politiek-ringen* (politics rings). Someone whose first language is Dutch can usually choose, they write, though sometimes only with the context.

    Koehn and Knight consider all splits into known words, the word itself among them, with an "exhaustive recursive search", and their known words have at least three letters. For *Aktionsplan* that gives *aktionsplan*, *aktion-plan*, *aktions-plan* and *akt-ion-plan*. One of their ways to pick one looks at how often each part occurs in text.

    A split can also be wrong although every part is a word. *kantoor* (office) has nothing to do with a *kant* (side) and an *oor* (ear). Koehn and Knight's German example is *Freitag* (Friday), which splits into *frei* (free) and *Tag* (day).
    """)
    return


@app.cell(hide_code=True)
def parts_cell():
    mo.Html(C.parts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **decompounding** | splitting a compound into its parts, such as *voetbalveld* into *voet*, *bal* and *veld* |
        | **linking sound** | a sound between the two words of a compound, such as -s- in *stadsmuur* and -en- in *hondenhok* |
        | `C.DUTCH` | the course's list of Dutch words, as a set: the 37,123 words of Open Dutch WordNet, a freely available Dutch dictionary, that are one word of three or more small letters |
        | `C.LINKS` | the course's linking sounds, `["", "s", "en"]`; `""`, the empty string, is no sound at all |
        | `x in a_set` | `True` when `x` is one of the items of the set; with many items, much faster than with a list |
        | `[a] + b` | for a list `b`: a new list with `a` before the items of `b`; `+` glues two lists, as it glues two strings |
        """),
        L.watch_out("The course's list has the base forms of words, such as *paard* (horse) and not *paarden* (horses), and it leaves out slurs and crude words. The course's program also leaves out the linking sound -e-, since that usually comes with a spelling change: *zon* becomes *zonne* in *zonnebril*. Even with -en- the spelling can change, and then the program finds no split: *peer* (pear) becomes *peren* in *perensap* (pear juice)."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Try every cut of the word into a left and a right piece.
    2. For each linking sound, take it off the end of the left piece, and check whether what is left and the right piece are known words.
    3. Split the rest of the word in the same way, with a function that calls itself; a known word on its own counts as a split too.
    4. Keep every split you find, not only the first.
    5. Look at the splits: the letters can fit where the meaning does not.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes each linking sound off the end of the piece *honden* with a slice, as *Words and their parts* took off an ending, and prints whether that gives the piece again and whether what is left is in the course's list.
    """)
    return


@app.cell
def example_link():
    piece = "honden"
    for sound in ["", "s", "en"]:
        stem = piece[:len(piece) - len(sound)]   # the piece without the sound
        print(f"sound '{sound}':", stem, stem + sound == piece, stem in C.DUTCH)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Only -en- fits and leaves a known word: *honden* (dogs) is not in the course's list, which has base forms, and *hond* (dog) is. With the empty sound, what is left is the whole piece, so the same check also covers a split without a sound. Exercise 1 does this for every cut of a word.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_link():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. link_splits: two words and a linking sound

        `C.LINKS` is `["", "s", "en"]`. Write `link_splits(word, words, links)`: give back a list of every split of `word` into a known word, a linking sound and a known word, each as a list `[stem, link, right]`. Start with an empty list. Try the cuts from left to right as in *Words and their parts*, with `left = word[:i]` and `right = word[i:]`, and for each cut, try the sounds of `links` in order, with `stem` the left piece without the sound, as in the worked example. When the left piece really ends in that sound, and both `stem` and `right` are in `words`, append `[stem, link, right]` to the list. Give back the list at the end, also when it stays empty. Replace the `...`. The check calls `link_splits("hondenhok", C.DUTCH, C.LINKS)`, then other words and lists of its own, also a German word.
        """),
        L.task("Try every cut and every linking sound, and keep each split into two known words."),
    ])
    return


@app.cell(hide_code=True)
def exercise_link():
    box_link = L.code_box("link_splits", lines=10)
    box_link
    return (box_link,)


@app.cell(hide_code=True)
def check_link(box_link):
    result_link = L.check("link_splits", box_link.value)
    result_link
    return (result_link,)


@app.cell(hide_code=True)
def hints_link():
    L.hints(
        "link_splits",
        [
            "Two loops, one inside the other: the outer one over the cuts, the inner one over `links`. Inside, do the steps of the worked example.",
            "When do you append a split to the list: what must be true of `stem` and `link` together, of `stem` and of `right`? And where does `return` go, so that it comes after both loops?",
            "`            if ___ and stem in words and right in words:`",
        ],
        why="`found = []`, then `for i in range(1, len(word)):` with `left = word[:i]` and `right = word[i:]`, and inside it `for link in links:` with `stem = left[:len(left) - len(link)]` and `if stem + link == left and stem in words and right in words:` with `found.append([stem, link, right])` under it; after both loops `return found`. `link_splits(\"hondenhok\", C.DUTCH, C.LINKS)` gives `[['hond', 'en', 'hok']]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_link():
    mo.md(r"""
    The picture under the check shows your function on more Dutch words. *huiswerk* (homework) has two splits: *huis* (house) and *werk* (work), and also *hui* (whey) with -s-. *zonnebril* has none, since its -e- comes with a spelling change. Exercise 2 splits into more than two parts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_all():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. all_splits: every way, with a function that calls itself

        Write `all_splits(word, words)`: give back a list of every way to cut `word` into known words, each as a list of parts, without linking sounds. Start with an empty list. When `word` itself is in `words`, append `[word]`: a known word is a split of one part. Then try the cuts from left to right. When the left piece `word[:i]` is in `words`, let `all_splits` split the rest of the word, and append each split it gives back, with the left piece glued in front with `+`. Give back the list at the end. The rest is shorter each time, so the calls stop. Replace the `...`. The check calls `all_splits("voetbalveld", C.DUTCH)`, then other words and lists of its own.
        """),
        L.task("Keep the word itself when it is known, and put each known left piece before every split of the rest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_all():
    box_all = L.code_box("all_splits", lines=10)
    box_all
    return (box_all,)


@app.cell(hide_code=True)
def check_all(box_all):
    result_all = L.check("all_splits", box_all.value)
    result_all
    return (result_all,)


@app.cell(hide_code=True)
def hints_all():
    L.hints(
        "all_splits",
        [
            "Work in two steps: first the word itself, then a `for` loop over the cuts, as in Exercise 1. The call for the rest gives back a list of splits, so go over it with a second `for` loop.",
            "What does `all_splits(\"veld\", C.DUTCH)` give back, and what do you put before each of its splits for *voetbalveld*? Why must `[word]` be in the list when the word itself is known?",
            "`            for rest in ___:`",
        ],
        why="`found = []`, then `if word in words:` with `found.append([word])` under it, and `for i in range(1, len(word)):` with `left = word[:i]`, and `if left in words:` with `for rest in all_splits(word[i:], words):` and `found.append([left] + rest)` under it; at the end `return found`. `all_splits(\"voetbalveld\", C.DUTCH)` gives `[['voetbalveld'], ['voet', 'bal', 'veld'], ['voetbal', 'veld']]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_all():
    mo.md(r"""
    The picture under the check shows your function on more words: *valkuil* and *politiekring* each have two splits into two words, and *kantoor* has one that only fits the letters. The demo below puts the two exercises together.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: split a word of your own

    Type a Dutch word, and see every split the course's own splitter finds with `C.DUTCH`: `all_splits`, with the linking sounds of `link_splits`. Try *politiekring*, *zeeroverschatten* (pirate treasures), or *koningspaarden* (the king's horses): the list has *paard* and not *paarden*, so the only split is *koning*, -s-, *paar* (pair) and *den* (pine).
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="verjaardagstaart", label="A Dutch word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_real():
    mo.md(r"""
    ### Exercise 3. A real compound, or only the letters?

    Each split below is made of words of the course's list, as the functions of this lesson find them. Put each split in its group: a real compound, whose parts give its meaning, or a split where only the letters fit.
    """)
    return


@app.cell(hide_code=True)
def exercise_real():
    sort_real = L.sort("real")
    sort_real
    return (sort_real,)


@app.cell(hide_code=True)
def check_real(sort_real):
    result_real = L.check("real", sort_real)
    result_real
    return (result_real,)


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 4. Which split is meant?

    Besides the word itself, `all_splits` gives two splits of *valkuil*: *val* and *kuil* (fall and pit), and *valk* and *uil* (falcon and owl).
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="What tells which of the two splits is meant?",
        options={
            "right": "The text around the word",
            "fewer": "The number of parts, since the right split has fewer",
            "longer": "The length of the first part: the longest known word at the start is the one meant",
        },
        correct="right",
        why={
            "right": "Both splits are two known words, so the letters cannot decide: the meaning does. In a sentence about a trap it is a pitfall, in one about birds it can be a falcon owl. Ordelman, van Hessen and de Jong write that someone whose first language is Dutch can usually choose, though sometimes only with the context.",
            "fewer": "Both splits have two parts, so counting parts does not decide here. What differs is the meaning, and the text around the word shows it. See *More than one split* in *The idea*.",
            "longer": "With that rule the program would take *valk* and *uil*, the falcon owl, also in a text about a trap. The length of a part says nothing about the meaning. See *More than one split* in *The idea*.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Words and their parts](/course/open/nlp-morphemes?lang=en) split a compound into two known words, at the first cut that worked. A Dutch compound can have more parts, since a part can itself be a compound, and a linking sound such as -s- or -en- can sit between two words. You found every split with a linking sound, and every split into any number of parts with a function that calls itself, as in [A function that calls itself](/course/open/py-recursion?lang=en). Splitting compounds, decompounding, helps search and translation. The letters often allow more than one split, and some splits fit only the letters: the meaning, from the text around a word, says which split is right. The next lesson looks at the kinds of words that a sentence is made of.

    **An everyday example.** A jigsaw piece can fit in two places, and only the picture on it tells you which place is right.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_all, result_link, result_real):
    L.wrap_up(result_link, result_all, result_real, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
