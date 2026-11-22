# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Two stemmers that chop the endings off words by rules: Martin Porter's stemmer for English, and his stemmer for
Dutch. Lessons of domain taal use them, starting with nlp-stemming (Stems and lemmas).

porter() follows "An algorithm for suffix stripping" (M.F. Porter, Program 14(3), pp. 130-137, 1980) the way Porter's
own ANSI C version does. That version, like the BCPL program of 1980, differs from the paper in three small points,
which Porter's page https://tartarus.org/martin/PorterStemmer/ lists: an extra rule logi -> log in step 2, the rule
bli -> ble in place of abli -> able, and words of one or two letters stay as they are. With paper=True, porter()
follows the paper itself.

dutch() follows the description of Porter's Dutch stemmer on https://snowballstem.org/algorithms/dutch/stemmer.html.
Since Snowball 3.0.0 (2025) that is Snowball's alternative stemmer for Dutch; its default is now the stemmer of Wessel
Kraaij and Renée Pohlmann.

The code is the course's own, written from those descriptions. tests/test_stemmers.py compares both stemmers with the
sample vocabularies of the Snowball project, in data/snowball-porter.tsv.gz and data/snowball-dutch-porter.tsv.gz."""

# --- Porter's stemmer for English ----------------------------------------------------------------------------------


def vowel(word: str, i: int) -> bool:
    """True when the letter at place i of `word` is a vowel in Porter's sense: a, e, i, o or u, or a y that follows a
    consonant. The loop goes over the letters up to place i and remembers whether the letter before was a vowel."""
    is_vowel = False
    for j in range(i + 1):
        is_vowel = word[j] in "aeiou" or (word[j] == "y" and j > 0 and not is_vowel)
    return is_vowel


def measure(word: str) -> int:
    """Porter's measure m of a word or part of a word: how often a vowel is followed by a consonant. TREE has m = 0,
    TROUBLE m = 1 and TROUBLES m = 2, the examples of the paper."""
    m = 0
    for i in range(1, len(word)):
        if vowel(word, i - 1) and not vowel(word, i):
            m = m + 1
    return m


def _has_vowel(stem: str) -> bool:
    """*v*: the stem holds a vowel."""
    return any(vowel(stem, i) for i in range(len(stem)))


def _double_consonant(stem: str) -> bool:
    """*d: the stem ends with two of the same consonant, such as -tt or -ss."""
    return len(stem) >= 2 and stem[-1] == stem[-2] and not vowel(stem, len(stem) - 1)


def _cvc(stem: str) -> bool:
    """*o: the stem ends consonant, vowel, consonant, and the last consonant is not w, x or y, as in -wil and -hop."""
    n = len(stem)
    return (n >= 3 and not vowel(stem, n - 3) and vowel(stem, n - 2) and not vowel(stem, n - 1)
            and stem[-1] not in "wxy")


STEP_1A = [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")]
STEP_2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("bli", "ble"),
          ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
          ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
          ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log")]
STEP_2_PAPER = [r for r in STEP_2 if r[0] not in ("bli", "logi")] + [("abli", "able")]
STEP_3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]
STEP_4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate",
          "iti", "ous", "ive", "ize"]


def _longest(word: str, suffixes):
    """The longest of `suffixes` that `word` ends with, or None: in a group of rules, only that one counts."""
    found = [s for s in suffixes if word.endswith(s)]
    return max(found, key=len) if found else None


def _group(word: str, rules, fits) -> str:
    """One group of rules (suffix, new ending): the rule with the longest suffix that the word ends with decides, and
    changes the word only when fits(stem) is True for what is left before the suffix."""
    rules = dict(rules)
    old = _longest(word, rules)
    if old is None:
        return word
    stem = word[:len(word) - len(old)]
    return stem + rules[old] if fits(stem) else word


def porter(word: str, paper: bool = False, measure=measure) -> str:
    """The stem of `word` by Porter's stemmer, in small letters. `measure` is the function that gives m: a lesson can
    give its learner's own."""
    w = word.lower()
    if len(w) <= 2 and not paper:
        return w
    w = _group(w, STEP_1A, lambda stem: True)
    # step 1b: -eed, -ed, -ing
    again = False
    if w.endswith("eed"):
        if measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w, again = w[:-2], True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w, again = w[:-3], True
    if again:
        if w.endswith(("at", "bl", "iz")):
            w = w + "e"
        elif _double_consonant(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif measure(w) == 1 and _cvc(w):
            w = w + "e"
    # step 1c: y -> i
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    w = _group(w, STEP_2_PAPER if paper else STEP_2, lambda stem: measure(stem) > 0)
    w = _group(w, STEP_3, lambda stem: measure(stem) > 0)
    # step 4: -ion only after s or t
    old = _longest(w, STEP_4)
    if old is not None:
        stem = w[:len(w) - len(old)]
        if measure(stem) > 1 and (old != "ion" or stem.endswith(("s", "t"))):
            w = stem
    # step 5a: a final e
    if w.endswith("e"):
        m = measure(w[:-1])
        if m > 1 or (m == 1 and not _cvc(w[:-1])):
            w = w[:-1]
    # step 5b: -ll
    if w.endswith("ll") and measure(w) > 1:
        w = w[:-1]
    return w


# --- Porter's stemmer for Dutch ------------------------------------------------------------------------------------

_DUTCH_VOWELS = "aeiouyè"
_PLAIN = str.maketrans("äëïöüáéíóú", "aeiouaeiou")


def _regions(w: str) -> tuple:
    """p1 and p2, where the regions R1 and R2 start: R1 starts after the first consonant that follows a vowel, but
    after at least 3 letters; R2 starts after the first consonant that follows a vowel in what comes after that
    consonant."""
    n = len(w)
    if n < 3:
        return n, n

    def after(start: int) -> int:
        i = start
        while i < n and w[i] not in _DUTCH_VOWELS:
            i += 1
        while i < n and w[i] in _DUTCH_VOWELS:
            i += 1
        return i + 1 if i < n else n

    p1 = after(0)
    p2 = after(p1) if p1 < n else n
    return max(p1, 3), p2


def _undouble(w: str) -> str:
    return w[:-1] if w.endswith(("kk", "dd", "tt")) else w


def _en_ending(w: str, start: int, p1: int) -> str:
    """Leaves out the -en or -ene at w[start:] when it is in R1 and follows a consonant, but not gem; then undoubles."""
    before = w[:start]
    if start >= p1 and before and before[-1] not in _DUTCH_VOWELS and not before.endswith("gem"):
        return _undouble(before)
    return w


def dutch(word: str) -> str:
    """The stem of `word` by Porter's stemmer for Dutch, in small letters and without accents but è."""
    chars = list(word.lower().translate(_PLAIN))
    # a y at the start, a y after a vowel and an i between vowels count as consonants: I and Y for now
    if chars and chars[0] == "y":
        chars[0] = "Y"
    for j in range(1, len(chars)):
        if chars[j - 1] in _DUTCH_VOWELS:
            if chars[j] == "i" and j + 1 < len(chars) and chars[j + 1] in _DUTCH_VOWELS:
                chars[j] = "I"
            elif chars[j] == "y":
                chars[j] = "Y"
    w = "".join(chars)
    p1, p2 = _regions(w)
    # step 1: -heden, -en and -ene, -s and -se
    old = _longest(w, ("heden", "ene", "en", "se", "s"))
    if old is not None:
        start = len(w) - len(old)
        if old == "heden":
            if start >= p1:
                w = w[:start] + "heid"
        elif old in ("en", "ene"):
            w = _en_ending(w, start, p1)
        elif start >= p1 and w[start - 1] not in _DUTCH_VOWELS + "j":
            w = w[:start]

    # step 2: a final e
    def e_ending(w: str) -> tuple:
        if w.endswith("e") and len(w) - 1 >= p1 and len(w) >= 2 and w[-2] not in _DUTCH_VOWELS:
            return _undouble(w[:-1]), True
        return w, False

    w, e_found = e_ending(w)
    # step 3a: -heid
    if w.endswith("heid") and len(w) - 4 >= p2 and not w[:-4].endswith("c"):
        w = w[:-4]
        if w.endswith("en"):
            w = _en_ending(w, len(w) - 2, p1)
    # step 3b: -end, -ing, -ig, -lijk, -baar, -bar
    old = _longest(w, ("lijk", "baar", "end", "ing", "bar", "ig"))
    if old is not None and len(w) - len(old) >= p2:
        start = len(w) - len(old)
        if old in ("end", "ing"):
            w = w[:start]
            if w.endswith("ig") and len(w) - 2 >= p2 and not w[:-2].endswith("e"):
                w = w[:-2]
            else:
                w = _undouble(w)
        elif old == "ig":
            if not w[:start].endswith("e"):
                w = w[:start]
        elif old == "lijk":
            w, _ = e_ending(w[:start])
        elif old == "baar" or (old == "bar" and e_found):
            w = w[:start]
    # step 4: one vowel of a double vowel goes, as in maan -> man
    if (len(w) >= 4 and w[-1] not in _DUTCH_VOWELS + "I" and w[-3:-1] in ("aa", "ee", "oo", "uu")
            and w[-4] not in _DUTCH_VOWELS):
        w = w[:-2] + w[-1]
    return w.replace("I", "i").replace("Y", "y")


def stem(word: str, lang: str = "en") -> str:
    """The stem of `word`: Porter's stemmer for English (en) or for Dutch (nl)."""
    return dutch(word) if lang == "nl" else porter(word)
