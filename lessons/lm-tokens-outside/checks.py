# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-tokens-outside: pieces, the tokens of a text as pieces of text; and tokens_per_word, how
many tokens a text costs per word. Both use the tokenizer of GPT-2, through tiktoken, from the course's own copy of
its two files (data/gpt2-tokenizer/), so that they work without internet."""
import re

from everything_ai import draw, gpt2_tokenizer
from everything_ai.check import FunctionBox, Mistake, same_value

gpt2_tokenizer.offline()   # before any tiktoken.get_encoding("gpt2"), also in the lesson's own cells

import tiktoken  # noqa: E402  (after offline(): tiktoken reads its cache folder when it loads a file)

ENC = tiktoken.get_encoding("gpt2")
EN = "Tokenization is surprisingly important."
NL = "Tokenisatie is verrassend belangrijk."


def tokens_picture(labels: dict) -> str:
    """The guess's two sentences cut into GPT-2's tokens, with the token ID of each under it."""
    rows = []
    for key, text in (("en", EN), ("nl", NL)):
        ids = ENC.encode(text)
        rows.append((labels[key].format(n=len(ids)), [(ENC.decode([i]), i) for i in ids]))
    return draw.token_rows(rows, title=labels["title"], alt=labels["alt"]).text


# The lesson is done when these are correct. "space" is the choice question in the lesson.
REQUIRED = ["space", "pieces", "tokens-per-word"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_HEAD = "import tiktoken\n\nenc = tiktoken.get_encoding(\"gpt2\")\n\n\n"

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# --- exercise 2: pieces ---------------------------------------------------------------------------------------------------

_P_START = _HEAD + "def pieces(text):\n    ids = ...\n    out = []\n    for i in ids:\n        out.append(...)\n    return out\n"


def _p(ids: str = "enc.encode(text)", piece: str = "enc.decode([i])") -> str:
    return _HEAD + f"def pieces(text):\n    ids = {ids}\n    out = []\n    for i in ids:\n        out.append({piece})\n    return out\n"


_P = _p()

# --- exercise 3: tokens_per_word ------------------------------------------------------------------------------------------

_W_START = _HEAD + "def tokens_per_word(text):\n    n_tokens = ...\n    n_words = ...\n    return n_tokens / n_words\n"


def _w(tokens: str = "len(enc.encode(text))", words: str = "len(text.split())", ret: str = "n_tokens / n_words") -> str:
    return _HEAD + f"def tokens_per_word(text):\n    n_tokens = {tokens}\n    n_words = {words}\n    return {ret}\n"


_W = _w()


def _close(a, b) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) < 1e-9


def _num(v):
    """A number for the messages: at most 4 decimals, without float noise. A text keeps its quotes, so that `"1.2"`
    does not look like the number 1.2."""
    if isinstance(v, float):
        return round(v, 4)
    return f'"{v}"' if isinstance(v, str) else v


EXERCISES = {
    "pieces": FunctionBox(
        name="pieces",
        function="pieces",
        start=_P_START,
        solution=_P,
        visible=[("Hello world",)],
        hidden=[("unbelievable",), ("Ik hou van grote taalmodellen",), ("the the  the",), ("",), ("🙂",)],
        compare=same_value,
        first=[_DOTS],
        mistakes=[
            Mistake("ids", wrong=_p(piece="i"),
                    en="`{call}` gives `{got}`: the token IDs themselves. Turn every ID back into its piece of text with `enc.decode`, as in step 4 of the recipe.",
                    nl="`{call}` geeft `{got}`: de token IDs zelf. Maak van elk ID weer zijn stuk tekst met `enc.decode`, zoals in stap 4 van het recept."),
            Mistake("whole", wrong=_p(piece="enc.decode(ids)"),
                    en="`{call}` gives `{got}`: the whole text again for every token. Decode only the one ID of this round, `i`, in a list of its own: `[i]`.",
                    nl="`{call}` geeft `{got}`: voor elk token weer de hele tekst. Decodeer alleen het ene ID van deze ronde, `i`, in een eigen lijst: `[i]`."),
            Mistake("characters", wrong=_p(ids="text", piece="i"),
                    en="`{call}` gives `{got}`: one piece for every character, as in the lesson before. Let the tokenizer cut the text: `enc.encode(text)` gives the token IDs.",
                    nl="`{call}` geeft `{got}`: één stuk per teken, zoals in de vorige les. Laat de tokenizer de tekst opknippen: `enc.encode(text)` geeft de token IDs."),
            Mistake("no-list", error=lambda e: isinstance(e, TypeError) and "Sequence" in str(e),
                    en="With `{call}`, `enc.decode` gets one number, and it wants a list of IDs, also for one ID. Put the ID in a list of its own: `enc.decode([i])`.",
                    nl="Met `{call}` krijgt `enc.decode` één getal, en hij wil een lijst IDs, ook voor één ID. Zet het ID in een eigen lijst: `enc.decode([i])`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the token IDs of the text, and for every ID its piece of text.",
                      "nl": "Vervang de twee `...`: de token IDs van de tekst, en voor elk ID zijn stuk tekst."},
            "arguments": {"en": "The check calls `{call}`, with one text, and the `def` line of your function has another number of names between its parentheses. Keep `(text)`.",
                          "nl": "De check roept `{call}` aan, met één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 2 and 4 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 2 en 4 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every text?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke tekst?"},
            "correct": {"en": "Your function shows how GPT-2 cuts any text: \"unbelievable\" becomes `['un', 'bel', 'iev', 'able']`, and of two spaces in a row, the first is a token of its own, `' '`. An empty text has no tokens.",
                        "nl": "Je functie laat zien hoe GPT-2 elke tekst opknipt: \"unbelievable\" wordt `['un', 'bel', 'iev', 'able']`, en van twee spaties na elkaar is de eerste een eigen token, `' '`. Een lege tekst heeft geen tokens."},
        },
    ),
    "tokens-per-word": FunctionBox(
        name="tokens-per-word",
        function="tokens_per_word",
        start=_W_START,
        solution=_W,
        visible=[("The weather is nice today.",)],
        hidden=[("Het weer is vandaag mooi.",), (EN,), (NL,), ("unbelievable",)],
        compare=_close,
        show=_num,
        first=[_DOTS],
        mistakes=[
            Mistake("characters", wrong=_w(tokens="len(text)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count characters, not tokens. The tokens are the IDs that `enc.encode(text)` gives.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt tekens, geen tokens. De tokens zijn de IDs die `enc.encode(text)` geeft."),
            Mistake("letters-as-words", wrong=_w(words="len(text)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the number of characters, not of words. `text.split()` gives the words.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door het aantal tekens, niet door het aantal woorden. `text.split()` geeft de woorden."),
            Mistake("spaces", wrong=_w(words="text.count(\" \")"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count the spaces, and a sentence has one space fewer than words. `text.split()` gives the words themselves.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de spaties, en een zin heeft één spatie minder dan woorden. `text.split()` geeft de woorden zelf."),
            Mistake("upside-down", wrong=_w(ret="n_words / n_tokens"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you give words per token. Tokens per word is the number of tokens divided by the number of words.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je geeft woorden per token. Tokens per woord is het aantal tokens gedeeld door het aantal woorden."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the number of tokens of the text, and its number of words.",
                      "nl": "Vervang de twee `...`: het aantal tokens van de tekst, en het aantal woorden."},
            "arguments": {"en": "The check calls `{call}`, with one text, and the `def` line of your function has another number of names between its parentheses. Keep `(text)`.",
                          "nl": "De check roept `{call}` aan, met één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Count the tokens as in step 3 of the recipe, the words with `text.split()`, and divide the one by the other with `/`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel de tokens zoals in stap 3 van het recept, de woorden met `text.split()`, en deel het ene door het andere met `/`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function count for every text?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt je functie voor elke tekst?"},
            "correct": {"en": "For \"The weather is nice today.\" your function gives 1.2 tokens per word, and for \"Het weer is vandaag mooi.\" 2.4: the same sentence costs twice as many tokens in Dutch. For the sentences of the guess, it is 1.5 and 3.75.",
                        "nl": "Voor \"The weather is nice today.\" geeft je functie 1,2 tokens per woord, en voor \"Het weer is vandaag mooi.\" 2,4: dezelfde zin kost in het Nederlands twee keer zoveel tokens. Voor de zinnen uit de gok is het 1,5 en 3,75."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("pieces", _P_START, "empty", None),
    ("pieces", _HEAD + "def pieces(text):\n    return [enc.decode([i]) for i in enc.encode(text)]\n", "correct", None),
    ("pieces", _p(piece="i"), "wrong", "ids"),
    ("pieces", _p(piece="enc.decode(ids)"), "wrong", "whole"),
    ("pieces", _p(ids="text", piece="i"), "wrong", "characters"),
    ("pieces", _p(piece="enc.decode(i)"), "error", "no-list"),
    ("pieces", _p(ids="text.split()", piece="i"), "wrong", None),
    ("tokens-per-word", _W_START, "empty", None),
    ("tokens-per-word", _w(ret="len(enc.encode(text)) / len(text.split(\" \"))"), "correct", None),
    ("tokens-per-word", _w(tokens="len(text)"), "wrong", "characters"),
    ("tokens-per-word", _w(words="len(text)"), "wrong", "letters-as-words"),
    ("tokens-per-word", _w(words="text.count(\" \")"), "wrong", "spaces"),
    ("tokens-per-word", _w(ret="n_words / n_tokens"), "wrong", "upside-down"),
    ("tokens-per-word", _w(words="5"), "almost", None),
    ("tokens-per-word", _w(ret="str(n_tokens / n_words)"), "wrong", None),
]
