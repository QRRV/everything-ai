# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""NTREX-128, by Christian Federmann, Tom Kocmi and Ying Xin (2022): the 1,997 English sentences of news of the test
set of WMT19, in 123 documents, translated by people into 128 languages, line by line, so that line i is the same
sentence in every language. The set is under CC BY-SA 4.0 at github.com/MicrosoftTranslator/NTREX; the English
sentences come from news sites, so the course hosts none of it (course/route/bronnen.yaml, ntrex-128):
everything_ai.hub downloads the files once, from a fixed commit, and keeps them only when their SHA-256 is the pinned
one. The course pins the English source and the Dutch whole, and the first 100 lines of 46 other languages. `sentences`
reads them as text: nothing in them is ever run."""
from everything_ai import hub

REPO = "github.com/MicrosoftTranslator/NTREX"
FIRST = 100   # the lines the course pins of every language but English and Dutch

# Every language the course pins: its code in NTREX, its name in English and in Dutch, whether it is written in the
# Latin alphabet, and whether a space comes between its words (Chinese, Japanese, Thai, Lao, Khmer and Tibetan write
# words without one).
LANGUAGES = {
    "eng": ("English", "Engels", True, True),
    "nld": ("Dutch", "Nederlands", True, True),
    "afr": ("Afrikaans", "Afrikaans", True, True),
    "amh": ("Amharic", "Amhaars", False, True),
    "arb": ("Arabic", "Arabisch", False, True),
    "ben": ("Bengali", "Bengaals", False, True),
    "bod": ("Tibetan", "Tibetaans", False, False),
    "cym": ("Welsh", "Welsh", True, True),
    "deu": ("German", "Duits", True, True),
    "ell": ("Greek", "Grieks", False, True),
    "eus": ("Basque", "Baskisch", True, True),
    "ewe": ("Ewe", "Ewe", True, True),
    "fao": ("Faroese", "Faeröers", True, True),
    "fas": ("Persian", "Perzisch", False, True),
    "fin": ("Finnish", "Fins", True, True),
    "fra": ("French", "Frans", True, True),
    "gle": ("Irish", "Iers", True, True),
    "hau": ("Hausa", "Hausa", True, True),
    "heb": ("Hebrew", "Hebreeuws", False, True),
    "hin": ("Hindi", "Hindi", False, True),
    "hun": ("Hungarian", "Hongaars", True, True),
    "hye": ("Armenian", "Armeens", False, True),
    "ind": ("Indonesian", "Indonesisch", True, True),
    "isl": ("Icelandic", "IJslands", True, True),
    "jpn": ("Japanese", "Japans", False, False),
    "kat": ("Georgian", "Georgisch", False, True),
    "khm": ("Khmer", "Khmer", False, False),
    "kor": ("Korean", "Koreaans", False, True),
    "lao": ("Lao", "Laotiaans", False, False),
    "ltz": ("Luxembourgish", "Luxemburgs", True, True),
    "mri": ("Maori", "Maori", True, True),
    "pol": ("Polish", "Pools", True, True),
    "rus": ("Russian", "Russisch", False, True),
    "smo": ("Samoan", "Samoaans", True, True),
    "som": ("Somali", "Somalisch", True, True),
    "spa": ("Spanish", "Spaans", True, True),
    "swa": ("Swahili", "Swahili", True, True),
    "tam": ("Tamil", "Tamil", False, True),
    "tel": ("Telugu", "Telugu", False, True),
    "tha": ("Thai", "Thai", False, False),
    "tir": ("Tigrinya", "Tigrinya", False, True),
    "tur": ("Turkish", "Turks", True, True),
    "urd": ("Urdu", "Urdu", False, True),
    "vie": ("Vietnamese", "Vietnamees", True, True),
    "wol": ("Wolof", "Wolof", True, True),
    "yor": ("Yoruba", "Yoruba", True, True),
    "zho-CN": ("Chinese", "Chinees", False, False),
    "zul": ("Zulu", "Zoeloe", True, True),
}


def file(code: str) -> tuple[str, str]:
    """The pinned file of a language, as (repo, filename) in everything_ai.hub.PINNED."""
    return REPO, "newstest2019-src.eng.txt" if code == "eng" else f"newstest2019-ref.{code}.txt"


FILES = [file(code) for code in LANGUAGES]


def name(code: str, lang: str) -> str:
    """The name of a language, in English (`lang` "en") or in Dutch ("nl")."""
    return LANGUAGES[code][0 if lang == "en" else 1]


def ready() -> bool:
    """Whether every pinned file is downloaded."""
    return all(hub.pinned(*pin) is not None for pin in FILES)


def sentences(code: str, n: int = FIRST) -> list[str]:
    """The first `n` sentences of a language, at most 100 for a language other than English and Dutch, as a new list
    of strings without their line ends. FileNotFoundError when the file is not downloaded yet: hub.download(*file(code))
    gets it."""
    if n > FIRST and code not in ("eng", "nld"):
        raise ValueError(f"the course pins only the first {FIRST} lines of {code}")
    path = hub.pinned(*file(code))
    if path is None:
        raise FileNotFoundError(f"NTREX is not downloaded yet: everything_ai.hub.download(*everything_ai.ntrex.file({code!r}))")
    return path.read_bytes().decode("utf-8").splitlines()[:n]
