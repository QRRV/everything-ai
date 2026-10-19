# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson tok-byte-level: train_byte_bpe, byte pair encoding on the UTF-8 bytes of a text, with
new ids from 256 on; and decode, which turns ids back into text through the bytes of every id. Both are saved in the
learner's project (work/projects/llm/byte_bpe.py and byte_decode.py) when right; the demo trains the learner's
byte-level BPE on Tiny Shakespeare and sends Dutch text with an emoji through it and back."""
import re
from functools import cache

from everything_ai import draw, paths, projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

# train_byte_bpe imports pair_counts from the learner's bpe.py of tok-bpe, which imports their merge; where those
# are missing or broken, the course's versions stand in (template 1.3).
projects.builds_on(("llm", "merge", "tok-bpe", "merge"), ("llm", "bpe", "tok-bpe", "train-bpe"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
TRAIN_CHARS, MERGES, TEST_CHARS = 20_000, 100, 5_000
LINES = ["To be, or not to be, that is the question", "Één ijsje 🍦 voor mij!"]


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def merge_ids(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i = i + 2
        else:
            out.append(ids[i])
            i = i + 1
    return out


def encode(text_, merges, merge=merge_ids):
    """The ids of a text: its bytes in UTF-8, with every merge rule applied in the order it was learned."""
    ids = list(text_.encode("utf-8"))
    for pair in merges:
        ids = merge(ids, pair, merges[pair])
    return ids


def bytes_picture(labels: dict):
    """One short text as characters, and as its bytes in UTF-8, each a number in a box: café and the space take 6
    bytes, é two of them, and the emoji 4, so 6 characters are 10 ids before any merge. Ink only: tokens are input."""
    word, emoji = "café ", "🙂"
    rows = [(labels["chars"].format(k=len(word + emoji)), [(ch, None) for ch in word + emoji]),
            (labels["word"].format(k=len(word.encode("utf-8"))), [(str(b), None) for b in word.encode("utf-8")]),
            (labels["emoji"].format(k=len(emoji.encode("utf-8"))), [(str(b), None) for b in emoji.encode("utf-8")])]
    return draw.token_rows(rows, title=labels["title"], alt=labels["alt"])


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:.{digits}f}"
    return text_.replace(".", ",") if lang == "nl" else text_


# The lesson is done when these are correct. "vocab-size" is the choice question in the lesson.
REQUIRED = ["vocab-size", "train-byte-bpe", "decode"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"

# --- exercise 2: train_byte_bpe --------------------------------------------------------------------------------------------

_T_HEAD = ("from llm.bpe import pair_counts\n\n\ndef merge_ids(ids, pair, new_id):\n    out = []\n    i = 0\n"
           "    while i < len(ids):\n        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:\n"
           "            out.append(new_id)\n            i = i + 2\n        else:\n            out.append(ids[i])\n"
           "            i = i + 1\n    return out\n\n\ndef train_byte_bpe(text, n):\n")
_T_IDS = "    ids = list(text.encode(\"utf-8\"))\n"
_T_LOOP = ("    merges = {}\n    for step in range(n):\n        counts = pair_counts(ids)\n        best = None\n"
           "        most = 0\n        for pair in counts:\n            if counts[pair] > most:\n                best = pair\n                most = counts[pair]\n")
_T_KEEP = "        new_id = 256 + step\n        merges[best] = new_id\n        ids = merge_ids(ids, best, new_id)\n"
_T_START = (_T_HEAD + "    # the bytes of the text in UTF-8, as numbers: the first ids\n    ids = ...\n" + _T_LOOP
            + "        # the next new id, 256 for the first rule; keep the rule in merges, under the pair best, and merge it everywhere\n"
            "        ...\n    return merges\n")


def _t(ids: str = _T_IDS, keep: str = _T_KEEP, loop: str = _T_LOOP) -> str:
    return _T_HEAD + ids + loop + keep + "    return merges\n"


# --- exercise 3: decode -------------------------------------------------------------------------------------------------------

_D_HEAD = ("def decode(ids, merges):\n    vocab = {}\n    for i in range(256):\n        vocab[i] = bytes([i])\n"
           "    for pair in merges:\n")
_D_VOCAB = "        vocab[merges[pair]] = vocab[pair[0]] + vocab[pair[1]]\n"
_D_JOIN = "    data = b\"\"\n    for i in ids:\n        data = data + vocab[i]\n"
_D_BACK = "    return data.decode(\"utf-8\", errors=\"replace\")\n"
_D_START = (_D_HEAD + "        # the bytes of the new id: those of the two halves of its pair, one after the other\n        ...\n"
            + _D_JOIN + "    # back to text, with � for bytes that are not a whole character\n    return ...\n")


def _d(vocab: str = _D_VOCAB, join: str = _D_JOIN, back: str = _D_BACK) -> str:
    return _D_HEAD + vocab + join + back


EXERCISES = {
    "train-byte-bpe": FunctionBox(
        name="train-byte-bpe",
        function="train_byte_bpe",
        start=_T_START,
        solution=_t(),
        visible=[("naïve naïve", 4)],
        hidden=[("low lower lowest", 3), ("ééé", 2), ("🙂🙂", 3), ("aaabdaaabac", 3), ("To be, or not to be", 5)],
        python_values=True,
        save_as=("llm", "byte_bpe"),
        first=[_DOTS],
        mistakes=[
            Mistake("chars", wrong=_t(ids="    ids = list(text)\n"),
                    en=_CALL + ": your ids are the characters of the text, not its bytes. Turn the text into bytes first, and those into numbers: `list(text.encode(\"utf-8\"))`.",
                    nl=_CALL_NL + ": je ids zijn de tekens van de tekst, niet zijn bytes. Maak eerst bytes van de tekst, en daarvan getallen: `list(text.encode(\"utf-8\"))`."),
            Mistake("code-points", wrong=_t(ids="    ids = [ord(c) for c in text]\n"),
                    en=_CALL + ": your ids are code points, such as 239 for ï, not bytes. A code point can be far above 255; the bytes of UTF-8 never are. `list(text.encode(\"utf-8\"))`.",
                    nl=_CALL_NL + ": je ids zijn code points, zoals 239 voor ï, geen bytes. Een code point kan ver boven 255 liggen; de bytes van UTF-8 nooit. `list(text.encode(\"utf-8\"))`."),
            Mistake("from-zero", wrong=_t(keep=_T_KEEP.replace("256 + step", "step")),
                    en=_CALL + ": your new ids start at 0, but 0 to 255 are the bytes. A new id comes after them: `256 + step`.",
                    nl=_CALL_NL + ": je nieuwe ids beginnen bij 0, maar 0 tot en met 255 zijn de bytes. Een nieuw id komt erna: `256 + step`."),
            Mistake("one-id", wrong=_t(keep=_T_KEEP.replace("256 + step", "256")),
                    en=_CALL + ": every rule gets the same new id, 256. Each new token needs an id of its own: `256 + step`.",
                    nl=_CALL_NL + ": elke regel krijgt hetzelfde nieuwe id, 256. Elk nieuw token heeft een eigen id nodig: `256 + step`."),
            Mistake("same-pair", wrong=_t(keep="        new_id = 256 + step\n        merges[best] = new_id\n"),
                    en=_CALL + ": every round finds the same pair. Merge it in the ids and keep the result, `ids = merge_ids(ids, best, new_id)`, so the next round counts the pairs of the new ids.",
                    nl=_CALL_NL + ": elke ronde vindt hetzelfde paar. Voeg het samen in de ids en bewaar de uitkomst, `ids = merge_ids(ids, best, new_id)`, zodat de volgende ronde de paren van de nieuwe ids telt."),
            Mistake("append", error=lambda e: isinstance(e, AttributeError) and "'dict' object has no attribute 'append'" in str(e),
                    en="Your function stops: `merges` is a dictionary now, not a list as in `train_bpe`, and a dictionary has no `append`. Keep each rule under its pair, with its new id as the value.",
                    nl="Je functie stopt: `merges` is nu een dictionary, geen lijst zoals in `train_bpe`, en een dictionary heeft geen `append`. Bewaar elke regel onder zijn paar, met zijn nieuwe id als waarde."),
            Mistake("id-key", wrong=_t(keep="        new_id = 256 + step\n        merges[new_id] = best\n        ids = merge_ids(ids, best, new_id)\n"),
                    en=_CALL + ": the keys of your `merges` are the new ids. A tokenizer looks up a pair it sees, to find what it becomes: make the pair the key, `merges[best] = new_id`.",
                    nl=_CALL_NL + ": de keys van je `merges` zijn de nieuwe ids. Een tokenizer zoekt een paar op dat hij ziet, om te vinden wat het wordt: maak het paar de key, `merges[best] = new_id`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the bytes of the text as a list of numbers; then the new id, the rule in `merges`, and the merge in `ids`.",
                      "nl": "Vervang de twee `...`: de bytes van de tekst als lijst van getallen; dan het nieuwe id, de regel in `merges`, en de merge in `ids`."},
            "wrong": {"en": _CALL + ". The ids are the bytes in UTF-8; every round, the best pair gets id `256 + step` in `merges`, and is merged in `ids`: steps 1 and 2 of the recipe?",
                      "nl": _CALL_NL + ". De ids zijn de bytes in UTF-8; elke ronde krijgt het beste paar id `256 + step` in `merges`, en wordt samengevoegd in `ids`: stap 1 en 2 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `text`, `n` and `step`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `text`, `n` en `step`, en geen vaste getallen?"},
            "correct": {"en": "Your function learns merge rules on bytes: the ï of \"naïve\" is two bytes, 195 and 175, and after 4 rules, id 259 stands for \"naïv\". It is in your project now, as `{saved}`.",
                        "nl": "Je functie leert merge rules op bytes: de ï van \"naïve\" bestaat uit de bytes 195 en 175, en na 4 regels staat id 259 voor \"naïv\". Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `train_byte_bpe` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `train_byte_bpe` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "decode": FunctionBox(
        name="decode",
        function="decode",
        start=_D_START,
        solution=_d(),
        visible=[([257, 256, 111], {(108, 108): 256, (104, 101): 257})],
        hidden=[([257, 32, 256], {(104, 105): 256, (256, 33): 257}), ([256, 256], {(195, 169): 256}), ([195], {}),
                ([], {}), ([258, 32, 104], {(240, 159): 256, (256, 153): 257, (257, 130): 258})],
        python_values=True,
        save_as=("llm", "byte_decode"),
        first=[_DOTS],
        mistakes=[
            Mistake("strict", error=lambda e: isinstance(e, UnicodeDecodeError),
                    en="Your function stops with a UnicodeDecodeError: some ids are only part of a character, such as the byte 195 without the byte after it. Decode with `errors=\"replace\"`, so such bytes become � instead of an error.",
                    nl="Je functie stopt met een UnicodeDecodeError: sommige ids zijn maar een deel van een teken, zoals de byte 195 zonder de byte erna. Decodeer met `errors=\"replace\"`, zodat zulke bytes � worden in plaats van een fout."),
            Mistake("added", error=lambda e: isinstance(e, TypeError) and ("can't concat int to bytes" in str(e) or "'int' and 'bytes'" in str(e)),
                    en="Your function stops: it tries to put a number and bytes together with `+`. A new id stands for the bytes of both halves of its pair: look them up in `vocab`, `vocab[pair[0]] + vocab[pair[1]]`, and do not add the ids themselves.",
                    nl="Je functie stopt: hij probeert een getal en bytes aan elkaar te zetten met `+`. Een nieuw id staat voor de bytes van beide helften van zijn paar: zoek ze op in `vocab`, `vocab[pair[0]] + vocab[pair[1]]`, en tel de ids zelf niet op."),
            Mistake("pair-key", error=lambda e: isinstance(e, KeyError) and isinstance(e.args[0], int) and e.args[0] >= 256,
                    en="Your function stops with a KeyError: it looks up a new id that has no bytes in `vocab`. Does every new id get its bytes under the new id itself, `vocab[merges[pair]]`, in the order the rules were learned? `ids` holds new ids, not pairs.",
                    nl="Je functie stopt met een KeyError: hij zoekt een nieuw id op dat geen bytes in `vocab` heeft. Krijgt elk nieuw id zijn bytes onder het nieuwe id zelf, `vocab[merges[pair]]`, in de volgorde waarin de regels geleerd zijn? `ids` bevat nieuwe ids, geen paren."),
            Mistake("raw-bytes", error=lambda e: isinstance(e, ValueError) and "range(0, 256)" in str(e),
                    en="Your function stops: `bytes(...)` only takes numbers up to 255, and a pair can hold new ids from 256 on. Take the bytes of each half from `vocab`: `vocab[pair[0]] + vocab[pair[1]]`.",
                    nl="Je functie stopt: `bytes(...)` neemt alleen getallen tot en met 255, en een paar kan nieuwe ids vanaf 256 bevatten. Neem de bytes van elke helft uit `vocab`: `vocab[pair[0]] + vocab[pair[1]]`."),
            Mistake("bytes-count", wrong=_d(vocab="        vocab[merges[pair]] = bytes(pair[0]) + bytes(pair[1])\n"),
                    en="With `{call}`, your text is full of invisible characters: `bytes(104)` is not the byte 104 but 104 zero bytes. The bytes of a new id are those of both halves of its pair: look them up in `vocab`.",
                    nl="Met `{call}` zit je tekst vol onzichtbare tekens: `bytes(104)` is niet de byte 104 maar 104 nul-bytes. De bytes van een nieuw id zijn die van beide helften van zijn paar: zoek ze op in `vocab`."),
            Mistake("bytes-out", wrong=_d(back="    return data\n"),
                    en=_CALL + ": your function gives bytes, not text. Turn them into text: `data.decode(\"utf-8\", errors=\"replace\")`.",
                    nl=_CALL_NL + ": je functie geeft bytes, geen tekst. Maak er tekst van: `data.decode(\"utf-8\", errors=\"replace\")`."),
            Mistake("latin", wrong=_d(back="    return data.decode(\"latin-1\")\n"),
                    en=_CALL + ": your function makes a character of every byte on its own, as Latin-1 does, so a character of more than one byte turns into garbled text. The bytes are UTF-8: decode them all together, in UTF-8.",
                    nl=_CALL_NL + ": je functie maakt van elke byte apart een teken, zoals Latin-1 doet, dus een teken van meer dan één byte wordt onleesbaar. De bytes zijn UTF-8: decodeer ze samen, in UTF-8."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the bytes of every new id in `vocab`; then the text of `data`, with `errors=\"replace\"`.",
                      "nl": "Vervang de twee `...`: de bytes van elk nieuw id in `vocab`; dan de tekst van `data`, met `errors=\"replace\"`."},
            "wrong": {"en": _CALL + ". Every new id gets the bytes of both halves of its pair, from `vocab`, and the bytes go back to text in UTF-8: step 4 of the recipe?",
                      "nl": _CALL_NL + ". Elk nieuw id krijgt de bytes van beide helften van zijn paar, uit `vocab`, en de bytes gaan in UTF-8 terug naar tekst: stap 4 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function work for any merges, also when a pair holds a new id, and for bytes that are only part of a character?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Werkt je functie voor alle merges, ook als een paar een nieuw id bevat, en voor bytes die maar een deel van een teken zijn?"},
            "correct": {"en": "Your function turns any ids back into text: 257, 256 and 111 are \"he\", \"ll\" and \"o\", and the byte 195 alone becomes �. It is in your project now, as `{saved}`.",
                        "nl": "Je functie maakt van alle ids weer tekst: 257, 256 en 111 zijn \"he\", \"ll\" en \"o\", en de byte 195 alleen wordt �. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `decode` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `decode` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: the learner's byte-level BPE on Tiny Shakespeare, and two lines there and back ---------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when byte_bpe.py or byte_decode.py is not in the project yet; "bpe" when they stop with an
    error; "slow" when training takes too long; else a dict with, for each line of LINES, its bytes, its ids and
    whether decode gives the line back, the bytes per id on 5,000 characters it did not learn from, and the modules
    the course's versions stood in for."""
    if not (_saved("byte_bpe") and _saved("byte_decode")):
        return None
    run = run_code("from llm.byte_bpe import merge_ids, train_byte_bpe\nfrom llm.byte_decode import decode")
    if run.error is not None:
        return "bpe"
    merge, train, decode = run.names["merge_ids"], run.names["train_byte_bpe"], run.names["decode"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def work():
        merges = dict(train(text()[:TRAIN_CHARS], MERGES))
        lines = []
        for line in LINES:
            ids = encode(line, merges, merge)
            lines.append({"bytes": len(line.encode("utf-8")), "ids": ids, "back": decode(ids, merges) == line})
        test = text()[TRAIN_CHARS:TRAIN_CHARS + TEST_CHARS]
        return {"lines": lines, "per_id": len(test.encode("utf-8")) / len(encode(test, merges, merge))}

    try:
        out = with_time_limit(work, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "bpe"
    out["stand_ins"] = stand_ins
    return out


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("train-byte-bpe", _T_START, "empty", None),
    ("train-byte-bpe", _t(), "correct", None),
    ("train-byte-bpe", _t(ids="    data = text.encode(\"utf-8\")\n    ids = []\n    for b in data:\n        ids.append(b)\n",
                          keep="        merges[best] = 256 + step\n        ids = merge_ids(ids, best, 256 + step)\n"), "correct", None),
    ("train-byte-bpe", _t(ids="    ids = list(text)\n"), "wrong", "chars"),
    ("train-byte-bpe", _t(ids="    ids = [ord(c) for c in text]\n"), "wrong", "code-points"),
    ("train-byte-bpe", _t(keep=_T_KEEP.replace("256 + step", "step")), "wrong", "from-zero"),
    ("train-byte-bpe", _t(keep=_T_KEEP.replace("256 + step", "256")), "wrong", "one-id"),
    ("train-byte-bpe", _t(keep="        new_id = 256 + step\n        merges[best] = new_id\n"), "wrong", "same-pair"),
    ("train-byte-bpe", _t(keep="        new_id = 256 + step\n        merges[new_id] = best\n        ids = merge_ids(ids, best, new_id)\n"), "wrong", "id-key"),
    ("train-byte-bpe", _t(keep="        new_id = 256 + step\n        merges.append(best)\n        ids = merge_ids(ids, best, new_id)\n"), "error", "append"),
    ("train-byte-bpe", _t().replace("for step in range(n):", "for step in range(4):"), "almost", None),
    ("decode", _D_START, "empty", None),
    ("decode", _d(), "correct", None),
    ("decode", _d(join="    data = b\"\".join([vocab[i] for i in ids])\n"), "correct", None),
    ("decode", _d(back="    return data.decode(\"utf-8\")\n"), "error", "strict"),
    ("decode", _d(vocab="        vocab[merges[pair]] = pair[0] + pair[1]\n"), "error", "added"),
    ("decode", _d(vocab="        vocab[pair] = vocab[pair[0]] + vocab[pair[1]]\n"), "error", "pair-key"),
    ("decode", _d(vocab="        vocab[merges[pair]] = bytes(pair)\n"), "error", "raw-bytes"),
    ("decode", _d(back="    return data\n"), "wrong", "bytes-out"),
    ("decode", _d(vocab="        vocab[merges[pair]] = bytes(pair[0]) + bytes(pair[1])\n"), "wrong", "bytes-count"),
    ("decode", _d(vocab="        vocab[merges[pair]] = vocab[pair]\n"), "error", None),
    ("decode", _d(vocab="        vocab[merges[pair]] = bytes(pair[0], pair[1])\n"), "error", None),
    ("decode", _d(back="    return \"\".join([chr(b) for b in data])\n"), "wrong", "latin"),
    ("decode", _d(back="    return data.decode(\"latin-1\")\n"), "wrong", "latin"),
    ("decode", _d(back="    return data.decode(\"utf-8\", errors=\"ignore\")\n"), "almost", None),
]
